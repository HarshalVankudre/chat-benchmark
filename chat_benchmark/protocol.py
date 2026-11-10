"""Fixed-length sweeps: what to generate next, when to stop, and what was proven."""
from __future__ import annotations
import time
from statistics import median
from .estimate import theil_sen
from .workload import rows_for, reference_digest

# Stages are nominal seconds; each maps to one fixed reference length for every model.
# `precise` repeats the lengths in mirrored order so slow drift cancels out.
PRESETS = {
    "quick": dict(durations=[5, 10], warmup=0, cap=60, words=10000),
    "standard": dict(durations=[10, 30, 60], warmup=5, cap=180, words=10000),
    "precise": dict(durations=[10, 30, 60, 60, 30, 10], warmup=5, cap=480, words=16000),
}
# Shorter fixed stages that may stand in for one that no longer fits the wall budget.
FALLBACK_SECONDS = (45, 40, 30, 25, 20, 15)
VALID_STRUCTURES = ("single_text_block", "single_file_payload", "unverified_file_payload")
SYNTHESIZED = ("not literal", "implausible token count")


def plan(preset="standard", durations=None, warmup_seconds=None, max_seconds=None, adaptive=True, **_):
    base = PRESETS.get(preset) or PRESETS["standard"]
    name = preset if preset in PRESETS else "standard"
    targets = list(durations if durations is not None else base["durations"])
    warmup = warmup_seconds if warmup_seconds is not None else base["warmup"]
    cap = max_seconds if max_seconds is not None else base["cap"]
    if not targets or len(targets) > 12 or any(not 2 <= t <= 90 for t in targets):
        raise ValueError("Use 1-12 stage targets, each between 2 and 90 seconds.")
    if not 0 <= warmup <= 10 or not 10 <= cap <= 900:
        raise ValueError("Warm-up must be 0-10 seconds and total wall budget 10-900 seconds.")
    if sum(targets) + warmup >= cap:
        raise ValueError("Stage targets plus warm-up must leave room for measurements in the total budget.")
    words = max(base["words"], (sum(rows_for(t) for t in targets) + rows_for(warmup or 2)) * 9 + 500)
    return dict(preset=name, mode="temporary_file", durations_seconds=targets,
                warmup_seconds=warmup, warmups=1 if warmup else 0, min_samples=len(targets), max_samples=len(targets),
                max_wall_seconds=cap, max_total_words=words, max_words_per_burst=6000, adaptive=bool(adaptive))


def stage_rows(seconds):
    return rows_for(seconds)


def stage_target(run, index):
    override = (run.get("stage_overrides") or {}).get(str(index))
    if override is not None:
        return override
    recorded = (run.get("samples") or {}).get(str(index), {}).get("target_seconds")
    if recorded is not None:
        return recorded
    p = run["plan"]
    return p["warmup_seconds"] if index <= p["warmups"] else p["durations_seconds"][index - p["warmups"] - 1]


def _predictor(requests):
    """Seconds a stage of `rows` should take, from this run's own samples."""
    usable = [r for r in requests if r.status == "completed" and r.reference_rows and r.duration_ms and r.duration_ms > 0]
    line = theil_sen([(r.reference_rows, r.duration_ms / 1000) for r in usable])
    if line and line[0] > 0:
        slope, intercept = line
        return (lambda rows: max(0.0, intercept) + slope * rows), "this run's length sweep (overhead + rows / speed)"
    rates = [r.visible_wps for r in requests if r.visible_wps and r.continuity != "interrupted"]
    fallback = [(r.payload_words or r.visible_words) * 1000 / r.duration_ms for r in requests
                if (r.payload_words or r.visible_words) and r.duration_ms and r.duration_ms > 0]
    wps = median(rates[-2:]) if rates else (median(fallback[-2:]) if fallback else 50)
    source = ("native text-item words/second" if rates else
              "request words/second (includes initial thinking)" if fallback else "initial 50 words/second estimate")
    return (lambda rows: rows * 9 / wps), source


def progress(run, requests, now_ms=None):
    p = run["plan"]
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    captured = len(run.get("samples", {}))
    elapsed = max(0, (now_ms - run["started_ms"]) / 1000)
    measured = [r for r in requests if r.phase != "warmup"]
    action, reason = "burst", "sampling"
    reference = dict(run.get("settings") or {})
    for r in requests:
        for key, value in r.settings().items():
            if reference.get(key) is None and value is not None:
                reference[key] = value
    drift = any(any(value is not None and getattr(r, key, None) is not None and getattr(r, key) != value
                    for key, value in reference.items()) for r in requests)
    words_used = sum(r.payload_words if r.payload_words is not None else (r.visible_words or 0) for r in requests)
    if drift:
        action, reason = "stop", "settings_changed"
    elif len({r.context_epoch for r in requests}) > 1:
        action, reason = "stop", "context_compacted"
    elif any(r.continuity == "interrupted" for r in requests):
        action, reason = "stop", "interrupted_output"
    elif any(r.reference_matches is False for r in requests):
        action, reason = "stop", "reference_mismatch"
    elif any(r.provenance in SYNTHESIZED for r in requests):
        action, reason = "stop", "synthesized_output"
    elif captured > len(requests):
        last = max((s.get("captured_ms", now_ms) for s in run["samples"].values()), default=now_ms)
        action, reason = ("stop", "missing_usage") if now_ms - last > 15000 else ("wait_for_usage", "awaiting_usage")
    elif len(measured) >= p["min_samples"]:
        action, reason = "stop", "stages_complete"
    elif elapsed >= p["max_wall_seconds"]:
        action, reason = "stop", "wall_time_cap"
    elif words_used >= p["max_total_words"]:
        action, reason = "stop", "word_budget"
    index = captured + 1
    target = stage_target(run, index) if action == "burst" else None
    predict, calibration = _predictor(requests)
    remaining_words = max(0, p["max_total_words"] - words_used)
    remaining_wall = max(0, p["max_wall_seconds"] - elapsed)

    def fits(seconds):
        rows = rows_for(seconds)
        return (remaining_wall >= predict(rows) + 5 and remaining_words >= rows * 9
                and rows * 9 <= p["max_words_per_burst"])

    substituted = None
    if action == "burst" and not fits(target):
        planned, used = target, {r.target_seconds for r in requests}
        overridden = str(index) in (run.get("stage_overrides") or {})
        # Keep a third length for the sweep rather than dropping the stage outright.
        choice = next((s for s in FALLBACK_SECONDS if s < planned and s not in used and fits(s)), None) \
            if p.get("adaptive") and index > p["warmups"] and not overridden else None
        if choice is None:
            action, reason, target = "stop", "remaining_budget", None
        else:
            target, substituted = choice, planned
    rows = rows_for(target) if target else None
    words = rows * 9 if rows else None
    return dict(action="write-sample" if action == "burst" and p.get("mode") == "temporary_file" else action,
                reason=reason, next_index=index, phase="warmup" if index <= p["warmups"] else "measured",
                target_seconds=target, target_words=max(100, words) if words is not None else None,
                target_rows=rows if action == "burst" else None,
                reference_sha256=reference_digest(rows) if action == "burst" else None,
                substituted_from=substituted,
                predicted_seconds=predict(rows) if rows else None, calibration=calibration,
                warmups_captured=sum(r.phase == "warmup" for r in requests), measured_captured=len(measured),
                active_seconds=sum(r.duration_ms or 0 for r in measured) / 1000, wall_seconds=elapsed,
                words_generated=words_used, targets_met=reason == "stages_complete", plan=p)


def quality(requests, summary, state):
    reasons = []
    if not state or not state["targets_met"]:
        reasons.append("not all duration stages completed")
    if summary["settings_drift"]:
        reasons.append("active settings changed")
    if any(r.continuity not in VALID_STRUCTURES for r in requests):
        reasons.append("some output was not one uninterrupted message or file-write payload")
    if any(r.continuity == "unverified_file_payload" for r in requests):
        reasons.append("no transcript: the single tool call and literal generation are unverified")
    if any(r.reference_matches is not True for r in requests):
        reasons.append("some output did not match the fixed text reference")
    if any(r.model is None for r in requests):
        reasons.append("the active model is unknown")
    if any(r.timing_basis != "native" for r in requests):
        reasons.append("some requests have no native timing (helper-clock or session-boundary timing used)")
    if len(requests) < 10:
        reasons.append("too few observations in one sweep for a per-stage reliability claim; consult same-duration history")
    if any(r.continuity == "interrupted" for r in requests):
        reasons.append("thinking/tool/text boundaries interrupted output")
    ttft_reasons = list(reasons)
    if summary["ttft_ms"]["n"] != len(requests):
        ttft_reasons.append("native TTFT coverage is incomplete")
    return dict(effective_tps=dict(status="provisional" if reasons else "observed", reasons=reasons),
                native_ttft=dict(status="provisional" if ttft_reasons else "observed", reasons=ttft_reasons),
                native_ttft_coverage_pct=summary["ttft_ms"]["n"] * 100 / len(requests) if requests else 0,
                tail_status="descriptive_only",
                interpretation="Generation speed comes from the length sweep and carries its own confidence level (summary.generation). Per-stage figures describe single requests: compare them within a sweep and across repeated runs. Individual token gaps are not exposed.")
