"""Compare like-duration bursts over time without pooling different settings."""
from __future__ import annotations
import json
from math import floor, log
from .metrics import stats
from .standard import standard_metrics
from .sweep import end_to_end, first_token_summary, generation_summary, valid_samples


CONFIGURATION = ("provider", "model", "effort", "service_tier", "speed")
REJECTED = ("settings_changed", "context_compacted", "interrupted_output", "reference_mismatch", "synthesized_output")
NOTE = ("Each row is one client/model/settings combination. Generation speed is the median of that combination's "
        "sweep estimates in the shared o200k_base reference unit, so rows are directly comparable. "
        "End-to-end and first-token figures depend on prompt size and client overhead; compare them with care.")


def load_reports(root, current=None, last_runs=None):
    reports = []
    for path in (root / "results").glob("*/benchmark.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("workload") in ("fixed-continuous-text-v1", "fixed-temp-file-v1"):
                reports.append(report)
        except (OSError, ValueError):
            continue
    reports.sort(key=lambda r: r.get("created_at", ""))
    if current:
        reports = [r for r in reports if r.get("run_id") != current.get("run_id")] + [current]
    return reports[-last_runs:] if last_runs else reports


def run_record(report):
    """One run reduced to its headline numbers, recomputed from its verified samples."""
    if report.get("status") in REJECTED:
        return None
    samples = [{**s, **standard_metrics(s)} for s in report.get("samples", [])]
    excluded = {d["request_id"] for d in report.get("summary", {}).get("settings_drift", [])}
    generation = generation_summary(samples, excluded)
    valid = valid_samples(samples, excluded)
    if generation.get("tps") is None or not valid:
        return None
    longest, first = end_to_end(samples, excluded), first_token_summary(samples, excluded)
    settings = {key: valid[0].get(key) for key in CONFIGURATION}
    settings["provider"] = settings["provider"] or (report.get("session") or {}).get("provider")
    return {"run_id": report.get("run_id"), "at": report.get("created_at") or "", "settings": settings,
            "tps": generation["tps"], "overhead_seconds": generation["overhead_seconds"],
            "confidence": generation["confidence"]["level"], "basis": generation["basis"],
            "end_to_end_tps": longest["tps"] if longest else None,
            "first_token_ms": first["median_ms"] if first else None}


def compare(root, last_runs=200, current=None):
    """Leaderboard: every client/model/settings combination with a generation-speed estimate."""
    from .providers import provider_for
    reports = load_reports(root, current, last_runs)
    groups = {}
    for report in reports:
        record = run_record(report)
        if record:
            groups.setdefault(tuple(record["settings"].get(k) for k in CONFIGURATION), []).append(record)
    order = ("none", "low", "medium", "high")
    result = []
    for key, records in groups.items():
        settings = dict(zip(CONFIGURATION, key))
        try:
            client = provider_for(settings["provider"]).title
        except ValueError:
            client = str(settings["provider"])
        result.append({"client": client, "settings": settings, "runs": len(records),
                       "generation_tps": stats((r["tps"] for r in records), intervals=True),
                       "overhead_seconds": stats((r["overhead_seconds"] for r in records), intervals=False),
                       "end_to_end_tps": stats((r["end_to_end_tps"] for r in records), intervals=False),
                       "first_token_ms": stats((r["first_token_ms"] for r in records), intervals=False),
                       "best_confidence": max((r["confidence"] for r in records), key=order.index),
                       "timing": sorted({r["basis"] for r in records if r["basis"]}),
                       "last_run": max(r["at"] for r in records),
                       "trend": [{"at": r["at"], "tps": r["tps"], "confidence": r["confidence"]} for r in records]})
    result.sort(key=lambda g: -g["generation_tps"]["median"])
    return {"groups": result, "runs_considered": len(reports), "note": NOTE}


def configuration_history(root, report):
    """The leaderboard row matching this report's client, model and settings."""
    record = run_record(report)
    if not record:
        return None
    return next((g for g in compare(root, 200, report)["groups"]
                 if all(g["settings"].get(k) == record["settings"].get(k) for k in CONFIGURATION)), None)


def compare_markdown(result):
    if not result["groups"]:
        return "No runs with a generation-speed estimate yet. Run the benchmark in a chat first.\n"
    out = ["| # | Client | Model | Effort | Tier | Runs | Generation tok/s | Range | Overhead s | End-to-end tok/s | First token ms | Confidence | Last run |",
           "|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---|---|"]
    def f(value, digits=2):
        return "N/A" if value is None else f"{value:,.{digits}f}"
    for rank, g in enumerate(result["groups"], 1):
        s, tps = g["settings"], g["generation_tps"]
        out.append(f"| {rank} | {g['client']} | {s.get('model') or 'not exposed'} | {s.get('effort') or '—'} | "
                   f"{s.get('service_tier') or '—'} | {g['runs']} | **{f(tps['median'])}** | {f(tps['min'])}–{f(tps['max'])} | "
                   f"{f(g['overhead_seconds']['median'])} | {f(g['end_to_end_tps']['median'])} | "
                   f"{f(g['first_token_ms']['median'], 0)} | {g['best_confidence']} | {g['last_run'][:10]} |")
    out += ["", result["note"], ""]
    return "\n".join(out)


def history(root, current=None, last_runs=20):
    reports = load_reports(root, current, last_runs)
    groups, seen = {}, set()
    for report in reports:
        if report.get("status") in REJECTED:
            continue
        for sample in report.get("samples", []):
            sample = {**sample, **standard_metrics(sample)}
            identity = (report.get("run_id"), sample["request_id"])
            if identity in seen or sample.get("continuity") not in ("single_text_block", "single_file_payload", "unverified_file_payload") or sample.get("status") != "completed" or sample.get("reference_matches") is not True:
                continue
            seen.add(identity)
            tokens, cache = sample.get("input_tokens"), sample.get("cache_hit_pct")
            context_band = floor(log(tokens, 1.25)) if tokens and tokens > 0 else None
            cache_band = min(9, int(cache // 10)) if cache is not None else None
            fields = ("provider", "model", "effort", "service_tier", "speed", "client_version", "target_seconds", "reference_rows")
            key = tuple(sample.get(k) for k in fields) + (context_band, cache_band, report["workload"], sample.get("standard_tokenizer"))
            group = groups.setdefault(key, {"workload": report["workload"], "settings": {k: sample.get(k) for k in fields},
                                           "context_band_tokens": ([round(1.25 ** context_band), round(1.25 ** (context_band + 1))] if context_band is not None else None),
                                           "reference_tokenizer": sample.get("standard_tokenizer"),
                                           "cache_band_pct": ([cache_band * 10, cache_band * 10 + 10] if cache_band is not None else None), "observations": []})
            group["observations"].append({"run_id": report.get("run_id"), "at": report.get("created_at"),
                                          "tps": sample.get("effective_tps"), "standard_tps": sample.get("standard_tps"), "ttft_ms": sample.get("ttft_ms"),
                                          "text_span_ms": sample.get("text_span_ms"), "request_id": sample["request_id"]})
    result = []
    for group in groups.values():
        observations = group["observations"]
        group["distinct_runs"] = len({o["run_id"] for o in observations})
        group["effective_tps"] = stats(o["tps"] for o in observations)
        group["standard_tps"] = stats(o["standard_tps"] for o in observations)
        group["ttft_ms"] = stats(o["ttft_ms"] for o in observations)
        group["status"] = "limited_history" if len(observations) < 10 else "descriptive_history"
        result.append(group)
    return result
