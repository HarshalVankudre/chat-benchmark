"""One report schema and layout for every client."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .metrics import Request, summarize, stats
from .protocol import quality
from .standard import metadata, standard_metrics
from .sweep import DEFINITION, end_to_end, first_token, first_token_summary, generation_summary
from .workload import rows_for

FILE_WORKLOADS = ("fixed-continuous-text-v1", "fixed-temp-file-v1")
DEFINITIONS = {
    "generation_tps": DEFINITION + " Counted in the frozen o200k_base reference unit, so it is comparable across providers.",
    "overhead_seconds": "Intercept of the same fit: seconds each request spends on everything that does not grow with the output (queueing, time to first token, thinking, tool-call syntax, and with helper-clock timing the client's own latency).",
    "standard_tps": "End-to-end rate of one request: verified text re-tokenized with the frozen o200k_base reference / full model-request seconds. Same text gives the same numerator for every provider. Includes initial delay, thinking, and tool-syntax generation time, so it rises with stage length; not isolated decode speed or billable tokens.",
    "effective_tps": "Provider output tokens / full model-request seconds. Includes reasoning and tool-call output; excludes tool execution with native request timing.",
    "post_ttft_tps": "(Output tokens - 1) / (request seconds - native TTFT seconds). Only shown with zero reported reasoning tokens. Client-side derived throughput, not engine decoding speed.",
    "ttft_ms": "Native client-reported request TTFT, using that client's own timing boundary. It can precede visible answer text and is not a universal first-visible-token measurement. Never inferred from completed messages, item timestamps, clocks around tools, or characters.",
    "prestream_ms": "Derived latency before the payload: native request seconds minus the client's native tool-argument stream duration. Covers time to first token plus any thinking.",
    "bracket_ms": "Helper-clock interval from the previous helper command's exit to this helper's process start. Brackets exactly one model response plus client and tool-dispatch overhead; available for every client.",
    "first_content_ms": "Claude's separately reported time to the first content block; can include thinking and is not time to first visible text.",
    "duration_ms": "Full model request latency, including prompt processing, queue/network time, generation, and retries when exposed. Helper-clock interval when the client exposes no native timing.",
    "input_tokens": "Total input including cache reads and writes. Codex counts those within input; Claude reports them separately and they are added here.",
    "p95": "Linearly interpolated sample percentile; small runs do not estimate production tail latency reliably.",
    "reference_rows_per_second": "Verified fixed-reference rows / full request seconds. This common workload unit can be compared across providers; it includes initial delay and request overhead, not isolated decoding.",
    "payload_bytes_per_second": "Verified file payload bytes / full request seconds. Provider tokenization and tool-call syntax are not part of the numerator.",
}
BASIS_LABEL = {"native": "native request timing", "bracket": "helper-clock timing",
               "boundary": "session-boundary timing", None: "no timing"}


def fmt(value, digits=2):
    return "N/A" if value is None else f"{value:,.{digits}f}"


def label(value):
    return str(value).replace("|", "\\|") if value is not None else "not exposed"


def client_title(session):
    from .providers import provider_for
    try:
        return provider_for(session.get("provider")).title
    except ValueError:
        return str(session.get("provider"))


def standardize_report(report):
    """Also supports old verified snapshots without regenerating model output."""
    report["reference_tokenizer"] = metadata()
    for sample in report.get("samples", []) + report.get("warmups", []):
        sample.update(standard_metrics(sample))
    excluded = {d["request_id"] for d in report.get("summary", {}).get("settings_drift", [])}
    valid = [s for s in report.get("samples", []) if s.get("phase") != "warmup"
             and s.get("request_id") not in excluded and s.get("standard_tps") is not None]
    summary = report.setdefault("summary", {})
    summary["standard_tps"] = stats((s["standard_tps"] for s in valid), intervals=False)
    seconds = sum(s["duration_ms"] for s in valid) / 1000
    summary["weighted_standard_tps"] = sum(s["standard_tokens"] for s in valid) / seconds if seconds else None
    samples = report.get("samples", [])
    summary["generation"] = generation_summary(samples, excluded)
    summary["end_to_end"] = end_to_end(samples, excluded)
    summary["first_token"] = first_token_summary(samples, excluded)
    report["definitions"] = {**report.get("definitions", {}), **DEFINITIONS}
    return report


def stage_rows(report):
    """Measured stages in plan order, with unrun stages kept as empty rows."""
    samples = [s for s in report.get("samples", []) if s.get("phase") != "warmup"]
    plan = report.get("plan") or {}
    targets = plan.get("durations_seconds")
    if targets:
        indexed = {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}
        stages = []
        for i, target in enumerate(targets):
            sample = indexed.get(i + plan.get("warmups", 0) + 1)
            actual = sample.get("target_seconds") if sample and sample.get("target_seconds") is not None else target
            stages.append((actual, sample, target if actual != target else None))
        return stages
    return [(s.get("target_seconds"), s, None) for s in samples]


def sample_valid(sample, excluded):
    return (bool(sample) and sample.get("status", "completed") == "completed" and sample.get("request_id") not in excluded
            and sample.get("reference_matches") is not False and sample.get("continuity") != "interrupted")


def confidence_line(generation):
    confidence = generation.get("confidence") or {}
    level = confidence.get("level", "none")
    mark = {"high": "High", "medium": "Medium", "low": "Low", "none": "Not established"}[level]
    return mark, "; ".join(confidence.get("reasons", []))


def headline(report):
    """The short result block that leads every chat summary."""
    summary = report.get("summary") or {}
    generation = summary.get("generation") or generation_summary(report.get("samples", []))
    session, settings = report.get("session") or {}, summary.get("settings") or {}
    out = []
    if session.get("provider"):
        parts = [f"**{client_title(session)}**"]
        parts += [f"{name} `{label(settings[key])}`" for name, key in
                  (("model", "model"), ("effort", "effort"), ("tier", "service_tier"), ("speed", "speed"))
                  if settings.get(key) is not None]
        out += [" · ".join(parts), ""]
    out += ["| Result | Value |", "|---|---:|"]
    if generation.get("tps") is not None:
        out.append(f"| **Generation speed** | **{fmt(generation['tps'])} tok/s** |")
        ci, spread = generation.get("ci95"), generation.get("pair_spread_pct")
        if ci:
            out.append(f"| 95% interval | {fmt(ci['low'])}–{fmt(ci['high'])} tok/s |")
        if spread is not None:
            out.append(f"| Agreement across lengths | within {fmt(spread, 1)}% ({generation['distinct_lengths']} lengths) |")
        out.append(f"| Fixed overhead per request | {fmt(generation['overhead_seconds'])} s |")
    else:
        single = generation.get("single_stage_tps")
        out.append("| **Generation speed** | N/A (needs two verified lengths) |")
        if single:
            out.append(f"| After first token, one stage | {fmt(single['tps'])} tok/s |")
    longest = summary.get("end_to_end") or end_to_end(report.get("samples", []))
    if longest:
        out.append(f"| End-to-end at {fmt(longest['tokens'], 0)} tokens | {fmt(longest['tps'])} tok/s |")
    first = summary.get("first_token") or first_token_summary(report.get("samples", []))
    if first:
        suffix = "" if first["source"] == "native" else " (derived)"
        out.append(f"| Time to first token, median | {fmt(first['median_ms'], 0)} ms{suffix} |")
    mark, _ = confidence_line(generation)
    out.append(f"| Confidence | {mark} · {BASIS_LABEL.get(generation.get('basis'), generation.get('basis'))} |")
    return out


def chat_table(report: dict) -> str:
    """Headline result, then the stable five-column stage table. Warm-ups never enter it."""
    summary = report.get("summary") or {}
    generation = summary.get("generation") or generation_summary(report.get("samples", []),
                                                                 {d["request_id"] for d in summary.get("settings_drift", [])})
    samples = [s for s in report.get("samples", []) if s.get("phase") != "warmup"]
    excluded = {d["request_id"] for d in summary.get("settings_drift", [])}
    out = headline({**report, "summary": {**summary, "generation": generation}})
    _, reasons = confidence_line(generation)
    if reasons:
        out += ["", "Confidence basis: " + reasons + "."]
    out += ["", "| Stage | Rows | Request time | Standard TPS | Native TTFT |",
            "|---|---:|---:|---:|---:|"]
    warnings, audit, substituted, derived = [], [], [], False
    for index, (target, sample, planned) in enumerate(stage_rows(report), 1):
        stage = f"{target:g} s" if target is not None else f"Sample {index}"
        if planned is not None:
            substituted.append(f"{stage} replaced the planned {planned:g} s stage to fit the time budget")
        rows = sample.get("reference_rows") if sample else None
        if rows is None and target is not None:
            rows = rows_for(target)
        valid = sample_valid(sample, excluded)
        duration = sample.get("duration_ms") if valid else None
        standard = standard_metrics(sample) if valid else {}
        tps = standard.get("standard_tps")
        ttft, kind = first_token(sample) if valid else (None, None)
        derived = derived or kind == "derived"
        out.append(f"| {stage} | {fmt(rows, 0)} | "
                   f"{fmt(duration / 1000) + ' s' if duration is not None else 'N/A'} | "
                   f"{fmt(tps)} | {fmt(ttft, 0) + ' ms' + ('†' if kind == 'derived' else '') if ttft is not None else 'N/A'} |")
        tokens = sample.get("output_tokens") if valid else None
        rows_per_second = (rows * 1000 / duration if valid and sample.get("reference_matches") is True
                           and rows is not None and duration and duration > 0 else None)
        native_tps = sample.get("effective_tps") if valid else None
        audit.append(f"| {stage} | {fmt(standard.get('standard_tokens'), 0)} | {fmt(tokens, 0)} | {fmt(native_tps)} | {fmt(rows_per_second)} |")
        if sample and not valid:
            warnings.append(stage)
    if derived:
        out += ["", "† Derived: request time minus the client's native tool-argument stream time."]
    if substituted:
        out += ["", "Adaptive stages: " + "; ".join(substituted) + "."]
    if warnings:
        out += ["", "Excluded invalid or mismatched samples: " + ", ".join(warnings) + "."]
    if report.get("status") not in (None, "complete"):
        out += ["", "Run status: " + str(report["status"]).replace("_", " ") + "."]
    if samples:
        out += ["", "| Stage | Standard tokens | Native output tokens | Native TPS | Fixed-text rows/s |", "|---|---:|---:|---:|---:|", *audit, "",
                "Generation speed = slope of request time against fixed-text length: tokens per second once output is flowing, with fixed per-request overhead separated out. "
                "Standard TPS = verified fixed-text tokens (frozen o200k_base) / full request seconds, so it includes that overhead. Same text uses one token unit across providers. Native counts are retained for usage/cost; native TTFT boundaries remain client-specific."]
    past = report.get("generation_history")
    if past and past.get("runs", 0) > 1:
        out += ["", f"History for this client/model/settings: median **{fmt(past['generation_tps']['median'])} tok/s** over "
                f"{past['runs']} runs (range {fmt(past['generation_tps']['min'])}–{fmt(past['generation_tps']['max'])})."]
    return "\n".join(out) + "\n"


def build_report(session: dict, requests: list[Request], settings: dict, turns=None, run=None, progress=None) -> dict:
    warmups = [r for r in requests if r.phase == "warmup"]
    requests = [r for r in requests if r.phase != "warmup"]
    summary = summarize(requests, settings, intervals=not (run and run.get("workload") in FILE_WORKLOADS))
    notes = []
    if not requests:
        notes.append("No completed samples are available yet.")
    native = [r for r in requests if r.timing_basis == "native"]
    if any(r.timing_basis == "bracket" for r in requests):
        notes.append("Helper-clock timing was used where no native request timing exists. It brackets one model response between two helper commands and includes client and tool-dispatch overhead; that overhead is constant per stage and cancels in generation speed, but it lowers per-stage end-to-end figures.")
    if any(r.ttft_ms is None for r in native):
        notes.append("Native request TTFT is not exposed for some samples. Where the client records a tool-argument stream duration, a derived first-token latency is shown instead. For native TTFT, run `install` and restart the client.")
    notes.append("Native TTFT uses each client's own boundary and can precede visible text. Exact first-visible-token latency and per-token gaps are not exposed by this integration.")
    if any(r.timing_basis == "boundary" for r in requests):
        notes.append("Some latencies include client overhead because this client version did not expose a native request start.")
    if any(r.reasoning_tokens != 0 for r in native):
        notes.append("Post-TTFT TPS is omitted when reasoning token counts are nonzero or unknown; effective TPS remains valid.")
    if summary["settings_drift"]:
        notes.append("Settings changed or the served model differed. Affected requests are excluded from aggregates; inspect their individual rows.")
    if len(requests) < 200:
        notes.append("p95 describes this small sample only. Compare runs with similar context length, cache hit rate, and workload.")
    if run and len(requests) < run["samples_expected"]:
        notes.append(f"Incomplete run: expected {run['samples_expected']} samples, captured {len(requests)}.")
    for request in warmups + requests:
        notes += [f"Sample {request.sample_index}: {note}" for note in request.notes]
    state = "complete"
    if not requests or run and len(requests) < run["samples_expected"]:
        state = "incomplete"
    elif summary["settings_drift"]:
        state = "settings_changed"
    if progress is not None:
        state = "complete" if progress["reason"] == "stages_complete" else (
            "running" if progress["action"] != "stop" else progress["reason"])
    reliability = quality(requests, summary, progress)
    notes += ["Warm-ups are excluded from every measured aggregate. Slow samples and outliers are retained.",
              "Each duration label selects a fixed reference length, identical across models and runs. Actual duration varies by model. The budget is checked between stages; a live message can overrun it.",
              "Native TPS counts provider output including reasoning and tool syntax. Standard TPS counts verified reference text over the full request. Generation speed removes the fixed per-request overhead by fitting across lengths.",
              "Per-stage figures come from single requests. Generation speed carries a confidence level; repeat runs (or `--preset precise`) and consult history for stability over time."]
    if run:
        notes.extend(run.get("operational_notes", []))
        if run.get("workload") == "fixed-temp-file-v1":
            notes += ["Samples are literal model-generated file-write arguments. Each is written to temporary sample.txt, validated, and deleted before the recorder returns; no workload text is printed in chat.",
                      "File-writing/deletion speed is excluded: the payload is fully generated before the helper starts."]
    return standardize_report({
        "schema_version": "3.0", "benchmark_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": state, "session": session,
        "run_id": run.get("run_id") if run else None,
        "workload": run.get("workload") if run else "observed-chat-requests",
        "expected_samples": run.get("samples_expected") if run else None,
        "plan": run.get("plan") if run else None, "progress": progress, "quality": reliability,
        "summary": summary, "samples": [r.to_dict() for r in requests],
        "warmups": [r.to_dict() for r in warmups],
        "completed_turns": turns or [], "definitions": DEFINITIONS, "notes": notes,
    })


def markdown(report: dict) -> str:
    summary, session = report["summary"], report["session"]
    settings = summary["settings"]
    generation = summary.get("generation") or {}
    fixed = report["workload"] in FILE_WORKLOADS
    out = ["# Chat benchmark", "",
           f"Client: **{client_title(session)}** · Model: **{label(settings.get('model'))}** · "
           f"Effort: **{label(settings.get('effort'))}** · Tier: **{label(settings.get('service_tier'))}**",
           f"Session: `{session['session_id']}` · Workload: `{report['workload']}` · "
           f"Status: **{report['status']}** · Valid samples: **{summary['valid_requests']}**", ""]
    if fixed:
        out += ["## Result", "", *headline(report)[2 if session.get("provider") else 0:], ""]
        mark, reasons = confidence_line(generation)
        out += [f"Confidence: **{mark}**" + (f" — {reasons}." if reasons else ""), ""]
        cross = generation.get("cross_check") or {}
        checks = [f"{name} {fmt(cross[key])} tok/s" for name, key in
                  (("native request timing", "native_tps"), ("helper clock", "helper_clock_tps"),
                   ("native argument stream", "native_stream_tps")) if cross.get(key)]
        if len(checks) > 1:
            out += ["Independent timing sources, same sweep: " + " · ".join(checks) + ".", ""]
        overhead = generation.get("helper_clock_overhead_ms")
        if overhead:
            out += [f"Client overhead seen by the helper clock (helper interval minus native request): median "
                    f"{fmt(overhead['median'], 0)} ms, range {fmt(overhead['min'], 0)}–{fmt(overhead['max'], 0)} ms.", ""]
        if generation.get("native_token_tps"):
            out += [f"In this provider's own token unit the same fit gives {fmt(generation['native_token_tps'])} native tok/s.", ""]
        if generation.get("r_squared") is not None:
            out += [f"Fit: {generation['points']} points over {generation['distinct_lengths']} lengths, "
                    f"R² {generation['r_squared']:.5f}, largest residual {fmt(generation['max_residual_pct'], 1)}% of a request, "
                    f"pairwise speeds {', '.join(fmt(v) for v in generation['pairwise_tps'])} tok/s.", ""]
        out += [DEFINITIONS["generation_tps"], "", "## Per-request statistics", ""]
    out += ["| Metric | Median | Approx. 95% median interval | Mean | p95 | CV % | Samples |",
            "|---|---:|---|---:|---:|---:|---:|"]
    for name, key in [("Standard TPS (end-to-end)", "standard_tps"), ("Native output TPS", "effective_tps"), ("Post-TTFT TPS (native, derived)", "post_ttft_tps"),
                      ("Native TTFT (ms)", "ttft_ms"), ("Request latency (ms)", "duration_ms")]:
        stat = summary[key]
        ci = stat.get("median_ci95")
        interval = f"{fmt(ci['low'])}–{fmt(ci['high'])}" if ci else "N/A"
        out.append(f"| {name} | {fmt(stat['median'])} | {interval} | {fmt(stat['mean'])} | {fmt(stat['p95'])} | {fmt(stat['cv_pct'])} | {stat['n']} |")
    status = report.get("progress")
    if status:
        out += ["", f"Warm-ups excluded: **{len(report['warmups'])}** · Active measured time: "
                f"**{fmt(status['active_seconds'] / 60)} min** · Wall time: **{fmt(status['wall_seconds'] / 60)} min** · "
                f"Stop/progress reason: **{status['reason']}**"]
    q = report["quality"]
    out += ["", f"Per-stage TPS status: **{q['effective_tps']['status']}** · "
            f"Native TTFT status: **{q['native_ttft']['status']}** · "
            f"Native TTFT coverage: **{fmt(q['native_ttft_coverage_pct'])}%**"]
    for key, title in (("effective_tps", "TPS"), ("native_ttft", "TTFT")):
        if q[key]["reasons"]:
            out += [f"{title} checks: " + "; ".join(q[key]["reasons"]) + "."]
    if fixed:
        file_mode = report["workload"] == "fixed-temp-file-v1"
        out += ["", "## Stages", "",
                "| Stage | Rows | Standard tokens | Request s | Helper-clock s | Stream s | First token ms | Standard TPS | Native tokens | Native TPS | Timing | Check |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        excluded = {d['request_id'] for d in summary.get('settings_drift', [])}
        for s in report["warmups"] + report["samples"]:
            valid = sample_valid(s, excluded) and s.get("reference_matches") is True
            first, kind = first_token(s)
            stage = (f"{s.get('target_seconds'):g} s" if s.get("target_seconds") is not None else "?") + (" (warm-up)" if s.get("phase") == "warmup" else "")
            check = "ok" if valid else "excluded"
            if valid and file_mode:
                check = {"single_file_payload": "verified", "unverified_file_payload": "hash only"}.get(s.get("continuity"), "ok")
            out.append(f"| {stage} | {fmt(s.get('reference_rows'), 0)} | {fmt(s.get('standard_tokens'), 0)} | "
                       f"{fmt(s['duration_ms'] / 1000 if s.get('duration_ms') else None)} | "
                       f"{fmt(s['bracket_ms'] / 1000 if s.get('bracket_ms') else None)} | "
                       f"{fmt(s['stream_ms'] / 1000 if s.get('stream_ms') else None)} | "
                       f"{fmt(first, 0) + ('†' if kind == 'derived' else '') if first is not None else 'N/A'} | "
                       f"{fmt(s.get('standard_tps') if valid else None)} | {fmt(s.get('output_tokens'), 0)} | "
                       f"{fmt(s.get('effective_tps') if valid else None)} | {s.get('timing_basis', 'native')} | {check} |")
        out += ["", "† Derived from the native tool-argument stream duration. “verified” = one tool call whose arguments contain the literal rows; “hash only” = content verified, call structure not inspectable without a transcript.",
                "", DEFINITIONS["standard_tps"], "Reference tokenizer: `" + report['reference_tokenizer']['id'] + "`."]
    if report.get("accounting"):
        a, tokens = report["accounting"], report["accounting"]["tokens"]
        out += ["", "## Run accounting (includes measurement overhead)", "",
                f"Requests: **{a['requests']}** · Input: **{fmt(tokens['input_tokens'], 0)}** · "
                f"Cache reads: **{fmt(tokens['cached_input_tokens'], 0)}** · Cache writes: **{fmt(tokens['cache_write_input_tokens'], 0)}** · "
                f"Output: **{fmt(tokens['output_tokens'], 0)}** · Reasoning: **{fmt(tokens['reasoning_tokens'], 0)}**",
                f"Client-estimated USD: **{fmt(a['client_estimated_usd'], 6)}** · Verified charge: **N/A**", "",
                a["scope"], a["billing_note"]]
    past = report.get("generation_history")
    if past:
        g = past["generation_tps"]
        out += ["", "## History for this client, model and settings", "",
                f"Runs with a generation-speed estimate: **{past['runs']}** · median **{fmt(g['median'])} tok/s** · "
                f"range {fmt(g['min'])}–{fmt(g['max'])} · CV {fmt(g['cv_pct'], 1)}%"]
    if report.get("history"):
        out += ["", "Matching-duration history (settings, client version, context size, and cache band kept separate):", "",
                "| Model / effort / tier | Target seconds | Runs | Median Standard TPS | Approx. 95% median interval |",
                "|---|---:|---:|---:|---|"]
        for group in report["history"]:
            s, st = group["settings"], group.get("standard_tps", stats([]))
            ci = st.get("median_ci95")
            out.append(f"| {label(s.get('model'))} / {label(s.get('effort'))} / {label(s.get('service_tier'))} | "
                       f"{fmt(s.get('target_seconds'))} | {group['distinct_runs']} | {fmt(st['median'])} | "
                       f"{fmt(ci['low']) + '–' + fmt(ci['high']) if ci else 'N/A (limited history)'} |")
    out += ["", "## Requests", "",
            f"Weighted native TPS: **{fmt(summary['weighted_effective_tps'])}** · "
            f"Output tokens: **{fmt(summary['output_tokens'], 0)}** · "
            f"Input tokens (sum of requests): **{fmt(summary['input_tokens'], 0)}**", "",
            "| Sample | Model | Effort | Tier | Native output tokens | Native TPS | TTFT ms | Latency ms | Cache hit |",
            "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for index, sample in enumerate(report["samples"], 1):
        out.append(f"| {sample.get('sample_index') or index} | {label(sample['model'])} | "
                   f"{label(sample['effort'])} | {label(sample['service_tier'])} | "
                   f"{label(sample['output_tokens'])} | {fmt(sample['effective_tps'])} | "
                   f"{fmt(sample['ttft_ms'])} | {fmt(sample['duration_ms'])} | "
                   f"{fmt(sample['cache_hit_pct']) + '%' if sample['cache_hit_pct'] is not None else 'N/A'} |")
    if report["completed_turns"]:
        out += ["", "Native completed-turn timings (separate from request samples):", "",
                "| Turn | TTFT ms | Full turn ms | Status |", "|---|---:|---:|---|"]
        for turn in report["completed_turns"][-5:]:
            out.append(f"| `{turn['turn_id']}` | {fmt(turn.get('ttft_ms'))} | "
                       f"{fmt(turn.get('duration_ms'))} | {turn.get('status', 'not exposed')} |")
    out += ["", "N/A means the measurement was not exposed; nothing is estimated in its place.", ""]
    if report["notes"]:
        out += ["## Notes", ""] + [f"- {note}" for note in report["notes"]] + [""]
    out += ["Timing evidence: " + "; ".join(sorted({s.get('timing_source', 'not exposed') for s in report['samples']})) + ".",
            "TTFT evidence: " + "; ".join(sorted({s.get('ttft_source', 'not exposed') for s in report['samples']})) + ".", ""]
    return "\n".join(out)


CSV_FIELDS = ["sample_index", "phase", "target_seconds", "reference_rows", "reference_matches", "standard_tokenizer",
              "standard_tokens", "standard_tps", "duration_ms", "timing_basis", "bracket_ms", "stream_ms", "prestream_ms",
              "ttft_ms", "first_content_ms", "reference_rows_per_second", "payload_bytes_per_second", "payload_sha256",
              "payload_words", "payload_bytes", "temporary_cleanup", "tool_call_count", "tool_input_rows", "provenance",
              "visible_sha256", "visible_words", "text_span_ms", "text_span_source", "visible_wps", "text_block_count",
              "continuity", "client_estimated_usd", "context_epoch", "client_version", "provider", "session_id",
              "request_id", "turn_id", "model", "effort", "service_tier", "speed", "input_tokens", "cached_input_tokens",
              "cache_write_input_tokens", "output_tokens", "reasoning_tokens", "effective_tps", "post_ttft_tps",
              "cache_hit_pct", "attempts", "status", "timing_source", "ttft_source", "stream_source", "usage_source"]


def _replace(path: Path, content: str):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_report(report: dict, directory: Path) -> dict:
    from .html import report_html
    standardize_report(report)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {kind: directory / f"benchmark.{kind}" for kind in ("md", "json", "csv", "html")}
    paths["chat"] = directory / "chat.md"
    for kind, content in (("md", markdown(report)), ("json", json.dumps(report, indent=2, ensure_ascii=False) + "\n"),
                          ("chat", chat_table(report)), ("html", report_html(report))):
        _replace(paths[kind], content)
    temporary = paths["csv"].with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report.get("warmups", []) + report["samples"])
    temporary.replace(paths["csv"])
    return {kind: str(path.resolve()) for kind, path in paths.items()}
