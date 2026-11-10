"""CLI and repeatable current-chat probe protocol."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .accounting import account
from .clock import bracket, process_start_ms, stamp
from .history import compare, compare_markdown, configuration_history, history
from .install import install
from .protocol import plan, progress, stage_rows, stage_target
from .providers import (PROVIDERS, capability_table, detect, late_bind, open_session, provider_for,
                        provider_status)
from .reports import build_report, chat_table, markdown, write_report
from .sessions import Session, parse_session, payload_digest
from .standard import metadata as standard_metadata, reference_tokens, tokenizer
from .telemetry import PORT, data_home, enrich, ensure_collector, health, read_events, serve
from .temporary import create_inbox, inbox_name, recover_stale, remove_inbox, sample_file, take_inbox_sample
from .workload import WORDS, canonical_text, reference_digest, text_digest

ROWS = 64
FILE_WORKLOADS = ("fixed-continuous-text-v1", "fixed-temp-file-v1")
BRACKET_SOURCE = ("helper bracket: previous helper exit to this helper's process start "
                  "(includes client and tool-dispatch overhead)")


def print_json(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))
    sys.stdout.flush()


def _run_path(root, run_id):
    if not re.fullmatch(r"[0-9a-f]{32}", run_id or ""):
        raise ValueError("Invalid run ID.")
    return root / "runs" / (run_id + ".json")


def load_run(root, run_id):
    path = _run_path(root, run_id)
    if not path.is_file():
        raise ValueError(f"Run {run_id} was not found in {root}.")
    return json.loads(path.read_text(encoding="utf-8"))


def save_run(root, run):
    path = _run_path(root, run["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stamp_exit(root, run_id, command):
    """Last act of a helper command: the next sample's bracket starts here."""
    try:
        run = load_run(root, run_id)
        run["clock"] = stamp(command)
        save_run(root, run)
    except (OSError, ValueError):
        pass


def run_session(run):
    return Session.from_dict(run["session"])


def validate_payload(payload: str, run_id: str, index: int, rows=ROWS):
    parts = payload.split("|")
    if len(parts) != rows + 1 or parts[0] != f"CBENCH:{run_id}:{index}":
        raise ValueError(f"Payload must start with CBENCH:{run_id}:{index} and contain exactly {rows} pipe-separated rows.")
    for n, text in enumerate(parts[1:], 1):
        if text != f"{n:04d} {WORDS}":
            raise ValueError(f"Payload row {n} must be '{n:04d} {WORDS}'.")


def brief(state, run=None):
    """Progress as printed to the agent: everything it must follow, nothing bulky."""
    result = {k: v for k, v in state.items() if k != "plan"}
    if state.get("target_rows"):
        result["first_row"] = f"0001 {WORDS}"
        result["last_row"] = f"{state['target_rows']:04d} {WORDS}"
        if run and run.get("inbox"):
            result["sample_path"] = (Path(run["inbox"]) / inbox_name(run["run_id"], state["next_index"])).as_posix()
    return result


def best_match(requests, run_id, index):
    """The response that carried the sample. With file delivery two responses name
    the same sample (the file write and the short helper call); the payload wins."""
    matching = [r for r in requests if r.run_id == run_id and r.sample_index == index]
    return max(matching, key=lambda r: r.tool_input_rows or 0) if matching else None


def recover_inboxes(root, age_seconds=3600):
    """Remove sample inboxes of runs that were abandoned mid-way."""
    now = time.time()
    for path in (root / "runs").glob("*.json"):
        try:
            age = now - path.stat().st_mtime
            if not age_seconds < age < 14 * 86400:
                continue
            run = json.loads(path.read_text(encoding="utf-8"))
            if run.get("inbox") and Path(run["inbox"]).exists():
                remove_inbox(Path(run["inbox"]), run["run_id"])
        except (OSError, ValueError, KeyError):
            continue


def prepare(args, root):
    tokenizer()  # Initialize locally before timing or generating any samples.
    sampling_plan = plan(args.preset, durations=args.durations, warmup_seconds=args.warmup_seconds,
                         max_seconds=args.max_seconds, adaptive=not args.strict_stages)
    # --trust-provider is for harnesses that drive the helper themselves (selftest).
    provider = provider_for(args.provider) if args.trust_provider and args.provider != "auto" else detect(args.provider)
    notes = []
    if args.provider not in ("auto", provider.key):
        notes.append(f"This chat runs in {provider.title}, not `{args.provider}`: the skill's client hint was overridden by the process chain.")
    recover_stale()
    recover_inboxes(root)
    session, reason = open_session(provider, args.session_id, args.session_path, args.client_home)
    if reason:
        notes.append(reason)
    collector = None
    if provider.telemetry:
        try:
            collector = ensure_collector(root)
        except (OSError, RuntimeError, ValueError) as exc:
            notes.append(f"Native telemetry receiver unavailable ({exc}); TTFT may be N/A.")
    existing_requests, settings, _ = parse_session(session)
    labels = {key: value for key, value in (("model", args.model), ("effort", args.effort),
                                            ("service_tier", args.tier)) if value}
    for key, value in labels.items():
        if settings.get(key) is None:
            settings[key] = value
    run_id = uuid.uuid4().hex
    delivery = provider.delivery() if args.delivery == "auto" else args.delivery
    sampling_plan["delivery"] = delivery
    stamp_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
           "session": session.to_dict(), "settings": settings, "labels": labels,
           "protocol_version": 3, "benchmark_version": __version__,
           "started_ms": time.time() * 1000, "plan": sampling_plan,
           "samples_expected": sampling_plan["min_samples"], "samples": {},
           "workload": "fixed-temp-file-v1",
           "reference_tokenizer": standard_metadata(),
           "accounting_start_request_id": existing_requests[-1].request_id if existing_requests else None,
           "operational_notes": notes,
           "results_directory": str(root / "results" / f"{stamp_name}-{session.provider}-{run_id[:8]}")}
    if delivery == "file":
        # Inside the chat's working directory, where the client's file tool may write.
        run["inbox"] = str(create_inbox(Path.cwd(), run_id))
    save_run(root, run)
    state = progress(run, [])
    how = ("Generate exactly target_rows reference rows as literal stdin in ONE write-sample tool call, with nothing else in that response."
           if delivery == "stdin" else
           "For each sample: FIRST create the file progress.sample_path with your file-writing tool, its content being exactly target_rows literal reference rows; THEN run write-sample (no stdin) as the next tool call. The helper reads, validates and deletes the file.")
    print_json({"run_id": run_id, "client": provider.title, "session": session.to_dict(),
                "delivery": delivery, "inherited_settings": settings,
                "timing": {"native_transcript": session.tracked, "helper_clock": True,
                           "native_telemetry_receiver": bool(collector)},
                "notes": notes, "plan": sampling_plan,
                "next_action": state["action"], "next_index": 1, "progress": brief(state, run),
                "workload": run["workload"], "results_directory": run["results_directory"],
                "sample_protocol": {
                    "command": "write-sample --run-id RUN_ID --index INDEX",
                    "reference_row": "NNNN " + WORDS,
                    "instructions": how + " Do not print the rows in chat. Run no other tool between helper commands: the helper clock times the gap between them. No scripts, loops, substitutions, copies, or reference-output tools may generate the rows for you; the measured workload is model-generated tool arguments."}})
    return run_id


def capture_sample(args, root, emit=True):
    def output(value):
        if emit:
            print_json(value)
        return value
    run = load_run(root, args.run_id)
    p = run.get("plan")
    maximum = p["warmups"] + p["max_samples"] if p else run["samples_expected"]
    if not 1 <= args.index <= maximum:
        raise ValueError("Sample index is outside this run's expected range.")
    is_burst = args.command in ("burst", "write-sample")
    if not is_burst:
        validate_payload(args.payload, args.run_id, args.index, p.get("rows", ROWS) if p else ROWS)
    file_metadata = getattr(args, "file_metadata", None)
    digest = file_metadata["payload_sha256"] if file_metadata else (None if is_burst else payload_digest(args.payload))
    if str(args.index) in run["samples"]:
        if run["samples"][str(args.index)].get("payload_sha256") != digest:
            raise ValueError("This sample index has a different recorded payload.")
        return output({"already_recorded": True, **checkpoint(root, run)})
    if args.index != len(run["samples"]) + 1:
        raise ValueError("Samples must be captured sequentially; use status to find the next index.")
    if run.get("finished_ms") is not None:
        raise ValueError("This run has stopped. Prepare a new run rather than extending its predeclared plan.")
    session = run_session(run)
    provider = provider_for(session.provider)
    # Keep the receiver alive/recover it during long runs, without dropping TPS data
    # if telemetry cannot be recovered. Missing native fields stay N/A.
    if provider.telemetry:
        try:
            ensure_collector(root)
        except (OSError, RuntimeError, ValueError) as exc:
            note = f"Native telemetry receiver unavailable ({exc}); TTFT may be N/A."
            if note not in run.setdefault("operational_notes", []):
                run["operational_notes"].append(note)
    invoked = getattr(args, "invoked_ms", None)
    clock = run.get("clock")
    sample = {"request_id": None, "captured_at": datetime.now(timezone.utc).isoformat(),
              "captured_ms": time.time() * 1000, "payload_sha256": digest,
              "target_seconds": stage_target(run, args.index),
              "invoked_ms": invoked, "bracket_ms": bracket(clock, invoked) if invoked else None,
              "bracket_after": (clock or {}).get("command")}
    if file_metadata:
        sample.update(file_metadata)
    run["samples"][str(args.index)] = sample
    save_run(root, run)  # The sample is durable before any transcript lookup.
    if not session.tracked:
        session = late_bind(run) or session
    selected = None
    if session.tracked:
        deadline = time.monotonic() + (3 if provider.live else 0)
        while True:
            requests, _, _ = parse_session(session)
            selected = best_match(requests, args.run_id, args.index)
            if selected or time.monotonic() >= deadline:
                break
            time.sleep(.1)
    sample["request_id"] = selected.request_id if selected else None
    state = checkpoint(root, run)
    return output({"sample": args.index, "mode": "temporary_file" if file_metadata else ("continuous_text" if is_burst else "legacy_payload"),
                   "payload_sha256": sample["payload_sha256"], "request_id": sample["request_id"],
                   "usage_available": selected is not None,
                   "helper_clock_ms": sample["bracket_ms"], **state})


def capture_file(args, root):
    run = load_run(root, args.run_id)
    if run.get("workload") != "fixed-temp-file-v1":
        raise ValueError("This run was prepared with another workload. Prepare a file-output run first.")
    if args.text is not None:
        text = args.text
    elif run.get("inbox"):
        text = take_inbox_sample(Path(run["inbox"]), args.run_id, args.index)
    else:
        text = sys.stdin.read(1024 * 1024 + 1)
    text = text.lstrip("\ufeff")  # Some shells prepend a byte-order mark to piped text.
    if not text or len(text.encode("utf-8")) > 1024 * 1024:
        raise ValueError("Supply the model-generated literal reference as stdin or --text (up to 1 MiB).")
    if not run.get("inbox") and str(args.index) not in run["samples"] and run.get("plan"):
        # A shell that cuts long commands delivers a prefix that stops mid-row. That is
        # a transport fault, not model output: reject it without recording a sample.
        expected = canonical_text(stage_rows(stage_target(run, args.index)))
        received = text.replace("\r\n", "\n").strip("\r\n")
        row = len(expected.split("\n")[0])
        if received != expected and expected.startswith(received) and len(received) % (row + 1) != row:
            raise ValueError(f"The sample arrived cut off after {len(received):,} characters (mid-row): this client's shell "
                             "truncates long commands. Do not retry this run and never use a script. Prepare a new run "
                             "with the same options plus --delivery file, which sends samples through your file-writing tool.")
    directory = None
    try:
        with sample_file(args.run_id, text) as path:
            directory = path.parent
            content = path.read_text(encoding="utf-8")
            args.file_metadata = {"payload_sha256": text_digest(content), "payload_words": len(content.split()),
                                  "payload_bytes": path.stat().st_size, "temporary_cleanup": "pending"}
            result = capture_sample(args, root, emit=False)
    except BaseException:
        # Preserve the real failure and make cleanup state reviewable if a sample
        # had already been recorded before deletion failed.
        try:
            failed_run = load_run(root, args.run_id)
            if str(args.index) in failed_run["samples"]:
                failed_run["samples"][str(args.index)]["temporary_cleanup"] = "failed" if directory and directory.exists() else "deleted"
                save_run(root, failed_run)
        except (OSError, ValueError):
            pass
        raise
    saved = load_run(root, args.run_id)
    saved["samples"][str(args.index)]["temporary_cleanup"] = "deleted"
    save_run(root, saved)
    result.update(checkpoint(root, saved))
    # Print only after cleanup succeeds; no sample body or filesystem timing leaks.
    print_json({**result, "temporary_files_deleted": True})


def bracket_request(run, session, index, sample):
    """A sample known only through the helper clock (no transcript row yet, or ever)."""
    from .metrics import Request
    request = Request(provider=session.provider, session_id=session.session_id,
                      request_id=f"{run['run_id'][:8]}-helper-{index}",
                      started_ms=sample["invoked_ms"] - sample["bracket_ms"], completed_ms=sample["invoked_ms"],
                      duration_ms=sample["bracket_ms"], timing_source=BRACKET_SOURCE,
                      settings_source="self-reported labels" if run.get("labels") else "not exposed",
                      usage_source="not exposed", run_id=run["run_id"], sample_index=index)
    # What the client recorded for this chat outranks self-reported labels.
    known = {**(run.get("labels") or {}), **{k: v for k, v in (run.get("settings") or {}).items() if v is not None}}
    for key in ("model", "effort", "service_tier", "speed"):
        if known.get(key) is not None:
            setattr(request, key, known[key])
    return request


def annotate(request, run, index, sample):
    """Attach the plan's stage, the recorded payload facts and the validity checks."""
    plan_data = run.get("plan") or {}
    request.sample_index = index
    request.payload_sha256 = sample.get("payload_sha256")
    request.phase = "warmup" if index <= plan_data.get("warmups", 0) else "measured"
    request.bracket_ms = sample.get("bracket_ms")
    for key, value in (run.get("labels") or {}).items():
        if getattr(request, key, None) is None:
            setattr(request, key, value)
    if plan_data.get("mode") not in ("continuous_text", "temporary_file"):
        return request
    request.target_seconds = sample.get("target_seconds") or stage_target(run, index)
    request.reference_rows = stage_rows(request.target_seconds)
    if plan_data.get("mode") != "temporary_file":
        request.reference_matches = request.visible_sha256 == reference_digest(request.reference_rows)
        if request.text_block_count != 1 or request.continuity == "interrupted":
            request.status, request.continuity = "invalid_output", "interrupted"
            request.notes.append("A legacy chat sample must be one uninterrupted text block.")
    else:
        request.payload_words = sample.get("payload_words")
        request.payload_bytes = sample.get("payload_bytes")
        request.temporary_cleanup = sample.get("temporary_cleanup")
        request.reference_matches = sample.get("payload_sha256") == reference_digest(request.reference_rows)
        request.text_span_ms = None
        request.text_span_source = "not exposed for tool-argument output; file I/O is not model generation"
        native = request.timing_basis != "bracket"
        if not native:
            # Structure cannot be inspected without a transcript row.
            request.continuity, request.provenance = "unverified_file_payload", "unverified (no transcript)"
        elif request.tool_call_count == 1:
            request.continuity = "single_file_payload"
        else:
            request.continuity, request.status = "interrupted", "invalid_output"
            request.notes.append("A sample must be exactly one write-sample tool call in its own response.")
        if native and request.visible_words:
            request.notes.append(f"The response also carried {request.visible_words} words of visible text; "
                                 "this adds to the fixed overhead, not to the generation rate.")
        if native and request.status == "completed" and request.reference_matches:
            expected = reference_tokens(request.reference_rows)
            visible_out = (request.output_tokens - (request.reasoning_tokens or 0)
                           if request.output_tokens is not None else None)
            found = request.tool_input_rows
            if found is not None and found < max(2, request.reference_rows // 8):
                request.provenance = "not literal"
            elif visible_out is not None and visible_out < expected * .5:
                request.provenance = "implausible token count"
            elif found is not None and found >= request.reference_rows:
                request.provenance = "literal tool argument"
            elif found is not None:
                request.provenance = "partially visible (transcript may truncate arguments)"
            if request.provenance in ("not literal", "implausible token count"):
                request.status = "invalid_output"
                request.notes.append("The reference rows were not generated literally by the model in the tool call "
                                     "(a script or copy produced them), so no model output was measured.")
    if not request.reference_matches:
        request.status = "invalid_output"
        request.notes.append("Output did not match the fixed reference.")
    return request


def collect_requests(root, run=None, session=None, last=10):
    session = run_session(run) if run else session
    if run and not session.tracked:
        bound = late_bind(run)
        if bound:
            session = bound
            save_run(root, run)
    requests, settings, turns = parse_session(session)
    if session.tracked:
        enrich(requests, read_events(root, session.provider, session.session_id))
    if not run:
        return session, requests[-last:], settings, turns
    selected = []
    for index in sorted(int(i) for i in run["samples"]):
        sample = run["samples"][str(index)]
        matching = [r for r in requests if (r.request_id == sample["request_id"] if sample.get("request_id")
                    else r.run_id == run["run_id"] and r.sample_index == index)]
        request = matching[-1] if matching and sample.get("request_id") else best_match(requests, run["run_id"], index)
        if request:
            sample["request_id"] = request.request_id
        elif sample.get("bracket_ms") is not None and sample.get("invoked_ms") is not None:
            request = bracket_request(run, session, index, sample)
        else:
            continue
        selected.append(annotate(request, run, index, sample))
    return session, selected, run["settings"], turns


def with_context(root, run, session, report_data):
    if run.get("workload") in FILE_WORKLOADS:
        report_data["history"] = history(root, report_data)
        report_data["generation_history"] = configuration_history(root, report_data)
        all_requests, _, _ = parse_session(session)
        if session.tracked:
            enrich(all_requests, read_events(root, session.provider, session.session_id))
        report_data["accounting"] = account(all_requests, run.get("accounting_start_request_id"))
    return report_data


def gather_report(root, run=None, session=None, last=10):
    session, requests, settings, turns = collect_requests(root, run, session, last)
    state = progress(run, requests, run.get("finished_ms")) if run and run.get("plan") else None
    result = build_report(session.to_dict(), requests, settings, turns, run, state)
    return with_context(root, run, session, result) if run else result


def checkpoint(root, run):
    session, requests, settings, turns = collect_requests(root, run)
    if run.get("plan"):
        state = progress(run, requests, run.get("finished_ms"))
        if state.get("substituted_from") is not None:
            # A shorter fixed stage replaces one that no longer fits the budget.
            run.setdefault("stage_overrides", {})[str(state["next_index"])] = state["target_seconds"]
        if state["action"] == "stop" and run.get("finished_ms") is None:
            run["finished_ms"] = time.time() * 1000
        if state["action"] == "stop" and run.get("inbox"):
            remove_inbox(Path(run["inbox"]), run["run_id"])
        run["progress"] = state
    else:
        done = len(run["samples"]) >= run["samples_expected"]
        state = {"action": "stop" if done else "sample", "reason": "legacy_run",
                 "next_index": len(run["samples"]) + 1}
    save_run(root, run)
    report_data = build_report(session.to_dict(), requests, settings, turns, run,
                               state if run.get("plan") else None)
    files = write_report(with_context(root, run, session, report_data), Path(run["results_directory"]))
    return {"next_action": state["action"], "next_index": state["next_index"],
            "phase": "warmup" if state["next_index"] <= run.get("plan", {}).get("warmups", 0) else "measured",
            "progress": brief(state, run), "files": files}


def run_status(args, root):
    run = load_run(root, args.run_id)
    deadline = time.monotonic() + args.wait
    while True:
        result = checkpoint(root, run)
        if result["next_action"] != "wait_for_usage" or time.monotonic() >= deadline:
            break
        time.sleep(.25)
    print_json({"run_id": args.run_id, **result})


def report(args, root):
    run = load_run(root, args.run_id) if args.run_id else None
    session = None
    if not run:
        provider = detect(args.provider)
        session, reason = open_session(provider, args.session_id, args.session_path, args.client_home)
        if not session.tracked:
            raise ValueError(reason or f"{provider.title} transcripts cannot be inspected; benchmark runs still work.")
    deadline = time.monotonic() + args.wait
    while True:
        result = gather_report(root, run, session, args.last)
        samples = result["samples"]
        # Native rows and telemetry can trail the tool call; wait for them only where they can arrive.
        waiting = [s for s in samples if s.get("timing_basis") != "native"
                   or (s.get("ttft_ms") is None and s.get("prestream_ms") is None)]
        tracked = result["session"].get("path") is not None
        if (samples and not waiting) or not tracked or time.monotonic() >= deadline:
            break
        time.sleep(.25)
    if args.output_dir:
        directory = Path(args.output_dir)
    elif run:
        directory = Path(run["results_directory"])
    else:
        stamp_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = root / "results" / f"{stamp_name}-{session.provider}-observed"
    paths = write_report(result, directory)
    if args.json:
        print_json({"report": result, "files": paths})
    else:
        print(chat_table(result) if args.chat else markdown(result))
        print("Files: " + json.dumps(paths))
    if args.open:
        import webbrowser
        webbrowser.open(Path(paths["html"]).as_uri())
    if args.follow_turn and run and run_session(run).tracked:
        start_follow(root, run)


def start_follow(root, run):
    entry = Path(__file__).resolve().parents[1] / "benchmark.py"
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS) if os.name == "nt" else 0
    with (root / "collector.log").open("ab") as log:
        subprocess.Popen([sys.executable, str(entry), "follow", "--run-id", run["run_id"],
                          "--data-home", str(root)], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         creationflags=flags, start_new_session=os.name != "nt")


def follow(args, root):
    run = load_run(root, args.run_id)
    session = run_session(run)
    if not session.tracked:
        return
    requests, _, previous = parse_session(session)
    active_turn = requests[-1].turn_id if requests else None
    initial_request = requests[-1].request_id if requests else None
    old_turns = {t["turn_id"] for t in previous}
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        result = gather_report(root, run)
        turn_done = any(t["turn_id"] == active_turn and t["turn_id"] not in old_turns for t in result["completed_turns"])
        if session.provider == "claude":
            latest, _, _ = parse_session(session)
            turn_done = bool(latest and latest[-1].turn_id == active_turn and
                             latest[-1].request_id != initial_request and latest[-1].stop_reason == "end_turn")
        if turn_done:
            time.sleep(3)  # Let asynchronously batched native usage/cost events arrive.
            write_report(gather_report(root, run), Path(run["results_directory"]))
            return
        time.sleep(.5)


def doctor(root):
    result = {"benchmark_version": __version__, "data_home": str(root), "python": sys.executable,
              "collector": health(), "clients": provider_status(root),
              "note": "A running client needs one restart after its skill or telemetry is first installed."}
    try:
        tokenizer()
        result["reference_tokenizer"] = "ok"
    except ValueError as exc:
        result["reference_tokenizer"] = str(exc)
    try:
        provider = detect("auto")
        session, reason = open_session(provider, None, None, None)
        result["current_client"] = provider.key
        result["current_session"] = session.to_dict()
        if reason:
            result["current_session_note"] = reason
        if session.tracked:
            requests, settings, turns = parse_session(session)
            enrich(requests, read_events(root, session.provider, session.session_id))
            result["current_settings"] = settings
            result["requests_with_usage"] = len(requests)
            result["requests_with_native_ttft"] = sum(r.ttft_ms is not None for r in requests)
            result["native_events_captured"] = len(read_events(root, session.provider, session.session_id))
    except (ValueError, OSError) as exc:
        result["current_session"] = str(exc)
    print_json(result)


def selftest(args, root):
    """Drive the real helper commands with a simulated client of known speed.

    Uses a throwaway data directory, so nothing reaches history. The simulated
    model sleeps overhead + tokens / speed before each write-sample; the sweep
    estimate must recover that speed from helper-clock timing alone.
    """
    import tempfile
    from .workload import canonical_text
    entry = Path(__file__).resolve().parents[1] / "benchmark.py"
    speed, overhead = float(args.tps), float(args.overhead)
    with tempfile.TemporaryDirectory(prefix="chat-benchmark-selftest-") as directory:
        def helper(*arguments, stdin=None):
            done = subprocess.run([sys.executable, str(entry), *arguments, "--data-home", directory],
                                  input=stdin, capture_output=True, text=True, encoding="utf-8", cwd=directory)
            if done.returncode:
                raise RuntimeError(f"helper {arguments[0]} failed: {done.stdout or done.stderr}")
            return json.loads(done.stdout)
        state = helper("prepare", "--provider", "selftest", "--trust-provider", "--durations", "2,5,10", "--warmup-seconds", "2",
                       "--max-seconds", "120", "--delivery", "stdin", "--model", f"simulated-{speed:g}-tps")
        run_id = state["run_id"]
        while state["next_action"] == "write-sample":
            rows = state["progress"]["target_rows"]
            time.sleep(overhead + reference_tokens(rows) / speed)  # The simulated model "generates".
            state = helper("write-sample", "--run-id", run_id, "--index", str(state["next_index"]),
                           stdin=canonical_text(rows) + "\n")
        report_data = json.loads(Path(state["files"]["json"]).read_text(encoding="utf-8"))
    generation = report_data["summary"]["generation"]
    error = abs(generation["tps"] - speed) / speed * 100 if generation.get("tps") else None
    passed = error is not None and error <= args.tolerance and report_data["status"] == "complete"
    print_json({"selftest": "passed" if passed else "failed", "simulated_tps": speed,
                "simulated_overhead_seconds": overhead, "measured_generation_tps": generation.get("tps"),
                "measured_overhead_seconds": generation.get("overhead_seconds"),
                "error_pct": error, "tolerance_pct": args.tolerance, "timing_basis": generation.get("basis"),
                "confidence": generation.get("confidence"), "run_status": report_data["status"],
                "note": "Measured overhead exceeds the simulated value by this machine's helper start-up cost."})
    if not passed:
        raise SystemExit(1)


def positive(value):
    number = int(value)
    if not 1 <= number <= 1020:
        raise argparse.ArgumentTypeError("Use a value between 1 and 1020.")
    return number


def client_name(value):
    if value != "auto" and not re.fullmatch(r"[a-z][a-z0-9_-]{1,31}", value):
        raise argparse.ArgumentTypeError("Use a known client key (see `providers`) or a short lowercase name.")
    return value


def build_parser():
    parser = argparse.ArgumentParser(prog="benchmark.py", description="Benchmark the current chat's generation speed in any agent CLI.")
    parser.add_argument("--version", action="version", version=f"chat-benchmark {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("install", "doctor", "providers", "collector", "prepare", "sample", "burst", "write-sample",
                    "status", "report", "follow", "history", "compare", "selftest"):
        sub = commands.add_parser(command)
        sub.add_argument("--data-home", type=Path, default=None)
        if command == "install":
            sub.add_argument("--clients", help="Comma-separated client keys; default: every detected client")
            sub.add_argument("--all-clients", action="store_true", help="Also install for clients that are not detected")
            sub.add_argument("--no-telemetry", action="store_true", help="Install skills only; leave client telemetry settings untouched")
        if command in ("prepare", "report"):
            sub.add_argument("--provider", type=client_name, default="auto",
                             help="Client key such as claude, codex, kimi, agy, gemini, grok; any other name runs with helper-clock timing")
            sub.add_argument("--session-id")
            sub.add_argument("--session-path")
            sub.add_argument("--client-home", type=Path)
        if command == "prepare":
            sub.add_argument("--preset", choices=("quick", "standard", "precise"), default="standard")
            sub.add_argument("--durations", type=lambda s: [int(v) for v in s.split(',')], help="Nominal stage seconds, e.g. 10,30,60")
            sub.add_argument("--warmup-seconds", type=int)
            sub.add_argument("--max-seconds", type=int, help="Between-stage wall-time budget; cannot interrupt a live message")
            sub.add_argument("--strict-stages", action="store_true", help="Never substitute a shorter stage when the budget runs low")
            sub.add_argument("--trust-provider", action="store_true", help=argparse.SUPPRESS)
            sub.add_argument("--delivery", choices=("auto", "stdin", "file"), default="auto",
                             help="How samples reach the helper: shell stdin, or a file written by the client's file tool")
            sub.add_argument("--model", help="Label for clients that do not expose the active model (self-reported)")
            sub.add_argument("--effort", help="Label for clients that do not expose thinking effort (self-reported)")
            sub.add_argument("--tier", help="Label for clients that do not expose a service tier (self-reported)")
        if command in ("sample", "burst", "write-sample", "status", "report", "follow"):
            sub.add_argument("--run-id", required=command != "report")
        if command in ("sample", "burst", "write-sample"):
            sub.add_argument("--index", type=positive, required=True)
            if command == "sample":
                sub.add_argument("--payload", required=True)
            if command == "write-sample":
                sub.add_argument("--text", help="Literal model-generated output; prefer quoted stdin for multiline content")
        if command == "status":
            sub.add_argument("--wait", type=float, default=0)
        if command == "report":
            sub.add_argument("--last", type=positive, default=10)
            sub.add_argument("--wait", type=float, default=0)
            sub.add_argument("--output-dir", type=Path)
            display = sub.add_mutually_exclusive_group()
            display.add_argument("--json", action="store_true")
            display.add_argument("--chat", action="store_true", help="Show the compact chat summary and stage tables")
            sub.add_argument("--follow-turn", action="store_true")
            sub.add_argument("--open", action="store_true", help="Open the HTML report in the default browser")
        if command == "compare":
            sub.add_argument("--last", type=positive, default=200, help="Most recent runs to consider")
            sub.add_argument("--json", action="store_true")
            sub.add_argument("--open", action="store_true", help="Write and open the HTML leaderboard")
        if command == "collector":
            sub.add_argument("--port", type=int, default=PORT)
            sub.add_argument("--idle-seconds", type=int, default=1200)
        if command == "follow":
            sub.add_argument("--timeout", type=int, default=600)
        if command == "selftest":
            sub.add_argument("--tps", type=float, default=400, help="Simulated generation speed")
            sub.add_argument("--overhead", type=float, default=.3, help="Simulated fixed seconds per request")
            sub.add_argument("--tolerance", type=float, default=8, help="Allowed error in percent")
    return parser


def main(started_ms=None):
    invoked_ms = process_start_ms(started_ms)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    args.invoked_ms = invoked_ms
    root = args.data_home or data_home()
    run_id, code = getattr(args, "run_id", None), 0
    try:
        if args.command == "install":
            clients = [c.strip() for c in args.clients.split(",") if c.strip()] if args.clients else None
            print_json(install(root, clients=clients, all_clients=args.all_clients, telemetry=not args.no_telemetry))
        elif args.command == "doctor":
            doctor(root)
        elif args.command == "providers":
            print(capability_table(root))
        elif args.command == "history":
            print_json({"groups": history(root), "note": "Grouped by target duration, client/settings/version, context band, and cache band; warm-ups excluded."})
        elif args.command == "compare":
            result = compare(root, args.last)
            if args.json:
                print_json(result)
            else:
                print(compare_markdown(result))
            if args.open:
                from .html import leaderboard_html
                path = root / "results" / "leaderboard.html"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(leaderboard_html(result), encoding="utf-8")
                import webbrowser
                webbrowser.open(path.as_uri())
        elif args.command == "collector":
            serve(root, args.port, args.idle_seconds)
        elif args.command == "prepare":
            run_id = prepare(args, root)
        elif args.command in ("sample", "burst"):
            capture_sample(args, root)
        elif args.command == "write-sample":
            capture_file(args, root)
        elif args.command == "status":
            if not 0 <= args.wait <= 30:
                raise ValueError("--wait must be between 0 and 30 seconds.")
            run_status(args, root)
        elif args.command == "report":
            if args.wait < 0 or args.wait > 30:
                raise ValueError("--wait must be between 0 and 30 seconds.")
            report(args, root)
        elif args.command == "follow":
            follow(args, root)
        elif args.command == "selftest":
            selftest(args, root)
    except (ValueError, OSError, RuntimeError) as exc:
        print_json({"error": str(exc), "status": "unavailable"})
        code = 2
    finally:
        # Stamp even after a rejected sample: a retry is timed from this exit.
        if run_id and args.command in ("prepare", "sample", "burst", "write-sample", "status"):
            stamp_exit(root, run_id, f"{args.command}:{getattr(args, 'index', '')}".rstrip(":"))
    if code:
        raise SystemExit(code)
