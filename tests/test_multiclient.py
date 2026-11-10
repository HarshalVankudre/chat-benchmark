import support  # noqa: F401  (isolates the home directory; keep first)

import argparse
import contextlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from chat_benchmark import adapters
from chat_benchmark.cli import capture_file, load_run, save_run
from chat_benchmark.clock import bracket, process_start_ms
from chat_benchmark.estimate import fit_sweep, grade
from chat_benchmark.history import compare, compare_markdown
from chat_benchmark.install import install
from chat_benchmark.protocol import plan, progress
from chat_benchmark.providers import PROVIDERS, detect, late_bind, open_session, provider_for
from chat_benchmark.reports import chat_table, standardize_report, write_report
from chat_benchmark.sessions import Session
from chat_benchmark.standard import reference_tokens
from chat_benchmark.sweep import generation_summary
from chat_benchmark.temporary import create_inbox, inbox_name, remove_inbox, take_inbox_sample
from chat_benchmark.workload import canonical_text, reference_digest

ROOT = Path(__file__).resolve().parents[1]
RUN = "c" * 32


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def stage(index, rows, seconds, **extra):
    """A verified measured stage as it appears in a saved report."""
    sample = dict(sample_index=index, request_id=f"r{index}", phase="measured", status="completed",
                  provider="kimi", model="k-test", effort="high", service_tier=None, speed=None,
                  reference_rows=rows, reference_matches=True, continuity="single_file_payload",
                  payload_sha256=reference_digest(rows), duration_ms=seconds * 1000,
                  timing_source="native fixture", output_tokens=reference_tokens(rows) + 60, reasoning_tokens=0,
                  target_seconds={64: 10, 192: 30, 384: 60}.get(rows), ttft_ms=900)
    sample.update(extra)
    return sample


def sweep(tps=80.0, overhead=2.0, rows=(64, 192, 384), **extra):
    return [stage(i + 2, n, overhead + reference_tokens(n) / tps, **extra) for i, n in enumerate(rows)]


class EstimatorTests(unittest.TestCase):
    def test_slope_recovers_speed_and_separates_overhead(self):
        fit = fit_sweep([(reference_tokens(n), 2.0 + reference_tokens(n) / 80) for n in (64, 192, 384)])
        self.assertAlmostEqual(fit["tps"], 80, places=6)
        self.assertAlmostEqual(fit["overhead_seconds"], 2.0, places=6)
        self.assertLess(fit["pair_spread_pct"], 1e-6)
        self.assertEqual(grade(fit, "native")["level"], "high")

    def test_end_to_end_rate_is_length_biased_but_the_slope_is_not(self):
        samples = sweep(80, 2.0)
        standardize_report(report := {"samples": samples, "summary": {}})
        rates = [s["standard_tps"] for s in report["samples"]]
        self.assertLess(rates[0], 67)           # Diluted by two seconds of overhead.
        self.assertGreater(rates[2], 77)
        self.assertAlmostEqual(report["summary"]["generation"]["tps"], 80, places=4)

    def test_one_slow_stage_does_not_drag_a_repeated_sweep(self):
        points = [(reference_tokens(n), 1.5 + reference_tokens(n) / 100) for n in (64, 192, 384, 384, 192, 64)]
        points[1] = (points[1][0], points[1][1] + 9)  # One stage stalled for nine seconds.
        fit = fit_sweep(points)
        self.assertAlmostEqual(fit["tps"], 100, delta=3)
        self.assertIsNotNone(fit["ci95"])
        self.assertLessEqual(fit["ci95"]["low"], fit["tps"])
        self.assertGreaterEqual(fit["ci95"]["high"], fit["tps"])

    def test_confidence_levels_follow_the_evidence(self):
        self.assertIsNone(fit_sweep([(767, 10.0)]))
        self.assertEqual(grade(None, None)["level"], "none")
        two = fit_sweep([(767, 11.0), (2303, 30.0)])
        self.assertEqual(grade(two, "native")["level"], "low")
        noisy = fit_sweep([(767, 11.0), (2303, 30.0), (4607, 56.0)])
        self.assertEqual(grade(noisy, "native")["level"], "medium")
        wild = fit_sweep([(767, 11.0), (2303, 40.0), (4607, 52.0)])
        self.assertEqual(grade(wild, "native")["level"], "low")
        good = fit_sweep([(767, 11.0), (2303, 30.2), (4607, 59.0)])
        self.assertEqual(grade(good, "native")["level"], "high")
        self.assertEqual(grade(good, "bracket", verified=False)["level"], "medium")
        self.assertEqual(grade(good, "native", cross_check_pct=25)["level"], "medium")
        flat = fit_sweep([(767, 30.0), (2303, 30.0), (4607, 29.0)])
        self.assertEqual(flat["status"], "non_positive_slope")
        self.assertEqual(grade(flat, "native")["level"], "none")

    def test_native_and_helper_clock_estimates_cross_check(self):
        samples = sweep(80, 2.0)
        for s in samples:
            s["bracket_ms"] = s["duration_ms"] + 1600  # Constant client overhead.
        summary = generation_summary([{**s, "standard_tokens": reference_tokens(s["reference_rows"])} for s in samples])
        self.assertEqual(summary["basis"], "native")
        self.assertLess(summary["cross_check"]["native_vs_helper_clock_pct"], 1e-6)
        self.assertAlmostEqual(summary["helper_clock_overhead_ms"]["median"], 1600)
        self.assertEqual(summary["confidence"]["level"], "high")


class ClockTests(unittest.TestCase):
    def test_bracket_is_previous_exit_to_this_start_and_rejects_nonsense(self):
        self.assertEqual(bracket({"exit_ms": 1000.0}, 13500.0), 12500.0)
        self.assertIsNone(bracket(None, 5.0))
        self.assertIsNone(bracket({"exit_ms": 2000.0}, 1000.0))             # Clock went backwards.
        self.assertIsNone(bracket({"exit_ms": 0.0}, 3 * 3600 * 1000.0))    # Hours apart: not one response.

    def test_process_start_precedes_now_by_a_plausible_margin(self):
        now = time.time() * 1000
        started = process_start_ms(now)
        self.assertLessEqual(started, now)
        self.assertLess(now - started, 60_000)


class DetectionTests(unittest.TestCase):
    def test_process_chain_outranks_a_stale_hint_from_a_shared_skill_folder(self):
        chain = [{"pwsh"}, {"grok"}, {"windowsterminal"}]
        self.assertEqual(detect("claude", chain, {}).key, "grok")     # Grok loaded Claude's skill.
        self.assertEqual(detect("auto", chain, {}).key, "grok")
        self.assertEqual(detect("kimi", [{"bash"}, {"kimi"}, {"claude"}], {}).key, "kimi")  # Nearest wins.

    def test_hints_for_clients_without_a_native_binary_are_trusted(self):
        chain = [{"powershell"}, {"node"}, {"claude"}]                 # Gemini (Node) started from a Claude terminal.
        self.assertEqual(detect("gemini", chain, {}).key, "gemini")
        self.assertEqual(detect("some-new-cli", chain, {}).key, "some-new-cli")
        self.assertFalse(provider_for("some-new-cli").known)
        # An unlisted native CLI is recognised by its own name in the chain.
        self.assertEqual(detect("crush", [{"bash"}, {"crush"}, {"pwsh"}, {"claude"}], {}).key, "crush")
        # Antigravity also reads Gemini's skill folder: no script runtime in between, so the hint is stale.
        self.assertEqual(detect("gemini", [{"powershell"}, {"agy"}, {"explorer"}], {}).key, "agy")
        self.assertEqual(detect("aider", [{"pwsh"}, {"claude"}], {}).key, "claude")
        self.assertEqual(detect("aider", [{"pwsh"}], {}).key, "aider")   # Nothing recognised: the name is taken as given.

    def test_auto_uses_environment_markers_and_never_guesses(self):
        self.assertEqual(detect("auto", [], {"PI_SESSION_ID": "x"}).key, "pi")
        self.assertEqual(detect("auto", [], {"OPENCODE": "1"}).key, "opencode")
        with self.assertRaisesRegex(ValueError, "Cannot identify"):
            detect("auto", [], {})
        with self.assertRaises(ValueError):
            provider_for("Not A Key!")

    def test_unidentified_chat_degrades_to_helper_clock_instead_of_failing(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, note = open_session(PROVIDERS["kimi"], "${KIMI_SESSION_ID}", None, tmp)
            self.assertFalse(session.tracked)
            self.assertTrue(session.session_id.startswith("untracked-"))    # A placeholder is not an id.
            self.assertIn("matched by run id", note)
            session, note = open_session(provider_for("aider"), None, None, None)
            self.assertFalse(session.tracked)
            self.assertIn("helper-clock", note)


class KimiAdapterTests(unittest.TestCase):
    def rows(self, command, start=1794168442670, first=2316, stream=482):
        return [
            {"type": "metadata", "protocol_version": "1.5", "created_at": start - 5000},
            {"type": "turn.prompt", "agentId": "main", "turnId": 0, "time": start - 100},
            {"type": "context.append_loop_event", "agentId": "main", "event": {"type": "step.begin", "uuid": "step-1", "turnId": "0", "step": 1}, "time": start - 7},
            {"type": "llm.request", "agentId": "main", "kind": "loop", "model": "k3.1-pro", "modelAlias": "kimi-code/k3.1-pro", "thinkingEffort": "high", "turnStep": "0.1", "attempt": "1/10", "time": start - 9000},
            {"type": "llm.request", "agentId": "main", "kind": "loop", "model": "k3.1-pro", "thinkingEffort": "high", "turnStep": "0.1", "attempt": "2/10", "time": start},
            {"type": "usage.record", "agentId": "main", "usage": {"inputOther": 747, "output": 900, "inputCacheRead": 13312, "inputCacheCreation": 0}, "time": start + first + stream},
            {"type": "llm.request", "agentId": "agent-1", "kind": "loop", "turnStep": "0.1", "time": start + 1},   # A subagent: ignored.
            {"type": "context.append_loop_event", "agentId": "main", "event": {"type": "content.part", "turnId": "0", "step": 1, "part": {"type": "think", "think": "private"}}, "time": start + first + stream + 3},
            {"type": "context.append_loop_event", "agentId": "main", "event": {"type": "tool.call", "turnId": "0", "step": 1, "stepUuid": "step-1", "name": "Bash", "args": {"command": command, "timeout": 60}}, "time": start + first + stream + 50},
            {"type": "context.append_loop_event", "agentId": "main", "event": {"type": "step.end", "uuid": "step-1", "turnId": "0", "step": 1, "finishReason": "tool_use", "llmFirstTokenLatencyMs": first, "llmStreamDurationMs": stream}, "time": start + first + stream + 900},
            {"type": "turn.ended", "agentId": "main", "turnId": 0, "reason": "completed", "durationMs": 9000, "time": start + 9000},
        ]

    def test_request_timing_usage_settings_and_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            wire = home / "sessions" / "wd_x" / "session_abc12345" / "agents" / "main" / "wire.jsonl"
            command = f"python benchmark.py write-sample --run-id {RUN} --index 2 <<'CB'\n{canonical_text(64)}\nCB"
            jsonl(wire, self.rows(command))
            self.assertEqual(adapters.kimi_find(home, "session_abc12345"), wire)
            self.assertEqual(adapters.kimi_find(home, "abc12345"), wire)
            self.assertEqual(adapters.kimi_session_id(wire), "session_abc12345")
            requests, settings, turns = adapters.parse_kimi(Session("kimi", "session_abc12345", wire, home))
        self.assertEqual(len(requests), 1)
        r = requests[0]
        self.assertEqual(r.duration_ms, 2316 + 482)           # Timed from the attempt that produced the output.
        self.assertEqual(r.attempts, 2)
        self.assertEqual((r.ttft_ms, r.stream_ms), (2316, 482))
        self.assertEqual((r.input_tokens, r.cached_input_tokens, r.output_tokens), (14059, 13312, 900))
        self.assertEqual((r.model, r.effort), ("k3.1-pro", "high"))
        self.assertEqual((r.run_id, r.sample_index), (RUN, 2))
        self.assertEqual((r.tool_call_count, r.tool_input_rows, r.text_block_count), (1, 64, 0))
        self.assertEqual(r.timing_basis, "native")
        self.assertEqual(settings["model"], "k3.1-pro")
        self.assertEqual(turns[0]["status"], "completed")
        self.assertNotIn("private", json.dumps(r.to_dict()))


class GrokAdapterTests(unittest.TestCase):
    def test_joins_log_timeline_and_tool_call_by_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            home, sid = Path(tmp), "01a1aaaa-bbbb-7ccc-8ddd-0123456789ab"
            directory = home / "sessions" / "C%3A%5Cwork" / sid
            end = 1794312142499
            jsonl(home / "logs" / "unified.jsonl", [
                {"ts": "2026-11-10T12:02:19.878Z", "src": "shell", "ver": "1.0.52", "sid": "other", "msg": "shell.turn.inference_done", "ctx": {"model_elapsed_ms": 1, "completion_tokens": 9}},
                {"ts": "2026-11-10T12:02:19.879Z", "src": "shell", "ver": "1.0.52", "sid": sid, "msg": "shell.turn.inference_start", "ctx": {"loop_index": 2}},
                {"ts": "2026-11-10T12:02:22.499Z", "src": "shell", "ver": "1.0.52", "sid": sid, "msg": "shell.turn.inference_done",
                 "ctx": {"loop_index": 2, "model_elapsed_ms": 2620, "attempts": 1, "prompt_tokens": 19533, "cached_prompt_tokens": 19200, "completion_tokens": 974, "reasoning_tokens": 161}},
            ])
            jsonl(directory / "events.jsonl", [
                {"ts": "2026-11-10T12:02:19.878Z", "type": "loop_started", "loop_index": 1},
                {"ts": "2026-11-10T12:02:21.694Z", "type": "first_token"},
            ])
            command = "@'\n" + canonical_text(64) + f"\n'@ | python benchmark.py write-sample --run-id {RUN} --index 2"
            jsonl(directory / "updates.jsonl", [
                {"timestamp": 1794312142, "method": "session/update", "params": {"sessionId": sid, "update": {"sessionUpdate": "tool_call", "toolCallId": "call-1", "rawInput": {"command": command, "description": "Benchmark sample 2"}}, "_meta": {"agentTimestampMs": end + 14}}},
                {"timestamp": 1794312200, "method": "_x.ai/session/update", "params": {"sessionId": sid, "update": {"sessionUpdate": "auto_compact_completed"}, "_meta": {"agentTimestampMs": end + 60000}}},
            ])
            jsonl(directory / "chat_history.jsonl", [{"type": "assistant", "tool_calls": [{"id": "call-1"}], "model_id": "grok-4.7-build-fast", "reasoning_effort": "xhigh"}])
            (directory / "summary.json").write_text(json.dumps({"current_model_id": "grok-4.7", "reasoning_effort": "low"}))
            self.assertEqual(adapters.grok_find(home, sid), directory / "updates.jsonl")
            requests, settings, _ = adapters.parse_grok(Session("grok", sid, directory / "updates.jsonl", home))
        self.assertEqual(len(requests), 1)                      # The other chat's row is never borrowed.
        r = requests[0]
        self.assertEqual(r.duration_ms, 2620)
        self.assertAlmostEqual(r.ttft_ms, 1816, delta=1)
        self.assertEqual((r.input_tokens, r.cached_input_tokens, r.output_tokens, r.reasoning_tokens), (19533, 19200, 974, 161))
        self.assertEqual((r.model, r.effort), ("grok-4.7-build-fast", "xhigh"))   # Per-message beats the session summary.
        self.assertEqual((r.run_id, r.sample_index, r.tool_call_count, r.tool_input_rows), (RUN, 2, 1, 64))
        self.assertEqual(r.client_version, "1.0.52")
        self.assertEqual(r.context_epoch, 0)


class PiAndOpenCodeAdapterTests(unittest.TestCase):
    def test_pi_start_is_inner_timestamp_and_end_is_append_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            home, sid = Path(tmp), "019faaaa-bbbb-7ccc-8ddd-0123456789ab"
            path = home / "sessions" / "--C--work--" / f"2026-07-26T11-23-09-299Z_{sid}.jsonl"
            jsonl(path, [
                {"type": "session", "version": 3, "id": sid, "timestamp": "2026-07-26T11:23:09.299Z"},
                {"type": "thinking_level_change", "id": "a1", "thinkingLevel": "max"},
                {"type": "message", "id": "u1", "timestamp": "2026-07-26T11:23:52.000Z", "message": {"role": "user", "content": "x"}},
                {"type": "message", "id": "f1", "timestamp": "2026-07-26T11:23:58.920Z", "message": {
                    "role": "assistant", "model": "gpt-test", "stopReason": "toolUse", "timestamp": 1785065032223, "responseId": "resp_1",
                    "content": [{"type": "thinking", "thinking": "private"},
                                {"type": "toolCall", "name": "write", "arguments": {"path": f"/w/.chat-benchmark-tmp/{RUN}-inbox/cbench-{RUN}-3.txt", "content": canonical_text(192)}}],
                    "usage": {"input": 1501, "output": 2531, "cacheRead": 400, "cacheWrite": 0, "reasoning": 88, "cost": {"total": 0.014435}}}},
            ])
            self.assertEqual(adapters.pi_find(home, sid), path)
            self.assertEqual(adapters.pi_session_id(path), sid)
            requests, settings, _ = adapters.parse_pi(Session("pi", sid, path, home))
        r = requests[0]
        self.assertEqual(r.duration_ms, 6697)
        self.assertEqual((r.input_tokens, r.cached_input_tokens, r.output_tokens, r.reasoning_tokens), (1901, 400, 2531, 88))
        self.assertEqual((r.model, r.effort, r.client_estimated_usd), ("gpt-test", "max", 0.014435))
        self.assertEqual((r.run_id, r.sample_index, r.tool_input_rows), (RUN, 3, 192))   # File delivery is recognised by path.

    def test_opencode_generation_ends_when_the_tool_starts_and_is_found_by_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with contextlib.closing(sqlite3.connect(home / "opencode.db")) as db:
                db.execute("CREATE TABLE session(id TEXT, version TEXT, time_updated INTEGER)")
                db.execute("CREATE TABLE message(id TEXT, session_id TEXT, time_created INTEGER, data TEXT)")
                db.execute("CREATE TABLE part(id TEXT, message_id TEXT, session_id TEXT, time_created INTEGER, data TEXT)")
                start = 1784487227055
                db.execute("INSERT INTO session VALUES ('ses_a','1.18.25',?)", (start + 9000,))
                db.execute("INSERT INTO session VALUES ('ses_b','1.18.25',?)", (start + 9999,))    # More recent, wrong chat.
                message = {"role": "assistant", "parentID": "msg_u", "modelID": "glm-test", "variant": "max", "cost": 0.012,
                           "tokens": {"input": 7764, "output": 834, "reasoning": 53, "cache": {"read": 100, "write": 0}},
                           "time": {"created": start, "completed": start + 9000}, "finish": "tool-calls"}
                db.execute("INSERT INTO message VALUES ('msg_1','ses_a',?,?)", (start, json.dumps(message)))
                command = f"python benchmark.py write-sample --run-id {RUN} --index 2 <<'CB'\n{canonical_text(64)}\nCB"
                for offset, part in ((2106, {"type": "step-start"}),
                                     (2200, {"type": "tool", "tool": "bash", "state": {"status": "completed", "input": {"command": command}, "time": {"start": 4948, "end": 5000}}}),
                                     (5100, {"type": "step-finish", "reason": "tool-calls"})):
                    if part.get("state"):
                        part["state"]["time"] = {"start": start + 4948, "end": start + 5000}
                    db.execute("INSERT INTO part VALUES (?,?,?,?,?)", (f"prt_{offset}", "msg_1", "ses_a", start + offset, json.dumps(part)))
                db.commit()
            self.assertEqual(adapters.opencode_locate(home, RUN, start), (home / "opencode.db", "ses_a"))
            self.assertIsNone(adapters.opencode_locate(home, "f" * 32, start))
            requests, settings, _ = adapters.parse_opencode(Session("opencode", "ses_a", home / "opencode.db", home))
        r = requests[0]
        self.assertEqual(r.duration_ms, 4948)                  # Not message.time.completed, which includes the tool.
        self.assertEqual(r.ttft_ms, 2106)
        self.assertEqual((r.input_tokens, r.output_tokens, r.reasoning_tokens), (7864, 887, 53))
        self.assertEqual((r.model, r.effort, r.client_version), ("glm-test", "max", "1.18.25"))
        self.assertEqual((r.run_id, r.sample_index, r.tool_input_rows), (RUN, 2, 64))


def pb(*fields):
    """Tiny protobuf encoder for fixtures: (number, int | bytes | str) pairs."""
    def varint(n):
        out = bytearray()
        while True:
            out.append((n & 0x7F) | (0x80 if n > 0x7F else 0))
            n >>= 7
            if not n:
                return bytes(out)
    data = b""
    for number, value in fields:
        if isinstance(value, int):
            data += varint(number << 3) + varint(value)
        else:
            raw = value.encode() if isinstance(value, str) else value
            data += varint(number << 3 | 2) + varint(len(raw)) + raw
    return data


class AntigravityAdapterTests(unittest.TestCase):
    def test_decodes_protobuf_steps_and_locates_the_chat_by_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            home, conversation = Path(tmp), "3e11aaaa-bbbb-4ccc-8ddd-0123456789ab"
            (home / "conversations").mkdir()
            database = home / "conversations" / f"{conversation}.db"
            stamp = lambda seconds, nanos: pb((1, seconds), (2, nanos))
            arguments = json.dumps({"TargetFile": f"C:/w/.chat-benchmark-tmp/{RUN}-inbox/cbench-{RUN}-2.txt", "CodeContent": canonical_text(64)})
            usage = pb((1, 1417), (2, 2494), (3, 993), (5, 172616), (9, 25), (10, 968), (11, "resp-example"))
            metadata = pb((1, stamp(1794254065, 978621200)), (6, stamp(1794254071, 563111800)), (7, stamp(1794254071, 602244500)),
                          (9, usage), (20, pb((2, 119), (3, 41))))
            payload = pb((1, 15), (20, pb((7, pb((1, "call_1"), (2, "write_to_file"), (3, arguments))))))
            generator = pb((1, pb((19, "gemini-test-model"), (21, "Gemini Test (High)"))))
            with contextlib.closing(sqlite3.connect(database)) as db:
                db.execute("CREATE TABLE steps(idx INTEGER PRIMARY KEY, step_type INTEGER, metadata BLOB, step_payload BLOB)")
                db.execute("CREATE TABLE gen_metadata(idx INTEGER PRIMARY KEY, data BLOB)")
                db.execute("INSERT INTO steps VALUES (118, 14, ?, ?)", (b"", b""))
                db.execute("INSERT INTO steps VALUES (119, 15, ?, ?)", (metadata, payload))
                db.execute("INSERT INTO steps VALUES (120, 132, ?, ?)", (b"\xff\xff", b"\x0a\x99"))   # Malformed rows are skipped.
                db.execute("INSERT INTO gen_metadata VALUES (41, ?)", (generator,))
                db.commit()
            log = home / "brain" / conversation / ".system_generated" / "logs" / "transcript_full.jsonl"
            jsonl(log, [{"step_index": 119, "type": "PLANNER_RESPONSE", "tool_calls": [{"name": "write_to_file", "args": {"TargetFile": f"cbench-{RUN}-2.txt"}}]}])
            self.assertEqual(adapters.agy_locate(home, RUN, time.time() * 1000), (database, conversation))
            self.assertIsNone(adapters.agy_locate(home, "e" * 32, time.time() * 1000))
            requests, settings, _ = adapters.parse_agy(Session("agy", conversation, database, home))
        self.assertEqual(len(requests), 1)
        r = requests[0]
        self.assertAlmostEqual(r.duration_ms, 5623.6233, places=2)
        self.assertEqual((r.input_tokens, r.cached_input_tokens, r.output_tokens, r.reasoning_tokens), (175110, 172616, 993, 25))
        self.assertEqual((r.model, r.effort), ("gemini-test-model", "high"))
        self.assertIsNone(r.ttft_ms)                            # Arguments arrive in one burst: no pre-generation latency.
        self.assertEqual((r.run_id, r.sample_index, r.tool_call_count, r.tool_input_rows), (RUN, 2, 1, 64))


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.root = Path(self.workspace.name)

    def tearDown(self):
        self.workspace.cleanup()

    def run_data(self, provider="aider", inbox=False, durations=(2,)):
        run = dict(run_id=RUN, started_ms=time.time() * 1000, created_at="2026-11-10T10:00:00Z",
                   plan=plan(durations=list(durations), warmup_seconds=0, max_seconds=120), samples={},
                   samples_expected=len(durations), settings={"model": "label-model"}, labels={"model": "label-model"},
                   workload="fixed-temp-file-v1", accounting_start_request_id=None,
                   session=dict(provider=provider, session_id="untracked-1", path=None, home=None),
                   clock={"exit_ms": time.time() * 1000 - 4000, "command": "prepare"},
                   results_directory=str(self.root / "results" / "run"))
        if inbox:
            run["inbox"] = str(create_inbox(self.root / "work", RUN))
        save_run(self.root, run)
        return run

    def write(self, index=1, text=None, invoked=None):
        args = argparse.Namespace(command="write-sample", run_id=RUN, index=index, text=text,
                                  invoked_ms=invoked or time.time() * 1000)
        output = io.StringIO()
        with patch("chat_benchmark.temporary.temp_root", return_value=self.root / "sample-temp"), contextlib.redirect_stdout(output):
            capture_file(args, self.root)
        return json.loads(output.getvalue())

    def test_untracked_client_is_measured_by_the_helper_clock(self):
        self.run_data()
        result = self.write(text=canonical_text(13))
        self.assertAlmostEqual(result["helper_clock_ms"], 4000, delta=500)
        self.assertEqual(result["progress"]["reason"], "stages_complete")
        report = json.loads((self.root / "results/run/benchmark.json").read_text(encoding="utf-8"))
        sample = report["samples"][0]
        self.assertEqual(sample["timing_basis"], "bracket")
        self.assertEqual(sample["continuity"], "unverified_file_payload")
        self.assertEqual(sample["model"], "label-model")
        self.assertEqual(sample["standard_tokens"], 155)
        self.assertAlmostEqual(sample["standard_tps"], 155 / 4, delta=5)
        self.assertIsNone(sample["output_tokens"])               # Nothing native is invented.
        self.assertTrue((self.root / "results/run/benchmark.html").is_file())

    def test_file_delivery_reads_validates_and_deletes_the_model_written_file(self):
        run = self.run_data(inbox=True)
        inbox = Path(run["inbox"])
        with self.assertRaisesRegex(ValueError, "No sample file yet"):
            self.write()
        self.assertEqual(load_run(self.root, RUN)["samples"], {})
        (inbox / inbox_name(RUN, 1)).write_bytes(("\ufeff" + canonical_text(13).replace("\n", "\r\n") + "\r\n").encode("utf-8"))
        result = self.write()
        self.assertEqual(result["progress"]["reason"], "stages_complete")
        self.assertFalse(inbox.exists())                         # Removed once the run stops.
        self.assertFalse((self.root / "work" / ".chat-benchmark-tmp").exists())

    def test_inbox_cleanup_never_touches_foreign_files(self):
        inbox = create_inbox(self.root / "work", RUN)
        (inbox / "notes.txt").write_text("keep")
        self.assertFalse(remove_inbox(inbox, RUN))
        self.assertTrue((inbox / "notes.txt").exists())
        with self.assertRaises(ValueError):
            take_inbox_sample(self.root / "work", RUN, 1)        # Not an inbox.
        with self.assertRaises(ValueError):
            take_inbox_sample(inbox, "d" * 32, 1)                # Another run's inbox.

    def test_shell_truncation_is_reported_without_recording_a_sample(self):
        self.run_data(durations=(10,))
        cut = canonical_text(64)[:1000]                          # Stops in the middle of a row.
        with self.assertRaisesRegex(ValueError, "--delivery file"):
            self.write(text=cut)
        self.assertEqual(load_run(self.root, RUN)["samples"], {})
        short = canonical_text(40)                               # A model that stopped early is a real mismatch.
        self.assertEqual(self.write(text=short)["progress"]["reason"], "reference_mismatch")

    def test_transcript_found_later_by_run_id_upgrades_the_run(self):
        home = self.root / "kimi-home"
        wire = home / "sessions" / "wd_x" / "session_late1234" / "agents" / "main" / "wire.jsonl"
        run = self.run_data("kimi")
        run["session"]["home"] = str(home)
        self.assertIsNone(late_bind(run))                        # Nothing on disk yet.
        jsonl(wire, [{"type": "llm.request", "agentId": "main", "kind": "loop", "turnStep": "0.1", "model": "k", "time": 1},
                     {"type": "context.append_loop_event", "agentId": "main", "event": {"type": "tool.call", "turnId": "0", "step": 1, "args": {"command": f"write-sample --run-id {RUN} --index 1"}}, "time": 2}])
        jsonl(home / "sessions" / "wd_x" / "session_other999" / "agents" / "main" / "wire.jsonl", [{"type": "metadata"}])
        bound = late_bind(run)
        self.assertEqual((bound.session_id, bound.path), ("session_late1234", wire))
        self.assertEqual(run["session"]["path"], str(wire))


class ProtocolTests(unittest.TestCase):
    def test_precise_preset_mirrors_lengths_and_fits_its_budgets(self):
        p = plan("precise")
        self.assertEqual(p["durations_seconds"], [10, 30, 60, 60, 30, 10])
        self.assertGreaterEqual(p["max_total_words"], (32 + 2 * 640) * 9)
        self.assertEqual(plan("standard")["durations_seconds"], [10, 30, 60])
        with self.assertRaises(ValueError):
            plan(durations=[10, 30, 60], max_seconds=60)

    def test_slow_model_gets_a_shorter_third_stage_instead_of_none(self):
        from test_sweep import sample
        run = dict(run_id=RUN, started_ms=0, plan=plan(), samples={str(i): {} for i in (1, 2, 3)}, settings=sample().settings())
        # About 3.9 rows per second: the 384-row stage needs ~100 s, but only ~80 s remain.
        requests = [sample(1, "warmup", 5), sample(2, seconds=10), sample(3, seconds=30)]
        for request, seconds in zip(requests, (10.9, 18.7, 49.3)):
            request.duration_ms = seconds * 1000
        state = progress(run, requests, 100_000)
        self.assertEqual(state["action"], "write-sample")
        self.assertEqual(state["substituted_from"], 60)
        self.assertLess(state["target_seconds"], 60)
        self.assertEqual(state["target_rows"], round(state["target_seconds"] * 6.4))
        strict = dict(run, plan=plan(adaptive=False))
        self.assertEqual(progress(strict, requests, 100_000)["reason"], "remaining_budget")


class ReportTests(unittest.TestCase):
    def report(self, **changes):
        report = {"status": "complete", "run_id": "run-1", "created_at": "2026-11-10T10:00:00+00:00",
                  "workload": "fixed-temp-file-v1", "session": {"provider": "kimi", "session_id": "s"},
                  "plan": {"durations_seconds": [10, 30, 60], "warmups": 1},
                  "summary": {"settings": {"model": "k-test", "effort": "high"}, "settings_drift": []},
                  "samples": sweep(80, 2.0), "warmups": [], "notes": ["a note"], "completed_turns": []}
        report.update(changes)
        return report

    def test_chat_summary_leads_with_generation_speed_and_confidence(self):
        table = chat_table(standardize_report(self.report()))
        self.assertIn("**Kimi Code** · model `k-test` · effort `high`", table)
        self.assertIn("| **Generation speed** | **80.00 tok/s** |", table)
        self.assertIn("| Fixed overhead per request | 2.00 s |", table)
        self.assertIn("| End-to-end at 4,607 tokens |", table)
        self.assertIn("| Confidence | High · native request timing |", table)
        self.assertIn("| Stage | Rows | Request time | Standard TPS | Native TTFT |", table)

    def test_substituted_stage_and_derived_first_token_are_labelled(self):
        samples = sweep(80, 2.0, rows=(64, 192, 256), ttft_ms=None, prestream_ms=740)
        samples[2]["target_seconds"] = 40
        table = chat_table(standardize_report(self.report(samples=samples)))
        self.assertIn("| 40 s | 256 |", table)
        self.assertIn("40 s replaced the planned 60 s stage", table)
        self.assertIn("740 ms†", table)
        self.assertIn("(derived)", table)

    def test_single_stage_never_claims_a_generation_speed(self):
        table = chat_table(standardize_report(self.report(samples=sweep(80, 2.0)[:1], status="remaining_budget")))
        self.assertIn("| **Generation speed** | N/A (needs two verified lengths) |", table)
        self.assertIn("| Confidence | Not established", table)

    def test_all_artifacts_are_written_and_contain_no_sample_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.report(quality={"effective_tps": {"status": "observed", "reasons": []},
                                          "native_ttft": {"status": "observed", "reasons": []}, "native_ttft_coverage_pct": 100})
            report["summary"].update({key: {"n": 0, "median": None, "mean": None, "p95": None, "cv_pct": None}
                                      for key in ("effective_tps", "post_ttft_tps", "ttft_ms", "duration_ms")},
                                     valid_requests=3, weighted_effective_tps=None, output_tokens=None, input_tokens=None)
            for sample in report["samples"]:
                sample.update(effective_tps=1.0, cache_hit_pct=None)
            paths = write_report(report, Path(tmp))
            html = Path(paths["html"]).read_text(encoding="utf-8")
            for kind in ("md", "json", "csv", "html", "chat"):
                self.assertTrue(Path(paths[kind]).is_file())
                self.assertNotIn("amber birch", Path(paths[kind]).read_text(encoding="utf-8"))
        self.assertIn("<svg", html)
        self.assertIn("80.0", html)
        self.assertNotIn("http://", html.replace("http://www.w3.org", ""))   # Self-contained: no external requests.
        self.assertNotIn("https://", html)

    def test_leaderboard_ranks_configurations_and_skips_rejected_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save(name, **changes):
                directory = root / "results" / name
                directory.mkdir(parents=True)
                (directory / "benchmark.json").write_text(json.dumps(self.report(run_id=name, **changes)), encoding="utf-8")
            save("a1", samples=sweep(80, 2.0), created_at="2026-11-01T00:00:00+00:00")
            save("a2", samples=sweep(84, 2.0), created_at="2026-11-02T00:00:00+00:00")
            save("b1", samples=sweep(150, 3.0, provider="codex", model="gpt-test"), created_at="2026-11-03T00:00:00+00:00")
            save("bad", samples=sweep(999, 1.0), status="reference_mismatch")
            save("one", samples=sweep(500, 1.0)[:1])
            result = compare(root)
        self.assertEqual([g["client"] for g in result["groups"]], ["Codex", "Kimi Code"])
        kimi = result["groups"][1]
        self.assertEqual(kimi["runs"], 2)
        self.assertAlmostEqual(kimi["generation_tps"]["median"], 82, places=3)
        self.assertEqual(len(kimi["trend"]), 2)
        text = compare_markdown(result)
        self.assertIn("| 1 | Codex | gpt-test |", text)
        self.assertNotIn("999", text)


class InstallTests(unittest.TestCase):
    def test_every_client_gets_the_same_skill_with_its_own_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            env = {"HOME": tmp, "USERPROFILE": tmp, "CODEX_HOME": str(home / ".codex"), "CLAUDE_CONFIG_DIR": str(home / ".claude"),
                   "KIMI_CODE_HOME": str(home / ".kimi-code"), "GROK_HOME": str(home / ".grok"),
                   "PI_CODING_AGENT_DIR": str(home / ".pi" / "agent"), "XDG_DATA_HOME": str(home / ".local" / "share")}
            with patch.dict(os.environ, env):
                result = install(home / "data", all_clients=True, telemetry=False)
                again = install(home / "data", clients=["kimi", "agents"], telemetry=False)
            installed = {item["client"]: Path(item["files"][0]) for item in result["installed"]}
            self.assertGreaterEqual(set(installed), {"claude", "codex", "kimi", "grok", "agy", "gemini", "qwen", "opencode", "pi", "cursor", "copilot", "droid"})
            self.assertNotIn("agents", installed)               # The shared folder is opt-in.
            self.assertEqual({item["client"] for item in again["installed"]}, {"kimi", "agents"})
            self.assertEqual(result["telemetry"], {})
            self.assertFalse((home / ".claude" / "settings.json").exists())
            for key, path in installed.items():
                self.assertTrue(str(path).startswith(tmp), path)
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("{{", text.replace("{{args}}", ""))
                self.assertIn(f"prepare --provider {key} ", text)
                self.assertLess(len(text.encode("utf-8")), 24 * 1024)   # Antigravity's skill size limit.
            self.assertIn('--session-id="${KIMI_SESSION_ID}"', installed["kimi"].read_text(encoding="utf-8"))
            self.assertEqual(installed["agy"], home / ".gemini" / "antigravity-cli" / "skills" / "chat-benchmark" / "SKILL.md")
            self.assertTrue((home / ".gemini" / "commands" / "chat-benchmark.toml").is_file())
            self.assertIn("prepare --provider auto ", (home / ".agents" / "skills" / "chat-benchmark" / "SKILL.md").read_text(encoding="utf-8"))
            with patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, "Unknown client"):
                install(home / "data", clients=["nope"])


class EndToEndTests(unittest.TestCase):
    def test_simulated_client_speed_is_recovered_through_the_real_commands(self):
        done = subprocess.run([sys.executable, str(ROOT / "benchmark.py"), "selftest", "--tps", "300", "--overhead", "0.25", "--tolerance", "12"],
                              capture_output=True, text=True, encoding="utf-8", timeout=180)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        result = json.loads(done.stdout)
        self.assertEqual(result["selftest"], "passed")
        self.assertEqual(result["timing_basis"], "bracket")
        self.assertLess(result["error_pct"], 12)
        self.assertGreater(result["measured_overhead_seconds"], 0.2)


if __name__ == "__main__":
    unittest.main()
