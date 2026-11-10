from __future__ import annotations

import support  # noqa: F401  (isolates the home directory; keep first)

import json
import os
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from chat_benchmark.cli import WORDS, gather_report, validate_payload
from chat_benchmark.install import install
from chat_benchmark.metrics import Request, summarize
from chat_benchmark.reports import build_report, write_report
from chat_benchmark.sessions import Session, parse_claude, parse_codex, resolve_session
from chat_benchmark.telemetry import enrich, health, normalize, read_events, serve, store_events


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def attr(key, value):
    kind = "boolValue" if isinstance(value, bool) else "intValue" if isinstance(value, int) else "stringValue"
    return {"key": key, "value": {kind: str(value) if kind == "intValue" else value}}


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_codex_uses_response_usage_and_native_sampling_start(self):
        session_id, run_id = "current-session-123", "a" * 32
        path = self.root / "session.jsonl"
        rows = [
            {"timestamp": "2026-10-30T10:00:00Z", "type": "session_meta", "payload": {"id": session_id}},
            {"timestamp": "2026-10-30T10:00:00Z", "type": "turn_context", "payload": {"turn_id": "t1", "model": "gpt-test", "effort": "max"}},
            {"type": "event_msg", "payload": {"type": "thread_settings_applied", "thread_settings": {"model": "gpt-test", "reasoning_effort": "max", "service_tier": "priority"}}},
            {"timestamp": "2026-10-30T10:00:00Z", "type": "event_msg", "payload": {"type": "task_started", "turn_id": "t1"}},
            {"timestamp": "2026-10-30T10:00:01Z", "type": "event_msg", "payload": {"type": "item_completed", "started_at_ms": 1, "completed_at_ms": 2, "item": {"type": "AgentMessage"}}},
            {"timestamp": "2026-10-30T10:00:04Z", "type": "response_item", "payload": {"type": "custom_tool_call", "input": f"sample --payload CBENCH:{run_id}:1"}},
            {"timestamp": "2026-10-30T10:00:04Z", "type": "token_usage_record", "payload": {"response_id": "r1", "turn_id": "t1", "usage": {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 30, "reasoning_output_tokens": 10}, "turn_token_usage": {"output_tokens": 9999}}},
            {"timestamp": "2026-10-30T10:00:04Z", "type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": {"output_tokens": 9999}, "last_token_usage": {"output_tokens": 30}}}},
            {"timestamp": "2026-10-30T10:00:09Z", "type": "response_item", "payload": {"type": "custom_tool_call_output", "output": "tool time"}},
            {"timestamp": "2026-10-30T10:00:12Z", "type": "token_usage_record", "payload": {"response_id": "r2", "turn_id": "t1", "usage": {"input_tokens": 200, "output_tokens": 60, "reasoning_output_tokens": 0}}},
            {"timestamp": "2026-10-30T10:00:12Z", "type": "event_msg", "payload": {"type": "task_complete", "turn_id": "t1", "duration_ms": 12000, "time_to_first_token_ms": 357}},
        ]
        write_rows(path, rows)
        # Simulate a client still writing its last JSONL line.
        with path.open("a", encoding="utf-8") as file:
            file.write('{"type":')
        db = sqlite3.connect(self.root / "logs_2.sqlite")
        db.execute("CREATE TABLE logs(id INTEGER,ts INTEGER,ts_nanos INTEGER,target TEXT,thread_id TEXT,feedback_log_body TEXT)")
        epoch = 1793354400
        for index, sec in enumerate((epoch + 1, epoch + 10)):
            db.execute("INSERT INTO logs VALUES (?,?,?,?,?,?)", (index, sec, 0, "feedback_tags", session_id,
                       'turn{}: model="gpt-test" approval_policy=Never tags_json={}'))
        db.commit()
        db.close()
        requests, settings, turns = parse_codex(Session("codex", session_id, path, self.root))
        self.assertEqual(len(requests), 2)
        self.assertEqual([r.output_tokens for r in requests], [30, 60])
        self.assertEqual([r.duration_ms for r in requests], [3000, 2000])
        self.assertEqual(requests[0].run_id, run_id)
        self.assertEqual(settings["service_tier"], "priority")
        self.assertIsNone(requests[0].ttft_ms)  # Item start is NOT first-token timing.
        self.assertEqual(turns[0]["ttft_ms"], 357)  # Turn timing stays separate.

    def test_claude_deduplicates_blocks_and_normalizes_cached_input(self):
        path = self.root / "claude.jsonl"
        usage = {"input_tokens": 10, "cache_read_input_tokens": 100,
                 "cache_creation_input_tokens": 20, "output_tokens": 50,
                 "output_tokens_details": {"thinking_tokens": 5}, "service_tier": "standard", "speed": "standard"}
        rows = [
            {"type": "user", "uuid": "turn-a", "sessionId": "claude-123", "timestamp": "2026-10-30T10:00:00Z"},
            {"type": "api-request", "id": "api1", "timestamp": "2026-10-30T10:00:02Z", "params": {"model": "claude-test", "output_config": {"effort": "xhigh"}}},
        ]
        for block, second in (("thinking", "05"), ("text", "07"), ("tool_use", "08")):
            rows.append({"type": "assistant", "sessionId": "claude-123", "requestId": "req1", "requestRef": "api1",
                         "perTurnEffort": "xhigh", "timestamp": f"2026-10-30T10:00:{second}Z",
                         "message": {"id": "msg1", "model": "claude-test", "usage": usage,
                                     "content": [{"type": block, "input": {"command": "x"}}]}})
        write_rows(path, rows)
        requests, settings, _ = parse_claude(Session("claude", "claude-123", path, self.root))
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.output_tokens, 50)
        self.assertEqual(request.input_tokens, 130)
        self.assertEqual(request.duration_ms, 6000)
        self.assertEqual(request.effort, "xhigh")
        self.assertIsNone(request.ttft_ms)
        self.assertIsNone(request.post_ttft_tps)
        self.assertAlmostEqual(request.cache_hit_pct, 10000 / 130)

    def test_legacy_cumulative_usage_is_not_double_counted(self):
        path = self.root / "legacy.jsonl"
        count = {"type": "event_msg", "timestamp": "2026-10-30T10:00:01Z", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"output_tokens": 50}, "last_token_usage": {"output_tokens": 50}}}}
        write_rows(path, [{"type": "event_msg", "timestamp": "2026-10-30T10:00:00Z", "payload": {"type": "task_started"}}, count, count])
        requests, _, _ = parse_codex(Session("codex", "legacy-123", path, self.root))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].output_tokens, 50)

    def test_metrics_do_not_mix_settings_or_invent_decode_speed(self):
        a = Request("codex", "s1", "r1", model="a", effort="high", service_tier="default",
                    output_tokens=101, reasoning_tokens=0, duration_ms=2000, ttft_ms=1000)
        b = Request("codex", "s1", "r2", model="b", effort="high", service_tier="default",
                    output_tokens=900, reasoning_tokens=20, duration_ms=1000, ttft_ms=500)
        self.assertEqual(a.post_ttft_tps, 100)
        self.assertIsNone(b.post_ttft_tps)
        result = summarize([a, b], a.settings())
        self.assertEqual(result["valid_requests"], 1)
        self.assertEqual(result["effective_tps"]["median"], 50.5)
        self.assertEqual(result["settings_drift"][0]["fields"], ["model"])

    def test_session_identity_is_never_inferred_by_recency(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "current client"):
                resolve_session(home=self.root)
        path = self.root / "foreign.jsonl"
        write_rows(path, [{"type": "session_meta", "payload": {"id": "foreign-123"}}])
        with self.assertRaisesRegex(ValueError, "different session"):
            resolve_session("codex", "current-123", path, self.root)

    def test_otlp_drops_content_and_enriches_only_exact_session_and_request(self):
        now = int(time.time() * 1_000_000_000)
        payload = {"resourceSpans": [{"resource": {"attributes": [attr("service.name", "claude-code")]},
                    "scopeSpans": [{"spans": [{"name": "claude_code.llm_request", "startTimeUnixNano": str(now - 2_000_000_000),
                        "endTimeUnixNano": str(now), "attributes": [attr("session.id", "s1"), attr("request_id", "r1"),
                        attr("model", "claude-test"), attr("duration_ms", 2000), attr("ttft_ms", 500),
                        attr("effort", "max"), attr("success", True), attr("user_prompt", "PRIVATE PROMPT"),
                        attr("full_command", "PRIVATE COMMAND"), attr("response", "PRIVATE RESPONSE")]}]}]}]}
        events = normalize(payload, "traces")
        self.assertEqual(len(events), 1)
        self.assertNotIn("PRIVATE", json.dumps(events))
        store_events(self.root, events)
        store_events(self.root, events)  # Export retry deduplication.
        stored = read_events(self.root, "claude", "s1")
        self.assertEqual(len(stored), 1)
        request = Request("claude", "s1", "r1", output_tokens=100, reasoning_tokens=0)
        foreign = Request("claude", "other", "r1", output_tokens=100)
        enrich([request, foreign], stored)
        self.assertEqual(request.ttft_ms, 500)
        self.assertEqual(request.duration_ms, 2000)
        self.assertIsNone(foreign.ttft_ms)

    def test_codex_completion_event_matches_once_without_response_id(self):
        event = {"id": "e1", "session_id": "s1", "name": "codex.sse_event", "end_ms": 10000,
                 "attributes": {"event.kind": "response.completed", "output_token_count": 30, "ttft_ms": 400}}
        requests = [Request("codex", "s1", "r1", completed_ms=10005, output_tokens=30),
                    Request("codex", "s1", "r2", completed_ms=10100, output_tokens=30)]
        enrich(requests, [event])
        self.assertEqual(requests[0].ttft_ms, 400)
        self.assertIsNone(requests[1].ttft_ms)

    def test_native_codex_uses_event_timestamp_when_otlp_timestamp_is_zero(self):
        payload = {"resourceLogs": [{"scopeLogs": [{"logRecords": [{"timeUnixNano": "0",
                    "attributes": [attr("event.name", "codex.sse_event"), attr("conversation.id", "s1"),
                                   attr("event.kind", "response.completed"), attr("output_token_count", "30"),
                                   attr("ttft_ms", 400), attr("event.timestamp", "2026-10-30T10:00:00Z")]}]}]}]}
        events = normalize(payload, "logs")
        self.assertEqual(events[0]["end_ms"], 1793354400000)
        request = Request("codex", "s1", "r1", completed_ms=1793354400005, output_tokens=30)
        enrich([request], events)
        self.assertEqual(request.ttft_ms, 400)

    def test_payload_requires_model_output_of_the_standard_workload(self):
        run_id = "b" * 32
        payload = f"CBENCH:{run_id}:1|" + "|".join(f"{n:04d} {WORDS}" for n in range(1, 65))
        validate_payload(payload, run_id, 1)
        with self.assertRaisesRegex(ValueError, "exactly 64"):
            validate_payload(payload.rsplit("|", 1)[0], run_id, 1)
        with self.assertRaises(ValueError):
            validate_payload(payload, run_id, 2)

    def test_install_preserves_client_settings_and_is_repeatable(self):
        codex, claude = self.root / "codex", self.root / "claude"
        codex.mkdir(); claude.mkdir()
        config = 'model = "my-active-default"\nmodel_reasoning_effort = "max"\nservice_tier = "priority"\n'
        (codex / "config.toml").write_text(config)
        settings = {"model": "opus", "modelSettings": {"effort": "xhigh"}, "hooks": {"Stop": []}, "permissions": {"allow": ["Read"]}}
        (claude / "settings.json").write_text(json.dumps(settings))
        with patch.dict(os.environ, {"CODEX_HOME": str(codex), "CLAUDE_CONFIG_DIR": str(claude)}):
            first = install(self.root / "installed", clients=["codex", "claude"])
            second = install(self.root / "installed", clients=["codex", "claude"])
        import tomllib
        parsed = tomllib.loads((codex / "config.toml").read_text())
        self.assertEqual(parsed["service_tier"], "priority")
        self.assertEqual(parsed["model_reasoning_effort"], "max")
        self.assertEqual((codex / "config.toml").read_text().count("[otel]"), 1)
        parsed_claude = json.loads((claude / "settings.json").read_text())
        self.assertEqual({k: v for k, v in parsed_claude.items() if k != "env"}, settings)
        self.assertFalse(first["warnings"])
        self.assertFalse(second["warnings"])
        for home in (codex, claude):
            text = (home / "skills/chat-benchmark/SKILL.md").read_text(encoding="utf-8")
            self.assertNotIn("{{", text)
            self.assertTrue(text.startswith("---\nname: chat-benchmark\n"))
        self.assertIn('--provider claude --session-id="${CLAUDE_SESSION_ID}" ', (claude / "skills/chat-benchmark/SKILL.md").read_text(encoding="utf-8"))
        self.assertEqual(first["telemetry"], {"codex": "added", "claude": "added"})
        self.assertEqual(second["telemetry"], {"codex": "already configured", "claude": "already configured"})

    def test_local_http_receiver_and_report_artifacts(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        thread = threading.Thread(target=serve, args=(self.root, port, 1), daemon=True)
        thread.start()
        for _ in range(30):
            if health(port):
                break
            time.sleep(.05)
        self.assertEqual(health(port)["service"], "chat-benchmark")
        now = int(time.time() * 1_000_000_000)
        payload = {"resourceLogs": [{"scopeLogs": [{"logRecords": [{"timeUnixNano": str(now),
                    "attributes": [attr("event.name", "codex.sse_event"), attr("conversation.id", "s1"),
                                   attr("event.kind", "response.completed"), attr("output_token_count", 100),
                                   attr("ttft_ms", 600), attr("prompt", "DO NOT STORE")]}]}]}]}
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/logs", json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 200)
        stored = read_events(self.root, "codex", "s1")
        self.assertEqual(stored[0]["attributes"]["ttft_ms"], 600)
        sample = Request("codex", "s1", "r1", model="test", output_tokens=100,
                         reasoning_tokens=0, duration_ms=2000, ttft_ms=600)
        result = build_report({"provider": "codex", "session_id": "s1"}, [sample], sample.settings())
        paths = write_report(result, self.root / "results")
        self.assertTrue(all(Path(p).is_file() for p in paths.values()))
        self.assertEqual(json.loads(Path(paths["json"]).read_text())["summary"]["effective_tps"]["median"], 50)
        self.assertNotIn("DO NOT STORE", json.dumps(stored))
        thread.join(3)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
