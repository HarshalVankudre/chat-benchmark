"""Loopback-only OTLP/HTTP JSON receiver. Store performance fields, not content."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import closing
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .metrics import Request

PORT = 43189
ALLOWED = {
    "service.name", "service.version", "app.version", "event.name", "event.kind", "event.timestamp", "kind", "type",
    "conversation.id", "thread.id", "thread_id", "turn.id", "turn_id", "session.id", "session_id",
    "ccr.session.id", "response.id", "response_id", "request_id", "client_request_id",
    "model", "model_name", "gen_ai.request.model", "gen_ai.response.model",
    "reasoning_effort", "model_reasoning_effort", "codex.request.reasoning_effort",
    "codex.turn.reasoning_effort", "effort", "service_tier", "speed",
    "duration_ms", "ttft_ms", "first_content_ms", "time_to_first_token_ms",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens",
    "input_token_count", "output_token_count", "cached_token_count", "cache_write_token_count",
    "reasoning_token_count", "tool_token_count", "attempt", "success", "status_code",
    "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.cache_write.input_tokens", "codex.usage.reasoning_output_tokens",
    "codex.usage.total_tokens", "codex.turn.ttft.duration_ms", "interaction.duration_ms",
    "response.has_tool_call", "stop_reason", "query_source_safe",
    "cost_usd", "cost_usd_micros",
}
LOG_NAMES = {"codex.sse_event", "codex.websocket_event", "codex.api_request",
             "codex.websocket_request", "codex.turn_ttft", "codex.conversation_starts",
             "claude_code.api_request", "claude_code.api_error"}
SPAN_NAMES = {"claude_code.llm_request", "claude_code.interaction", "turn", "session_task.turn",
              "run_sampling_request", "try_run_sampling_request", "responses_websocket.stream_request"}


def data_home() -> Path:
    return Path(os.environ.get("CHAT_BENCHMARK_HOME") or Path.home() / ".chat-benchmark")


def _value(value):
    for key in ("stringValue", "intValue", "doubleValue", "boolValue",
                "string_value", "int_value", "double_value", "bool_value"):
        if key in value:
            result = value[key]
            if key in ("intValue", "int_value"):
                try:
                    return int(result)
                except (TypeError, ValueError):
                    return None
            return result
    return None


def attributes(items) -> dict:
    return {item["key"]: value for item in items or [] if item.get("key") in ALLOWED
            and (value := _value(item.get("value", {}))) is not None}


def nanos_to_ms(value):
    try:
        return int(value) / 1_000_000 if value and int(value) > 0 else None
    except (TypeError, ValueError):
        return None


def normalize(payload: dict, signal: str) -> list[dict]:
    """Allowlist at ingestion; arbitrary prompt, response, or command text is discarded."""
    events = []
    roots = payload.get("resourceSpans" if signal == "traces" else "resourceLogs", [])
    for root in roots:
        resource = attributes(root.get("resource", {}).get("attributes", []))
        scopes = root.get("scopeSpans" if signal == "traces" else "scopeLogs", [])
        for scope in scopes:
            for item in scope.get("spans" if signal == "traces" else "logRecords", []):
                attrs = {**resource, **attributes(item.get("attributes", []))}
                name = item.get("name") if signal == "traces" else attrs.get("event.name")
                if signal == "logs" and isinstance(name, str) and "claude" in str(attrs.get("service.name", "")):
                    if "claude_code." + name in LOG_NAMES:
                        name = "claude_code." + name
                if not name and signal == "logs":
                    body = _value(item.get("body", {}))
                    if body in LOG_NAMES:
                        name = body
                if name not in (SPAN_NAMES if signal == "traces" else LOG_NAMES):
                    continue
                provider = "claude" if name.startswith("claude_code.") else "codex"
                sid = next((attrs.get(k) for k in ("conversation.id", "thread.id", "thread_id", "session.id", "session_id")
                            if attrs.get(k)), None)
                if not sid:
                    # Unattributed data must never be assigned to whichever chat ran last.
                    continue
                start = nanos_to_ms(item.get("startTimeUnixNano"))
                end = nanos_to_ms(item.get("endTimeUnixNano")) or nanos_to_ms(item.get("timeUnixNano"))
                # Native Codex currently exports timeUnixNano=0 and puts its real
                # emission timestamp in an attribute. Never substitute receipt time.
                if end is None and attrs.get("event.timestamp"):
                    try:
                        end = datetime.fromisoformat(str(attrs["event.timestamp"]).replace("Z", "+00:00")).timestamp() * 1000
                    except ValueError:
                        pass
                end = end or nanos_to_ms(item.get("observedTimeUnixNano"))
                record = {"provider": provider, "session_id": sid,
                          "turn_id": attrs.get("turn.id") or attrs.get("turn_id"),
                          "name": name, "start_ms": start, "end_ms": end, "attributes": attrs}
                record["id"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
                events.append(record)
    return events


def connect_store(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "telemetry.sqlite", timeout=10)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY, provider TEXT NOT NULL, session_id TEXT NOT NULL,
        turn_id TEXT, name TEXT NOT NULL, start_ms REAL, end_ms REAL, attributes TEXT NOT NULL)""")
    connection.execute("CREATE INDEX IF NOT EXISTS session_time ON events(session_id,end_ms)")
    return connection


def store_events(root: Path, events: list[dict]):
    with closing(connect_store(root)) as conn:
        with conn:
            conn.executemany("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?)",
                             [(e["id"], e["provider"], e["session_id"], e["turn_id"], e["name"],
                               e["start_ms"], e["end_ms"], json.dumps(e["attributes"])) for e in events])
            conn.execute("DELETE FROM events WHERE end_ms < ?", ((time.time() - 14 * 86400) * 1000,))


def read_events(root: Path, provider: str, session_id: str) -> list[dict]:
    db = root / "telemetry.sqlite"
    if not db.exists():
        return []
    conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM events WHERE provider=? AND session_id=? ORDER BY end_ms",
                            (provider, session_id)).fetchall()
        return [{**dict(r), "attributes": json.loads(r["attributes"])} for r in rows]
    finally:
        conn.close()


def _number(attrs, *keys):
    for key in keys:
        if attrs.get(key) is not None:
            try:
                return float(attrs[key])
            except (TypeError, ValueError):
                continue
    return None


def enrich(requests: list[Request], events: list[dict]) -> None:
    used = set()
    for request in requests:
        for event in events:
            a = event["attributes"]
            if event["session_id"] == request.session_id and a.get("request_id") == request.request_id:
                cost = _number(a, "cost_usd")
                if cost is None:
                    micros = _number(a, "cost_usd_micros")
                    cost = micros / 1_000_000 if micros is not None else None
                if cost is not None:
                    request.client_estimated_usd = cost
        candidates = []
        for event in events:
            if event["id"] in used or event["session_id"] != request.session_id:
                continue
            attrs = event["attributes"]
            if request.provider == "claude":
                if event["name"] != "claude_code.llm_request" or attrs.get("request_id") != request.request_id:
                    continue
                candidates.append((0, event))
            elif event["name"] in ("codex.sse_event", "codex.websocket_event"):
                kind = attrs.get("event.kind") or attrs.get("kind") or attrs.get("type")
                if kind not in ("response.completed", "completed"):
                    continue
                rid = attrs.get("response.id") or attrs.get("response_id")
                exact = rid == request.request_id if rid else False
                distance = abs((event.get("end_ms") or 0) - (request.completed_ms or 0))
                count = _number(attrs, "output_token_count", "output_tokens", "gen_ai.usage.output_tokens")
                same_count = count is not None and count == request.output_tokens
                if not exact and same_count:
                    # The event must prefer this response, too. Otherwise a missing
                    # event on a nearby response could steal the next response's TTFT.
                    eligible = [r for r in requests if r.session_id == request.session_id
                                and r.output_tokens == count and r.completed_ms is not None]
                    nearest = min(eligible, key=lambda r: abs(r.completed_ms - (event.get("end_ms") or 0)))
                    same_count = nearest is request
                if exact or (same_count and distance < 1500):
                    candidates.append((0 if exact else distance + 1, event))
        if not candidates:
            continue
        event = min(candidates, key=lambda x: x[0])[1]
        used.add(event["id"])
        attrs = event["attributes"]
        request.client_version = attrs.get("app.version") or attrs.get("service.version") or request.client_version
        ttft = _number(attrs, "ttft_ms", "time_to_first_token_ms")
        if ttft is not None and ttft >= 0:
            request.ttft_ms = ttft
            request.ttft_source = "native " + event["name"]
        request.first_content_ms = _number(attrs, "first_content_ms")
        if request.provider == "claude":
            duration = _number(attrs, "duration_ms")
            if duration is None and event.get("start_ms") is not None and event.get("end_ms") is not None:
                duration = event["end_ms"] - event["start_ms"]
            if duration is not None and duration >= 0:
                request.duration_ms = duration
                request.timing_source = "native claude_code.llm_request span"
                request.started_ms = event.get("start_ms", request.started_ms)
                request.completed_ms = event.get("end_ms", request.completed_ms)
            if "success" in attrs:
                success = attrs["success"]
                request.status = "completed" if success in (True, "true", "True", 1) else "failed"
        for field, keys in {
            "effort": ("effort", "model_reasoning_effort", "reasoning_effort", "codex.request.reasoning_effort"),
            "service_tier": ("service_tier",), "speed": ("speed",),
            "model": ("model", "gen_ai.response.model", "gen_ai.request.model"),
        }.items():
            value = next((attrs[k] for k in keys if attrs.get(k) is not None), None)
            if value is not None:
                setattr(request, field, "normal" if field == "speed" and value == "standard" else str(value))
        attempt = _number(attrs, "attempt")
        if attempt is not None:
            request.attempts = int(attempt)


def health(port=PORT) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
            state = json.load(response)
            return state if state.get("service") == "chat-benchmark" else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def ensure_collector(root: Path, port=PORT) -> dict:
    state = health(port)
    if state:
        if Path(state.get("data_home", "")).resolve() != root.resolve():
            raise ValueError("The local benchmark port is serving a different data directory.")
        return state
    root.mkdir(parents=True, exist_ok=True)
    entry = Path(__file__).resolve().parents[1] / "benchmark.py"
    flags = (subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS) if os.name == "nt" else 0
    with (root / "collector.log").open("ab") as log:
        subprocess.Popen([sys.executable, str(entry), "collector", "--data-home", str(root),
                          "--port", str(port)], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         creationflags=flags, start_new_session=os.name != "nt")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = health(port)
        if state:
            return state
        time.sleep(.1)
    raise RuntimeError("Could not start the local collector. Check collector.log or a port conflict.")


def serve(root: Path, port=PORT, idle_seconds=1200):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, result):
            body = json.dumps(result).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # The exporter hung up first; nothing to answer.

        def do_GET(self):
            self.server.last_activity = time.monotonic()
            if self.path == "/health":
                self.reply(200, {"service": "chat-benchmark", "pid": os.getpid(),
                                 "data_home": str(root.resolve()), "content_storage": False})
            else:
                self.reply(404, {})

        def do_POST(self):
            self.server.last_activity = time.monotonic()
            if self.path not in ("/v1/logs", "/v1/traces"):
                self.reply(404, {})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 8 * 1024 * 1024:
                    self.reply(413, {})
                    return
                content_type = self.headers.get("Content-Type", "")
                if "json" not in content_type:
                    self.reply(415, {"error": "Configure OTLP/HTTP JSON, not protobuf."})
                    return
                body = self.rfile.read(size)
                if self.headers.get("Content-Encoding") == "gzip":
                    # Bound decompression too; exporters only send small metadata batches.
                    import io
                    with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
                        body = stream.read(8 * 1024 * 1024 + 1)
                if len(body) > 8 * 1024 * 1024:
                    self.reply(413, {})
                    return
                payload = json.loads(body)
                store_events(root, normalize(payload, self.path.rsplit("/", 1)[-1]))
                self.reply(200, {})
            except (ValueError, TypeError, KeyError, OSError):
                self.reply(400, {})

    root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.timeout = 1
    server.last_activity = time.monotonic()
    try:
        while time.monotonic() - server.last_activity < idle_seconds:
            server.handle_request()
    finally:
        server.server_close()
