"""Read performance metadata, never choose another conversation by recency."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .metrics import Request
from .workload import WORDS, text_digest

MARKER = re.compile(r"CBENCH:([0-9a-f]{32}):([1-9][0-9]*)")


def epoch_ms(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) * (1000 if value < 100_000_000_000 else 1)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000
    except (ValueError, TypeError):
        return None


def read_jsonl(path: Path) -> list[dict]:
    # The final line can be partially written while the client is running.
    rows = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as source:
        for line in source:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except json.JSONDecodeError:
                continue
    return rows


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def claude_home() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def readonly_db(path: Path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)


@dataclass
class Session:
    """A chat. `path` is None when the client's transcript is unknown (untracked):
    helper-clock timing still works, native usage and settings do not."""
    provider: str
    session_id: str
    path: Path | None
    home: Path | None

    @property
    def tracked(self) -> bool:
        return self.path is not None

    def to_dict(self):
        return {"provider": self.provider, "session_id": self.session_id,
                "path": str(self.path) if self.path else None,
                "home": str(self.home) if self.home else None}

    @classmethod
    def from_dict(cls, value):
        return cls(value["provider"], value["session_id"],
                   Path(value["path"]) if value.get("path") else None,
                   Path(value["home"]) if value.get("home") else None)


CLAUDE_SESSION_ENV = ("CLAUDE_SESSION_ID", "CLAUDE_CODE_SESSION_ID")
CODEX_SESSION_ENV = ("CODEX_THREAD_ID", "CODEX_SESSION_ID")


def first_env(names):
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


def reference_rows_in(text: str) -> int:
    """Reference rows present literally in tool-call arguments (provenance check)."""
    return text.count(WORDS)


def resolve_session(provider="auto", session_id=None, path=None, home=None) -> Session:
    if provider == "auto":
        if first_env(CLAUDE_SESSION_ENV):
            provider = "claude"
        elif first_env(CODEX_SESSION_ENV):
            provider = "codex"
        else:
            raise ValueError("Cannot identify the current client. Supply --provider and --session-id; "
                             "the most recently modified chat is deliberately never guessed.")
    root = Path(home) if home else (codex_home() if provider == "codex" else claude_home())
    if not session_id:
        session_id = first_env(CODEX_SESSION_ENV if provider == "codex" else CLAUDE_SESSION_ENV)
    if path:
        path = Path(path).expanduser().resolve()
        rows = read_jsonl(path)
        observed = None
        if provider == "codex":
            observed = next((r.get("payload", {}).get("id") for r in rows
                             if r.get("type") == "session_meta"), None)
        else:
            observed = next((r.get("sessionId") for r in rows if r.get("sessionId")), None)
        if session_id and observed and session_id != observed:
            raise ValueError("The transcript belongs to a different session.")
        session_id = session_id or observed
    else:
        if not session_id or not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", session_id):
            raise ValueError("Supply the current chat's exact session ID, not a model name or 'latest'.")
        path = None
        if provider == "codex":
            for db in sorted(root.glob("state_*.sqlite"), reverse=True):
                try:
                    with closing(readonly_db(db)) as conn:
                        row = conn.execute("SELECT rollout_path FROM threads WHERE id=?", (session_id,)).fetchone()
                    if row and Path(row[0]).is_file():
                        path = Path(row[0])
                        break
                except sqlite3.Error:
                    continue
            if path is None:
                candidates = list((root / "sessions").glob(f"**/*{session_id}.jsonl"))
                candidates += list((root / "archived_sessions").glob(f"*{session_id}.jsonl"))
        else:
            candidates = list((root / "projects").glob(f"*/{session_id}.jsonl"))
        if path is None:
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one transcript for {session_id}, found {len(candidates)}. "
                                 "Pass --session-path explicitly.")
            path = candidates[0]
    if not session_id:
        raise ValueError("No session identity is present in the selected transcript.")
    return Session(provider, session_id, Path(path), root)


def _probe(text: str) -> tuple[str | None, int | None]:
    markers = set(MARKER.findall(text))
    markers.update(re.findall(r"\b(?:burst|write-sample)\s+--run-id[\s=]+([0-9a-f]{32})\s+--index[\s=]+([1-9][0-9]*)", text))
    # File delivery: the sample travels in a file-write call whose path names the run and index.
    markers.update(re.findall(r"cbench-([0-9a-f]{32})-([1-9][0-9]*)\.txt", text))
    if len(markers) == 1:
        run, index = markers.pop()
        return run, int(index)
    return None, None


def _codex_starts(session: Session) -> list[float]:
    starts = []
    for db in sorted(session.home.glob("logs_*.sqlite"), reverse=True)[:1]:
        try:
            with closing(readonly_db(db)) as conn:
                # This is a native sampling-request log; it precedes network send.
                # Do not include other feedback_tags entries or tool execution time.
                rows = conn.execute(
                    "SELECT ts,ts_nanos,feedback_log_body FROM logs "
                    "WHERE thread_id=? AND target='feedback_tags' ORDER BY id", (session.session_id,))
                for sec, nanos, body in rows:
                    if body and re.search(r': model="[^"]+" approval_policy=', body):
                        starts.append(sec * 1000 + nanos / 1_000_000)
        except sqlite3.Error:
            pass
    return starts


def parse_codex(session: Session) -> tuple[list[Request], dict, list[dict]]:
    rows, requests, turns = read_jsonl(session.path), [], []
    settings, turn_id, boundary, pending, previous_end = {}, None, None, [], float("-inf")
    starts = _codex_starts(session)
    seen_responses, seen_turns = set(), set()
    old_total, context_epoch = None, 0
    messages, item_timings, generation_items = [], {}, []
    modern = any(r.get("type") == "token_usage_record" for r in rows)
    for row in rows:
        typ, payload, at = row.get("type"), row.get("payload", {}), epoch_ms(row.get("timestamp"))
        kind = payload.get("type")
        if kind == "item_completed":
            item = payload.get("item", {})
            interval = (payload.get("started_at_ms"), payload.get("completed_at_ms"), item.get("type"))
            item_timings[item.get("id")] = interval
            generation_items.append(interval)
        if (typ == "compacted" or kind in ("context_compacted", "context_compaction") or
                kind == "item_completed" and payload.get("item", {}).get("type") in ("ContextCompaction", "contextCompaction")):
            context_epoch += 1
        if typ == "turn_context":
            turn_id = payload.get("turn_id", turn_id)
            for source, target in [("model", "model"), ("effort", "effort"),
                                   ("reasoning_effort", "effort"), ("service_tier", "service_tier")]:
                if source in payload:
                    settings[target] = payload[source]
        if kind == "thread_settings_applied":
            applied = payload.get("thread_settings", {})
            settings.update({target: applied[source] for source, target in
                             [("model", "model"), ("reasoning_effort", "effort"),
                              ("service_tier", "service_tier")] if source in applied})
        if kind == "task_started":
            turn_id = payload.get("turn_id", turn_id)
            boundary = at
        if typ == "response_item":
            if kind == "message" and payload.get("role") == "assistant":
                text = "".join(c.get("text", "") for c in payload.get("content", []) if isinstance(c, dict))
                if text.strip():
                    messages.append((payload.get("id"), text))
            if kind in ("function_call_output", "custom_tool_call_output", "local_shell_call_output"):
                boundary = at
            elif kind in ("function_call", "custom_tool_call"):
                text = payload.get("input") or payload.get("arguments") or ""
                pending.append(text if isinstance(text, str) else json.dumps(text))
        if kind in ("task_complete", "turn_complete", "task_completed"):
            tid = payload.get("turn_id", turn_id)
            if tid not in seen_turns:
                turns.append({"turn_id": tid, "duration_ms": payload.get("duration_ms"),
                              "ttft_ms": payload.get("time_to_first_token_ms"),
                              "completed_ms": at, "status": "failed" if payload.get("error") else "completed"})
                seen_turns.add(tid)
        usage = None
        response_id = payload.get("response_id")
        if typ == "token_usage_record":
            usage = payload.get("usage")
        elif not modern and kind == "token_count":
            info = payload.get("info") or {}
            total = info.get("total_token_usage")
            if total and total != old_total:
                usage, response_id = info.get("last_token_usage"), f"legacy-{at}"
            old_total = total
        if not usage or response_id in seen_responses:
            continue
        seen_responses.add(response_id)
        candidates = [s for s in starts if previous_end < s <= (at or 0)]
        # Multiple attempts are one model response; count all latency from its first attempt.
        start = min(candidates) if candidates else boundary
        source = "native sampling-request log + response usage timestamp" if candidates else "session boundary (includes client overhead)"
        run, index = _probe("\n".join(pending))
        request = Request(
            provider="codex", session_id=session.session_id, request_id=response_id or f"response-{at}",
            turn_id=payload.get("turn_id", turn_id), **{k: settings.get(k) for k in
                ("model", "effort", "service_tier")},
            started_ms=start, completed_ms=at,
            duration_ms=(at - start if at is not None and start is not None and at >= start else None),
            input_tokens=usage.get("input_tokens"), cached_input_tokens=usage.get("cached_input_tokens"),
            cache_write_input_tokens=usage.get("cache_write_input_tokens"),
            output_tokens=usage.get("output_tokens"), reasoning_tokens=usage.get("reasoning_output_tokens"),
            attempts=len(candidates) or None, timing_source=source,
            settings_source="current-turn session settings", run_id=run, sample_index=index,
            context_epoch=context_epoch)
        request.text_block_count = len(messages)
        request.tool_call_count = len(pending)
        request.tool_input_rows = reference_rows_in("\n".join(pending))
        request.visible_words = sum(len(text.split()) for _, text in messages)
        request.visible_sha256 = text_digest(messages[0][1]) if len(messages) == 1 else None
        request.continuity = "single_text_block" if len(messages) == 1 else ("interrupted" if messages else "unknown")
        if len(messages) == 1:
            interval = item_timings.get(messages[0][0])
            if interval and interval[0] is not None and interval[1] is not None and interval[1] > interval[0]:
                start_text, end_text, _ = interval
                request.text_span_ms = end_text - start_text
                request.text_span_source = "native text-item start/end (not individual token arrivals)"
                if any(a is not None and b is not None and a < end_text and b > start_text and
                       t in ("Reasoning", "CommandExecution", "McpToolCall", "DynamicToolCall")
                       for a, b, t in generation_items):
                    request.continuity = "interrupted"
        requests.append(request)
        previous_end, boundary, pending = at or previous_end, at, []
        messages, generation_items = [], []
    return requests, settings, turns


def parse_claude(session: Session) -> tuple[list[Request], dict, list[dict]]:
    rows, groups, api_requests = read_jsonl(session.path), {}, {}
    boundary, turn_id, settings, context_epoch = None, None, {}, 0
    for row in rows:
        typ, at = row.get("type"), epoch_ms(row.get("timestamp"))
        if typ == "system" and row.get("subtype") == "compact_boundary":
            context_epoch += 1
        if row.get("isSidechain"):
            continue
        if typ == "api-request":
            api_requests[row.get("id")] = row
        if typ == "user":
            boundary = at
            content = row.get("message", {}).get("content", [])
            tool_result = isinstance(content, list) and any(
                isinstance(x, dict) and x.get("type") == "tool_result" for x in content)
            if not tool_result:
                turn_id = row.get("uuid")
        if typ != "assistant":
            continue
        msg = row.get("message", {})
        key = row.get("requestId") or msg.get("id")
        if not key or not isinstance(msg.get("usage"), dict):
            continue
        group = groups.setdefault(key, {"rows": [], "boundary": boundary, "turn_id": turn_id,
                                       "context_epoch": context_epoch})
        group["rows"].append(row)
    requests = []
    for key, group in groups.items():
        entries = group["rows"]
        last = entries[-1]
        msg, usage = last["message"], last["message"]["usage"]
        raw_api = api_requests.get(last.get("requestRef"), {})
        params = raw_api.get("params", {})
        start = epoch_ms(raw_api.get("timestamp")) if raw_api else group["boundary"]
        end = max((epoch_ms(x.get("timestamp")) or 0) for x in entries) or None
        settings = {"model": msg.get("model"),
                    "effort": last.get("perTurnEffort") or last.get("effort") or
                              params.get("output_config", {}).get("effort"),
                    "service_tier": usage.get("service_tier"),
                    "speed": ("normal" if usage.get("speed") == "standard" else usage.get("speed"))}
        # Anthropic's input_tokens excludes BOTH cache-read and cache-write tokens.
        parts = [usage.get(k) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")]
        total_input = sum(p or 0 for p in parts) if parts[0] is not None else None
        details = usage.get("output_tokens_details") or {}
        tool_inputs = [block.get("input", {}) for x in entries for block in x["message"].get("content", [])
                       if isinstance(block, dict) and block.get("type") == "tool_use"]
        run, index = _probe(json.dumps(tool_inputs))
        request = Request(
            provider="claude", session_id=session.session_id, request_id=key,
            turn_id=group["turn_id"], **settings, requested_model=last.get("requestedModel") or params.get("model"),
            started_ms=start, completed_ms=end,
            duration_ms=(end - start if end is not None and start is not None and end >= start else None),
            input_tokens=total_input, cached_input_tokens=usage.get("cache_read_input_tokens"),
            cache_write_input_tokens=usage.get("cache_creation_input_tokens"),
            output_tokens=usage.get("output_tokens"), reasoning_tokens=details.get("thinking_tokens"),
            timing_source=("native api-request + assistant completion timestamps" if raw_api else
                           "session boundary (includes client overhead)"),
            settings_source="actual response usage + per-request session metadata", run_id=run, sample_index=index,
            context_epoch=group["context_epoch"], client_version=last.get("version"))
        requests.append(request)
        text_blocks = {}
        for entry_index, entry in enumerate(entries):
            for block_index, block in enumerate(entry["message"].get("content", [])):
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text", "").strip():
                    key_block = (entry.get("apiBlockIndex", entry_index), block_index)
                    text_blocks[key_block] = block["text"]
        request.text_block_count = len(text_blocks)
        request.tool_call_count = len(tool_inputs)
        request.tool_input_rows = reference_rows_in(json.dumps(tool_inputs, ensure_ascii=False))
        # Newer clients record how long each tool call's arguments streamed.
        streams = [x["toolInputStreamMs"] for x in entries
                   if isinstance(x.get("toolInputStreamMs"), (int, float)) and x["toolInputStreamMs"] > 0]
        if len(streams) == 1 and len(tool_inputs) == 1:
            request.stream_ms = float(streams[0])
            request.stream_source = "native tool-argument stream duration (includes tool-call syntax)"
        request.stop_reason = msg.get("stop_reason")
        request.visible_words = sum(len(text.split()) for text in text_blocks.values())
        request.visible_sha256 = text_digest(next(iter(text_blocks.values()))) if len(text_blocks) == 1 else None
        request.continuity = "single_text_block" if len(text_blocks) == 1 else ("interrupted" if text_blocks else "unknown")
    return requests, settings, []


def parse_session(session: Session):
    """Requests, settings and completed turns; empty for an untracked session."""
    if not session.tracked or not session.path.exists():
        # Untracked, or the client has since moved/archived the transcript.
        return [], {}, []
    from .providers import provider_for
    parser = provider_for(session.provider).parse
    return parser(session) if parser else ([], {}, [])


def payload_digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
