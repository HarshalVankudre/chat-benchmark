"""Transcript adapters for clients beyond Codex and Claude Code.

Each parser returns (requests, settings, turns) like sessions.parse_codex. They
read only timing, usage, settings and tool-call arguments; a format they do not
recognise yields no requests, and the run falls back to helper-clock timing.
Formats were verified against real sessions of the versions named per adapter.
"""
from __future__ import annotations

import json
from pathlib import Path

from .metrics import Request
from .sessions import Session, _probe, epoch_ms, read_jsonl, reference_rows_in
from .workload import text_digest


def _finish(request, texts, tool_inputs):
    """Shared structure facts: visible text blocks, tool calls, literal rows, probe."""
    blob = "\n".join(tool_inputs)
    request.run_id, request.sample_index = _probe(blob)
    request.text_block_count = len(texts)
    request.tool_call_count = len(tool_inputs)
    request.tool_input_rows = reference_rows_in(blob)
    request.visible_words = sum(len(text.split()) for text in texts)
    request.visible_sha256 = text_digest(texts[0]) if len(texts) == 1 else None
    request.continuity = "single_text_block" if len(texts) == 1 else ("interrupted" if texts else "unknown")
    return request


def _duration(start, end):
    return end - start if start is not None and end is not None and end >= start else None


# --------------------------------------------------------------------------- Kimi Code
# Verified on Kimi Code CLI 2.5.x, wire protocol 1.4/1.5:
#   ~/.kimi-code/sessions/<workspace>/<session>/agents/main/wire.jsonl
# llm.request.time = request start, usage.record.time = stream end (epoch ms);
# step.end carries llmFirstTokenLatencyMs / llmStreamDurationMs once tools finish.

def kimi_home() -> Path:
    import os
    return Path(os.environ.get("KIMI_CODE_HOME") or Path.home() / ".kimi-code")


def kimi_find(home: Path, session_id: str):
    matches = list((home / "sessions").glob(f"*/{session_id}/agents/main/wire.jsonl"))
    if not matches and not session_id.startswith("session_"):
        matches = list((home / "sessions").glob(f"*/session_{session_id}/agents/main/wire.jsonl"))
    return matches[0] if len(matches) == 1 else None


def kimi_candidates(home: Path):
    return (home / "sessions").glob("*/*/agents/main/wire.jsonl")


def kimi_session_id(path: Path) -> str:
    return path.parents[2].name


def parse_kimi(session: Session):
    steps, order, settings, turns = {}, [], {}, []
    context_epoch, latest = 0, None
    for row in read_jsonl(session.path):
        if row.get("agentId") not in (None, "main"):
            continue
        kind, at = row.get("type"), row.get("time")
        if kind in ("context.apply_compaction", "full_compaction.complete", "context.clear", "context.undo"):
            context_epoch += 1
        elif kind == "llm.request" and row.get("kind", "loop") == "loop":
            key = str(row.get("turnStep"))
            step = steps.get(key)
            if step is None or step["usage"] is not None:
                # A finished step with the same label means the turn was re-run.
                key = key if step is None else f"{key}#{len(order)}"
                step = steps[key] = {"key": key, "start": None, "end": None, "usage": None, "attempts": 0,
                                     "texts": [], "tools": [], "end_event": None, "uuid": None,
                                     "context_epoch": context_epoch}
                order.append(key)
            # A retried request restarts the clock: time the attempt that produced the output.
            step.update(start=at, attempts=step["attempts"] + 1, model=row.get("model") or row.get("modelAlias"),
                        effort=row.get("thinkingEffort"))
            settings = {"model": step["model"], "effort": step["effort"], "service_tier": None}
            latest = key
        elif kind == "usage.record" and latest and steps[latest]["usage"] is None:
            steps[latest].update(usage=row.get("usage") or {}, end=at)
        elif kind == "context.append_loop_event":
            event = row.get("event") or {}
            step = steps.get(latest) if latest else None
            if step is None:
                continue
            etype = event.get("type")
            if etype == "step.begin":
                step["uuid"] = event.get("uuid")
            elif etype == "content.part":
                part = event.get("part") or {}
                if part.get("type") == "text" and str(part.get("text", "")).strip():
                    step["texts"].append(part["text"])
            elif etype == "tool.call":
                step["tools"].append(json.dumps(event.get("args") or {}, ensure_ascii=False))
            elif etype == "step.end":
                step["end_event"] = event
                if step["usage"] is None and event.get("usage"):  # Protocol 1.4 ordering.
                    step.update(usage=event["usage"], end=at)
        elif kind == "turn.ended":
            turns.append({"turn_id": str(row.get("turnId")), "duration_ms": row.get("durationMs"), "ttft_ms": None,
                          "completed_ms": at, "status": row.get("reason") or "completed"})
    requests = []
    for key in order:
        step = steps[key]
        usage = step["usage"]
        if usage is None:
            continue
        parts = [usage.get(k) for k in ("inputOther", "inputCacheRead", "inputCacheCreation")]
        end_event = step["end_event"] or {}
        request = Request(
            provider=session.provider, session_id=session.session_id,
            request_id=step["uuid"] or f"{session.session_id}:{key}", turn_id=key.split(".")[0],
            model=step.get("model"), effort=step.get("effort"),
            started_ms=step["start"], completed_ms=step["end"], duration_ms=_duration(step["start"], step["end"]),
            input_tokens=sum(p or 0 for p in parts) if any(p is not None for p in parts) else None,
            cached_input_tokens=usage.get("inputCacheRead"), cache_write_input_tokens=usage.get("inputCacheCreation"),
            output_tokens=usage.get("output"), reasoning_tokens=None, attempts=step["attempts"],
            timing_source="native llm.request to usage.record timestamps",
            settings_source="per-request llm.request record", context_epoch=step["context_epoch"],
            stop_reason=end_event.get("finishReason"))
        ttft, stream = end_event.get("llmFirstTokenLatencyMs"), end_event.get("llmStreamDurationMs")
        if isinstance(ttft, (int, float)) and ttft >= 0:
            request.ttft_ms, request.ttft_source = float(ttft), "native step.end llmFirstTokenLatencyMs (first stream part of any kind)"
        if isinstance(stream, (int, float)) and stream > 0:
            request.stream_ms = float(stream)
            request.stream_source = "native step.end llmStreamDurationMs (first stream part to stream end; includes thinking)"
        requests.append(_finish(request, step["texts"], step["tools"]))
    return requests, settings, turns


# --------------------------------------------------------------------------- Grok CLI
# Verified on Grok CLI 1.0.3-1.0.52 data:
#   ~/.grok/sessions/<encoded cwd>/<session>/updates.jsonl   tool calls (rawInput.command)
#   ~/.grok/sessions/<encoded cwd>/<session>/events.jsonl    loop_started / first_token
#   ~/.grok/logs/unified.jsonl                               per-request tokens and elapsed time
# No single file has everything, so requests are joined on session id and time.

def grok_home() -> Path:
    import os
    return Path(os.environ.get("GROK_HOME") or Path.home() / ".grok")


def grok_find(home: Path, session_id: str):
    matches = list((home / "sessions").glob(f"*/{session_id}/updates.jsonl"))
    return matches[0] if len(matches) == 1 else None


def grok_candidates(home: Path):
    return (home / "sessions").glob("*/*/updates.jsonl")


def grok_session_id(path: Path) -> str:
    return path.parent.name


def parse_grok(session: Session):
    directory, home = session.path.parent, session.home or grok_home()
    summary = {}
    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    settings = {"model": summary.get("current_model_id"), "effort": summary.get("reasoning_effort"), "service_tier": None}
    by_call = {}  # tool_call_id -> (model, effort) from the model context, when still present
    history = directory / "chat_history.jsonl"
    for row in read_jsonl(history) if history.is_file() else []:
        if row.get("type") == "assistant":
            for call in row.get("tool_calls") or []:
                by_call[call.get("id")] = (row.get("model_id"), row.get("reasoning_effort"))
    requests, pending, version = [], None, None
    log = home / "logs" / "unified.jsonl"
    for row in read_jsonl(log) if log.is_file() else []:
        if row.get("sid") != session.session_id:
            continue
        message, ctx, at = row.get("msg"), row.get("ctx") or {}, epoch_ms(row.get("ts"))
        version = row.get("ver") or version
        if message == "shell.turn.inference_start":
            pending = at
        elif message == "shell.turn.inference_done":
            elapsed = ctx.get("model_elapsed_ms")
            start = at - elapsed if isinstance(elapsed, (int, float)) and at is not None else pending
            requests.append(Request(
                provider=session.provider, session_id=session.session_id,
                request_id=f"{session.session_id}:{len(requests) + 1}", **settings,
                started_ms=start, completed_ms=at,
                duration_ms=float(elapsed) if isinstance(elapsed, (int, float)) and elapsed > 0 else _duration(start, at),
                input_tokens=ctx.get("prompt_tokens"), cached_input_tokens=ctx.get("cached_prompt_tokens"),
                output_tokens=ctx.get("completion_tokens"), reasoning_tokens=ctx.get("reasoning_tokens"),
                attempts=ctx.get("attempts"), timing_source="native shell.turn.inference_done model_elapsed_ms",
                settings_source="session summary / per-message model context", client_version=version))
            pending = None
    if not requests:
        return [], settings, []
    # First streamed chunk per request, from the phase timeline.
    loops, events = [], directory / "events.jsonl"
    for row in read_jsonl(events) if events.is_file() else []:
        at = epoch_ms(row.get("ts"))
        if row.get("type") == "loop_started":
            loops.append([at, None])
        elif row.get("type") == "first_token" and loops and loops[-1][1] is None:
            loops[-1][1] = at
    for request in requests:
        match = min(loops, key=lambda loop: abs((loop[0] or 0) - (request.started_ms or 0)), default=None)
        if match and match[0] is not None and abs(match[0] - (request.started_ms or 0)) < 500 and match[1] is not None:
            request.ttft_ms = match[1] - match[0]
            request.ttft_source = "native events.jsonl loop_started to first_token (first streamed chunk)"
    # Tool calls and visible text, attached to the request whose response ended when they were emitted.
    tools, texts, epoch_marks, turns = {}, {}, [], []
    for row in read_jsonl(session.path):
        update = (row.get("params") or {}).get("update") or {}
        meta = (row.get("params") or {}).get("_meta") or update.get("_meta") or {}
        at = meta.get("agentTimestampMs")
        kind = update.get("sessionUpdate")
        if kind == "auto_compact_completed" and at:
            epoch_marks.append(at)
        if kind == "turn_completed":
            turns.append({"turn_id": update.get("prompt_id"), "duration_ms": update.get("elapsed_ms"), "ttft_ms": None,
                          "completed_ms": at, "status": update.get("stop_reason") or "completed"})
        if not at or kind not in ("tool_call", "agent_message_chunk"):
            continue
        target = min(requests, key=lambda r: abs((r.completed_ms or 0) - at))
        if kind == "tool_call":
            if abs((target.completed_ms or 0) - at) > 2000:
                continue
            tools.setdefault(target.request_id, []).append(json.dumps(update.get("rawInput") or {}, ensure_ascii=False))
            model, effort = by_call.get(update.get("toolCallId"), (None, None))
            target.model, target.effort = model or target.model, effort or target.effort
        else:
            owner = next((r for r in requests if (r.started_ms or 0) <= at <= (r.completed_ms or 0) + 250), None)
            content = update.get("content") or {}
            text = content.get("text") if isinstance(content, dict) else None
            if owner and isinstance(text, str) and text.strip():
                texts.setdefault(owner.request_id, []).append(text)
    for request in requests:
        request.context_epoch = sum(mark <= (request.started_ms or 0) for mark in epoch_marks)
        _finish(request, ["".join(texts[request.request_id])] if request.request_id in texts else [],
                tools.get(request.request_id, []))
    return requests, settings, turns


# --------------------------------------------------------------------------- Antigravity CLI (agy)
# Verified on agy 1.4.x data with 1.5.3 schemas:
#   ~/.gemini/antigravity-cli/conversations/<conversation>.db   SQLite, protobuf payloads
#   ~/.gemini/antigravity-cli/brain/<conversation>/.system_generated/logs/transcript_full.jsonl
# A PLANNER_RESPONSE step (type 15) is one model request: metadata.created_at is the
# request start and finished_generating_at the stream end. Tool-call arguments arrive
# in one burst, so the client's "first token" is not a pre-generation latency here.

def agy_home() -> Path:
    return Path.home() / ".gemini" / "antigravity-cli"


def agy_find(home: Path, session_id: str):
    path = home / "conversations" / f"{session_id}.db"
    return path if path.is_file() else None


def agy_locate(home: Path, nonce: str, since_ms: float):
    needle = nonce.encode()
    for log in (home / "brain").glob("*/.system_generated/logs/transcript_full.jsonl"):
        try:
            if log.stat().st_mtime * 1000 < since_ms - 120_000 or needle not in log.read_bytes():
                continue
        except OSError:
            continue
        conversation = log.parents[2].name
        database = agy_find(home, conversation)
        if database:
            return database, conversation
    return None


def _varint(data, i):
    value = shift = 0
    while True:
        byte = data[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _pb(data) -> dict:
    """Minimal protobuf wire reader: field number -> list of ints or bytes. {} if malformed."""
    out, i = {}, 0
    try:
        data = bytes(data or b"")
        while i < len(data):
            key, i = _varint(data, i)
            field, wire = key >> 3, key & 7
            if wire == 0:
                value, i = _varint(data, i)
            elif wire == 2:
                size, i = _varint(data, i)
                if i + size > len(data):
                    return {}
                value, i = data[i:i + size], i + size
            elif wire in (1, 5):
                size = 8 if wire == 1 else 4
                value, i = data[i:i + size], i + size
            else:
                return {}
            out.setdefault(field, []).append(value)
    except (IndexError, ValueError):
        return {}
    return out


def _sub(message, field):
    value = message.get(field, [None])[0]
    return _pb(value) if isinstance(value, bytes) else {}


def _text(message, field):
    value = message.get(field, [None])[0]
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else None


def _number(message, field):
    value = message.get(field, [None])[0]
    return value if isinstance(value, int) else None


def _stamp(message, field):
    stamp = _sub(message, field)
    seconds = _number(stamp, 1)
    return seconds * 1000 + (_number(stamp, 2) or 0) / 1_000_000 if seconds else None


def parse_agy(session: Session):
    import re
    import sqlite3
    from contextlib import closing
    from .sessions import readonly_db
    try:
        with closing(readonly_db(session.path)) as conn:
            steps = conn.execute("SELECT idx, step_type, metadata, step_payload FROM steps ORDER BY idx").fetchall()
            generators = dict(conn.execute("SELECT idx, data FROM gen_metadata").fetchall())
    except (sqlite3.Error, OSError):
        return [], {}, []
    label = None
    try:
        label = json.loads((session.path.parents[1] / "settings.json").read_text(encoding="utf-8-sig")).get("model")
    except (OSError, ValueError, AttributeError):
        pass
    requests, settings, context_epoch, turn_id = [], {}, 0, None
    for idx, step_type, metadata, payload in steps:
        if step_type == 23:  # CHECKPOINT: the context was summarized.
            context_epoch += 1
        elif step_type == 14:  # USER_INPUT
            turn_id = str(idx)
        if step_type != 15:
            continue
        body = _pb(payload)
        meta = _pb(metadata) or _sub(body, 5)
        start, shown, end = _stamp(meta, 1), _stamp(meta, 6), _stamp(meta, 7)
        usage, planner = _sub(meta, 9), _sub(body, 20)
        chat = _sub(_pb(generators.get(_number(_sub(meta, 20), 3))), 1)
        display = _text(chat, 21) or label
        effort = re.search(r"\(([^)]+)\)\s*$", display or "")
        model = _text(chat, 19) or (re.sub(r"\s*\([^)]*\)\s*$", "", display) if display else None)
        if model is None and _number(usage, 1) is not None:
            model = f"model-enum-{_number(usage, 1)}"
        fresh, cached, written = (_number(usage, n) for n in (2, 5, 4))
        request = Request(
            provider=session.provider, session_id=session.session_id, request_id=_text(usage, 11) or f"{session.session_id}:{idx}",
            turn_id=turn_id, model=model, effort=effort.group(1).lower() if effort else None,
            started_ms=start, completed_ms=end, duration_ms=_duration(start, end),
            input_tokens=(fresh or 0) + (cached or 0) + (written or 0) if fresh is not None else None,
            cached_input_tokens=cached if cached is not None else (0 if fresh is not None else None),
            cache_write_input_tokens=written, output_tokens=_number(usage, 3),
            reasoning_tokens=_number(usage, 9) if _number(usage, 3) is not None else None,
            timing_source="native step created_at to finished_generating_at",
            settings_source="per-request generator metadata, else the client's model setting",
            context_epoch=context_epoch)
        if request.reasoning_tokens is None and request.output_tokens is not None:
            request.reasoning_tokens = 0
        calls = [_text(_pb(call), 3) or "" for call in planner.get(7, []) if isinstance(call, bytes)]
        if shown is not None and end is not None and start is not None and not calls:
            request.ttft_ms, request.ttft_source = shown - start, "native step viewable_at (first streamed chunk)"
        elif calls:
            request.ttft_source = "not exposed: tool-call arguments arrive in one burst"
        if shown is not None and end is not None and end > shown:
            request.stream_source = "not used: the client streams tool-call arguments as one burst"
        settings = {"model": request.model, "effort": request.effort, "service_tier": None}
        answer = _text(planner, 1)
        requests.append(_finish(request, [answer] if answer and answer.strip() else [], calls))
    return requests, settings, []


# --------------------------------------------------------------------------- pi coding agent
# Verified on pi 0.84.x session format 3:
#   ~/.pi/agent/sessions/--<cwd>--/<time>_<uuid>.jsonl
# Assistant entries are appended before their tools run. The inner message.timestamp
# (epoch ms) is the request start; the outer entry timestamp (ISO) is its completion.

def pi_home() -> Path:
    import os
    return Path(os.environ.get("PI_CODING_AGENT_DIR") or Path.home() / ".pi" / "agent")


def pi_find(home: Path, session_id: str):
    matches = list((home / "sessions").glob(f"*/*_{session_id}.jsonl"))
    return matches[0] if len(matches) == 1 else None


def pi_candidates(home: Path):
    return (home / "sessions").glob("*/*.jsonl")


def pi_session_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def parse_pi(session: Session):
    requests, settings, turns = [], {"model": None, "effort": None, "service_tier": None}, []
    context_epoch, turn_id, effort = 0, None, None
    for row in read_jsonl(session.path):
        kind = row.get("type")
        if kind == "compaction":
            context_epoch += 1
        elif kind == "thinking_level_change":
            effort = row.get("thinkingLevel")
        if kind != "message":
            continue
        message = row.get("message") or {}
        role, end = message.get("role"), epoch_ms(row.get("timestamp"))
        if role == "user":
            turn_id = row.get("id")
        usage = message.get("usage")
        if role != "assistant" or not isinstance(usage, dict):
            continue
        start = message.get("timestamp") if isinstance(message.get("timestamp"), (int, float)) else None
        parts = [usage.get(k) for k in ("input", "cacheRead", "cacheWrite")]
        content = [c for c in message.get("content") or [] if isinstance(c, dict)]
        request = Request(
            provider=session.provider, session_id=session.session_id,
            request_id=message.get("responseId") or row.get("id") or f"{session.session_id}:{len(requests)}",
            turn_id=turn_id, model=message.get("model"), effort=effort,
            started_ms=start, completed_ms=end, duration_ms=_duration(start, end),
            input_tokens=sum(p or 0 for p in parts) if parts[0] is not None else None,
            cached_input_tokens=usage.get("cacheRead"), cache_write_input_tokens=usage.get("cacheWrite"),
            output_tokens=usage.get("output"), reasoning_tokens=usage.get("reasoning"),
            timing_source="native message start timestamp to transcript append time",
            settings_source="per-message model and latest thinking-level entry", context_epoch=context_epoch,
            stop_reason=message.get("stopReason"),
            status="failed" if message.get("stopReason") in ("error", "aborted") else "completed")
        cost = (usage.get("cost") or {}).get("total")
        if isinstance(cost, (int, float)):
            request.client_estimated_usd = float(cost)
        settings = {"model": request.model, "effort": effort, "service_tier": None}
        if message.get("stopReason") == "stop":
            turns.append({"turn_id": turn_id, "duration_ms": None, "ttft_ms": None, "completed_ms": end, "status": "completed"})
        requests.append(_finish(request,
                                [c["text"] for c in content if c.get("type") == "text" and str(c.get("text", "")).strip()],
                                [json.dumps(c.get("arguments") or {}, ensure_ascii=False) for c in content if c.get("type") == "toolCall"]))
    return requests, settings, turns


# --------------------------------------------------------------------------- OpenCode
# Verified on OpenCode 1.18.x: one SQLite database, one assistant `message` row per
# model request, its stream in `part` rows. message.time.completed includes tool
# execution, so generation ends where the tool part starts running. Usage is written
# only after the tool returns and is therefore filled in on a later read.

def opencode_home() -> Path:
    import os
    base = os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base else Path.home() / ".local" / "share") / "opencode"


def _opencode_db(home: Path):
    from .sessions import readonly_db
    return readonly_db(home / "opencode.db")


def opencode_find(home: Path, session_id: str):
    import sqlite3
    from contextlib import closing
    try:
        with closing(_opencode_db(home)) as conn:
            found = conn.execute("SELECT 1 FROM session WHERE id=?", (session_id,)).fetchone()
        return home / "opencode.db" if found else None
    except (sqlite3.Error, OSError):
        return None


def opencode_locate(home: Path, nonce: str, since_ms: float):
    """The session whose recent tool input contains this run id; never simply the latest."""
    import sqlite3
    from contextlib import closing
    try:
        with closing(_opencode_db(home)) as conn:
            recent = conn.execute("SELECT id FROM session WHERE time_updated>=? ORDER BY time_updated DESC LIMIT 8",
                                  (since_ms - 60_000,)).fetchall()
            for (session_id,) in recent:
                rows = conn.execute("SELECT data FROM part WHERE session_id=? AND time_created>=?",
                                    (session_id, since_ms - 60_000))
                if any(nonce in (data or "") for (data,) in rows):
                    return home / "opencode.db", session_id
    except (sqlite3.Error, OSError):
        pass
    return None


def parse_opencode(session: Session):
    import sqlite3
    from contextlib import closing
    try:
        with closing(_opencode_db(session.path.parent)) as conn:
            messages = conn.execute("SELECT id, data FROM message WHERE session_id=? ORDER BY time_created, id",
                                    (session.session_id,)).fetchall()
            parts = conn.execute("SELECT message_id, time_created, data FROM part WHERE session_id=? ORDER BY time_created, id",
                                 (session.session_id,)).fetchall()
            version = conn.execute("SELECT version FROM session WHERE id=?", (session.session_id,)).fetchone()
    except (sqlite3.Error, OSError):
        return [], {}, []
    by_message = {}
    for message_id, created, data in parts:
        try:
            by_message.setdefault(message_id, []).append((created, json.loads(data)))
        except ValueError:
            continue
    requests, settings, context_epoch = [], {}, 0
    for message_id, data in messages:
        try:
            message = json.loads(data)
        except ValueError:
            continue
        if message.get("role") != "assistant":
            continue
        if message.get("mode") == "compaction" or message.get("summary") is True:
            context_epoch += 1
            continue
        start = (message.get("time") or {}).get("created")
        first, generation_end, texts, tools = None, None, [], []
        for created, part in by_message.get(message_id, []):
            kind = part.get("type")
            if kind == "step-start" and first is None:
                first = created
            elif kind in ("text", "reasoning"):
                generation_end = max(generation_end or 0, (part.get("time") or {}).get("end") or 0) or generation_end
                if kind == "text" and str(part.get("text", "")).strip():
                    texts.append(part["text"])
            elif kind == "tool":
                state = part.get("state") or {}
                tools.append(json.dumps(state.get("input") or {}, ensure_ascii=False))
                began = (state.get("time") or {}).get("start")
                if began:  # The call is fully generated when it starts running.
                    generation_end = max(generation_end or 0, began)
        finished = message.get("finish") is not None
        tokens = message.get("tokens") if finished else None
        cache = (tokens or {}).get("cache") or {}
        end = generation_end or ((message.get("time") or {}).get("completed") if finished and not tools else None)
        if start is None or end is None:
            continue
        request = Request(
            provider=session.provider, session_id=session.session_id, request_id=message_id,
            turn_id=message.get("parentID"), model=message.get("modelID"), effort=message.get("variant"),
            started_ms=start, completed_ms=end, duration_ms=_duration(start, end),
            input_tokens=(tokens["input"] + (cache.get("read") or 0) + (cache.get("write") or 0)) if tokens and tokens.get("input") is not None else None,
            cached_input_tokens=cache.get("read") if tokens else None,
            cache_write_input_tokens=cache.get("write") if tokens else None,
            output_tokens=((tokens.get("output") or 0) + (tokens.get("reasoning") or 0)) if tokens else None,
            reasoning_tokens=tokens.get("reasoning") if tokens else None,
            timing_source="native message creation to tool-call start (or last content delta)",
            settings_source="per-message model and variant", context_epoch=context_epoch,
            client_version=version[0] if version else None, stop_reason=message.get("finish"))
        if first is not None and first >= start:
            request.ttft_ms = float(first - start)
            request.ttft_source = "native first stream event (step-start) after message creation"
        if finished and isinstance(message.get("cost"), (int, float)):
            request.client_estimated_usd = float(message["cost"])
        settings = {"model": request.model, "effort": request.effort, "service_tier": None}
        requests.append(_finish(request, texts, tools))
    return requests, settings, []
