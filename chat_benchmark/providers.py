"""Registry of agent CLIs.

Every client can be benchmarked: helper-clock timing needs nothing but a shell
tool. An adapter adds native request timing, token usage and settings for clients
whose transcript format has been verified. Unknown client names are accepted and
run with helper-clock timing.
"""
from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import adapters
from .ancestry import ancestors, nearest
from .sessions import (CLAUDE_SESSION_ENV, CODEX_SESSION_ENV, Session, claude_home, codex_home, first_env,
                       parse_claude, parse_codex, read_jsonl, resolve_session)

SKILL_NAME = "chat-benchmark"
MAX_SCAN_BYTES = 200 * 1024 * 1024


def _home(*parts):
    return lambda: Path.home().joinpath(*parts)


@dataclass(frozen=True)
class Provider:
    key: str
    title: str
    home: Callable[[], Path] | None = None          # client data home
    skills: Callable[[], Path] | None = None        # user-level Agent Skills directory
    binaries: tuple = ()                            # executables looked up on PATH
    processes: tuple = ()                           # process names that identify the client
    session_env: tuple = ()                         # environment variables carrying the chat id
    marker_env: tuple = ()                          # environment variables that mark its shell
    session_placeholder: str = ""                   # text the client substitutes into skill bodies
    parse: Callable | None = None                   # transcript -> (requests, settings, turns)
    find: Callable | None = None                    # (home, session id) -> transcript path
    candidates: Callable | None = None              # home -> transcripts that may hold a run id
    session_id_of: Callable | None = None           # transcript path -> session id
    locate: Callable | None = None                  # (home, run id, since ms) -> (path, session id)
    live: bool = True                               # rows are written while the turn runs
    telemetry: bool = False                         # the local OTLP receiver adds native fields
    windows_delivery: str = "file"                  # how samples travel on Windows: stdin | file
    invoke: str = "/chat-benchmark"
    native: str = "helper clock only"
    known: bool = True

    def delivery(self) -> str:
        return self.windows_delivery if os.name == "nt" else "stdin"


def _codex_session_id(path: Path) -> str | None:
    return next((r.get("payload", {}).get("id") for r in read_jsonl(path) if r.get("type") == "session_meta"), None)


PROVIDERS: dict[str, Provider] = {p.key: p for p in (
    Provider("claude", "Claude Code", claude_home, lambda: claude_home() / "skills", ("claude",), ("claude",),
             CLAUDE_SESSION_ENV, ("CLAUDECODE",), "${CLAUDE_SESSION_ID}", parse_claude,
             candidates=lambda home: (home / "projects").glob("*/*.jsonl"), session_id_of=lambda path: path.stem,
             telemetry=True, windows_delivery="stdin",
             native="request timing, token usage, settings, tool-argument stream time; TTFT with telemetry"),
    Provider("codex", "Codex", codex_home, lambda: codex_home() / "skills", ("codex",), ("codex",),
             CODEX_SESSION_ENV, (), "", parse_codex,
             candidates=lambda home: (home / "sessions").glob("**/*.jsonl"), session_id_of=_codex_session_id,
             telemetry=True, windows_delivery="stdin", invoke="$chat-benchmark",
             native="request timing, token usage, settings; TTFT with telemetry"),
    Provider("kimi", "Kimi Code", adapters.kimi_home, lambda: adapters.kimi_home() / "skills", ("kimi",), ("kimi",),
             (), (), "${KIMI_SESSION_ID}", adapters.parse_kimi, adapters.kimi_find, adapters.kimi_candidates,
             adapters.kimi_session_id, invoke="/skill:chat-benchmark",
             native="request timing, token usage, settings, TTFT, stream time"),
    Provider("grok", "Grok CLI", adapters.grok_home, lambda: adapters.grok_home() / "skills", ("grok",), ("grok",),
             ("GROK_SESSION_ID",), (), "${SESSION_ID}", adapters.parse_grok, adapters.grok_find,
             adapters.grok_candidates, adapters.grok_session_id, windows_delivery="stdin",
             native="request timing, token usage, settings, first-chunk latency"),
    Provider("agy", "Antigravity CLI", adapters.agy_home, lambda: adapters.agy_home() / "skills", ("agy",),
             ("agy", "antigravity"), ("ANTIGRAVITY_CONVERSATION_ID",), ("ANTIGRAVITY_AGENT",), "",
             adapters.parse_agy, adapters.agy_find, locate=adapters.agy_locate,
             native="request timing, token usage incl. thinking, model"),
    Provider("gemini", "Gemini CLI", _home(".gemini"), _home(".gemini", "skills"), ("gemini",), ("gemini",),
             (), ("GEMINI_CLI",), ""),
    Provider("qwen", "Qwen Code", _home(".qwen"), _home(".qwen", "skills"), ("qwen",), ("qwen",), (), ("QWEN_CODE",), ""),
    Provider("opencode", "OpenCode", adapters.opencode_home, _home(".config", "opencode", "skills"), ("opencode",),
             ("opencode",), (), ("OPENCODE",), "", adapters.parse_opencode, adapters.opencode_find,
             locate=adapters.opencode_locate,
             native="request timing, token usage (after the tool returns), settings, first-event latency"),
    Provider("pi", "pi", adapters.pi_home, lambda: adapters.pi_home() / "skills", ("pi",), ("pi",),
             ("PI_SESSION_ID",), (), "", adapters.parse_pi, adapters.pi_find, adapters.pi_candidates,
             adapters.pi_session_id, invoke="/skill:chat-benchmark",
             native="request timing, token usage, settings, client cost estimate"),
    Provider("cursor", "Cursor Agent CLI", _home(".cursor"), _home(".cursor", "skills"), ("cursor-agent",),
             ("cursor-agent",), ("CURSOR_CONVERSATION_ID",), ("CURSOR_AGENT",), "", windows_delivery="stdin"),
    Provider("copilot", "GitHub Copilot CLI", _home(".copilot"), _home(".copilot", "skills"), ("copilot",), ("copilot",)),
    Provider("droid", "Factory Droid", _home(".factory"), _home(".factory", "skills"), ("droid",), ("droid",)),
    Provider("goose", "Goose", _home(".config", "goose"), _home(".config", "goose", "skills"), ("goose",), ("goose",)),
    Provider("auggie", "Augment CLI", _home(".augment"), _home(".augment", "skills"), ("auggie",), ("auggie",)),
    Provider("iflow", "iFlow CLI", _home(".iflow"), _home(".iflow", "skills"), ("iflow",), ("iflow",)),
    Provider("codebuddy", "CodeBuddy Code", _home(".codebuddy"), _home(".codebuddy", "skills"), ("codebuddy",), ("codebuddy",)),
    Provider("agents", "Shared ~/.agents/skills", _home(".agents"), _home(".agents", "skills")),
)}
# Clients that read ~/.agents/skills also read their own folder; installing there is opt-in.
OPT_IN = ("agents",)
# Distributed as executables (not Node scripts), so their process name is reliable evidence.
NATIVE_BINARIES = ("claude", "codex", "kimi", "grok", "agy", "copilot", "droid", "opencode", "goose")
SCRIPT_RUNTIMES = {"node", "bun", "deno"}


def provider_for(key: str | None) -> Provider:
    if key in PROVIDERS:
        return PROVIDERS[key]
    if not key or not re.fullmatch(r"[a-z][a-z0-9_-]{1,31}", key):
        raise ValueError("Unknown client. Use a key from `providers` or a short lowercase name.")
    # Any other agent CLI: helper-clock timing, self-reported labels.
    return Provider(key, key, known=False)


def detect(hint: str = "auto", chain=None, environ=None) -> Provider:
    """The client that is really running this helper.

    Process ancestry outranks the hint, because clients load each other's skill
    folders. A hint for a client that cannot be seen in the process chain
    (Node-based CLIs, unknown names) is trusted.
    """
    environ = os.environ if environ is None else environ
    chain = ancestors() if chain is None else chain
    names = {p.key: p.processes for p in PROVIDERS.values() if p.processes}
    seen = nearest(names, chain)
    if hint and hint != "auto":
        hinted = provider_for(hint)
        if seen and seen != hinted.key:
            # Everything between this helper and the nearest recognised client.
            depth = next(i for i, stems in enumerate(chain) if stems & set(names[seen]))
            between = chain[:depth]
            if hinted.key in NATIVE_BINARIES:
                # Its executable would be the nearest client if it were running: the hint is stale.
                return PROVIDERS[seen]
            # A script-based or unlisted client shows up as its own name or as a JS
            # runtime sitting between the helper and the outer client (for example
            # Gemini CLI started from a terminal inside another agent).
            hosted = any(stems & (SCRIPT_RUNTIMES | {hinted.key} | set(hinted.processes)) for stems in between)
            if not hosted:
                return PROVIDERS[seen]
        return hinted
    if seen:
        return PROVIDERS[seen]
    for provider in PROVIDERS.values():
        if any(environ.get(name) for name in provider.session_env + provider.marker_env):
            return provider
    raise ValueError("Cannot identify the current client. Supply --provider (any client name works; see `providers`). "
                     "The most recently modified chat is deliberately never guessed.")


def _clean_id(value):
    value = (value or "").strip()
    # An unsubstituted skill placeholder is not an id.
    return value if re.fullmatch(r"[A-Za-z0-9_-]{8,100}", value) else None


def open_session(provider: Provider, session_id=None, path=None, home=None) -> tuple[Session, str | None]:
    """The chat's transcript when it can be identified exactly, else an untracked session.

    Returns (session, note). An untracked session is not an error: the run uses
    helper-clock timing and binds to its transcript later by run id if one appears.
    """
    root = Path(home) if home else (provider.home() if provider.home else None)
    given = _clean_id(session_id)
    env_id = _clean_id(first_env(provider.session_env)) if provider.session_env else None
    fallback = Session(provider.key, given or env_id or f"untracked-{uuid.uuid4().hex[:12]}", None, root)
    if provider.parse is None:
        return fallback, None if provider.known and provider.native != "helper clock only" else (
            f"{provider.title} has no transcript adapter: helper-clock timing is used; model and effort come from --model/--effort labels.")
    if provider.key in ("claude", "codex"):
        if path:
            return resolve_session(provider.key, given, path, root), None  # Explicit paths fail loudly.
        for candidate in dict.fromkeys(i for i in (given, env_id) if i):
            try:
                return resolve_session(provider.key, candidate, None, root), None
            except ValueError:
                continue
    else:
        if path:
            target = Path(path).expanduser().resolve()
            if not target.is_file():
                raise ValueError("The transcript path does not exist.")
            observed = provider.session_id_of(target) if provider.session_id_of else None
            return Session(provider.key, given or observed or fallback.session_id, target, root), None
        direct = os.environ.get("PI_SESSION_FILE") if provider.key == "pi" else None
        if direct and Path(direct).is_file():
            return Session(provider.key, given or env_id or adapters.pi_session_id(Path(direct)), Path(direct), root), None
        for candidate in dict.fromkeys(i for i in (given, env_id) if i):
            found = provider.find(root, candidate) if provider.find and root else None
            if found:
                observed = provider.session_id_of(found) if provider.session_id_of else candidate
                return Session(provider.key, observed or candidate, found, root), None
    return fallback, (f"{provider.title}'s transcript for this chat is not identified yet; it will be matched by run id "
                      "after the first sample. Until then helper-clock timing is used.")


def late_bind(run: dict) -> Session | None:
    """Find the transcript that contains this run's id and attach it to the run.

    The run id is a fresh random value that appears only in this chat's tool
    calls, so a match identifies the chat exactly; recency alone never does.
    """
    session = Session.from_dict(run["session"])
    try:
        provider = provider_for(session.provider)
    except ValueError:
        return None
    root = session.home or (provider.home() if provider.home else None)
    if session.tracked or provider.parse is None or root is None or not Path(root).exists():
        return None
    nonce, since = run["run_id"], run.get("started_ms", 0)
    found = None
    try:
        if provider.locate:
            found = provider.locate(Path(root), nonce, since)
        elif provider.candidates:
            needle = nonce.encode()
            for path in provider.candidates(Path(root)):
                try:
                    stat = path.stat()
                    if stat.st_mtime * 1000 < since - 120_000 or stat.st_size > MAX_SCAN_BYTES:
                        continue
                    if needle in path.read_bytes():
                        found = (path, provider.session_id_of(path) if provider.session_id_of else None)
                        break
                except OSError:
                    continue
    except OSError:
        return None
    if not found:
        return None
    path, session_id = found
    bound = Session(provider.key, session_id or session.session_id, Path(path), Path(root))
    run["session"] = bound.to_dict()
    note = "Transcript matched by run id after the first sample; native timing and usage are attached from there."
    if note not in run.setdefault("operational_notes", []):
        run["operational_notes"].append(note)
    return bound


def skill_path(provider: Provider) -> Path | None:
    return provider.skills() / SKILL_NAME / "SKILL.md" if provider.skills else None


def is_detected(provider: Provider) -> bool:
    if any(shutil.which(name) for name in provider.binaries):
        return True
    try:
        return bool(provider.home and provider.home().exists())
    except OSError:
        return False


def provider_status(root: Path | None = None) -> dict:
    result = {}
    for key, provider in PROVIDERS.items():
        skill = skill_path(provider)
        binary = next((shutil.which(name) for name in provider.binaries if shutil.which(name)), None)
        result[key] = {"title": provider.title, "detected": is_detected(provider), "binary": binary,
                       "home": str(provider.home()) if provider.home else None,
                       "skill": str(skill) if skill else None,
                       "skill_installed": bool(skill and skill.is_file()),
                       "native_data": provider.native, "sample_delivery": provider.delivery(),
                       "invoke": provider.invoke}
        if provider.telemetry:
            result[key]["telemetry"] = telemetry_state(key)
    return result


def telemetry_state(key: str):
    import json
    try:
        if key == "codex":
            import tomllib
            config = tomllib.loads((codex_home() / "config.toml").read_text(encoding="utf-8-sig"))
            return "configured" if config.get("otel", {}).get("exporter") else "not configured (TTFT stays N/A; run install)"
        env = json.loads((claude_home() / "settings.json").read_text(encoding="utf-8-sig")).get("env", {})
        on = env.get("CLAUDE_CODE_ENABLE_TELEMETRY") == "1" and "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" in env
        return "configured" if on else "not configured (native TTFT stays N/A; a derived value is shown; run install)"
    except (OSError, ValueError):
        return "not configured"


def capability_table(root: Path | None = None) -> str:
    out = ["| Client | Key | Detected | Skill | Native data | Samples travel by | Start with |",
           "|---|---|---|---|---|---|---|"]
    for key, state in provider_status(root).items():
        out.append(f"| {state['title']} | `{key}` | {'yes' if state['detected'] else 'no'} | "
                   f"{'installed' if state['skill_installed'] else 'not installed'} | {state['native_data']} | "
                   f"{state['sample_delivery']} | `{state['invoke']}` |")
    out += ["", "Every client, listed or not, is measured with the helper clock; native data refines it where an adapter exists. "
            "For an unlisted CLI use `prepare --provider <name>`."]
    return "\n".join(out)
