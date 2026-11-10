"""Install the benchmark skill for every agent CLI, without model overrides."""
from __future__ import annotations

import json
import shutil
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from .providers import OPT_IN, PROVIDERS, SKILL_NAME, is_detected
from .sessions import claude_home, codex_home
from .standard import tokenizer
from .telemetry import PORT

TEMPLATE = "SKILL.template.md"


def render_skill(template: str, provider, runtime: Path) -> str:
    """One skill text for every client; only the client hint and session argument differ.

    Clients load each other's skill folders, so each copy must work anywhere: the
    helper verifies the real client itself and treats the hint as a fallback.
    """
    session = f'--session-id="{provider.session_placeholder}" ' if provider.session_placeholder else ""
    return (template.replace("{{PYTHON}}", sys.executable.replace("\\", "/"))
            .replace("{{CLI}}", str(runtime / "benchmark.py").replace("\\", "/"))
            .replace("{{PROVIDER}}", provider.key if provider.key != "agents" else "auto")
            .replace("{{SESSION_ARG}}", session))


def install_runtime(root: Path, source: Path) -> Path:
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    if source.resolve() != runtime.resolve():
        shutil.copy2(source / "benchmark.py", runtime / "benchmark.py")
        shutil.copy2(source / "requirements.txt", runtime / "requirements.txt")
        for folder in ("chat_benchmark", "skills"):
            # Replace rather than merge, so files removed upstream do not linger.
            target = runtime / folder
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source / folder, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return runtime


def install_skill(provider, template: str, runtime: Path) -> dict:
    skill_dir = provider.skills() / SKILL_NAME
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(render_skill(template, provider, runtime), encoding="utf-8")
    written = [str(skill_dir / "SKILL.md")]
    if provider.key == "codex":
        agents = skill_dir / "agents"
        agents.mkdir(exist_ok=True)
        (agents / "openai.yaml").write_text(
            'interface:\n  display_name: "Chat Benchmark"\n'
            '  short_description: "Measure this chat\'s generation speed, TTFT and token usage"\n'
            '  default_prompt: "Use $chat-benchmark to benchmark this chat with its active settings."\n',
            encoding="utf-8")
    if provider.key == "gemini":
        # Gemini CLI activates skills itself; a custom command gives the user an explicit /chat-benchmark.
        commands = provider.home() / "commands"
        commands.mkdir(parents=True, exist_ok=True)
        command = commands / f"{SKILL_NAME}.toml"
        command.write_text('description = "Benchmark this chat\'s generation speed"\n'
                           f'prompt = "Read the file {(skill_dir / "SKILL.md").as_posix()} and follow it exactly to benchmark this chat. {{{{args}}}}"\n',
                           encoding="utf-8")
        written.append(str(command))
    return {"client": provider.key, "title": provider.title, "files": written, "start_with": provider.invoke}


def install_telemetry(root: Path, backup: Path, warnings: list) -> dict:
    """Point Codex and Claude Code telemetry at the local receiver (adds native TTFT)."""
    codex_root, claude_root = codex_home(), claude_home()
    codex_config, claude_config = codex_root / "config.toml", claude_root / "settings.json"
    original_codex = codex_config.read_bytes() if codex_config.exists() else b""
    original_claude = claude_config.read_bytes() if claude_config.exists() else b"{}"
    config = tomllib.loads(original_codex.decode("utf-8-sig"))
    claude = json.loads(original_claude.decode("utf-8-sig"))
    endpoint = f"http://127.0.0.1:{PORT}"
    (backup / "codex-config.toml").write_bytes(original_codex)
    (backup / "claude-settings.json").write_bytes(original_claude)
    changed = {}
    if is_detected(PROVIDERS["codex"]):
        otel = config.get("otel", {})
        ours = otel.get("exporter", {}).get("otlp-http", {}).get("endpoint") if isinstance(otel.get("exporter"), dict) else None
        if not otel:
            addition = (
                "\n\n# Local current-chat benchmark. No model, effort, or service-tier overrides.\n"
                "[otel]\n"
                f'exporter = {{ otlp-http = {{ endpoint = "{endpoint}/v1/logs", protocol = "json" }} }}\n'
                "log_user_prompt = false\nlog_agent_responses = false\n"
                "[otel.tool_result]\nmax_bytes = 0\n")
            updated = original_codex.decode("utf-8-sig").rstrip() + addition
            parsed = tomllib.loads(updated)
            assert {k: v for k, v in parsed.items() if k != "otel"} == config
            codex_root.mkdir(parents=True, exist_ok=True)
            codex_config.write_text(updated, encoding="utf-8", newline="")
            changed["codex"] = "added"
        elif ours != endpoint + "/v1/logs":
            warnings.append("Existing Codex telemetry was preserved. Native request TTFT needs an OTLP fan-out to the local endpoint.")
        else:
            changed["codex"] = "already configured"
    if is_detected(PROVIDERS["claude"]):
        env = claude.setdefault("env", {})
        desired = {
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1", "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
            "OTEL_TRACES_EXPORTER": "otlp", "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/json",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint + "/v1/traces",
            "OTEL_TRACES_EXPORT_INTERVAL": "1000",
            "OTEL_LOGS_EXPORTER": "otlp", "OTEL_EXPORTER_OTLP_LOGS_PROTOCOL": "http/json",
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": endpoint + "/v1/logs", "OTEL_LOGS_EXPORT_INTERVAL": "1000",
        }
        conflicting = [k for k, v in desired.items() if k in env and env[k] != v]
        if conflicting:
            warnings.append("Existing Claude tracing settings were preserved: " + ", ".join(conflicting))
        elif all(env.get(k) == v for k, v in desired.items()):
            changed["claude"] = "already configured"
        else:
            env.update(desired)
            claude_root.mkdir(parents=True, exist_ok=True)
            # No session/global model settings, hooks, permissions, status line, or plugins are replaced.
            claude_config.write_text(json.dumps(claude, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            changed["claude"] = "added"
    return changed


def install(root: Path, source: Path | None = None, clients=None, all_clients=False, telemetry=True) -> dict:
    tokenizer()  # Fail before writing config if the offline reference is unavailable.
    source = source or Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=True)
    runtime = install_runtime(root, source)
    template = (runtime / "skills" / TEMPLATE).read_text(encoding="utf-8")
    unknown = [c for c in clients or [] if c not in PROVIDERS]
    if unknown:
        raise ValueError("Unknown client key(s): " + ", ".join(unknown) + ". See `providers`.")
    backup = root / "backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup.mkdir(parents=True)
    warnings, installed, skipped = [], [], []
    for key, provider in PROVIDERS.items():
        if provider.skills is None:
            continue
        wanted = key in clients if clients else (all_clients or is_detected(provider)) and key not in OPT_IN
        if not wanted:
            skipped.append(key)
            continue
        try:
            installed.append(install_skill(provider, template, runtime))
        except OSError as exc:
            warnings.append(f"{provider.title}: could not write the skill ({exc}).")
    telemetry_changes = install_telemetry(root, backup, warnings) if telemetry else {}
    result = {"runtime": str(runtime), "installed": installed,
              "skills": [str(Path(item["files"][0]).parent) for item in installed],
              "not_installed": skipped, "backup": str(backup), "telemetry": telemetry_changes,
              "telemetry_endpoint": f"http://127.0.0.1:{PORT}", "warnings": warnings,
              "restart_required": "Restart each client once so it loads the skill (and, for Codex and Claude Code, the telemetry exporter). Then say \"run the benchmark\" in any chat.",
              "note": "Clients not detected on this machine were skipped; use --clients or --all-clients to include them. Any other CLI works with `prepare --provider <name>`."}
    (root / "installation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
