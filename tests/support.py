"""Test isolation. Import this first in every test module.

The installer and the adapters resolve client homes from the user's home
directory. Tests must never write there, so the whole test process is pointed at
a throwaway home before any benchmark module is imported.
"""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

REAL_HOME = Path.home()
SANDBOX = Path(tempfile.mkdtemp(prefix="chat-benchmark-tests-"))
for _name in ("HOME", "USERPROFILE"):
    os.environ[_name] = str(SANDBOX)
for _name, _sub in (("CODEX_HOME", ".codex"), ("CLAUDE_CONFIG_DIR", ".claude"), ("KIMI_CODE_HOME", ".kimi-code"),
                    ("GROK_HOME", ".grok"), ("XDG_DATA_HOME", ".local/share"), ("PI_CODING_AGENT_DIR", ".pi/agent"),
                    ("CHAT_BENCHMARK_HOME", ".chat-benchmark")):
    os.environ[_name] = str(SANDBOX / _sub)
# The suite may itself be launched from inside an agent CLI; do not inherit its identity.
for _name in ("CLAUDE_SESSION_ID", "CLAUDE_CODE_SESSION_ID", "CLAUDECODE", "CODEX_THREAD_ID", "CODEX_SESSION_ID",
              "GROK_SESSION_ID", "PI_SESSION_ID", "PI_SESSION_FILE", "ANTIGRAVITY_CONVERSATION_ID", "ANTIGRAVITY_AGENT",
              "CURSOR_CONVERSATION_ID", "CURSOR_AGENT", "OPENCODE", "GEMINI_CLI", "QWEN_CODE"):
    os.environ.pop(_name, None)
atexit.register(shutil.rmtree, SANDBOX, True)

assert Path.home() == SANDBOX and Path.home() != REAL_HOME, "test isolation failed: refusing to run against the real home"
