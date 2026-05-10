#!/usr/bin/env bash
# UserPromptSubmit hook — injects session context into every Claude prompt.
# The || true ensures a broken hook never blocks the session.
export PYTHONIOENCODING=utf-8
PY="c:/Projects/MrTrader/venv/Scripts/python.exe"
[ -x "$PY" ] || PY=python
"$PY" "c:/Projects/MrTrader/.claude/hooks/session_context.py" 2>/dev/null || true
