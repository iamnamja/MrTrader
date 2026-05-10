#!/usr/bin/env bash
# PreToolUse hook for Bash — blocks direct pushes to main/master.
# Receives tool input JSON on stdin. Exit 2 = block + show stderr to Claude.

export PYTHONIOENCODING=utf-8
PY="c:/Projects/MrTrader/venv/Scripts/python.exe"
[ -x "$PY" ] || PY=python

# Extract the bash command from the JSON input
input="$(cat)"
cmd=$( printf '%s' "$input" | "$PY" -c \
  "import json,sys; d=json.load(sys.stdin); print(d.get('tool_input',d).get('command',''))" \
  2>/dev/null || echo "" )

if [ -z "$cmd" ]; then
  exit 0
fi

# Block: explicit push to main or master (covers -u, --set-upstream, HEAD:main, origin main)
if echo "$cmd" | grep -qE \
  'git[[:space:]]+push.*(origin[[:space:]]+(main|master)|HEAD:(main|master)|(main|master)[[:space:]]*$)'; then
  echo "BLOCKED: Direct push to main/master is forbidden (feedback_branch_workflow.md)." >&2
  echo "Use a feature branch + PR instead." >&2
  exit 2
fi

# Block: bare 'git push' while currently on main/master
if echo "$cmd" | grep -qE '^[[:space:]]*git[[:space:]]+push([[:space:]]+-[a-zA-Z]+)*[[:space:]]*$'; then
  branch=$(git -C c:/Projects/MrTrader rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
  if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
    echo "BLOCKED: You are on branch '$branch'. Create a feature branch first." >&2
    exit 2
  fi
fi

# Warn (non-blocking): branch is behind origin/main before a push
if echo "$cmd" | grep -qE '^[[:space:]]*git[[:space:]]+push'; then
  behind=$(git -C c:/Projects/MrTrader rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
  if [ "$behind" -gt 0 ] 2>/dev/null; then
    echo "WARNING: Branch is $behind commit(s) behind origin/main." >&2
    echo "Run: git fetch origin main && git merge origin/main  (per feedback_merge_before_push.md)" >&2
  fi
fi

exit 0
