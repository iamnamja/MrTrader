#!/usr/bin/env bash
# Stop hook — fires after Claude finishes a turn.
# Checks if app/ml/ was modified this session but ML_EXPERIMENT_LOG.md was not.
# Non-blocking (exit 0 always) — just prints a reminder Claude will see.

export PYTHONIOENCODING=utf-8

ml_changed=$(git -C c:/Projects/MrTrader diff --name-only HEAD 2>/dev/null | \
  grep -E '^app/ml/' | wc -l | tr -d ' ')

log_changed=$(git -C c:/Projects/MrTrader diff --name-only HEAD 2>/dev/null | \
  grep -c 'ML_EXPERIMENT_LOG' || echo 0)

if [ "$ml_changed" -gt 0 ] && [ "$log_changed" -eq 0 ]; then
  echo "REMINDER: app/ml/ was modified but docs/ML_EXPERIMENT_LOG.md has no new entry." >&2
  echo "Run /log-model or /log-wf before merging (feedback_model_documentation.md)." >&2
fi

exit 0
