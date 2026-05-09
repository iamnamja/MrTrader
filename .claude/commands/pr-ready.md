---
description: Pre-PR checklist — pytest, flake8, branch sync, stale branches, change summary
---

Run the pre-PR checklist. Do steps 1-3 in parallel, then 4-5 sequentially after.

## Step 1 (parallel) — Branch checks
- Current branch: `git -C c:/Projects/MrTrader rev-parse --abbrev-ref HEAD`
  - FAIL if branch is `main` or `master` — you cannot PR from main.
- Sync status: `git -C c:/Projects/MrTrader fetch origin main && git -C c:/Projects/MrTrader rev-list --left-right --count origin/main...HEAD`
  - WARN if behind > 0 — recommend: `git fetch origin main && git merge origin/main`
- Working tree: `git -C c:/Projects/MrTrader status --porcelain`
  - List any uncommitted files. Ask user if they intend to include them.

## Step 2 (parallel) — Code quality
- Flake8: `c:/Projects/MrTrader/venv/Scripts/python.exe -m flake8 c:/Projects/MrTrader/app c:/Projects/MrTrader/scripts c:/Projects/MrTrader/tests --max-line-length=120 --count`
  - FAIL if any errors (count > 0).
- Check flake8 config: look for `c:/Projects/MrTrader/.flake8` or `[flake8]` section in `setup.cfg` — use those settings if present.

## Step 3 (parallel) — Documentation check
- ML code changed? `git -C c:/Projects/MrTrader diff --name-only origin/main...HEAD | grep "^app/ml/"`
- Experiment log updated? `git -C c:/Projects/MrTrader diff --name-only origin/main...HEAD | grep "ML_EXPERIMENT_LOG"`
- If ml/ changed but log NOT updated → FAIL with: "Model code changed but ML_EXPERIMENT_LOG.md has no new entry. Run /log-model or /log-wf first."

## Step 4 — Full test suite
Run: `c:/Projects/MrTrader/venv/Scripts/python.exe -m pytest c:/Projects/MrTrader/tests -x -q --tb=short`
- FAIL if any test failures (per feedback_merge_policy.md: 0 failures required).
- This takes several minutes — warn the user before starting.
- Show the final "N passed" line.

## Step 5 — Stale branch cleanup
`git -C c:/Projects/MrTrader branch --merged main | grep -v "^\* " | grep -v "^  main$"`
- List any merged branches that haven't been deleted.
- Recommend: `git branch -d <branch>` for each (do NOT auto-delete).

## Step 6 — Change summary
`git -C c:/Projects/MrTrader log --oneline origin/main..HEAD`
`git -C c:/Projects/MrTrader diff --stat origin/main...HEAD`
Summarize what this PR contains in 2-3 sentences.

## Final report
Print a clear PASS / FAIL table:
```
✅/❌  Not on main branch
✅/❌  Branch up to date with origin/main
✅/❌  Flake8 clean
✅/❌  ML changes documented
✅/❌  All tests pass (N passed)
⚠️    N stale merged branches (optional cleanup)
```

If all ✅: suggest the exact `gh pr create` command:
```
gh pr create --title "<concise title>" --body "$(cat <<'EOF'
## Summary
- <bullet 1>
- <bullet 2>

## Test plan
- [ ] Full pytest suite: N passed, 0 failed
- [ ] Flake8 clean
- [ ] Walk-forward / CPCV: <result>

🤖 Generated with Claude Code
EOF
)"
```

Enable auto-merge: `gh pr merge --auto --squash --delete-branch`
