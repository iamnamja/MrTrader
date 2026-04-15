# Git Workflow for MrTrader

## Branch Strategy

```
main (stable, production-ready)
  └── develop (integration branch)
       ├── feature/phase-2-risk-manager
       ├── feature/phase-3-trader-agent
       └── feature/phase-4-portfolio-ml
```

---

## Setup Instructions

### 1. Create `develop` branch (integration)

```powershell
git checkout -b develop
git push -u origin develop
```

### 2. Create feature branches for each phase

```powershell
# For Phase 2
git checkout -b feature/phase-2-risk-manager

# For Phase 3
git checkout -b feature/phase-3-trader-agent

# For Phase 4
git checkout -b feature/phase-4-portfolio-ml

# etc.
```

---

## Workflow for Each Phase

### When Starting a Phase:

```powershell
# 1. Make sure you're on develop (latest)
git checkout develop
git pull origin develop

# 2. Create feature branch from develop
git checkout -b feature/phase-X-name

# 3. Work on phase (commits, changes, etc)
git add .
git commit -m "Phase X: Description of work"

# 4. When done, push to GitHub
git push -u origin feature/phase-X-name
```

### When Phase is Complete:

```powershell
# 1. Push all commits
git push origin feature/phase-X-name

# 2. Go to GitHub and create Pull Request:
#    - Base: develop (merge into)
#    - Compare: feature/phase-X-name (merge from)

# 3. Review your own PR (check diffs)

# 4. Merge PR on GitHub (or via CLI):
git checkout develop
git pull origin develop
git merge feature/phase-X-name
git push origin develop

# 5. Delete feature branch
git branch -D feature/phase-X-name
git push origin --delete feature/phase-X-name
```

### When Ready for Production (Phase 9 complete):

```powershell
# Create PR from develop → main
# After review/testing, merge to main
# Tag the release

git checkout main
git pull origin main
git merge develop
git tag -a v1.0.0 -m "Phase 9: Full system complete"
git push origin main --tags
```

---

## Current Status

You're currently on **main**. Here's what to do now:

1. Create `develop` branch (below)
2. Move Phase 1 work to develop
3. Start Phase 2 on a feature branch

---

## Do This Now:

```powershell
# 1. Verify you're on main
git branch

# 2. Create develop from main
git checkout -b develop

# 3. Push develop to GitHub
git push -u origin develop

# 4. Switch back to main
git checkout main

# 5. Verify branches
git branch -a
# Should show:
# * main
#   develop
```

---

## Why This Matters

| Scenario | Without Branches | With Branches |
|----------|------------------|---------------|
| **Bug in Phase 2** | Breaks main/live | Only affects feature branch, main stays clean |
| **Need to rollback** | Revert entire commits | Just don't merge the PR |
| **Multiple phases** | Conflicts everywhere | Each phase isolated |
| **Code review** | Harder to track changes | PR shows exact diff |
| **Production issue** | Can't hotfix safely | Create hotfix branch from main |

---

## GitHub Configuration (Optional but Recommended)

On GitHub repo settings:

1. **Branch protection rules**:
   - Protect `main` branch
   - Require pull request reviews before merge
   - Require status checks to pass

2. **Default branch**:
   - Set to `develop` (for development)

This prevents accidental direct pushes to main.
