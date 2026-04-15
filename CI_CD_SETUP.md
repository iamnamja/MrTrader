# GitHub CI/CD & Branch Protection Setup

## Step 1: Push the CI Workflow

First, commit and push the `.github/workflows/ci.yml` file:

```powershell
git add .github/
git commit -m "Add GitHub Actions CI pipeline"
git push origin main
```

---

## Step 2: Configure Branch Protection for `develop`

Go to: **GitHub.com** → Your Repo → **Settings** → **Branches**

### Add branch protection rule for `develop`:

1. Click **"Add rule"**
2. **Branch name pattern**: `develop`
3. Enable these:
   - ✅ **Require a pull request before merging**
     - Require approvals: `1`
     - Dismiss stale PR approvals: ✅
   - ✅ **Require status checks to pass before merging**
     - Require branches to be up to date: ✅
     - Required status checks: Select `lint-and-test` and `check-security`
   - ✅ **Require code reviews before merging**
   - ✅ **Allow auto-merge**: Enable (for convenience)
   - ✅ **Allow force pushes**: Disable
   - ✅ **Allow deletions**: Disable

4. Click **"Create"**

---

## Step 3: Configure Branch Protection for `main`

Repeat for `main` branch (even stricter):

1. Click **"Add rule"**
2. **Branch name pattern**: `main`
3. Enable these:
   - ✅ **Require a pull request before merging**
     - Require approvals: `2` (for production)
     - Require review from Code Owners: ✅
     - Dismiss stale PR approvals: ✅
   - ✅ **Require status checks to pass**
   - ✅ **Require branches to be up to date**
   - ✅ **Allow auto-merge**: Enable
   - ✅ **Allow force pushes**: Disable
   - ✅ **Allow deletions**: Disable

4. Click **"Create"**

---

## Step 4: Create `.github/CODEOWNERS` (Optional)

This file specifies who must review code changes:

```
# Default owners for all files
* @iamnamja

# You can add more specific rules later
# app/agents/* @agent-reviewer
# app/ml/* @ml-reviewer
```

---

## Step 5: Enable Auto-Merge on PRs

Settings → **General** → **Pull Requests** section

Enable:
- ✅ **Allow auto-merge**
- ✅ **Delete head branches** (auto-cleanup after merge)

---

## Now Your Workflow Is:

### For Each Phase:

```powershell
# 1. Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/phase-X-name

# 2. Work and commit
git add .
git commit -m "Phase X: Description"

# 3. Push to GitHub
git push -u origin feature/phase-X-name
```

### On GitHub:

1. **PR automatically created** (or create manually)
   - Base: `develop`
   - Compare: `feature/phase-X-name`

2. **CI Pipeline runs automatically**:
   - Tests run
   - Linting runs
   - Security check runs
   - Status shows on PR

3. **If all checks pass**:
   - Green ✅ shows "All checks passed"
   - You can **enable auto-merge**

4. **Auto-merge happens**:
   - PR automatically merges when all requirements met
   - Branch auto-deletes
   - `develop` updated

### Then for Production Release:

```powershell
# Create PR from develop → main
# Same process, but requires 2 approvals
# Once merged to main, tag a release
git checkout main
git pull origin main
git tag -a v1.0.0 -m "Release Phase 9"
git push origin --tags
```

---

## What Gets Checked

**On every PR:**
- ✅ Code formatting (Black)
- ✅ Import sorting (isort)
- ✅ Linting (flake8)
- ✅ Tests pass (pytest)
- ✅ No secrets in code
- ✅ Branch up to date with base
- ✅ Code review approved

**If any fail:**
- ❌ PR shows red ✗
- ❌ Can't merge until fixed

---

## Quick Checklist

- [ ] Commit and push `.github/workflows/ci.yml` to main
- [ ] Go to GitHub Settings → Branches
- [ ] Create branch protection rule for `develop`
- [ ] Create branch protection rule for `main`
- [ ] (Optional) Create `.github/CODEOWNERS`
- [ ] Enable auto-merge in Settings → General
- [ ] Test with a PR: create feature branch → push → create PR
- [ ] Watch CI pipeline run automatically

---

## Example: First PR with CI

```powershell
# 1. Create feature branch
git checkout develop
git checkout -b feature/test-ci

# 2. Make a small change
echo "# Test" > test.md

# 3. Commit and push
git add test.md
git commit -m "Test: CI pipeline"
git push -u origin feature/test-ci

# 4. On GitHub, create PR
# → See CI pipeline run automatically
# → All checks pass/fail
# → See status on PR

# 5. Once approved and checks pass, enable auto-merge
# → PR auto-merges to develop
# → Branch auto-deletes
```

---

## Troubleshooting

**"CI pipeline failing on my PR?"**
- Check the logs: Click "Details" next to failed check
- Common issues: Tests fail, linting errors, import sorting
- Fix locally, push again, CI reruns automatically

**"Can't merge PR?"**
- Make sure branch is up to date: `git pull origin develop`
- Ensure all checks are passing (green ✅)
- For main: need 2 approvals, not just 1

**"Want to bypass protection temporarily?"**
- Don't! That defeats the purpose
- Fix the underlying issue instead
- Only admins can bypass (not recommended for main)
