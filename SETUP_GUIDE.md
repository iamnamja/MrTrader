# Complete MrTrader Setup Guide (Windows)

## Step 1: Fix Virtual Environment & Dependencies

### 1a. Clean up and reinstall

```powershell
# Remove old venv
Remove-Item -Recurse -Force venv

# Create fresh venv
python -m venv venv

# Activate
.\venv\Scripts\Activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies (this should now work)
pip install -r requirements.txt
```

**Expected output**: All packages install successfully without errors.

---

## Step 2: Install & Setup Docker Desktop (Windows)

### 2a. Install Docker Desktop

1. Download: https://www.docker.com/products/docker-desktop
2. Run installer
3. Restart your computer (required!)
4. Verify installation:

```powershell
docker --version
# Should output: Docker version 25.x.x ...

docker ps
# Should show "CONTAINER ID   IMAGE..." (no errors)
```

### 2b. Use `docker compose` (new way, not `docker-compose`)

**Important**: Windows Docker Desktop uses `docker compose` (space, not dash)

```powershell
# Start PostgreSQL + Redis
docker compose up -d

# Verify they're running
docker ps
# Should show 2 containers: mrtrader_postgres and mrtrader_redis
```

If containers don't start, check Docker Desktop is actually running (look for whale icon in system tray).

---

## Step 3: Setup GitHub

### 3a. Initialize Git Repository

```powershell
cd C:\Projects\MrTrader

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Phase 1 foundation"
```

### 3b. Create GitHub Repository

1. Go to https://github.com/new
2. Create repository named `MrTrader` (or your choice)
3. **Don't** initialize with README (you already have one)
4. Click "Create repository"

### 3c. Connect Local to GitHub

```powershell
# Add remote (replace YOUR_USERNAME and YOUR_REPO)
git remote add origin https://github.com/YOUR_USERNAME/MrTrader.git

# Rename branch to main (GitHub default)
git branch -M main

# Push to GitHub
git push -u origin main
```

**After this**: Your code is backed up on GitHub!

### 3d. Verify Setup

```powershell
git remote -v
# Should show:
# origin  https://github.com/YOUR_USERNAME/MrTrader.git (fetch)
# origin  https://github.com/YOUR_USERNAME/MrTrader.git (push)

git log
# Should show your commit
```

---

## Step 4: Environment Configuration

### 4a. Create `.env` from template

```powershell
# Copy template
Copy-Item .env.example .env

# Edit .env
notepad .env
```

### 4b. Fill in required values

```
# Database (leave as-is if using Docker)
DATABASE_URL=postgresql://mrtrader:mrtrader_password@localhost:5432/mrtrader

# Redis (leave as-is if using Docker)
REDIS_URL=redis://localhost:6379

# Alpaca API (you MUST add these)
ALPACA_API_KEY=your_actual_api_key_here
ALPACA_SECRET_KEY=your_actual_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Optional (can skip for now)
NEWS_API_KEY=
SLACK_WEBHOOK_URL=
```

**How to get Alpaca keys:**
1. Sign up: https://alpaca.markets
2. Go to Account → API Keys
3. Copy "Key ID" and "Secret Key"
4. Paste into .env

### 4c. Add .env to .gitignore (don't commit secrets!)

```powershell
# .env is already in .gitignore, but verify:
cat .gitignore | grep ".env"
# Should show: ".env" and ".env.local"
```

---

## Step 5: Test Everything Works

### 5a. Initialize Database

```powershell
# With venv activated
python -c "from app.database import init_db; init_db()"

# Should output:
# Initializing database...
# Database initialization complete
```

### 5b. Start the app

```powershell
uvicorn app.main:app --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### 5c. Test the app

Open browser and go to: http://localhost:8000/docs

You should see:
- ✅ FastAPI Swagger UI
- ✅ All endpoints listed
- ✅ Try-it-out buttons

### 5d. Check system health

Click "Try it out" on `/api/status` endpoint:

**Expected response:**
```json
{
  "status": "healthy",
  "database": "✓ connected",
  "redis": "✓ connected",
  "alpaca": "✓ connected",
  "mode": "paper"
}
```

If any shows `✗ disconnected`:
- Database: Check `docker ps`, restart if needed
- Redis: Check `docker ps`, restart if needed  
- Alpaca: Check your API keys in `.env`

---

## Step 6: Commit Working Setup to GitHub

```powershell
# Check what changed
git status

# Add changes
git add .

# Commit
git commit -m "Setup Phase 1: Dependencies, Docker, env config"

# Push to GitHub
git push origin main
```

---

## Complete Checklist

- ✅ Virtual environment created and activated
- ✅ All Python dependencies installed (`pip list | grep fastapi` shows packages)
- ✅ Docker Desktop running (check system tray)
- ✅ Postgres + Redis containers running (`docker ps` shows 2 containers)
- ✅ `.env` file created with Alpaca keys
- ✅ Database initialized (`python -c "from app.database import init_db; init_db()"`)
- ✅ App starts (`uvicorn app.main:app --reload` runs without errors)
- ✅ Dashboard loads (`http://localhost:8000/docs` works)
- ✅ Health check passes (all systems show ✓)
- ✅ Git repository initialized
- ✅ GitHub repository created and connected
- ✅ Code pushed to GitHub

---

## Troubleshooting

### "pip install" still fails
```powershell
# Try upgrading pip first
python -m pip install --upgrade pip
# Then try again
pip install -r requirements.txt
```

### Docker containers won't start
```powershell
# Check if Docker Desktop is actually running
docker ps

# If error: start Docker Desktop from Start menu or system tray

# If containers crashed, remove and restart
docker compose down
docker compose up -d

# Check logs
docker logs mrtrader_postgres
docker logs mrtrader_redis
```

### "Could not connect to Alpaca" error
- Verify `ALPACA_BASE_URL=https://paper-api.alpaca.markets` (check for typos)
- Verify API keys are correct (copy-paste from Alpaca dashboard)
- Verify internet connection

### "Port 8000 already in use"
```powershell
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# Then try running uvicorn again
```

---

## Next: Go to Claude Code!

Once all checks pass:

1. **Keep terminal running with the app** (don't close uvicorn)
2. **Switch to Claude Code tab** in VS Code
3. **Tell Claude**: "I've completed Phase 1 setup. All systems healthy. Let's build Phase 2 (Risk Manager Agent)."

Claude Code will have full context from your memory files and will build Phase 2 with complete code examples.

---

## Git Workflow Going Forward

**After each phase:**
```powershell
# Check changes
git status

# Add all changes
git add .

# Commit with meaningful message
git commit -m "Phase 2: Risk Manager Agent implementation"

# Push to GitHub
git push origin main
```

This keeps your code backed up and creates a history of your work!
