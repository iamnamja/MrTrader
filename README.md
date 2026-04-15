# MrTrader - Automated Trading System

Automated day trading system powered by AI agents using FastAPI, PostgreSQL, and Redis.

## Architecture

- **FastAPI**: REST API + WebSocket for real-time updates
- **PostgreSQL**: Persistent storage (trades, decisions, audit logs, models)
- **Redis**: Message queue for agent communication
- **Alpaca**: Broker integration for paper/live trading
- **ML Models**: Daily-retrained models for instrument selection

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Alpaca API credentials (get at https://alpaca.markets)

### 1. Setup Environment

```bash
# Clone and setup
cd c:\Projects\MrTrader

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
copy .env.example .env
# Edit .env with your Alpaca API keys and configuration
```

### 2. Start Database & Redis

```bash
# Start PostgreSQL and Redis
docker-compose up -d

# Verify they're running
docker-compose ps
```

### 3. Initialize Database

```bash
python -c "from app.database import init_db; init_db()"
```

### 4. Run the Application

```bash
# Start FastAPI server
uvicorn app.main:app --reload --port 8000

# Open browser to http://localhost:8000/docs for API documentation
```

## Project Structure

```
MrTrader/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration settings
│   ├── database/
│   │   ├── models.py           # SQLAlchemy models
│   │   ├── session.py          # DB connection
│   │   └── __init__.py
│   ├── agents/                 # (To be created)
│   │   ├── base.py
│   │   ├── portfolio_manager.py
│   │   ├── risk_manager.py
│   │   └── trader.py
│   ├── integrations/
│   │   ├── alpaca.py           # Alpaca API client
│   │   ├── redis_queue.py      # Redis message queue
│   │   └── __init__.py
│   ├── indicators/             # (To be created)
│   │   └── technical.py
│   ├── ml/                     # (To be created)
│   │   ├── features.py
│   │   ├── model.py
│   │   └── training.py
│   ├── monitoring/             # (To be created)
│   │   ├── alerts.py
│   │   └── logger.py
│   ├── api/                    # (To be created)
│   │   ├── routes.py
│   │   └── schemas.py
│   └── utils/                  # (To be created)
│       └── constants.py
├── tests/                      # Unit and integration tests
├── scripts/                    # Utility scripts
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── .env                        # (Create from .env.example)
└── README.md
```

## API Endpoints (Available Now)

### Health & Status
- `GET /` - Welcome page
- `GET /health` - Health check
- `GET /api/status` - System status (DB, Redis, Alpaca)

### Account & Trading
- `GET /api/account` - Account information
- `GET /api/positions` - All open positions
- `GET /api/position/{symbol}` - Position for specific symbol

## Configuration

Edit `.env` file to configure:

```env
# Database
DATABASE_URL=postgresql://mrtrader:password@localhost:5432/mrtrader

# Redis
REDIS_URL=redis://localhost:6379

# Alpaca (required!)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # paper or live

# Trading
TRADING_MODE=paper
INITIAL_CAPITAL=20000
MAX_POSITION_SIZE_PCT=0.05
MAX_DAILY_LOSS_PCT=0.02

# Alerts (optional)
SLACK_WEBHOOK_URL=...
ALERT_EMAIL=...
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black app/ tests/
isort app/ tests/
flake8 app/ tests/
```

### Database Migrations

```bash
# (Using Alembic - to be setup)
alembic upgrade head
```

## Implementation Phases

### ✅ Phase 1: Foundation (Current)
- FastAPI server
- Database schema
- Alpaca integration
- Redis queue
- Basic API endpoints

### ⏳ Phase 2: Risk Manager Agent
- Risk rule validation
- Veto power logic
- Risk metrics tracking

### ⏳ Phase 3: Trader Agent
- Technical indicators
- Entry/exit logic
- Order management

### ⏳ Phase 4: Portfolio Manager + ML
- ML model training
- Instrument selection
- Daily rebalancing

### ⏳ Phase 5: Orchestration
- Agent scheduling
- Message routing
- Error handling

### ⏳ Phase 6: Dashboard
- Monitoring UI
- Real-time metrics
- Alert system

### ⏳ Phase 7: Backtesting
- Backtrader integration
- Strategy testing
- Performance evaluation

### ⏳ Phase 8: Paper Trading
- Live market testing
- Metrics collection

### ⏳ Phase 9: Go-Live Workflow
- Approval system
- Capital increase logic

## Troubleshooting

### Cannot connect to PostgreSQL
```bash
# Check if Docker container is running
docker ps

# View logs
docker logs mrtrader_postgres

# Restart services
docker-compose restart postgres
```

### Cannot connect to Redis
```bash
# Check if Redis is running
docker logs mrtrader_redis

# Test connection
redis-cli ping
```

### Alpaca API errors
- Verify API keys in `.env`
- Check market hours (US market only, 9:30 AM - 4:00 PM EST)
- Ensure paper trading mode for testing

## Next Steps

1. Create `.env` file with your Alpaca credentials
2. Start Docker services: `docker-compose up -d`
3. Run the app: `uvicorn app.main:app --reload`
4. Check status: `http://localhost:8000/api/status`
5. Continue with Phase 2: Build Risk Manager Agent

## References

- [Alpaca API Docs](https://docs.alpaca.markets/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLAlchemy Docs](https://docs.sqlalchemy.org/)
- [Redis Docs](https://redis.io/documentation)

## License

Proprietary - MrTrader Trading System

## Support

For issues or questions, review the plan file at `../claude/plans/nested-wobbling-bunny.md`
