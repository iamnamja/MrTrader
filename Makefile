.PHONY: up down build logs shell test frontend

# Start full stack (Postgres + Redis + app)
up:
	docker compose up -d

# Start only infrastructure (Postgres + Redis), run app locally
infra:
	docker compose up -d postgres redis

# Build the app image
build:
	docker compose build app

# Stop and remove containers
down:
	docker compose down

# Tail app logs
logs:
	docker compose logs -f app

# Open a shell in the app container
shell:
	docker compose exec app bash

# Build the React frontend
frontend:
	cd frontend && npm install && npm run build

# Run tests (requires infra running on test ports)
test:
	docker compose -f docker-compose.test.yml up -d
	sleep 3
	pytest tests/ -x -q
	docker compose -f docker-compose.test.yml down -v

# Full rebuild: frontend + docker image
rebuild: frontend build
