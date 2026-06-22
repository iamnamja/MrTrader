# P2 — IBKR Futures Execution-Truth Design (Alpha-v10)

**Purpose:** turnkey spec so that on IBKR approval, building the futures-execution adapter is a
fast, planned sequence — not a multi-week unknown. Bakes in the external panel's production-trading
guidance (idempotency, broker-as-source-of-truth, immutable snapshots, replay parity, kill switch).

**Scope:** take the validated **futures multi-factor book** (`futures_carry` + `futures_xsmom`,
weekly) — and optionally `vix_vrp` — from paper-validated to **paper-traded on IBKR with real fills,
roll, and margin**, the gate to ever trading it with capital. Alpaca stays the equities/ETF venue;
IBKR is futures-only. **Build against the PAPER account first.**

> Why not build it now (pre-approval): the contract specs, order behaviors, and reconciliation must
> be **verified against the live API** (reqContractDetails, real fills) — building blind risks rework.
> The broker-agnostic logic (sizing, reconciliation, idempotency) is testable against a mock now if
> we want a head start, but the binding + integration test need the live paper gateway.

---

## 0. Connection
- **Library:** `ib_insync` (mature, async, well-documented) over **IB Gateway** (preferred for
  headless) or TWS, running in **paper** mode. Paper ports: IB Gateway **4002** / TWS **7497**
  (live 4001/7496). Config: `ibkr_host`, `ibkr_port`, `ibkr_client_id` in agent_config.
- Connection manager: connect on demand, auto-reconnect with backoff, heartbeat; **never trade if
  the connection/market-data/clock is stale** (fail-closed, mirrors the cash/trend sleeves).

## 1. Contract master (verify-on-connect)
- A static spec table for our ~38-market universe: `{norgate_sym: {ibkr_symbol, exchange,
  multiplier, currency, sec_type='FUT'}}` (e.g. ES→ES/CME/$50, NQ→NQ/CME/$20, CL→CL/NYMEX/1000,
  GC→GC/COMEX/100, ZN→ZN/CBOT, 6E→EUR/CME, ...). Public/stable values.
- **On connect, VERIFY the table against `reqContractDetails`** (multiplier, exchange, expiry) and
  **alert + block on any mismatch** — so a hand-entered error can never size an order wrong.
- **Roll calendar:** reuse `app/research/futures_roll.py` (scheduled-expiry front + 5-day buffer) to
  pick the front contract to hold + know roll windows; resolve the concrete front contract via IBKR
  (`reqContractDetails` for the active month) at trade time.

## 2. Execution flow (weekly, after signal compute)
```
target weights (futures_book) ──▶ target LOTS  ──▶ reconcile vs BROKER positions ──▶ orders
   (per-market, vol-targeted)      (w·NAV /         (broker = source of truth)        (idempotent)
                                    (price·mult),
                                    integer-rounded)
                              + ROLL: if held contract is within the roll window, add a
                                close-old + open-new calendar order (the roll transaction).
```
- **target_lots** = round(weight · NAV / (price · multiplier)); skip dust (< 1 lot); respect
  per-market + asset-class notional caps.
- **reconcile(target_lots, broker_positions)** → order deltas. Broker positions/cash/margin are
  pulled live; **block trading on any DB↔broker mismatch** ("the DB is not reality").

## 3. Safety machinery (the panel's production checklist)
- **Idempotency:** every order keyed by `run_id = strategy_id + signal_date + rebalance_ts +
  config_hash + code_version`; safe to retry after a crash (no double-send). Use IBKR
  `clientOrderId`/`orderRef` = run_id.
- **Immutable snapshots:** persist per run — data hash, signals, target weights, target lots,
  orders sent, broker responses/fills, final positions, margin. Reconstruct any position later.
- **Margin-aware order preview:** check initial/maintenance margin + buying power BEFORE sending;
  block if it would breach a margin/notional/stress-loss cap.
- **Risk layer (futures-specific):** contract-multiplier/tick sanity, max contracts/market,
  notional caps by asset class, exchange-hours/holiday gate, roll-window limits.
- **Kill switch:** app-level (existing) PLUS a broker-level flatten/cancel-all; **dead-man switch**
  — no new orders if heartbeat / reconciliation / market-data / margin checks fail.

## 4. Architecture (decouple before live capital)
- Split the **scheduler/execution out of the FastAPI web server** into a standalone daemon
  (systemd/process): FastAPI becomes a read-only view into Postgres for the dashboard; an
  **execution worker** + a **reconciliation worker** run independently. (The panel was unanimous:
  fine for ETF paper today, mandatory before live futures capital.)
- **Backtest/live parity (replay):** a mode that feeds historical data into the live execution
  engine and confirms it reproduces the research book on golden dates — catches research↔live drift.

## 5. Rollout
1. **Paper** (IBKR paper account): run the futures book weekly, accrue the real-fill OOS record;
   confirm fills/roll/margin behave + the replay parity holds + crash-survival of vix_vrp (the
   stress behavior the CPCV couldn't test).
2. **Tiny live** (1-2 contracts/market) once paper is clean for ≥1-2 months incl. ≥1 roll cycle.
3. **Scale** via the capital-allocation layer (P5) once trend + futures book run together.

## 6. Phased build checklist (turnkey on approval)
- **P2.1** — broker-agnostic core: contract-spec table + `target_lots` + `reconcile` + run-id/
  idempotency + snapshot schema. *(Mock-testable pre-approval if we want a head start.)*
- **P2.2** — `ib_insync` binding: connect/heartbeat, contract resolution + **verify-on-connect**,
  place/modify/cancel, positions/cash/margin pull, fills capture. *(Needs the live paper gateway.)*
  ✅ **READ side DONE 2026-06-22** (`ibkr_adapter.IBKRReadOnlyAdapter`): connect (`readonly=True`) +
  health + account + positions + **verify-on-connect** (`verify_contracts`), validated live vs TWS
  paper (DUQ869409) — ALL 16 futures verified after correcting ZC/ZS mult 50→5000 and the FX/VIX
  request symbols (6E→EUR/6J→JPY/VX→VIX). Order placement (place/modify/cancel) + fills capture remain
  R1 (behind the whole-book gate + the live-paper soak).
- **P2.3** — the execution + reconciliation workers + decouple from FastAPI + kill switch/dead-man.
- **P2.4** — replay/parity tests + paper go-live (weekly futures-book rebalance on IBKR paper).

## What I need from you on approval
IB Gateway (or TWS) running in **paper** mode + the **host:port** (e.g. `127.0.0.1:4002`) and
confirmation **Futures** permissions are active (CME/CBOT/NYMEX/COMEX/ICE/Eurex). Then P2.2→P2.4 go fast.
