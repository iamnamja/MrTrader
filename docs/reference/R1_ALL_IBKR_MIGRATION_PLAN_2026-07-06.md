# R1 — All-IBKR Single-Venue Migration Plan (Alpha-v10)

**Decision (2026-07-06, owner):** consolidate the ENTIRE live book — ETFs + cash + futures — onto **IBKR as
the single venue.** Rationale: the real-money end-state is one broker, one book, one reconciliation, one
margin pool (portfolio margin across equities + futures) — so do the migration once, in paper, rather
than run two venues and migrate later.

**Relationship to prior docs:** supersedes the *cross-venue* framing (Alpaca-ETFs + IBKR-futures) in
[`R1_VENUE_FLEXIBLE_EXECUTION_DESIGN.md`](R1_VENUE_FLEXIBLE_EXECUTION_DESIGN.md) and
[`P2_IBKR_EXECUTION_DESIGN.md`](P2_IBKR_EXECUTION_DESIGN.md) §"Scope". The *mechanics* in those docs
(broker-write abstraction, verify-on-connect, idempotency, snapshots, roll, whatIfOrder margin, the
Opus-reviewed R1.1 hardening list) still apply verbatim — the only change is the **target: all sleeves
route to IBKR**, and Alpaca is retained only as a **shadow-comparison + instant rollback** during
migration, not a second operational venue.

---

## 1. What this actually means (eyes open)

It is **more than "add IBKR for futures" — it migrates the currently-LIVE, validated Alpaca ETF + cash
paper book onto IBKR.** So R1 must build **equity/ETF execution on IBKR**, not just futures.

**IBKR-equity specifics that differ from Alpaca** (small but real — handle explicitly):
- **No fractional shares, no notional-dollar orders** → orders round to whole shares (Alpaca had both).
- **Small commissions** on IBKR equities (Alpaca was free) — negligible at this turnover, but real.
- All current ETFs incl. SGOV trade on IBKR → no coverage gap.

## 2. Guardrails (non-negotiable — the whole session's discipline)

- **Shadow-first, always.** Every new order path exists + logs what it *would* do, places nothing, until
  proven — then flips behind a config flag, instantly reversible.
- **Alpaca stays as a rollback belt** through the migration: run IBKR in shadow *against* the live Alpaca
  book, compare, cut over, and keep the Alpaca adapter one config-flip away for the soak. This is NOT
  "operating two venues" — it's a safe reversible migration. Retire/remove Alpaca only after IBKR is proven.
- **No IBKR capital** until: the enforce-flip soak is clean (a few live rebalances), `kill_switch_sm` is
  in enforce, and the daemon subprocess decouple (R0.2 P4) is done (mandatory before live *capital* per
  P2 §4; paper can precede).
- **"Fix things as they happen"** (owner directive): iterate — ship the safe slice, observe, fix the next
  surprise. Do NOT try to pre-solve every IBKR quirk; the shadow phases exist to surface them cheaply.

## 3. Where we start FROM (already built)

- IBKR Gateway is **up + logged in**; `IBKRReadOnlyAdapter` connects read-only, and **verify-on-connect
  validated all 16 futures contracts (0 critical)**. The READ side + contract master are done.
- The safety layer is **load-bearing**: reconciliation + whole-book gate are LIVE in **enforce** (flipped
  2026-07-06), and both are already venue-aware (`(venue, instrument_id)` / `venue=`).
- The consolidated cross-venue `BookState` + reconciliation + kill-switch state machine exist (R0.4).

---

## 4. The sequence — **START: get IBKR up and running**

### R1.0 — Get IBKR up and running (the foundation) ← START HERE
The execution seam + a real IBKR write connection. All inert / shadow (places nothing).
- **R1.0a — write-surface foundation** (Alpaca-validated first, lowest risk): canonical `OrderIntent`
  (venue, instrument_id, sec_type, side, qty, order_type=MARKET, tif, client_ref), **ack-only**
  `OrderResult` (broker_order_id, accepted_status, idempotent_reuse), and `FillEvent` (fills are a
  SEPARATE async capture, never the `place()` return — per the Opus R1 review). Add a
  `WritableBrokerAdapter(BrokerAdapter)` Protocol (`place`/`cancel`/`get_open_orders`/`preview`) and a
  **`WritableAlpacaAdapter`** that wraps the EXISTING `alpaca.place_market_order` byte-identically (H3
  fat-finger + H6 idempotent-reuse preserved). **No caller — inert.** Validates the abstraction on the
  working venue before IBKR.
- **R1.0b — robust IBKR connection**: ✅ **DONE** — `app/live_trading/ibkr_connection.py`
  `IBKRConnectionManager`: single dedicated `ib_insync` thread owning its own loop + `IB()`; all ops
  dispatched onto it via `run_coroutine_threadsafe`; `clientId` single-flight, idempotent connect,
  rate-limited `ensure_connected`, fail-closed, clean `stop()`. Validated LIVE (paper DUQ869409:
  connect / managedAccounts / reqCurrentTimeAsync / cross-thread dispatch / teardown). Inert (no caller,
  no auto-connect, **no order method**; read-capable only, `readonly=True`). Opus deep-dive folded in:
  timed-out coroutines are cancelled; a re-entrant call from the loop thread fails loud (never
  deadlocks); the data `call()` dispatch is NOT under the manager lock (so a hung op can't stall
  `stop()`/`is_connected()`); `stop()` is terminal. Read-Only API stays ON here; **flipping it OFF is
  the owner step at R1.0c.**
  **`call()` contract (carried into R1.0c):** the thunk MUST be non-blocking and MUST NOT re-enter the
  manager; every dispatch carries a bounded timeout (cancel-on-timeout is built in).
- **R1.0c-1 — `WritableIBKRAdapter` (write surface, SHADOW)**: ✅ **DONE** —
  `app/live_trading/writable_ibkr_adapter.py`. Implements the write Protocol on the R1.0b manager:
  OrderIntent → IBKR `Stock`/`Future` contract + `MarketOrder`, **multiplier ONLY from
  `instrument_master`** (the #1-killer guard; unmapped → fail-closed), `client_ref` → `orderRef`
  verbatim. `place()`/`cancel()` are SHADOW by default (`mode="shadow"`): build + validate + LOG the
  would-be order, **place NOTHING** (`mode="live"` is R1.1+, and Read-Only rejects it anyway). Reads
  delegate to `IBKRReadOnlyAdapter` bound to the manager's `ib`. Validated LIVE (SPY STK mult 1.0;
  FUT.ES → symbol ES / exch CME / **mult 50** from the spec; unmapped fail-closed; shadow placed
  nothing). **Finding: `whatIfOrder` margin preview is BLOCKED by the Read-Only API** (Error 321 →
  empty OrderState → `preview.ok=False`), so real margin preview needs Read-Only-OFF. (This is also
  what pops the gateway's "an API client is attempting to place an order" dialog — a what-if counts
  as an order attempt; **Read-Only correctly blocks it, nothing is placed**; keep Read-Only ON.)
  Independent Opus deep-dive: **SAFE-TO-MERGE** (no reachable live-order path). Two findings folded in:
  (1) the FUT multiplier now **fails CLOSED** on a 0/None/fractional multiplier — raises instead of
  emitting an empty string that would let IBKR substitute a default (was the one fail-*open* hole in the
  #1-killer guard); (2) the shadow snapshot now records `contract_multiplier` (what's actually bound to
  the IBKR `Contract`), and the multiplier test asserts on the built contract, not just the master echo.
  11 tests.
- **R1.0c-2 — live path (owner-gated, Read-Only-OFF)**: async fill capture (`execDetails` +
  `commissionReport` → `FillEvent`), disconnect-gap fill recovery, futures front-month qualification
  (`reqContractDetails`), working `whatIfOrder` margin, and the actual `mode="live"` place smoke-test.
  **Owner step: turn the Gateway Read-Only API OFF** (the one hands-on gate before any IBKR order).

### R1.1 — ETF/cash on IBKR (SHADOW)
Route the trend + cash order CONSTRUCTION through `WritableIBKRAdapter` in **shadow**, alongside the
live Alpaca path: compute the IBKR orders it *would* place, log + snapshot, **place nothing on IBKR**.
Compare vs the actual Alpaca fills/positions — this is where whole-share rounding + any execution
surprise surface, before a single real IBKR order. (The enforce gates already evaluate `venue=IBKR`.)

### R1.2 — Cut over ETF/cash to IBKR (tiny-live)
Flip `trend`/`cash` venue → IBKR (whole book-state now reads IBKR positions). Alpaca adapter stays as
the **instant rollback**. Tiny sizes first; soak; reconcile IBKR ↔ DB clean under enforce.

### R1.3 — Futures on IBKR (SHADOW → tiny-live)
Wire the LIVE carry/xsmom signal → `futures_target_weights` (replaces the stub). Futures rebalance
shadow → tiny-live (1–2 lots/market) behind the enforce whole-book gate; require **≥1 clean roll cycle**
(the crash-governor / roll mechanics P2 couldn't CPCV-test).

### R1.4 — Consolidate + scale (single venue)
Collapse to single-venue book-state / reconciliation / kill-switch / margin reserve (drop the cross-venue
plumbing that's now moot). Retire the Alpaca operational path (keep the adapter as a dormant fallback OR
delete). Scale via the R2 ERC/covariance sizing stack (data-gated on ≥2 strategies × ≥6mo joint live).

### Parallel / prerequisite hardening (gates R1 *capital*, not the build)
- Finish **`kill_switch_sm` reduce-only refinement + enforce flip** (last shadow-hardening item).
- **Enforce-flip soak** — a few clean live rebalances (first test 2026-07-07).
- **R0.2 P4 daemon subprocess decouple** — mandatory before live futures *capital* (paper can precede).

---

## 5. What stays unchanged until each owner-present flip
The live book = Alpaca trend + cash (paper) until R1.2 cutover. Every R1.0/R1.1 step is inert/shadow —
no live order changes venue or hits IBKR until an explicit, owner-present config flip, each instantly
reversible. Each slice ships with tests + an independent Opus deep-dive, per this session's discipline.
