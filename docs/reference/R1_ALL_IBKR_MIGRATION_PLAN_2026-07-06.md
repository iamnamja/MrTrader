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
  **Cutover-readiness hardening (2026-07-06, 2nd Opus deep-dive):** `_call_on_loop` now snapshots the
  loop and **fails closed with `ConnectionError`** on the `stop()`/dead-loop race (no more raw
  `AttributeError`/`RuntimeError` or ~30s stall) and translates op timeouts to `ConnectionError` too
  (uniform fail-closed); a failed/timed-out `connect` **hard-closes the half-open socket** so a claimed
  `clientId` can't wedge the next reconnect; `stop()` warns on a leaked loop thread; the `...Async`-thunk
  requirement is documented on `call()`.
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
- **R1.0c-2a — inert fill + qualification surface (READ-only-safe)**: ✅ **DONE** — no Read-Only flip
  needed. `qualify_future(instrument_id)` resolves a FUT to its **front-month** via `reqContractDetails`
  (filters by `tradingClass==root`, nearest non-expired expiry) and **RE-VERIFIES the multiplier on the
  RESOLVED contract** against the master (the #1-killer guard on the real contract, fail-closed) — wired
  into the live `place()` FUT path so a live futures order can never go out as a bare generic contract.
  `get_fills(client_ref=None)` is the **SEPARATE async fill channel** via `reqExecutions` → `FillEvent`
  (BOT/SLD→BUY/SELL, `orderRef`→`client_ref`, symbol→canonical `instrument_id`, `execId`→`exec_id`,
  `commissionReport` joined); because `reqExecutions` replays the whole session, it **doubles as
  disconnect-gap recovery** (consumer dedupes on `exec_id`). `FillEvent` gained `exec_id`/`side`/`venue`.
  Both reads **degrade** (return `[]`/raise fail-closed) on error, never crash the caller. Validated LIVE
  read-only (paper): FUT.ES → **ESU6, expiry 20260918, tradingClass ES, mult 50, exch CME**; `get_fills`
  ran clean (0 fills). 20 tests; independent Opus deep-dive. **Places nothing** (still shadow).
- **Cutover-readiness hardening (2026-07-06, 2nd Opus deep-dive of the write surface):** (1) **M1** —
  the ETF universe now maps to the IBKR venue (`_etf` registers `{ALPACA, IBKR}`); without it every
  IBKR-held ETF read `mapped=False` → a whole-book-gate breach that would HOLD the trend rebalance the
  moment equities move to IBKR. (2) **M2** — `preview()` now qualifies a FUT to its front-month before
  `whatIfOrder` (else futures margin was always empty/ok=False — the margin gate blind on the one asset
  class that needs it). (3) fail-closed order build: `_build_order` rejects a non-BUY/SELL side and a
  0/negative quantity (mirrored in `WritableAlpacaAdapter`); `qualify_future` fails closed when only
  EXPIRED contracts resolve (never trade a dead contract); `preview` nulls the IBKR UNSET_DOUBLE margin
  sentinel (no giant-garbage `ok=True`). +6 tests; independent Opus verify. Still inert.
  *(Deferred: gating unmapped **equity** fills to the known universe — needs a `FillEvent.mapped` flag +
  consumer policy; after M1 the whole R1 ETF universe maps, so no phantom for the actual migration.)*
- **R1.0c-2b — live path (owner-gated, Read-Only-OFF)**: the actual `mode="live"` place smoke-test and a
  real (non-empty) `whatIfOrder` margin — both need the gateway's Read-Only API OFF.
  **Owner step: turn the Gateway Read-Only API OFF** (the one hands-on gate before any IBKR order).

### R1.1 — ETF/cash on IBKR (SHADOW) ✅ **built (gated OFF)**
`app/live_trading/ibkr_shadow_router.py` — the trend + cash sleeves now reconstruct each Alpaca ETF/cash
order as the IBKR order it *would* build (`OrderIntent` → shadow `WritableIBKRAdapter.place()`), log +
compare (`ibkr_qty` vs `alpaca_qty`, into the rebalance `summary["ibkr_shadow_route"]`), and **place
nothing on IBKR**. The live value is a weekly **reconstruction / instrument-mapping check** on real
rebalance data — an unmapped symbol fails closed and is recorded — before a single real IBKR order.
(Whole-share rounding is identical on both sides for whole-share ETFs, so that only really bites for
futures lots at R1.3; here it's the mapping-gap + reconstruction parity that matters.)
- **Gated OFF** by the `ibkr.shadow_routing` agent-config flag (default off; instantly reversible).
- **Structurally cannot reach the gateway**: a shadow place returns before any dispatch, so the router
  needs no IBKR connection — and its connection stub RAISES if anyone ever tries to dispatch.
- **Never raises into the live loop**: per-order fail-closed; the Alpaca placement is untouched.
- Runs in BOTH shadow and live rebalance modes (so the comparison accrues during the Alpaca-shadow soak).
- Owner step to activate: set `ibkr.shadow_routing=on`. (The enforce gates already evaluate `venue=IBKR`.)

### R1.2 — Cut over ETF/cash to IBKR (tiny-live)
Flip `trend`/`cash` venue → IBKR (whole book-state now reads IBKR positions). Alpaca adapter stays as
the **instant rollback**. Tiny sizes first; soak; reconcile IBKR ↔ DB clean under enforce.

### R1.3 — Futures on IBKR (SHADOW → tiny-live)
Wire the LIVE carry/xsmom signal → `futures_target_weights` (replaces the stub). Futures rebalance
shadow → tiny-live (1–2 lots/market) behind the enforce whole-book gate; require **≥1 clean roll cycle**
(the crash-governor / roll mechanics P2 couldn't CPCV-test).

**Roll policy — recon + design (2026-07-07):** `app/live_trading/futures_roll_policy.py` (SHADOW
instrumentation). Recon against the live paper gateway established:
- **IBKR `reqContractDetails` exposes NO First Notice Day** — only `contractMonth` (delivery month),
  `lastTradeDateOrContractMonth`, `realExpirationDate`, `timeZoneId`. So an FND floor must be **derived**
  per-market, not read from the broker. Norgate DOES provide per-contract `volume`/`open_interest`
  (feasible vol/OI-crossover roll).
- **A fixed "N days before last-trade" roll is unsafe for physically-delivered contracts.** Live proof:
  the qualifier's nearest-expiry "front" for **ZS (soybeans)** today is the **July** contract, whose
  **First Notice Day (~Jun 30) is ~2 weeks before last-trade and already PAST** — i.e. a naive roll
  would sit in the delivery window, in an illiquid rolled-away contract.
- **Design (per-market, earliest-binding):** `roll = min( fixed 5d-before-scheduled-expiry [backtest
  convention, all markets], FND floor [physical grain/metal/rate, derived from the delivery month],
  last_trade cap [ALWAYS: 3 trading days before the IBKR last-trade date], liquidity vol/OI crossover
  [INSTRUMENTED, not yet binding] )`. Cash-settled (ES/NQ/RTY/VX) → calendar only; FX (6E/6J) →
  settle-on-expiry. **Energy (CL/NG) is protected by the last_trade cap, NOT the FND estimate** — their
  true expiry falls BEFORE the prior-month-end estimate (Opus review caught this: without the cap the
  policy would roll crude ~10 days after it stopped trading). Independent Opus review verified the
  settlement taxonomy (all 16 correct) + grain/metal/Treasury FND; energy fix folded in.
- **Empirical answer (2026-07-07, `app/research/futures_roll_analysis.py`):** measured `fixed_roll −
  liquidity_roll` (OI-crossover) over the last 16 rolls/market. **Cash (ES/NQ/RTY −5d), FX (6E/6J −2d),
  VX (+5d) → fixed ≈ liquidity (calendar is fine).** **Physical commodities → fixed rolls 2–5 WEEKS LATE**
  (CL +34, NG +35, ZC +30, ZS +30, GC +22, SI +23, HG +26, ZN/ZB/ZF +16; even FND is +11 to +26d late).
  **DECISION: HYBRID roll** — calendar for cash+FX, **dynamic OI-crossover for physical markets**, bounded
  by the FND/last-trade safety floor. (See `DECISIONS.md` 2026-07-07.)
- **Delivery-safety monitor (2026-07-07, `app/live_trading/futures_roll_monitor.py`):** hardens the
  "bot down 1–2 days near roll time" case. (1) `catch_up_on_restart(held, today)` — on daemon startup,
  roll any future whose roll is DUE/overdue IMMEDIATELY (not at the next weekly rebalance), so an outage
  over the roll date can't leave a position in an expiring contract till Monday. (2) Downtime-buffered
  urgency tiers — `APPROACHING` pre-alerts `DOWNTIME_ALERT_BDAYS` before the roll; `CRITICAL` escalates
  `CRITICAL_MARGIN_BDAYS` BEFORE the hard FND/last-trade floor (so a multi-day outage can't silently
  cross it). (3) `notify_delivery_risk` → CATASTROPHIC `futures_delivery_risk` (off-box snitch webhook).
  Inert until the futures executor wires it in (no held-futures source yet). Answer to "would we take
  delivery from a short outage?": **no** — the hybrid roll fires weeks before FND (~22–26d buffer, from
  the analysis), catch-up re-rolls on restart, and IBKR auto-liquidation is the last resort. Independent
  Opus review (SAFE) folded in: a PHYSICAL position with no computable floor now **fails safe to CRITICAL**
  (never silent).
- **Exchange holiday calendar ✅ DONE (2026-07-07, `app/live_trading/exchange_calendar.py`):** the FND /
  last-trade / CRITICAL-margin business-day math is now **holiday-aware** (was weekday-only — the Opus
  MINOR). Computes the 10 US futures-exchange full-close holidays algorithmically (any year, with the
  weekend-observance rule + Good Friday via computus); `futures_roll_policy` + `futures_roll_monitor`
  delegate their trading-day math to it, so a month-end holiday pulls the FND estimate ~1 day earlier
  (the SAFE direction) instead of compressing the margin.
- **R1.3 is measure-first:** `roll_report()` logs all candidate dates so we quantify the fixed-vs-FND-vs
  -liquidity gaps per product on real data BEFORE choosing fixed-vs-dynamic. **Any live roll that diverges
  from the backtested `fixed` rule must be re-validated against the signal** (a roll change silently
  alters returns). **Before live futures:** verify the per-market FND rules against exchange calendars
  (esp. rates/energy — the estimate is "last business day of the month before delivery," standard for
  grains/metals) and wire the vol/OI crossover. Independent Opus review of the settlement/FND taxonomy.

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
