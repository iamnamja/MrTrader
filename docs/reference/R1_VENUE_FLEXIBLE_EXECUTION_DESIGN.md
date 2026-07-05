# R1 — Venue-Flexible Live Execution Layer (Alpha-v10)

**Status:** DESIGN (planning only — no live change). Owner-gated; execution begins only after the
Phase-H enforce flips are done (the hard IBKR no-go gate).
**Supersedes the scope line in** [`P2_IBKR_EXECUTION_DESIGN.md`](P2_IBKR_EXECUTION_DESIGN.md): P2 framed IBKR
as *futures-only* with Alpaca permanently the equities venue. R1 keeps that as the **default** but makes
**venue a per-sleeve config choice**, so "consolidate everything onto IBKR" (single-venue, for real
money later) OR "switch venues / fail over" becomes a config flip, not a rewrite. P2's futures
execution mechanics (§0–§3) still hold and are reused verbatim; R1 is the layer that makes them
venue-neutral and turns on the live order path.

---

## 1. Why this shape (the user's ask, made precise)

Goal: make an eventual live-trading flip easy, ideally single-venue, or at least venue-flexible (switch
back/forth, or to a third venue). **Single-venue and venue-flexible are the same build** — both fall out
of finishing the broker **write** abstraction. Once order placement goes through a venue-selected
adapter, the destination (all-IBKR / Alpaca-ETF+IBKR-futures / failover) is a *setting*, not an
architecture. So we build the abstraction and defer the destination.

**Recommendation:** build the abstraction (flexibility); treat **all-IBKR as the operational default for
real money**; do **not** hard-code single-venue (the read side is already abstracted, so venue-neutrality
is cheap and buys failover + no broker lock-in).

---

## 2. Current state (what's built vs missing)

**Built (read side + shadow):**
- `BrokerAdapter` Protocol ([broker_adapter.py:61](../../app/live_trading/broker_adapter.py#L61)) —
  **read-only**: `health / get_account / get_positions / normalize_instrument`. `place_order` is
  *deliberately absent* (enforced by an **import-time `assert`** + a runtime read-only client proxy —
  Python has no true compile-time check).
- Two read adapters: `AlpacaReadOnlyAdapter`, `IBKRReadOnlyAdapter` (+ `verify_contracts`, validated live
  — all 16 futures matched, 0 critical).
- Futures order-*construction* shadow path (`futures_sizing.py`, `futures_sleeve.run_futures_rebalance`)
  — inert by 4 locks + no write API.
- **Venue-neutral** safety layers already `venue=`/`(venue, instrument_id)`-aware: reconciliation,
  whole-book gate, kill-switch SM. **Alpaca-specific** (NOT venue-aware — must be generalized in R1):
  H3 fat-finger caps + H6 idempotent-reuse both live *inside* `alpaca.place_market_order`.

**Missing (the R1 work):**
- **No write surface on the Protocol**, and the **live path is hardwired to Alpaca**: trend/cash call
  `alpaca.place_market_order` directly ([trend_sleeve.py:816](../../app/live_trading/trend_sleeve.py#L816),
  [cash_sleeve.py:334](../../app/live_trading/cash_sleeve.py#L334)).
- No IBKR order placement / fills capture / margin preview / roll emission.
- No live futures signal path (carry/xsmom → weights); `ibkr.futures_target_weights_json` is a stub.
- No per-sleeve venue routing / config.

---

## 3. Target architecture — one execution layer, many venues

```
 sleeve (trend/cash/futures)                     venue-neutral                         venue-specific
 ─ compute target exposure                          │                                        │
 ─ reconcile-before-trade  (venue, instr) ──────────┤  [ already venue-aware ]               │
 ─ whole-book risk gate    (venue=…)      ──────────┤  [ already venue-aware ]               │
 ─ kill-switch SM                          ─────────┤  [ already venue-aware ]               │
 ─ build OrderIntent[]  (canonical) ────────────────┤                                        │
 ─ pre-trade caps (H3, venue-neutral) ──────────────┤                                        │
        │                                            │                                        │
        ▼                                            ▼                                        ▼
   get_execution_adapter(venue)  ──►  ExecutionAdapter.place(intent)  ──►  broker-native order + fills
                                       (Protocol write surface)            (symbol/contract resolution,
                                                                            multiplier, margin, roll)
```

### 3.1 Canonical types (venue-neutral)
- `OrderIntent{ venue, instrument_id, sec_type(STK|ETF|FUT), side(BUY|SELL), quantity, order_type(MARKET
  first), tif, client_ref }` — `quantity` = shares (equities) or lots (futures); `client_ref` = the
  idempotency key (see §3.6 — the **sleeve** computes it, the adapter passes it through verbatim).
- **`place()` returns ACK only, never a fill** (CRITICAL — a market order's `filled_qty/avg_price` are
  unknowable at submit on *both* venues: Alpaca returns `status=accepted, filled_qty=0, price=None`;
  IBKR delivers fills asynchronously via `execDetails`, commissions later via `commissionReport`).
  `OrderResult{ broker_order_id, accepted_status, idempotent_reuse, raw }` — **no fill fields.**
- **Fills are a separate async capture** (`FillEvent{ client_ref, instrument_id, filled_qty, avg_price,
  commission, ts }`) flowing through an `on_fill`/reconcile path for both venues — NOT the `place()`
  return. The sync-vs-async shape of `place()` + the ib_insync thread/loop model (§3.3, §5 R1.1) are a
  **hard prerequisite for freezing this signature** — do not defer them to R1.4.
- `MarginImpact{ init_margin, maint_margin, buying_power_after, ok }` (from a preview).

### 3.2 Write surface added to `BrokerAdapter` (R1)
- `place(intent) -> OrderResult` (ACK only, per §3.1)
- `cancel(order_ref_or_id) -> OrderResult`
- `get_open_orders() -> list`
- `on_fill` / fill-capture hook (async; see §3.3) — fills do NOT come back through `place()`
- `preview(intent) -> MarginImpact` (IBKR `whatIfOrder`; Alpaca = buying-power/notional check)
- `flatten(scope) -> report` (reuse per-venue emergency flatten; already exists for Alpaca)

### 3.3 Adapters
- **`WritableAlpacaAdapter`** — wraps the **existing** `alpaca.place_market_order` so H3 (fat-finger) +
  H6 (dup-`client_order_id` idempotent reuse) are preserved. **"Byte-identical" must cover BOTH the
  Alpaca wire request AND the return-value contract the sleeves consume:** today the sleeves read a
  **dict** — `order.get("idempotent_reuse")` / `order.get("order_id")`
  ([trend_sleeve.py:821](../../app/live_trading/trend_sleeve.py#L821),
  [cash_sleeve.py:337](../../app/live_trading/cash_sleeve.py#L337)) — and the cash reuse branch also does
  a `alpaca.get_position(sym)` **read** ([cash_sleeve.py:354](../../app/live_trading/cash_sleeve.py#L354)),
  a read not on the write surface. So `OrderResult` must be dict-compatible (or both call-sites are
  updated in the SAME step), the H6 reuse branch tested explicitly, and the sleeve keeps a read adapter
  handle. `client_ref` → `client_order_id`.
- **`WritableIBKRAdapter`** — runs on a **single dedicated ib_insync thread owning the connection + event
  loop**, fed by a queue (the orchestrator dispatches sleeves via `run_in_executor` onto a multi-worker
  pool → `IB()`'s thread/loop affinity + one `clientId` would otherwise interleave/reject; single-flight
  + unique `clientId`). `place`/`cancel`, **fills via `execDetails` AND commissions via the separate
  async `commissionReport`**, plus a **disconnect-gap fill reconciliation** (`reqExecutions` is only
  current+prior day → a fill that lands while disconnected must be recovered from broker positions, not
  a naive replay). `whatIfOrder` margin preview; contract resolution + verify-on-connect (done);
  multiplier ONLY from `instrument_master`. `client_ref` → `orderRef`. Roll orders per P2 §2.

### 3.4 Where each safety layer lives
| Layer | Placement | Rationale |
|---|---|---|
| Reconciliation-before-trade | venue-neutral (sleeve) | already `(venue, instrument_id)`-keyed |
| Whole-book risk gate | venue-neutral (sleeve) | already `venue=`-aware |
| Kill-switch SM | venue-neutral (sleeve) | cross-venue by design |
| Pre-trade caps (H3) | **lift to venue-neutral** pre-trade + keep per-adapter as defense-in-depth | today it lives inside `alpaca.place_market_order`; futures need it too |
| Idempotency | venue-neutral concept, venue-specific field | `client_ref` → `client_order_id`/`orderRef` |
| Symbol/contract, multiplier, margin, fills, roll | venue-specific (adapter) | genuinely differ per broker |

### 3.5 Per-sleeve venue routing
- Config: `pm.trend_venue` / `pm.cash_venue` / `pm.futures_venue` (default `ALPACA/ALPACA/IBKR`).
- `get_execution_adapter(venue)` factory resolves the adapter at trade time (fail-closed if the venue is
  disabled / disconnected / stale).
- **Single-venue** = set all three to `IBKR`. **Failover / switch** = change one value. No code change.

### 3.6 Idempotency-key ownership (pin this down — it defeats H6 if wrong)
The **sleeve** computes `client_ref` and the adapter passes it through **verbatim** (the adapter must NOT
re-derive it). The derivation is sleeve-specific: **trend omits side** (`idempotency_key("trend", sym)`,
[trend_sleeve.py:818](../../app/live_trading/trend_sleeve.py#L818)) while **cash includes side**
(`idempotency_key("cash", sym, side=side)`, [cash_sleeve.py:335](../../app/live_trading/cash_sleeve.py#L335)).
If the adapter re-derived it without knowing that rule it would produce a *different* `client_order_id`
→ H6 dedup silently defeated → double-fill risk on retry.

---

## 4. The delicate part — refactoring the LIVE Alpaca write path

The trend/cash order path is the code that **actually trades**. It must be changed shadow-first and
**byte-identical-by-default**:

1. **R1.1** introduce `WritableAlpacaAdapter` wrapping the *exact* existing call — no caller uses it yet.
   The parity harness asserts identity on **both** (a) the Alpaca wire request AND (b) the return-value
   fields the callers consume (`idempotent_reuse`, `order_id`) — including the **H6 reuse branch** and
   the cash `get_position` read (§3.3). A request-only parity check is insufficient.
2. **R1.2** route trend/cash placement through `get_execution_adapter("ALPACA").place(intent)` behind
   `pm.execution_router_mode` (default **`direct`** = calls the same `alpaca.place_market_order` under
   the hood → byte-identical; **`adapter`** = through the Protocol). Run `adapter` in the parity
   harness, Opus deep-dive, THEN flip the default. Instant rollback = set back to `direct`.
3. Only after the Alpaca path is proven through the adapter do we add IBKR as a second implementation —
   so the abstraction is validated on the working venue before the new one.

> **Note on the current pattern:** the existing Alpaca live path **books the trade pre-fill** —
> `_sync_trend_trade(...)` records `target_shares` right after submit with no fill poll
> ([trend_sleeve.py:839](../../app/live_trading/trend_sleeve.py#L839)). Tolerable for RTH equity market
> orders, but the canonical layer must acknowledge this so §6's "never infer fill from send" isn't
> quietly violated when generalized to IBKR (where pre-fill booking is unsafe — use the async fill path).

---

## 5. Phased checklist (each: shadow-first · tests · independent Opus deep-dive · full suite green)

- **Prereq (no-go gate):** Phase-H enforce flips done — `reconciliation_mode` + `whole_book_gate_mode` +
  `kill_switch_sm_mode` = `enforce`, clean. *No R1 live step before this.*
- **R1.1** — canonical types (`OrderIntent`, **ack-only** `OrderResult`, `FillEvent`, `MarginImpact`) +
  write surface on `BrokerAdapter` + `WritableAlpacaAdapter` (wraps existing path; no caller; parity
  tests per §4). **Prerequisite decisions frozen HERE, not R1.4:** (i) `place()` sync/async shape +
  the async fill-capture path; (ii) the single dedicated ib_insync thread/loop + `clientId`
  single-flight model. The write-surface signature cannot be frozen before these. *No live change.*
- **R1.2** — route trend/cash through the adapter behind `pm.execution_router_mode` (default `direct`);
  parity harness; flip default after a clean shadow window. *Touches the live path — highest scrutiny.*
- **R1.3** — lift H3 pre-trade caps to a venue-neutral guard (keep the Alpaca in-adapter guard as
  defense-in-depth); add futures notional/lot caps to the same guard.
- **R1.4** — `WritableIBKRAdapter`: place/cancel + `execDetails` fills + **`commissionReport` capture** +
  **disconnect-gap fill reconciliation** (broker positions, not naive replay) + `whatIfOrder` margin
  preview + roll emission, on the dedicated ib thread from R1.1. Built against the **live paper
  gateway**, **Read-Only API OFF**, owner-present.
- **R1.5** — live futures signal path (carry/xsmom weights → `futures_target_weights`) + scheduler
  wiring + immutable per-run snapshot table (P2 §3).
- **R1.6** — replay/parity tests (research book reproduced on golden dates) → **IBKR paper go-live**
  (futures book weekly), accrue real-fill OOS record incl. ≥1 roll cycle.
- **R1.7 (optional, owner choice)** — enable `*_venue=IBKR` for ETFs to validate single-venue; run as a
  **shadow parity** vs the live Alpaca book first; promote only if parity holds.
- **Then:** tiny-live futures → scale via R2 ERC sizing (data-gated: ≥2 strategies × ≥6 mo joint live).

---

## 6. Risks / loss vectors → mitigation

- **Double-send / missed order from the refactor** → adapter wraps the *exact* idempotent call;
  `direct` default + parity tests (request AND return contract) + deep-dive; instant flag rollback.
- **Single-shot `place()` that times out but actually RESTED** → idempotency only helps on a *retry*;
  a first submit that raises after the order rested currently books no Trade row (the sleeve's
  `except` logs `order_error`), leaving an **orphan live position until the next reconcile-before-trade
  — up to ~1 week for trend, ~1 day for cash**. Mitigate: `place()` does a **place-then-verify**
  (query by `client_ref` on timeout) before giving up; the residual orphan window is an explicit,
  bounded accepted risk.
- **IBKR order semantics differ** (partial fills; `orderRef` isn't uniqueness-enforced like Alpaca's
  `client_order_id`) → `orderRef` + **reconcile-before-trade + broker-as-truth** as the real dedup;
  **async fills (`execDetails` + `commissionReport`)**; on IBKR never infer fill from send (unlike the
  current Alpaca pre-fill booking, §4).
- **IBKR ≠ Alpaca order primitives for R1.7** (IBKR has **no fractional shares** and no notional-dollar
  orders; Alpaca positions may be fractional) → an ETF shadow-parity will show quantity divergence at
  the integer-rounding boundary that is **not a bug**; account for it before declaring parity failed.
- **Futures multiplier error (the #1 killer)** → multiplier ONLY from `instrument_master` +
  verify-on-connect (done, 0 critical).
- **Margin reject / auto-liquidation** → `whatIfOrder` preview before send + per-venue margin reserve
  (Alpaca cash ≠ IBKR buying power) + no auto-liquidate (kill-switch capped at CANCEL_ONLY).
- **Single-venue concentration** (all-IBKR outage) → keep the abstraction; don't hard-commit; Alpaca
  stays a configured failover.
- **Connection/clock/data staleness** → fail-closed at the adapter (mirror the sleeves' existing gates).
- **Cross-venue book confusion** → R0.4 consolidated `(venue, instrument_id)` book-state (done).

---

## 7. Open decisions for the owner

1. **End-state:** build abstraction + default all-IBKR-for-real-money, keep flexible (recommended) — vs
   commit single-venue now?
2. **ETF migration timing:** keep ETFs on Alpaca live through R1 and only *shadow-parity* IBKR-ETF (rec),
   vs migrate ETF paper to IBKR during R1?
3. **Daemon decouple (R0.2 P4 subprocess flip):** do it before or alongside R1 live futures? (P2 §4 says
   mandatory before live futures *capital*; paper can precede.)
4. **ib_insync event-loop/threading** — **resolved to R1.1** (single dedicated ib thread/loop +
   `clientId` single-flight) because the write-surface signature depends on it; no longer an R1.4
   follow-up.

---

## 8. What stays unchanged until the owner says go
Live book = Alpaca trend + cash (paper). Every R1 step is shadow-first and default-off; the first change
that touches the live order path is R1.2 behind a `direct`-default flag. Nothing here trades futures or
moves a venue without an explicit, owner-present flip **after** the enforce flips land.
