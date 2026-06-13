"""
Research registry — the pre-registration ledger for the Alpha-v6 research program.

WHY THIS EXISTS (blueprint Phase 0 / section 6, consensus item C9):
CPCV protects a single run from within-run overfitting; it does NOT protect the
research PROGRAM from repeated human/LLM iteration. The program's true trial
count is far above any DSR constant (DSR is report-only). The defenses are:
  1. Pre-registration: a confirmatory run's acceptance criteria and parameters
     are fixed BEFORE the run (preregistered_at < run_at, enforced here).
  2. This registry: every hypothesis is a row — the program's true N_TRIALS.
  3. The forward sacred holdout (exists elsewhere, 2026-11-09).

INTEGRITY RULES (enforced in code; each has a dedicated test):
  R1  Duplicate hypothesis_id on register() -> RegistryIntegrityError.
  R2  Confirmatory ordering: record_result() on a 'confirmatory' or
      'live_confirm' row raises unless preregistered_at is set AND strictly
      earlier than run_at AND non-empty acceptance_criteria were fixed at
      pre-registration time. Criteria/params fixed before the run, provably.
      preregister() refuses empty/absent criteria; register() refuses a
      confirmatory preregistered_at without non-empty criteria.
  R3  Exploratory cannot promote: record_result() on an 'exploratory' row
      raises if decision is 'promote_paper' or 'live'. Exploratory runs are
      unlimited and can inspire, but only {kill, park, exploratory_only}.
  R4  One confirmatory shot: a second record_result() on a row that already
      has a result raises. The sanctioned path to run again is a RE-TEST: a
      NEW hypothesis_id with parent_id set to the original and a
      cooling_off_until that is strictly earlier than the re-test's run_at —
      the time the run EXECUTED, not when the result is recorded.
  R5  preregister() raises if the row already has a recorded result (no
      post-hoc "pre"-registration) or is already preregistered (no moving
      the goalposts — register a re-test instead).
  Unknown labels fail CLOSED: record_result() on a row whose label is not in
  LABELS (schema drift / manual edit) raises instead of skipping R2/R3.
  parent_id, when given to register(), must reference an existing row.

CONCURRENCY: R4/R5 are also enforced at the storage layer — the result/
pre-registration UPDATEs are conditional ("... AND run_at IS NULL ...") and
a rowcount != 1 raises, so two concurrent writers cannot both commit even
though the friendly pre-checks read first (no check-then-write TOCTOU).

TIMESTAMPS: all stored as ISO-8601 TEXT. Integrity comparisons parse with
datetime.fromisoformat and treat tz-naive values as UTC, so naive and aware
strings compare correctly. No wall-clock is read inside any integrity check —
callers pass run_at / preregistered_at explicitly (testable, replayable).

STORE: standalone sqlite, mirrors app/live_trading/pead_tracker.py — schema
auto-created on connect, WAL, env-overridable path so pytest-xdist workers are
isolated (MRTRADER_RESEARCH_REGISTRY_DB; see tests/conftest.py pattern).
JSON columns (features/params/acceptance_criteria/result_json) are always
json.dumps'd on write and json.loads'd on read, so values round-trip with
their type (a string stays a string); None is stored as SQL NULL.
Unlike the live trackers this module RAISES on integrity violations — that is
its entire point — but is never imported by the live agent loop.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_ENV_VAR = "MRTRADER_RESEARCH_REGISTRY_DB"
_DEFAULT_DB = _ROOT / "data" / "research_registry.db"

LABELS = ("exploratory", "confirmatory", "live_confirm")
DECISIONS = ("kill", "park", "promote_paper", "live", "exploratory_only")
EXPLORATORY_ALLOWED_DECISIONS = ("kill", "park", "exploratory_only")
_PREREG_REQUIRED_LABELS = ("confirmatory", "live_confirm")

# Columns whose values are JSON-encoded in storage (always dumps/loads — F7).
_JSON_COLUMNS = ("features", "params", "acceptance_criteria", "result_json")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    hypothesis_id        TEXT PRIMARY KEY,
    parent_id            TEXT,
    family               TEXT,
    label                TEXT NOT NULL,
    features             TEXT,
    params               TEXT,
    universe             TEXT,
    window               TEXT,
    folds                TEXT,
    cost_model           TEXT,
    code_commit          TEXT,
    data_hash            TEXT,
    mechanism            TEXT,
    acceptance_criteria  TEXT,
    preregistered_at     TEXT,
    run_at               TEXT,
    result_json          TEXT,
    decision             TEXT,
    cooling_off_until    TEXT,
    created_at           TEXT
);
"""


class RegistryIntegrityError(ValueError):
    """An integrity rule of the research registry was violated."""


def _parse_ts(value: str, field: str) -> datetime:
    """Parse an ISO-8601 string; treat tz-naive as UTC so comparisons never mix."""
    try:
        dt = datetime.fromisoformat(str(value))
    except (ValueError, TypeError) as exc:
        raise RegistryIntegrityError(
            f"{field}={value!r} is not a valid ISO-8601 timestamp: {exc}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _encode(col: str, value: Any) -> Any:
    """JSON-encode JSON-column values; None stays SQL NULL (never the string 'null')."""
    if value is None:
        return None
    if col in _JSON_COLUMNS:
        return json.dumps(value, default=str, sort_keys=True)
    return value


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for col in _JSON_COLUMNS:
        v = d.get(col)
        if isinstance(v, str):
            try:
                d[col] = json.loads(v)
            except (ValueError, TypeError):
                pass  # legacy raw-string value (pre-F7 rows) — keep as-is
    return d


def _require_nonempty_criteria(acceptance_criteria: Any, context: str) -> None:
    """R2: pre-registration is meaningless without concrete acceptance criteria."""
    if not acceptance_criteria:
        raise RegistryIntegrityError(
            f"[R2 pre-registration requires acceptance criteria] {context}: "
            f"acceptance_criteria={acceptance_criteria!r} is empty/absent. "
            f"Pre-registration fixes the pass/fail bar BEFORE the run; an empty "
            f"criteria set proves nothing and cannot gate a confirmatory result."
        )


class ResearchRegistry:
    """Sqlite-backed pre-registration ledger. See module docstring for the rules.

    db_path resolution: explicit arg > MRTRADER_RESEARCH_REGISTRY_DB env var
    > data/research_registry.db. Resolved at construction (not import) so test
    monkeypatching of the env var takes effect per-instance.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = Path(db_path or os.environ.get(_ENV_VAR, str(_DEFAULT_DB)))

    def _conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(self.db_path), timeout=10)
        c.execute("PRAGMA journal_mode=WAL;")
        c.executescript(_SCHEMA)
        c.row_factory = sqlite3.Row
        return c

    @staticmethod
    def _fetch(c: sqlite3.Connection, hypothesis_id: str) -> dict[str, Any] | None:
        r = c.execute(
            "SELECT * FROM experiments WHERE hypothesis_id=?", (hypothesis_id,)
        ).fetchone()
        return _decode_row(r) if r is not None else None

    def _fetch_or_raise(self, c: sqlite3.Connection, hypothesis_id: str) -> dict[str, Any]:
        row = self._fetch(c, hypothesis_id)
        if row is None:
            raise RegistryIntegrityError(
                f"hypothesis_id={hypothesis_id!r} is not registered. Call register() "
                f"first — unregistered runs do not exist for this program."
            )
        return row

    # ------------------------------------------------------------------ write
    def register(
        self,
        hypothesis_id: str,
        *,
        label: str,
        family: str | None = None,
        parent_id: str | None = None,
        features: Any = None,
        params: dict[str, Any] | None = None,
        universe: str | None = None,
        window: str | None = None,
        folds: str | int | None = None,
        cost_model: str | None = None,
        code_commit: str | None = None,
        data_hash: str | None = None,
        mechanism: str | None = None,
        acceptance_criteria: dict[str, Any] | None = None,
        preregistered_at: str | None = None,
        cooling_off_until: str | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Insert a new hypothesis row. Raises on duplicate hypothesis_id (R1),
        on a confirmatory preregistered_at without non-empty criteria (R2), and
        on a parent_id that does not reference an existing row."""
        if label not in LABELS:
            raise RegistryIntegrityError(
                f"label={label!r} invalid; must be one of {LABELS}"
            )
        if preregistered_at is not None:
            _parse_ts(preregistered_at, "preregistered_at")
            if label in _PREREG_REQUIRED_LABELS:
                _require_nonempty_criteria(
                    acceptance_criteria,
                    f"register(hypothesis_id={hypothesis_id!r}, label={label!r}) "
                    f"with preregistered_at={preregistered_at!r}",
                )
        if cooling_off_until is not None:
            _parse_ts(cooling_off_until, "cooling_off_until")
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()
        row = {
            "hypothesis_id": hypothesis_id,
            "parent_id": parent_id,
            "family": family,
            "label": label,
            "features": _encode("features", features),
            "params": _encode("params", params),
            "universe": universe,
            "window": window,
            "folds": (str(folds) if folds is not None else None),
            "cost_model": cost_model,
            "code_commit": code_commit,
            "data_hash": data_hash,
            "mechanism": mechanism,
            "acceptance_criteria": _encode("acceptance_criteria", acceptance_criteria),
            "preregistered_at": preregistered_at,
            "run_at": None,
            "result_json": None,
            "decision": None,
            "cooling_off_until": cooling_off_until,
            "created_at": created_at,
        }
        cols = ", ".join(row)
        ph = ", ".join("?" for _ in row)
        with closing(self._conn()) as c, c:
            if parent_id is not None:
                parent = c.execute(
                    "SELECT 1 FROM experiments WHERE hypothesis_id=?", (parent_id,)
                ).fetchone()
                if parent is None:
                    raise RegistryIntegrityError(
                        f"[parent] parent_id={parent_id!r} is not a registered "
                        f"hypothesis. A re-test must reference the actual original "
                        f"row — register the parent first or fix the id."
                    )
            try:
                c.execute(
                    f"INSERT INTO experiments ({cols}) VALUES ({ph})",
                    tuple(row.values()),
                )
            except sqlite3.IntegrityError as exc:
                raise RegistryIntegrityError(
                    f"[R1 duplicate hypothesis] hypothesis_id={hypothesis_id!r} is already "
                    f"registered. Every run is a new trial: register a NEW hypothesis_id "
                    f"(set parent_id={hypothesis_id!r} if this is a re-test)."
                ) from exc
        return self.get(hypothesis_id)  # type: ignore[return-value]

    def preregister(
        self,
        hypothesis_id: str,
        *,
        acceptance_criteria: dict[str, Any],
        preregistered_at: str,
    ) -> dict[str, Any]:
        """Fix acceptance criteria + preregistered_at BEFORE the run (R2/R5).

        Raises if the criteria are empty/absent (R2), if the hypothesis is
        unknown, already has a recorded result (pre-registration cannot follow
        the run), or is already preregistered (criteria are immutable once
        fixed — register a re-test instead). The UPDATE is conditional on the
        row still being clean, so a concurrent writer cannot slip through.
        """
        _parse_ts(preregistered_at, "preregistered_at")
        _require_nonempty_criteria(
            acceptance_criteria, f"preregister(hypothesis_id={hypothesis_id!r})"
        )
        with closing(self._conn()) as c, c:
            row = self._fetch_or_raise(c, hypothesis_id)
            if row["run_at"] is not None or row["result_json"] is not None:
                raise RegistryIntegrityError(
                    f"[R5 post-hoc preregistration] hypothesis_id={hypothesis_id!r} already "
                    f"has a recorded result (run_at={row['run_at']}). Pre-registration must "
                    f"precede the run; register a new hypothesis (re-test) instead."
                )
            if row["preregistered_at"] is not None:
                raise RegistryIntegrityError(
                    f"[R5 criteria immutable] hypothesis_id={hypothesis_id!r} was already "
                    f"preregistered at {row['preregistered_at']}. Criteria cannot be changed "
                    f"after pre-registration; register a NEW hypothesis_id with "
                    f"parent_id={hypothesis_id!r} to test revised criteria."
                )
            cur = c.execute(
                "UPDATE experiments SET acceptance_criteria=?, preregistered_at=? "
                "WHERE hypothesis_id=? AND preregistered_at IS NULL "
                "AND run_at IS NULL AND result_json IS NULL",
                (
                    _encode("acceptance_criteria", acceptance_criteria),
                    preregistered_at,
                    hypothesis_id,
                ),
            )
            if cur.rowcount != 1:
                raise RegistryIntegrityError(
                    f"[R5 criteria immutable] hypothesis_id={hypothesis_id!r}: "
                    f"concurrent/duplicate pre-registration detected — the row was "
                    f"preregistered or got a result between check and write. "
                    f"Criteria were NOT changed."
                )
        return self.get(hypothesis_id)  # type: ignore[return-value]

    def record_result(
        self,
        hypothesis_id: str,
        *,
        run_at: str,
        result: dict[str, Any] | None = None,
        decision: str | None = None,
    ) -> dict[str, Any]:
        """Record the outcome of a run. Enforces R2, R3, R4 (and fails closed
        on unknown labels).

        `run_at` is the (caller-supplied) timestamp the run EXECUTED — no
        wall-clock is read here. The re-test cooling-off check (R4) compares
        cooling_off_until against run_at: a run that executed during the
        cooling-off window can never be recorded, no matter when the result
        is written. The UPDATE is conditional on the row having no result yet,
        so a concurrent second writer cannot also commit (R4 race backstop).
        """
        run_dt = _parse_ts(run_at, "run_at")

        if decision is not None and decision not in DECISIONS:
            raise RegistryIntegrityError(
                f"decision={decision!r} invalid; must be one of {DECISIONS}"
            )

        with closing(self._conn()) as c, c:
            row = self._fetch_or_raise(c, hypothesis_id)
            label = row["label"]

            # Fail CLOSED on labels outside the known set (schema drift /
            # manual DB edit) — an unknown label would silently skip R2 and R3.
            if label not in LABELS:
                raise RegistryIntegrityError(
                    f"[label] unknown label {label!r} on "
                    f"hypothesis_id={hypothesis_id!r}; must be one of {LABELS}. "
                    f"Refusing to record a result for a row whose integrity "
                    f"rules cannot be determined (fail closed)."
                )

            # R4 — one shot per hypothesis.
            if (
                row["run_at"] is not None
                or row["result_json"] is not None
                or row["decision"] is not None
            ):
                raise RegistryIntegrityError(
                    f"[R4 one shot] hypothesis_id={hypothesis_id!r} already has a recorded "
                    f"result (run_at={row['run_at']}, decision={row['decision']}). A "
                    f"hypothesis gets exactly one result. To run again, register a NEW "
                    f"hypothesis_id with parent_id={hypothesis_id!r} and a cooling_off_until "
                    f"date, and run it after that date."
                )

            # R4 — a re-test (parent_id set) must respect its registered
            # cooling-off: the run must EXECUTE after cooling_off_until.
            if row["parent_id"] is not None:
                if row["cooling_off_until"] is None:
                    raise RegistryIntegrityError(
                        f"[R4 re-test cooling-off] hypothesis_id={hypothesis_id!r} is a "
                        f"re-test of {row['parent_id']!r} but has no cooling_off_until. "
                        f"Re-tests must register a cooling-off date at registration time."
                    )
                cool_dt = _parse_ts(row["cooling_off_until"], "cooling_off_until")
                if cool_dt >= run_dt:
                    raise RegistryIntegrityError(
                        f"[R4 re-test run must execute after the cooling-off date] "
                        f"hypothesis_id={hypothesis_id!r} re-tests {row['parent_id']!r} "
                        f"but run_at={run_at} is not strictly after cooling_off_until="
                        f"{row['cooling_off_until']}. The run itself must wait out the "
                        f"cooling-off period; recording the result later does not help."
                    )

            # R2 — confirmatory runs must have been preregistered strictly
            # before the run, with non-empty acceptance criteria.
            if label in _PREREG_REQUIRED_LABELS:
                if row["preregistered_at"] is None:
                    raise RegistryIntegrityError(
                        f"[R2 pre-registration] hypothesis_id={hypothesis_id!r} is "
                        f"label={label!r} but was never preregistered. Confirmatory runs "
                        f"must fix acceptance_criteria via preregister() BEFORE the run; "
                        f"this result cannot be recorded."
                    )
                # Defense-in-depth: empty criteria should be unreachable via the
                # public API (preregister/register both refuse them) but a
                # manual DB edit must not promote either.
                _require_nonempty_criteria(
                    row["acceptance_criteria"],
                    f"record_result(hypothesis_id={hypothesis_id!r}, label={label!r})",
                )
                prereg_dt = _parse_ts(row["preregistered_at"], "preregistered_at")
                if prereg_dt >= run_dt:
                    raise RegistryIntegrityError(
                        f"[R2 pre-registration ordering] hypothesis_id={hypothesis_id!r}: "
                        f"preregistered_at={row['preregistered_at']} is not strictly before "
                        f"run_at={run_at}. Criteria must be fixed BEFORE the run executes; "
                        f"a same-instant or later pre-registration proves nothing."
                    )

            # R3 — exploratory runs can inspire but can never promote.
            if label == "exploratory" and decision is not None:
                if decision not in EXPLORATORY_ALLOWED_DECISIONS:
                    raise RegistryIntegrityError(
                        f"[R3 exploratory cannot promote] hypothesis_id={hypothesis_id!r} is "
                        f"exploratory; decision={decision!r} is not allowed (only "
                        f"{EXPLORATORY_ALLOWED_DECISIONS}). To promote, register a NEW "
                        f"confirmatory hypothesis with preregistered criteria and run it."
                    )

            cur = c.execute(
                "UPDATE experiments SET run_at=?, result_json=?, decision=? "
                "WHERE hypothesis_id=? AND run_at IS NULL "
                "AND result_json IS NULL AND decision IS NULL",
                (run_at, _encode("result_json", result), decision, hypothesis_id),
            )
            if cur.rowcount != 1:
                raise RegistryIntegrityError(
                    f"[R4 one shot] hypothesis_id={hypothesis_id!r}: concurrent/"
                    f"duplicate result detected — another writer recorded a result "
                    f"between check and write. This result was NOT recorded."
                )
        return self.get(hypothesis_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------- read
    def get(self, hypothesis_id: str) -> dict[str, Any] | None:
        with closing(self._conn()) as c:
            return self._fetch(c, hypothesis_id)

    def list(
        self,
        *,
        label: str | None = None,
        family: str | None = None,
        decision: str | None = None,
        parent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rows matching the given filters, ordered by created_at then id."""
        clauses, args = [], []
        for col, val in (
            ("label", label),
            ("family", family),
            ("decision", decision),
            ("parent_id", parent_id),
        ):
            if val is not None:
                clauses.append(f"{col}=?")
                args.append(val)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with closing(self._conn()) as c:
            rows = c.execute(
                f"SELECT * FROM experiments{where} ORDER BY created_at, hypothesis_id",
                args,
            ).fetchall()
        return [_decode_row(r) for r in rows]

    def trial_count(self, *, label: str | None = None,
                    family: str | None = None) -> int:
        """The number of registered trials in a multiple-testing universe — the count
        of experiment rows matching the filter (no filter → the program's true
        N_TRIALS). This is the HONEST per-hypothesis trial count the Ruler-v2 Bayesian
        prior should consume (design risk R7): a hypothesis is penalized for the shots
        taken at ITS family, not the saturated global 300 that broke the DSR."""
        clauses, args = [], []
        for col, val in (("label", label), ("family", family)):
            if val is not None:
                clauses.append(f"{col}=?")
                args.append(val)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with closing(self._conn()) as c:
            return int(c.execute(
                f"SELECT COUNT(*) FROM experiments{where}", args).fetchone()[0])

    def summary(self) -> dict[str, Any]:
        """Trial accounting: total rows + counts by label and by decision.

        `total` is the program's true N_TRIALS. Rows without a decision are
        counted under 'pending'.
        """
        with closing(self._conn()) as c:
            total = c.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            by_label = {
                r[0]: r[1]
                for r in c.execute(
                    "SELECT label, COUNT(*) FROM experiments GROUP BY label"
                ).fetchall()
            }
            by_decision = {
                (r[0] if r[0] is not None else "pending"): r[1]
                for r in c.execute(
                    "SELECT decision, COUNT(*) FROM experiments GROUP BY decision"
                ).fetchall()
            }
        return {"total": total, "by_label": by_label, "by_decision": by_decision}
