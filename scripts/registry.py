"""
Research-registry CLI — thin wrapper over app.research.registry.ResearchRegistry.

Usage:
  python scripts/registry.py register HYP-001 --label confirmatory --family pead \
      --params '{"threshold": 0.04}' --mechanism "implied-move filters PEAD entries"
  python scripts/registry.py preregister HYP-001 \
      --criteria '{"track_a_t": 2.0}' [--at 2026-06-10T12:00:00+00:00]
  python scripts/registry.py record-result HYP-001 --run-at 2026-06-11T03:00:00+00:00 \
      --result '{"sharpe": 0.61}' --decision promote_paper
  python scripts/registry.py list [--label exploratory] [--family pead] [--decision kill]
  python scripts/registry.py summary

All output is ASCII. --db overrides the sqlite path (else env
MRTRADER_RESEARCH_REGISTRY_DB, else data/research_registry.db).
Integrity violations exit 2 with the rule named on stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.research.registry import RegistryIntegrityError, ResearchRegistry  # noqa: E402


def _json_arg(s: str | None):
    if s is None:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: not valid JSON: {s!r} ({exc})")


def _ascii(v) -> str:
    """ASCII-safe text for cp1252/legacy consoles (no UnicodeEncodeError)."""
    return str(v).encode("ascii", "backslashreplace").decode("ascii")


def _print_row(row: dict) -> None:
    for k, v in row.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, sort_keys=True)  # ensure_ascii=True by default
        print(f"  {k:<20} {_ascii(v)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Research pre-registration ledger.")
    p.add_argument("--db", default=None, help="sqlite path override")
    sub = p.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="register a new hypothesis")
    reg.add_argument("hypothesis_id")
    reg.add_argument("--label", required=True,
                     choices=["exploratory", "confirmatory", "live_confirm"])
    reg.add_argument("--family", default=None)
    reg.add_argument("--parent-id", default=None)
    reg.add_argument("--features", default=None, help="JSON list or plain string")
    reg.add_argument("--params", default=None, help="JSON dict")
    reg.add_argument("--universe", default=None)
    reg.add_argument("--window", default=None)
    reg.add_argument("--folds", default=None)
    reg.add_argument("--cost-model", default=None)
    reg.add_argument("--code-commit", default=None)
    reg.add_argument("--data-hash", default=None)
    reg.add_argument("--mechanism", default=None)
    reg.add_argument("--cooling-off-until", default=None, help="ISO-8601 (re-tests)")

    pre = sub.add_parser("preregister", help="fix acceptance criteria BEFORE the run")
    pre.add_argument("hypothesis_id")
    pre.add_argument("--criteria", required=True, help="JSON dict of acceptance criteria")
    pre.add_argument("--at", default=None, help="ISO-8601 (default: now UTC)")

    rec = sub.add_parser("record-result", help="record a run's outcome")
    rec.add_argument("hypothesis_id")
    rec.add_argument("--run-at", required=True, help="ISO-8601 time the run executed")
    rec.add_argument("--result", default=None, help="JSON dict of metrics")
    rec.add_argument("--decision", default=None,
                     choices=["kill", "park", "promote_paper", "live", "exploratory_only"])

    ls = sub.add_parser("list", help="list registered hypotheses")
    ls.add_argument("--label", default=None)
    ls.add_argument("--family", default=None)
    ls.add_argument("--decision", default=None)
    ls.add_argument("--parent-id", default=None)

    sub.add_parser("summary", help="trial accounting: counts by label and decision")

    args = p.parse_args(argv)
    r = ResearchRegistry(db_path=args.db)

    try:
        if args.cmd == "register":
            feats = args.features
            if feats is not None:
                try:
                    feats = json.loads(feats)
                except json.JSONDecodeError:
                    pass  # plain string is fine
            row = r.register(
                args.hypothesis_id,
                label=args.label,
                family=args.family,
                parent_id=args.parent_id,
                features=feats,
                params=_json_arg(args.params),
                universe=args.universe,
                window=args.window,
                folds=args.folds,
                cost_model=args.cost_model,
                code_commit=args.code_commit,
                data_hash=args.data_hash,
                mechanism=args.mechanism,
                cooling_off_until=args.cooling_off_until,
            )
            print(_ascii(f"registered {args.hypothesis_id}"))
            _print_row(row)

        elif args.cmd == "preregister":
            at = args.at or datetime.now(timezone.utc).isoformat()
            row = r.preregister(
                args.hypothesis_id,
                acceptance_criteria=_json_arg(args.criteria),
                preregistered_at=at,
            )
            print(_ascii(f"preregistered {args.hypothesis_id} at {at}"))
            _print_row(row)

        elif args.cmd == "record-result":
            row = r.record_result(
                args.hypothesis_id,
                run_at=args.run_at,
                result=_json_arg(args.result),
                decision=args.decision,
            )
            print(_ascii(f"recorded result for {args.hypothesis_id} "
                         f"(decision={row['decision']})"))
            _print_row(row)

        elif args.cmd == "list":
            rows = r.list(label=args.label, family=args.family,
                          decision=args.decision, parent_id=args.parent_id)
            if not rows:
                print("(no rows)")
            for row in rows:
                print(_ascii(
                    f"{row['hypothesis_id']:<24} label={row['label']:<13} "
                    f"family={str(row['family']):<12} "
                    f"decision={str(row['decision']):<16} "
                    f"prereg={str(row['preregistered_at'])} "
                    f"run={str(row['run_at'])}"
                ))

        elif args.cmd == "summary":
            s = r.summary()
            print(f"total trials: {s['total']}")
            print("by label:")
            for k in sorted(s["by_label"]):
                print(f"  {_ascii(k):<18} {s['by_label'][k]}")
            print("by decision:")
            for k in sorted(s["by_decision"]):
                print(f"  {_ascii(k):<18} {s['by_decision'][k]}")

    except RegistryIntegrityError as exc:
        print(f"INTEGRITY VIOLATION: {_ascii(exc)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
