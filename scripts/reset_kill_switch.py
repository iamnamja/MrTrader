"""Clear a stuck kill switch state in the persistent configuration store.

Usage:
    python -m scripts.reset_kill_switch
    python -m scripts.reset_kill_switch --status   # show current value only

Safe to run while the server is stopped. After running, restart the server
so the singleton reloads load_state() and observes the cleared flag.
"""
from __future__ import annotations

import argparse
import sys


def _current(db) -> object:
    from app.database.config_store import get_config
    return get_config(db, "kill_switch.active")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset persisted kill switch state")
    parser.add_argument("--status", action="store_true",
                        help="Print current kill_switch.active value and exit")
    args = parser.parse_args(argv)

    from app.database.session import get_session
    from app.database.config_store import set_config

    db = get_session()
    try:
        before = _current(db)
        print(f"kill_switch.active (before): {before!r}")
        if args.status:
            return 0
        set_config(db, "kill_switch.active", False, "Kill switch active flag")
        after = _current(db)
        print(f"kill_switch.active (after):  {after!r}")
        print("OK — restart the server (uvicorn) so the singleton picks up the cleared state.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
