"""Numeric version selection for ``<name>_v<N>.pkl`` model artifacts (2026-08-22).

WHY THIS EXISTS
---------------
``sorted(MODEL_DIR.glob("regime_model_v*.pkl"))[-1]`` sorts LEXICALLY, so once v10 exists
the "latest" file is ``v9`` — "v9" > "v40" as text. This was not hypothetical: it silently
pinned the live regime scorer to ``regime_model_v9.pkl`` (trained through 2026-07-02) from
the moment v10 landed on 2026-07-10, while 31 newer models were trained and ignored.

It also broke the retrain-interval guard in the portfolio manager, which stat()'d the same
wrongly-chosen file. Because v9's mtime was always older than REGIME_RETRAIN_INTERVAL_DAYS,
the "skip, model is fresh" branch never fired and the regime model retrained DAILY instead
of weekly — the exact cadence `retrain_config` argues against ("daily retraining adds noise
without benefit").

Both failures are silent: no error, no warning, just the wrong file. Hence one helper that
parses the integer rather than four call sites each re-deriving it from a string sort.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

# `swing_v223.pkl` -> 223. Anchored at the end so a prefix containing "_v" cannot match.
_VERSION_RE = re.compile(r"_v(\d+)$")


def parse_version(path: Path) -> Optional[int]:
    """Version integer from a ``<name>_v<N>.pkl`` path, or None if it does not match."""
    m = _VERSION_RE.search(path.stem)
    return int(m.group(1)) if m else None


def versioned_files(directory: Path, prefix: str) -> List[Path]:
    """All ``<prefix>_v<N>.pkl`` files, sorted ASCENDING BY NUMERIC VERSION.

    Files whose suffix is not an integer are skipped rather than crashing the caller —
    a stray ``regime_model_vBACKUP.pkl`` must not take down model loading.
    """
    pairs = []
    for p in Path(directory).glob(f"{prefix}_v*.pkl"):
        v = parse_version(p)
        if v is not None:
            pairs.append((v, p))
    return [p for _, p in sorted(pairs, key=lambda vp: vp[0])]


def latest_versioned_file(directory: Path, prefix: str) -> Optional[Path]:
    """Highest-numbered ``<prefix>_v<N>.pkl``, or None when there are none."""
    files = versioned_files(directory, prefix)
    return files[-1] if files else None


def latest_version(directory: Path, prefix: str) -> int:
    """Highest version number present, or 0 when there are none."""
    latest = latest_versioned_file(directory, prefix)
    if latest is None:
        return 0
    return parse_version(latest) or 0


def next_version(directory: Path, prefix: str) -> int:
    """Next version to write: max + 1.

    Deliberately max-based, not ``len(files) + 1``: counting collides the moment any
    artifact is deleted or archived, silently overwriting an existing version.
    """
    return latest_version(directory, prefix) + 1
