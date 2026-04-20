#!/usr/bin/env bash
# Sequential training: v17 then v18 (v19 already running separately)
# Run after v19 completes. Each run clears the feature store first.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/app/ml/models/feature_store.db"

clear_feature_store() {
    echo "[seq] Clearing feature store..."
    python -c "
import sqlite3
c = sqlite3.connect('$DB')
deleted = c.execute('DELETE FROM features').rowcount
c.commit()
c.close()
print(f'[seq] Deleted {deleted} stale rows from feature store')
"
}

echo "========================================"
echo "  Sequential Experiment Runner"
echo "  Running: v17 -> v18"
echo "========================================"

# --- v17 ---
echo ""
echo "[seq] === v17: Symmetric ATR + 12mo Momentum (LightGBM) ==="
git -C "$ROOT" checkout experiment/v17-symmetric-atr-12mo-momentum
clear_feature_store
python "$ROOT/scripts/train_model.py" \
    --years 5 \
    --model-type lgbm \
    --label-scheme atr \
    --top-features 25 \
    2>&1 | tee /tmp/v17_training.log
echo "[seq] v17 DONE"

# --- v18 ---
echo ""
echo "[seq] === v18: XGBoost+LGBM Ensemble + Sector-Neutral Momentum ==="
git -C "$ROOT" checkout experiment/v18-ensemble-sector-neutral-momentum
clear_feature_store
python "$ROOT/scripts/train_model.py" \
    --years 5 \
    --model-type lgbm_ensemble \
    --label-scheme atr \
    --top-features 30 \
    2>&1 | tee /tmp/v18_training.log
echo "[seq] v18 DONE"

echo ""
echo "========================================"
echo "  Both experiments complete!"
echo "  v17 log: /tmp/v17_training.log"
echo "  v18 log: /tmp/v18_training.log"
echo "========================================"
