"""End-to-end pipeline: load -> clean -> features -> split -> preprocess -> train.

Each stage is a pure function/class imported from its own module, so any stage
can be unit-tested, swapped, or reused independently.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib

from src.config import load_config, resolve
from src.data.load import load_raw
from src.data.clean import (
    compute_target, remove_cancelled_and_diverted, select_and_deleak,
)
from src.data.split import split
from src.features.build_features import build_features, Preprocessor
from src.models.registry import build_models
from src.models.train import train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run(config_path: str | None = None) -> dict:
    cfg = load_config(config_path) if config_path else load_config()
    dcfg, seed = cfg["data"], cfg["project"]["random_state"]

    # 1. Load + parse
    df = load_raw(resolve(dcfg["raw_path"]), dcfg["date_columns"])

    # 2. Clean + target (target is clipped to a plausible range)
    df = compute_target(df, dcfg["target_clip_min"], dcfg["target_clip_max"])
    df = remove_cancelled_and_diverted(df)
    df = select_and_deleak(df, dcfg["safe_columns"], dcfg["leakage_columns"])

    # 3. Features (timestamps kept so we can split on them)
    df = build_features(df, cfg)

    # 4. Temporal split BEFORE any fitting — uses the real departure timestamp
    scfg = cfg["split"]
    X_tr, X_te, y_tr, y_te = split(
        df, dcfg["target"], scfg["strategy"], scfg["test_size"],
        scfg["time_column"], seed,
    )

    # Drop the raw timestamps now that the split is done — they were only needed
    # as feature inputs and as the temporal sort key.
    drop_after = ["SCHD_LEG_DEP_LCL_TMS", "SCHD_LEG_ARVL_LCL_TMS"]
    X_tr = X_tr.drop(columns=[c for c in drop_after if c in X_tr.columns])
    X_te = X_te.drop(columns=[c for c in drop_after if c in X_te.columns])

    # 5. Fit preprocessor on TRAIN ONLY
    pre = Preprocessor(cfg["features"]["categorical_fill"]).fit(X_tr)

    # 6. Train + evaluate
    models = build_models(cfg["models"], seed)
    leaderboard, fitted = train_and_evaluate(models, pre, X_tr, X_te, y_tr, y_te)

    # 7. Persist artifacts
    out = resolve(cfg["output"]["reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(out / "leaderboard.csv", index=False)

    champion = leaderboard.iloc[0]["Model"]
    models_dir = resolve(cfg["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[champion], models_dir / f"{champion}.joblib")
    joblib.dump(pre, models_dir / "preprocessor.joblib")
    (out / "summary.json").write_text(json.dumps(
        {"champion": champion, "leaderboard": leaderboard.to_dict("records")}, indent=2
    ))

    logger.info("Done. Champion: %s", champion)
    return {"champion": champion, "leaderboard": leaderboard}


if __name__ == "__main__":
    run()
