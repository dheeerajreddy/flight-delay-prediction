"""Unit tests for the leakage-sensitive pieces.

Run with:  pytest -q
These are the tests that would have caught the reviewed notebook's bugs.
"""
import numpy as np
import pandas as pd

from src.data.clean import compute_target
from src.data.split import split
from src.features.build_features import Preprocessor
from src.features.temporal import add_temporal_features


def _toy_df():
    ts = pd.to_datetime(pd.date_range("2023-01-01", periods=10, freq="6h"))
    return pd.DataFrame({
        "SCHD_LEG_DEP_LCL_TMS": ts,
        "SCHD_LEG_ARVL_LCL_TMS": ts + pd.Timedelta(hours=2),
        "ACTL_LEG_ARVL_LCL_TMS": ts + pd.Timedelta(hours=2, minutes=15),
    })


def test_target_drops_implausible_delays():
    df = _toy_df()
    df.loc[0, "ACTL_LEG_ARVL_LCL_TMS"] += pd.Timedelta(hours=40)  # 35h+ artifact
    out = compute_target(df, clip_min=-180, clip_max=600)
    assert out["ARR_DELAY_MIN"].max() <= 600
    assert len(out) == len(df) - 1  # the artifact row is removed


def test_temporal_split_preserves_time_order():
    df = _toy_df()
    df = compute_target(df, -180, 600)
    X_tr, X_te, _, _ = split(
        df, "ARR_DELAY_MIN", "temporal", 0.2, "SCHD_LEG_DEP_LCL_TMS", 42
    )
    assert X_tr["SCHD_LEG_DEP_LCL_TMS"].max() <= X_te["SCHD_LEG_DEP_LCL_TMS"].min()


def test_preprocessor_uses_train_medians_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "c": ["a", "b", "a"]})
    test = pd.DataFrame({"x": [np.nan, 100.0], "c": ["a", "z"]})  # 'z' unseen
    pre = Preprocessor("Unknown").fit(train)
    out = pre.transform(test)
    assert out.loc[0, "x"] == 2.0        # train median, not test's
    assert out.loc[1, "c"] == -1         # unseen category -> safe fallback


def test_temporal_features_added():
    out = add_temporal_features(_toy_df())
    for col in ("dep_hour", "is_weekend", "scheduled_duration_min", "dep_time_period"):
        assert col in out.columns
