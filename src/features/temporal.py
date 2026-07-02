"""Calendar / time-of-day features derived from scheduled timestamps."""
from __future__ import annotations

import pandas as pd

_PERIOD_BINS = [
    (5, 12, "Morning"),
    (12, 17, "Afternoon"),
    (17, 22, "Evening"),
]


def _time_period(hour: int) -> str:
    for lo, hi, name in _PERIOD_BINS:
        if lo <= hour < hi:
            return name
    return "Night"


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Explode scheduled departure/arrival timestamps into modelling features."""
    df = df.copy()
    dep = df["SCHD_LEG_DEP_LCL_TMS"].dt
    arr = df["SCHD_LEG_ARVL_LCL_TMS"].dt

    df["dep_hour"], df["dep_day"] = dep.hour, dep.day
    df["dep_month"], df["dep_dayofweek"] = dep.month, dep.dayofweek
    df["is_weekend"] = df["dep_dayofweek"].isin([5, 6]).astype(int)

    df["arr_hour"], df["arr_day"] = arr.hour, arr.day
    df["arr_month"], df["arr_dayofweek"] = arr.month, arr.dayofweek

    df["scheduled_duration_min"] = (
        df["SCHD_LEG_ARVL_LCL_TMS"] - df["SCHD_LEG_DEP_LCL_TMS"]
    ).dt.total_seconds() / 60

    df["dep_time_period"] = df["dep_hour"].apply(_time_period)
    return df
