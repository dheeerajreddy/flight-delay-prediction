"""Feature orchestration + a leakage-safe preprocessor.

The reviewed notebook imputed medians over the WHOLE dataset before splitting,
leaking test statistics into training. Here, `Preprocessor.fit` sees only the
train fold; `transform` re-applies those learned values to any fold.
"""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.features.temporal import add_temporal_features
from src.features import weather as weather_mod

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run all feature steps, then drop raw timestamps used only for engineering."""
    df = add_temporal_features(df)

    if cfg["weather"]["enabled"]:
        airports = weather_mod.load_airports(cfg["weather"]["airports_reference"])
        df = weather_mod.attach_coords(df, airports)
        df = weather_mod.add_weather(df, cfg["weather"])

    # Timestamps are intentionally KEPT here — the pipeline needs the departure
    # timestamp as the temporal-split sort key. They are dropped after the split.
    return df


class Preprocessor:
    """Fit imputation + ordinal encoding on TRAIN ONLY, apply to any fold."""

    def __init__(self, categorical_fill: str):
        self.categorical_fill = categorical_fill
        self.medians_: dict[str, float] = {}
        self.cat_cols_: list[str] = []
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        num = X.select_dtypes(include=["int64", "float64"]).columns
        self.medians_ = {c: X[c].median() for c in num}
        self.cat_cols_ = X.select_dtypes(include=["object"]).columns.tolist()
        if self.cat_cols_:
            filled = X[self.cat_cols_].astype(str).fillna(self.categorical_fill)
            self.encoder_.fit(filled)
        return self

    def transform(self, X: pd.DataFrame, encode: bool = True) -> pd.DataFrame:
        """encode=False returns raw string categoricals (for CatBoost)."""
        X = X.copy()
        for c, med in self.medians_.items():
            if c in X.columns:
                X[c] = X[c].fillna(med)
        if self.cat_cols_:
            X[self.cat_cols_] = X[self.cat_cols_].astype(str).fillna(self.categorical_fill)
            if encode:
                X[self.cat_cols_] = self.encoder_.transform(X[self.cat_cols_])
        return X
