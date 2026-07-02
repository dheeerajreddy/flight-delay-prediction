"""Train/test splitting.

The reviewed notebook used a random shuffle, which is invalid for a
predict-before-departure problem: it lets future flights leak into training.
The default here is a *temporal* split — train on the earliest rows, test on
the latest. A random split is kept only for A/B comparison.
"""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split(
    df: pd.DataFrame,
    target: str,
    strategy: str,
    test_size: float,
    time_column: str,
    random_state: int,
):
    """Return X_train, X_test, y_train, y_test."""
    y = df[target]
    X = df.drop(columns=[target])

    if strategy == "temporal":
        order = df[time_column].argsort()
        cut = int(len(df) * (1 - test_size))
        train_idx, test_idx = order[:cut], order[cut:]
        logger.info(
            "Temporal split at %s | train<= %s | test> %s",
            time_column,
            df[time_column].iloc[order[cut - 1]],
            df[time_column].iloc[order[cut]],
        )
        return (
            X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
        )

    if strategy == "random":
        logger.warning("Using RANDOM split — invalid for production, comparison only.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    raise ValueError(f"Unknown split strategy: {strategy}")
