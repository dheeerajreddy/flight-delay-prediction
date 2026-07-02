"""Raw data loading and timestamp parsing."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw(path: str | Path, date_columns: list[str]) -> pd.DataFrame:
    """Read the raw flight CSV and parse timestamp columns.

    Parameters
    ----------
    path : path to the raw CSV.
    date_columns : columns to coerce to datetime (bad values -> NaT).
    """
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Raw shape: %s", df.shape)

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
