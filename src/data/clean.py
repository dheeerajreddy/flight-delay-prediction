"""Cleaning, target construction, and leakage removal.

Two of the reviewed notebook's biggest risks are addressed here:
  * The target is *clipped* to a sane range so timezone/parse artifacts
    (e.g. a 2114-minute "delay") don't silently poison training.
  * Leakage columns are dropped up front.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_target(df: pd.DataFrame, clip_min: float, clip_max: float) -> pd.DataFrame:
    """Arrival delay (minutes) = actual arrival - scheduled arrival, in local time.

    Rows outside [clip_min, clip_max] are treated as data errors and dropped,
    not clipped-in-place, so they can't distort the distribution.
    """
    df = df.copy()
    df["ARR_DELAY_MIN"] = (
        df["ACTL_LEG_ARVL_LCL_TMS"] - df["SCHD_LEG_ARVL_LCL_TMS"]
    ).dt.total_seconds() / 60

    before = len(df)
    df = df.dropna(subset=["ARR_DELAY_MIN", "SCHD_LEG_DEP_LCL_TMS", "SCHD_LEG_ARVL_LCL_TMS"])
    df = df[(df["ARR_DELAY_MIN"] >= clip_min) & (df["ARR_DELAY_MIN"] <= clip_max)]
    logger.info("Target computed; dropped %d implausible/blank rows", before - len(df))
    return df


def remove_cancelled_and_diverted(df: pd.DataFrame) -> pd.DataFrame:
    """Cancelled and diverted flights are a different problem — exclude them."""
    df = df.copy()
    for ind in ("CANCEL_IND", "DIVERT_IND"):
        if ind in df.columns:
            df = df[df[ind].fillna(0) == 0]
    return df


def select_and_deleak(
    df: pd.DataFrame, safe_columns: list[str], leakage_columns: list[str]
) -> pd.DataFrame:
    """Keep only pre-departure-safe columns; explicitly drop known leakage columns."""
    keep = [c for c in safe_columns if c in df.columns] + ["ARR_DELAY_MIN"]
    df_model = df[keep].copy()
    df_model = df_model.drop(
        columns=[c for c in leakage_columns if c in df_model.columns], errors="ignore"
    )
    logger.info("Selected %d safe feature columns", len(df_model.columns) - 1)
    return df_model
