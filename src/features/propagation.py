"""Delay-propagation feature: is the inbound aircraft already running late?

The single strongest real-world predictor of a departure delay is the state of
the *previous* flight flown by the same physical aircraft. If the plane arrives
late, the next leg leaves late.

LEAKAGE GUARD (critical):
  The previous leg's *actual* arrival time is only known once it has landed. We
  may only use it if that landing happened BEFORE the current flight's scheduled
  departure — otherwise we'd be using information from the future, exactly the
  bug this project fixed. When the inbound leg hasn't landed in time, the feature
  is marked unknown (NaN + a 0 flag) instead of leaking.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_inbound_delay(df: pd.DataFrame, tail_column: str, enabled: bool = True) -> pd.DataFrame:
    """Add `inbound_delay_min` and `inbound_known` using the aircraft's prior leg.

    Requires (already present at this stage of the pipeline):
      - `tail_column`            : aircraft identifier
      - `ARR_DELAY_MIN`          : delay of every leg (our target, computed earlier)
      - `ACTL_LEG_ARVL_LCL_TMS`  : when each leg actually landed
      - `SCHD_LEG_DEP_LCL_TMS`   : scheduled departure of the current leg
    """
    if not enabled or tail_column not in df.columns:
        logger.info("Inbound-delay feature skipped (disabled or tail column missing)")
        return df

    # Order each aircraft's legs chronologically, then look one leg back.
    df = df.sort_values([tail_column, "SCHD_LEG_DEP_LCL_TMS"]).copy()
    grp = df.groupby(tail_column, sort=False)
    prev_delay = grp["ARR_DELAY_MIN"].shift(1)              # previous leg's delay
    prev_landed = grp["ACTL_LEG_ARVL_LCL_TMS"].shift(1)     # when it actually landed

    # Only usable if the previous leg landed before this leg was scheduled to leave.
    # NaT comparisons (first leg of an aircraft) evaluate to False -> unknown.
    known = prev_landed < df["SCHD_LEG_DEP_LCL_TMS"]

    df["inbound_delay_min"] = prev_delay.where(known)        # NaN when unknown
    df["inbound_known"] = known.fillna(False).astype(int)    # 1 = known, 0 = unknown

    pct_known = 100 * df["inbound_known"].mean()
    logger.info("Inbound-delay feature added (known for %.1f%% of flights)", pct_known)
    return df
