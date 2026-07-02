"""Weather features from the Open-Meteo archive API.

Improvements over the reviewed notebook:
  * Airport coordinates come from a CSV lookup (config/airports.csv), not a
    hardcoded dict — add an airport by editing the CSV, no code change.
  * Responses are cached to parquet so repeated runs are instant and offline-safe.
  * KNOWN LIMITATION (documented, not hidden): the archive endpoint returns
    *observed* weather. A truly pre-departure model should use the *forecast*
    available at scheduling time. Swap `archive_url` for the forecast endpoint
    when moving to production.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def load_airports(reference_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(reference_csv)


def attach_coords(df: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Map destination IATA -> lat/lon; drop flights we have no coords for."""
    df = df.merge(
        airports.rename(columns={"iata": "SCHD_LEG_ARVL_AIRPRT_IATA_CD",
                                 "lat": "dest_lat", "lon": "dest_lon"}),
        on="SCHD_LEG_ARVL_AIRPRT_IATA_CD", how="left",
    )
    before = len(df)
    df = df.dropna(subset=["dest_lat", "dest_lon"])
    logger.info("Coord filter kept %d of %d rows", len(df), before)
    return df


def _fetch_one(url, lat, lon, start, end, hourly_vars, timeout) -> pd.DataFrame:
    resp = requests.get(url, params={
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(hourly_vars), "timezone": "auto",
    }, timeout=timeout)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    out = pd.DataFrame({"weather_time": pd.to_datetime(hourly.get("time", []))})
    for var in hourly_vars:
        out[var] = hourly.get(var, [])
    out["dest_lat"], out["dest_lon"] = lat, lon
    return out


def add_weather(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Join hourly weather onto flights, using a parquet cache when available."""
    df = df.copy()
    df["weather_time"] = df["SCHD_LEG_ARVL_LCL_TMS"].dt.floor("h")

    cache_path = Path(cfg["cache_path"])
    if cache_path.exists():
        logger.info("Loading weather from cache %s", cache_path)
        weather_all = pd.read_parquet(cache_path)
    else:
        start = df["weather_time"].min().date().isoformat()
        end = df["weather_time"].max().date().isoformat()
        unique = df[["dest_lat", "dest_lon"]].drop_duplicates()
        frames = []
        for _, row in unique.iterrows():
            try:
                frames.append(_fetch_one(
                    cfg["archive_url"], row["dest_lat"], row["dest_lon"],
                    start, end, cfg["hourly_vars"], cfg["request_timeout_sec"],
                ))
            except Exception as e:  # noqa: BLE001 — skip a failing airport, keep going
                logger.warning("Weather fetch failed for %s: %s", row.to_dict(), e)
        weather_all = pd.concat(frames, ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        weather_all.to_parquet(cache_path, index=False)

    df = df.merge(weather_all, on=["weather_time", "dest_lat", "dest_lon"], how="left")
    return df.drop(columns=["weather_time"], errors="ignore")
