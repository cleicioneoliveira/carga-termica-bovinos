#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("build_hourly_cta_from_heat_stress")

ALIASES = {
    "timestamp": "timestamp",
    "data": "timestamp",
    "data_hora": "timestamp",
    "datetime": "timestamp",
    "dispositivo": "dispositivo",
    "device": "dispositivo",
    "sensor": "dispositivo",
    "temperature": "temperature",
    "temperatura": "temperature",
    "humidity": "humidity",
    "umidade": "humidity",
    "umidade_relativa": "humidity",
}
REQUIRED = ("timestamp", "dispositivo", "temperature", "humidity")


def norm_col(name: str) -> str:
    txt = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return txt.strip().lower().replace(" ", "_").replace(".", "").replace("/", "_").replace("-", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [norm_col(c) for c in out.columns]
    rename = {c: ALIASES[c] for c in out.columns if c in ALIASES and ALIASES[c] not in out.columns}
    if rename:
        LOGGER.info("Renaming columns: %s", rename)
        out = out.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}. Available: {list(out.columns)}")
    return out


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Input must be .csv or .parquet")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError("Output must be .csv or .parquet")


def calculate_itu(temp_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    temp_f = 1.8 * temp_c.astype(float) + 32.0
    rh = humidity_pct.astype(float) / 100.0
    return temp_f - (0.55 - 0.55 * rh) * (temp_f - 58.0)


def prepare_heat_data(path: Path, humidity_unit: str) -> pd.DataFrame:
    if humidity_unit not in {"auto", "pct", "fraction"}:
        raise ValueError("humidity_unit must be auto, pct or fraction")

    df = normalize_columns(read_table(path))
    out = df.loc[:, list(REQUIRED)].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["temperature"] = pd.to_numeric(out["temperature"], errors="coerce")
    out["humidity"] = pd.to_numeric(out["humidity"], errors="coerce")
    out = out.dropna(subset=list(REQUIRED)).copy()

    if humidity_unit == "fraction":
        out["humidity"] = out["humidity"] * 100.0
    elif humidity_unit == "auto":
        q95 = out["humidity"].quantile(0.95)
        if q95 <= 1.5:
            LOGGER.info("Humidity appears to be fractional. Converting to percent.")
            out["humidity"] = out["humidity"] * 100.0
        else:
            LOGGER.info("Humidity appears to be already in percent.")

    return out.sort_values(["dispositivo", "timestamp"]).reset_index(drop=True)


def add_segments(df: pd.DataFrame, max_gap_minutes: int) -> pd.DataFrame:
    out = df.sort_values(["dispositivo", "timestamp"]).copy()
    gap = pd.Timedelta(minutes=max_gap_minutes)
    out["time_diff"] = out.groupby("dispositivo", observed=False)["timestamp"].diff()
    out["new_segment"] = out["time_diff"].isna() | (out["time_diff"] > gap)
    out["segment_id"] = out.groupby("dispositivo", observed=False)["new_segment"].cumsum()
    LOGGER.info("Segments by device: %s", out.groupby("dispositivo", observed=False)["segment_id"].nunique().to_dict())
    return out


def compute_metrics(df: pd.DataFrame, threshold: float, windows: Iterable[int], freq_min: int, max_gap_min: int) -> pd.DataFrame:
    if freq_min <= 0 or 60 % freq_min != 0:
        raise ValueError("input frequency must divide 60")
    if max_gap_min < freq_min:
        raise ValueError("max gap must be >= input frequency")
    out = df.copy()
    dt = freq_min / 60.0
    records_per_hour = int(60 / freq_min)
    out["itu"] = calculate_itu(out["temperature"], out["humidity"])
    out["heat_excess"] = np.maximum(out["itu"] - threshold, 0.0)
    out = add_segments(out, max_gap_minutes=max_gap_min)
    for w in windows:
        wr = int(w * records_per_hour)
        col = f"cta_{w}h"
        out[col] = out.groupby(["dispositivo", "segment_id"], observed=False)["heat_excess"].transform(
            lambda s, wr=wr: (s * dt).rolling(wr, min_periods=1).sum()
        )
    return out


def aggregate_hourly(df: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    out["data_hora"] = out["timestamp"].dt.floor("h")
    agg = {"temperature": "mean", "humidity": "mean", "itu": "mean", "heat_excess": "mean", "timestamp": "count", "segment_id": "last"}
    for w in windows:
        agg[f"cta_{w}h"] = "last"
    result = out.groupby(["dispositivo", "data_hora"], observed=False, as_index=False).agg(agg)
    result = result.rename(columns={"temperature": "temperatura", "humidity": "umidade", "timestamp": "n_registros_hora"})
    cols = ["data_hora", "dispositivo", "segment_id", "n_registros_hora", "temperatura", "umidade", "itu", "heat_excess", *[f"cta_{w}h" for w in windows]]
    return result.loc[:, cols].sort_values(["dispositivo", "data_hora"])


def build_hourly_cta(input_path: Path, output_path: Path, windows: list[int], threshold: float, freq_min: int, max_gap_min: int, humidity_unit: str, output_csv: Path | None) -> pd.DataFrame:
    heat = prepare_heat_data(input_path, humidity_unit)
    thermal = compute_metrics(heat, threshold, windows, freq_min, max_gap_min)
    hourly = aggregate_hourly(thermal, windows)
    write_table(hourly, output_path)
    if output_csv is not None:
        write_table(hourly, output_csv)
    LOGGER.info("Generated %s hourly rows", len(hourly))
    return hourly


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build hourly CTA table from heat_stress_report")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--output-csv", default=None, type=Path)
    p.add_argument("--windows", nargs="+", type=int, default=[6, 9, 12, 15, 18, 24])
    p.add_argument("--threshold", type=float, default=72.0)
    p.add_argument("--input-frequency-minutes", type=int, default=5)
    p.add_argument("--max-gap-minutes", type=int, default=60)
    p.add_argument("--humidity-unit", choices=["auto", "pct", "fraction"], default="auto")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s", force=True)
    build_hourly_cta(
        input_path=args.input,
        output_path=args.output,
        windows=args.windows,
        threshold=args.threshold,
        freq_min=args.input_frequency_minutes,
        max_gap_min=args.max_gap_minutes,
        humidity_unit=args.humidity_unit,
        output_csv=args.output_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
