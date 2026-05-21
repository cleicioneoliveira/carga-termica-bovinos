#!/usr/bin/env python3
"""Build hourly accumulated heat-load series from heat_stress_report.

This script is intended for the environmental stage of the dissertation
workflow. It reads the sub-hourly heat_stress_report used by the
``environment_correction`` repository, computes ITU, heat excess and
accumulated heat load using time weighting, then aggregates the result to an
hourly environmental table.

The output does not include animal-level behavioral variables. It should be
joined later with the hourly animal monitoring dataset containing panting,
rumination, activity and idle time.

Expected heat_stress_report columns after normalization:

- timestamp
- dispositivo
- temperature
- humidity

Accepted aliases include data, data_hora, datetime, device, sensor,
temperatura and umidade.
"""

from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("build_hourly_cta_from_heat_stress")

COLUMN_ALIASES = {
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

REQUIRED_COLUMNS = ("timestamp", "dispositivo", "temperature", "humidity")


def normalize_col(name: str) -> str:
    """Normalize a column name to ASCII snake-like lowercase."""
    normalized = unicodedata.normalize("NFKD", str(name))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return (
        normalized.strip()
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("/", "_")
        .replace("-", "_")
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize heat_stress_report columns and apply known aliases."""
    normalized = df.copy()
    normalized.columns = [normalize_col(column) for column in normalized.columns]

    rename_map = {
        column: COLUMN_ALIASES[column]
        for column in normalized.columns
        if column in COLUMN_ALIASES and COLUMN_ALIASES[column] not in normalized.columns
    }

    if rename_map:
        LOGGER.info("Renaming columns: %s", rename_map)
        normalized = normalized.rename(columns=rename_map)

    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(
            "Missing required heat_stress_report columns after normalization: "
            f"{missing}. Available columns: {list(normalized.columns)}"
        )

    return normalized


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet table based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}. Use .csv or .parquet.")


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write CSV or Parquet table based on file extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {path}. Use .csv or .parquet.")


def calculate_itu(temp_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """Calculate ITU using the same Fahrenheit-based formula used by the pipeline."""
    temp_f = (1.8 * temp_c.astype(float)) + 32.0
    humidity_fraction = humidity_pct.astype(float) / 100.0
    return temp_f - (0.55 - 0.55 * humidity_fraction) * (temp_f - 58.0)


def prepare_heat_data(path: Path, humidity_unit: str) -> pd.DataFrame:
    """Load, normalize and clean the heat_stress_report."""
    LOGGER.info("Reading heat_stress_report: %s", path)
    df = read_table(path)
    df = normalize_columns(df)

    prepared = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    prepared["temperature"] = pd.to_numeric(prepared["temperature"], errors="coerce")
    prepared["humidity"] = pd.to_numeric(prepared["humidity"], errors="coerce")

    before = len(prepared)
    prepared = prepared.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    removed = before - len(prepared)
    if removed:
        LOGGER.warning("Removed %s invalid rows from heat_stress_report.", removed)

    if humidity_unit == "auto":
        q95 = prepared["humidity"].quantile(0.95)
        if q95 <= 1.5:
            LOGGER.info("Humidity appears to be fractional. Converting to percent.")
            prepared["humidity"] = prepared["humidity"] * 100.0
    elif humidity_unit == "fraction":
        LOGGER.info("Converting humidity from fraction to percent.")
        prepared["humidity"] = prepared["humidity"] * 100.0
    elif humidity_unit != "pct":
        raise ValueError("humidity_unit must be one of: auto, pct, fraction")

    prepared = prepared.sort_values(["dispositivo", "timestamp"]).reset_index(drop=True)
    return prepared


def compute_thermal_metrics(
    df: pd.DataFrame,
    threshold: float,
    windows: Iterable[int],
    input_frequency_minutes: int,
) -> pd.DataFrame:
    """Compute ITU, heat excess and time-weighted CTA by device."""
    enriched = df.copy()
    delta_t_hours = input_frequency_minutes / 60.0

    if input_frequency_minutes <= 0:
        raise ValueError("input_frequency_minutes must be greater than zero.")
    if 60 % input_frequency_minutes != 0:
        raise ValueError("input_frequency_minutes must divide 60 exactly.")

    records_per_hour = int(60 / input_frequency_minutes)

    enriched["itu"] = calculate_itu(enriched["temperature"], enriched["humidity"])
    enriched["heat_excess"] = np.maximum(enriched["itu"] - threshold, 0.0)

    for window in windows:
        if window <= 0:
            raise ValueError(f"Window must be positive. Received: {window}")

        window_records = int(window * records_per_hour)
        column = f"cta_{window}h"
        LOGGER.info(
            "Computing %s using %s records and dt=%s h.",
            column,
            window_records,
            delta_t_hours,
        )

        enriched[column] = (
            enriched.groupby("dispositivo", observed=False)["heat_excess"]
            .transform(
                lambda series, wr=window_records: (series * delta_t_hours)
                .rolling(wr, min_periods=1)
                .sum()
            )
        )

    return enriched


def aggregate_hourly(df: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    """Aggregate sub-hourly thermal metrics to hourly records by device."""
    hourly = df.copy()
    hourly["data_hora"] = hourly["timestamp"].dt.floor("h")

    agg_map: dict[str, str] = {
        "temperature": "mean",
        "humidity": "mean",
        "itu": "mean",
        "heat_excess": "mean",
    }
    for window in windows:
        agg_map[f"cta_{window}h"] = "last"

    result = (
        hourly.groupby(["dispositivo", "data_hora"], observed=False, as_index=False)
        .agg(agg_map)
        .rename(columns={"temperature": "temperatura", "humidity": "umidade"})
    )

    ordered_columns = [
        "data_hora",
        "dispositivo",
        "temperatura",
        "umidade",
        "itu",
        "heat_excess",
        *[f"cta_{window}h" for window in windows],
    ]
    return result.loc[:, ordered_columns].sort_values(["dispositivo", "data_hora"])


def build_hourly_cta(
    input_path: Path,
    output_path: Path,
    windows: list[int],
    threshold: float,
    input_frequency_minutes: int,
    humidity_unit: str,
    output_csv: Path | None,
) -> pd.DataFrame:
    """Build and save hourly CTA table from heat_stress_report."""
    heat = prepare_heat_data(input_path, humidity_unit=humidity_unit)
    thermal = compute_thermal_metrics(
        heat,
        threshold=threshold,
        windows=windows,
        input_frequency_minutes=input_frequency_minutes,
    )
    hourly = aggregate_hourly(thermal, windows=windows)

    LOGGER.info("Writing hourly CTA table: %s", output_path)
    write_table(hourly, output_path)

    if output_csv is not None:
        LOGGER.info("Writing hourly CTA CSV copy: %s", output_csv)
        write_table(hourly, output_csv)

    LOGGER.info("Generated %s hourly rows.", f"{len(hourly):,}")
    return hourly


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build hourly CTA table from heat_stress_report sub-hourly data."
    )
    parser.add_argument("--input", required=True, type=Path, help="heat_stress_report CSV/Parquet.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output hourly CTA table (.parquet or .csv).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        type=Path,
        help="Optional CSV copy of the output table.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[6, 9, 12, 15, 18, 24],
        help="Accumulation windows in hours.",
    )
    parser.add_argument("--threshold", type=float, default=72.0, help="ITU threshold.")
    parser.add_argument(
        "--input-frequency-minutes",
        type=int,
        default=5,
        help="Minutes represented by each heat_stress_report record.",
    )
    parser.add_argument(
        "--humidity-unit",
        choices=["auto", "pct", "fraction"],
        default="auto",
        help="Unit of relative humidity in the heat file.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
        force=True,
    )

    build_hourly_cta(
        input_path=args.input,
        output_path=args.output,
        output_csv=args.output_csv,
        windows=args.windows,
        threshold=args.threshold,
        input_frequency_minutes=args.input_frequency_minutes,
        humidity_unit=args.humidity_unit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
