#!/usr/bin/env python3
"""Analyze the impact of replacing environmental variables.

This diagnostic script focuses on one question:

    What changed when corrected temperature and humidity were incorporated?

It compares an original monitoring dataset against a corrected/integrated dataset
using the common key ``brinco + data_hora``. For matched records, it compares
raw environmental values, recalculates THI and heat excess in both versions, and
identifies whether the correction changed the thermal classification relative to
the THI threshold.

Outputs include summaries by variable, animal, month, hour of day, validity
transition and temporal continuity.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("environment_correction_impact")

KEY_COLUMNS = ["brinco", "data_hora"]

COLUMN_ALIASES = {
    "animal_id": "brinco",
    "id_animal": "brinco",
    "timestamp": "data_hora",
    "datetime": "data_hora",
    "data": "data_hora",
    "temperatura_compost1": "temperatura_compost_1",
    "temperatura_compost_1": "temperatura_compost_1",
    "humidade_compost1": "humidade_compost_1",
    "humidade_compost_1": "humidade_compost_1",
    "umidade_compost_1": "humidade_compost_1",
    "thi_compost_1": "thi_compost1",
    "thi_compost1": "thi_compost1",
    "ofegacao_hora": "ofegacao_hora",
}

DEFAULT_TEMP_COL = "temperatura_compost_1"
DEFAULT_RH_COL = "humidade_compost_1"
DEFAULT_PANTING_COL = "ofegacao_hora"


def normalize_col(name: str) -> str:
    """Normalize a column name to ASCII snake_case-like format."""
    import unicodedata

    value = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("/", "_")
        .replace("-", "_")
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame columns and apply known aliases."""
    normalized = df.copy()
    normalized.columns = [normalize_col(column) for column in normalized.columns]
    rename_map = {
        column: COLUMN_ALIASES[column]
        for column in normalized.columns
        if column in COLUMN_ALIASES and COLUMN_ALIASES[column] not in normalized.columns
    }
    if rename_map:
        LOGGER.info("Applying aliases: %s", rename_map)
        normalized = normalized.rename(columns=rename_map)
    return normalized


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or Parquet based on extension."""
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path, low_memory=False)

    raise ValueError(f"Unsupported file format: {file_path}. Use .csv or .parquet")


def prepare_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize key columns and collapse duplicate keys."""
    prepared = normalize_columns(df)
    missing = [column for column in KEY_COLUMNS if column not in prepared.columns]
    if missing:
        raise ValueError(f"Missing key columns in {name}: {missing}")

    prepared["brinco"] = prepared["brinco"].astype(str).str.strip()
    prepared["data_hora"] = pd.to_datetime(prepared["data_hora"], errors="coerce")

    before = len(prepared)
    prepared = prepared.dropna(subset=KEY_COLUMNS)
    after = len(prepared)
    if before != after:
        LOGGER.info("%s: removed %s rows with invalid keys.", name, before - after)

    duplicates = int(prepared.duplicated(KEY_COLUMNS).sum())
    if duplicates:
        LOGGER.warning("%s: found %s duplicated keys; keeping the last record.", name, duplicates)
        prepared = prepared.sort_values(KEY_COLUMNS).drop_duplicates(KEY_COLUMNS, keep="last")

    return prepared.sort_values(KEY_COLUMNS).reset_index(drop=True)


def calculate_thi(temperature_c: pd.Series, relative_humidity: pd.Series) -> pd.Series:
    """Calculate THI using the same operational formula used by the pipeline.

    Formula equivalent to common dairy-cattle THI implementation:

        THI = (1.8*T + 32) - (0.55 - 0.0055*RH) * (1.8*T - 26)
    """
    t = pd.to_numeric(temperature_c, errors="coerce")
    rh = pd.to_numeric(relative_humidity, errors="coerce")
    return (1.8 * t + 32.0) - (0.55 - 0.0055 * rh) * (1.8 * t - 26.0)


def add_environment_metrics(
    df: pd.DataFrame,
    *,
    temp_col: str,
    rh_col: str,
    panting_col: str,
    thi_threshold: float,
    suffix: str,
) -> pd.DataFrame:
    """Add numeric environmental metrics for one dataset version."""
    enriched = df.copy()

    for column in [temp_col, rh_col, panting_col]:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    if temp_col not in enriched.columns or rh_col not in enriched.columns:
        raise ValueError(
            f"Missing environmental columns for {suffix}: temp={temp_col!r}, rh={rh_col!r}"
        )

    enriched[f"thi_calc_{suffix}"] = calculate_thi(enriched[temp_col], enriched[rh_col])
    enriched[f"heat_excess_{suffix}"] = np.maximum(
        enriched[f"thi_calc_{suffix}"] - thi_threshold,
        0,
    )
    enriched[f"above_threshold_{suffix}"] = enriched[f"thi_calc_{suffix}"] > thi_threshold
    enriched[f"thermal_valid_{suffix}"] = (
        enriched[temp_col].notna()
        & enriched[rh_col].notna()
        & enriched[panting_col].notna()
        if panting_col in enriched.columns
        else enriched[temp_col].notna() & enriched[rh_col].notna()
    )

    return enriched


def summarize_numeric(series: pd.Series) -> dict[str, float | int | None]:
    """Summarize numeric series."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p05": None,
            "p95": None,
        }
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": float(values.quantile(0.05)),
        "p95": float(values.quantile(0.95)),
    }


def compare_delta(
    matched: pd.DataFrame,
    original_col: str,
    corrected_col: str,
    label: str,
) -> dict[str, object]:
    """Build delta summary for one original/corrected variable pair."""
    valid = matched[[original_col, corrected_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if valid.empty:
        return {"variable": label, "n_paired": 0}

    original = valid[original_col]
    corrected = valid[corrected_col]
    delta = corrected - original
    abs_delta = delta.abs()
    changed = abs_delta > 1e-12

    return {
        "variable": label,
        "n_paired": int(len(valid)),
        "n_changed": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "original_mean": float(original.mean()),
        "corrected_mean": float(corrected.mean()),
        "delta_mean": float(delta.mean()),
        "delta_median": float(delta.median()),
        "delta_std": float(delta.std()),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "abs_delta_mean": float(abs_delta.mean()),
        "abs_delta_p95": float(abs_delta.quantile(0.95)),
        "abs_delta_max": float(abs_delta.max()),
    }


def validity_transition_summary(matched: pd.DataFrame) -> pd.DataFrame:
    """Summarize valid/invalid transitions between versions."""
    original_valid = matched["thermal_valid_original"].fillna(False)
    corrected_valid = matched["thermal_valid_corrected"].fillna(False)

    labels = np.select(
        [
            original_valid & corrected_valid,
            original_valid & ~corrected_valid,
            ~original_valid & corrected_valid,
            ~original_valid & ~corrected_valid,
        ],
        [
            "valid_in_both",
            "valid_only_original",
            "valid_only_corrected",
            "invalid_in_both",
        ],
        default="unknown",
    )

    counts = pd.Series(labels).value_counts().rename_axis("transition").reset_index(name="n")
    counts["fraction"] = counts["n"] / counts["n"].sum()
    return counts


def threshold_transition_summary(matched: pd.DataFrame) -> pd.DataFrame:
    """Summarize THI-threshold transitions after correction."""
    valid = matched[
        matched["thi_calc_original"].notna() & matched["thi_calc_corrected"].notna()
    ].copy()

    if valid.empty:
        return pd.DataFrame(columns=["transition", "n", "fraction"])

    original_above = valid["above_threshold_original"].fillna(False)
    corrected_above = valid["above_threshold_corrected"].fillna(False)

    labels = np.select(
        [
            ~original_above & ~corrected_above,
            ~original_above & corrected_above,
            original_above & ~corrected_above,
            original_above & corrected_above,
        ],
        [
            "below_to_below",
            "below_to_above",
            "above_to_below",
            "above_to_above",
        ],
        default="unknown",
    )

    counts = pd.Series(labels).value_counts().rename_axis("transition").reset_index(name="n")
    counts["fraction"] = counts["n"] / counts["n"].sum()
    return counts


def summarize_by_group(matched: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Summarize deltas by a grouping variable."""
    valid = matched.dropna(subset=["delta_temperature", "delta_humidity", "delta_thi"])
    if valid.empty:
        return pd.DataFrame()

    summary = (
        valid.groupby(group_cols, observed=False)
        .agg(
            n=("delta_thi", "size"),
            mean_delta_temperature=("delta_temperature", "mean"),
            mean_abs_delta_temperature=("delta_temperature", lambda s: s.abs().mean()),
            mean_delta_humidity=("delta_humidity", "mean"),
            mean_abs_delta_humidity=("delta_humidity", lambda s: s.abs().mean()),
            mean_delta_thi=("delta_thi", "mean"),
            mean_abs_delta_thi=("delta_thi", lambda s: s.abs().mean()),
            mean_delta_heat_excess=("delta_heat_excess", "mean"),
            mean_abs_delta_heat_excess=("delta_heat_excess", lambda s: s.abs().mean()),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    return summary


def continuity_summary(df: pd.DataFrame, label: str, *, valid_col: str) -> pd.DataFrame:
    """Summarize temporal continuity of valid records by animal."""
    rows = []
    valid = df[df[valid_col].fillna(False)].copy()

    for animal_id, group in valid.groupby("brinco", observed=False, sort=False):
        times = group["data_hora"].sort_values().dropna()
        if times.empty:
            continue

        diffs = times.diff().dropna().dt.total_seconds() / 3600.0
        gaps_gt_1h = int((diffs > 1.01).sum()) if not diffs.empty else 0
        rows.append(
            {
                "dataset": label,
                "brinco": animal_id,
                "n_valid_records": int(len(times)),
                "time_min": str(times.min()),
                "time_max": str(times.max()),
                "median_gap_h": float(diffs.median()) if not diffs.empty else None,
                "mean_gap_h": float(diffs.mean()) if not diffs.empty else None,
                "max_gap_h": float(diffs.max()) if not diffs.empty else None,
                "n_gaps_gt_1h": gaps_gt_1h,
            }
        )

    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze what changed when corrected temperature/humidity were incorporated."
    )
    parser.add_argument("--original", required=True, type=Path, help="Original dataset CSV/Parquet.")
    parser.add_argument("--corrected", required=True, type=Path, help="Corrected dataset CSV/Parquet.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--temp-col", default=DEFAULT_TEMP_COL, help="Temperature column to compare.")
    parser.add_argument("--rh-col", default=DEFAULT_RH_COL, help="Relative humidity column to compare.")
    parser.add_argument("--panting-col", default=DEFAULT_PANTING_COL, help="Panting column used in validity checks.")
    parser.add_argument("--thi-threshold", type=float, default=72.0, help="THI threshold for heat excess.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
        force=True,
    )

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading original dataset: %s", args.original)
    original = prepare_dataset(read_table(args.original), "original")

    LOGGER.info("Reading corrected dataset: %s", args.corrected)
    corrected = prepare_dataset(read_table(args.corrected), "corrected")

    original = add_environment_metrics(
        original,
        temp_col=args.temp_col,
        rh_col=args.rh_col,
        panting_col=args.panting_col,
        thi_threshold=args.thi_threshold,
        suffix="original",
    )
    corrected = add_environment_metrics(
        corrected,
        temp_col=args.temp_col,
        rh_col=args.rh_col,
        panting_col=args.panting_col,
        thi_threshold=args.thi_threshold,
        suffix="corrected",
    )

    matched = original.merge(
        corrected,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_original", "_corrected"),
    )

    LOGGER.info("Original rows: %s", f"{len(original):,}")
    LOGGER.info("Corrected rows: %s", f"{len(corrected):,}")
    LOGGER.info("Matched rows: %s", f"{len(matched):,}")

    temp_o = f"{args.temp_col}_original"
    temp_c = f"{args.temp_col}_corrected"
    rh_o = f"{args.rh_col}_original"
    rh_c = f"{args.rh_col}_corrected"

    matched["delta_temperature"] = pd.to_numeric(matched[temp_c], errors="coerce") - pd.to_numeric(matched[temp_o], errors="coerce")
    matched["delta_humidity"] = pd.to_numeric(matched[rh_c], errors="coerce") - pd.to_numeric(matched[rh_o], errors="coerce")
    matched["delta_thi"] = matched["thi_calc_corrected"] - matched["thi_calc_original"]
    matched["delta_heat_excess"] = matched["heat_excess_corrected"] - matched["heat_excess_original"]
    matched["month"] = matched["data_hora"].dt.to_period("M").astype(str)
    matched["hour"] = matched["data_hora"].dt.hour

    coverage = {
        "original_rows": int(len(original)),
        "corrected_rows": int(len(corrected)),
        "matched_rows": int(len(matched)),
        "original_unique_keys": int(len(original[KEY_COLUMNS].drop_duplicates())),
        "corrected_unique_keys": int(len(corrected[KEY_COLUMNS].drop_duplicates())),
        "matched_unique_keys": int(len(matched[KEY_COLUMNS].drop_duplicates())),
        "original_animals": int(original["brinco"].nunique(dropna=True)),
        "corrected_animals": int(corrected["brinco"].nunique(dropna=True)),
        "matched_animals": int(matched["brinco"].nunique(dropna=True)),
        "original_valid_rows": int(original["thermal_valid_original"].sum()),
        "corrected_valid_rows": int(corrected["thermal_valid_corrected"].sum()),
        "matched_valid_original_rows": int(matched["thermal_valid_original"].sum()),
        "matched_valid_corrected_rows": int(matched["thermal_valid_corrected"].sum()),
        "original_time_min": str(original["data_hora"].min()) if not original.empty else None,
        "original_time_max": str(original["data_hora"].max()) if not original.empty else None,
        "corrected_time_min": str(corrected["data_hora"].min()) if not corrected.empty else None,
        "corrected_time_max": str(corrected["data_hora"].max()) if not corrected.empty else None,
    }

    with (output_dir / "coverage_and_validity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(coverage, handle, indent=2, ensure_ascii=False)

    delta_summary = pd.DataFrame(
        [
            compare_delta(matched, temp_o, temp_c, args.temp_col),
            compare_delta(matched, rh_o, rh_c, args.rh_col),
            compare_delta(matched, "thi_calc_original", "thi_calc_corrected", "thi_recalculated"),
            compare_delta(matched, "heat_excess_original", "heat_excess_corrected", "heat_excess"),
        ]
    )
    delta_summary.to_csv(output_dir / "environment_delta_summary.csv", index=False)

    validity_transition_summary(matched).to_csv(
        output_dir / "validity_transition_summary.csv",
        index=False,
    )
    threshold_transition_summary(matched).to_csv(
        output_dir / "thi_threshold_transition_summary.csv",
        index=False,
    )

    by_animal = summarize_by_group(matched, ["brinco"])
    by_animal.to_csv(output_dir / "environment_delta_by_animal.csv", index=False)

    by_month = summarize_by_group(matched, ["month"])
    by_month.to_csv(output_dir / "environment_delta_by_month.csv", index=False)

    by_hour = summarize_by_group(matched, ["hour"])
    by_hour.to_csv(output_dir / "environment_delta_by_hour.csv", index=False)

    original_continuity = continuity_summary(original, "original", valid_col="thermal_valid_original")
    corrected_continuity = continuity_summary(corrected, "corrected", valid_col="thermal_valid_corrected")
    pd.concat([original_continuity, corrected_continuity], ignore_index=True).to_csv(
        output_dir / "temporal_continuity_by_animal.csv",
        index=False,
    )

    matched.sort_values(KEY_COLUMNS).to_parquet(output_dir / "matched_environment_impact.parquet", index=False)

    LOGGER.info("Wrote correction-impact reports to: %s", output_dir)
    LOGGER.info("Key outputs: environment_delta_summary.csv, validity_transition_summary.csv, thi_threshold_transition_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
