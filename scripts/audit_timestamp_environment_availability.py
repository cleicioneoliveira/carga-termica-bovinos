#!/usr/bin/env python3
"""Audit timestamp preservation and environmental availability by animal.

This diagnostic separates two different mechanisms that may increase the amount
of valid thermal data after correction:

1. Existing animal timestamps in the original dataset where temperature and/or
   humidity were missing and became available after correction.
2. New animal timestamps present only in the corrected dataset.

The audit is performed by ``brinco + data_hora`` and also by ``brinco + date``
to distinguish completely new animal-days from new timestamps inserted inside
animal-days that already existed in the original dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("timestamp_environment_audit")

KEY_COLUMNS = ["brinco", "data_hora"]
DAY_KEY_COLUMNS = ["brinco", "data"]

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
DEFAULT_BEHAVIOR_COLS = [
    "ruminacao_hora",
    "atividade_hora",
    "ocio_hora",
    "ofegacao_hora",
    "ruminacao_acumulado",
    "atividade_acumulado",
    "ocio_acumulado",
    "ofegacao_acumulado",
]


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
    """Read CSV or Parquet by extension."""
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path, low_memory=False)

    raise ValueError(f"Unsupported file format: {file_path}. Use .csv or .parquet")


def prepare_dataset(
    df: pd.DataFrame,
    name: str,
    *,
    temp_col: str,
    rh_col: str,
    behavior_cols: list[str],
) -> pd.DataFrame:
    """Normalize keys and add availability flags."""
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

    for column in [temp_col, rh_col, *behavior_cols]:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared["data"] = prepared["data_hora"].dt.date.astype(str)
    prepared[f"has_{temp_col}"] = prepared[temp_col].notna() if temp_col in prepared.columns else False
    prepared[f"has_{rh_col}"] = prepared[rh_col].notna() if rh_col in prepared.columns else False
    prepared["has_temperature_humidity"] = prepared[f"has_{temp_col}"] & prepared[f"has_{rh_col}"]

    available_behavior = [column for column in behavior_cols if column in prepared.columns]
    if available_behavior:
        prepared["has_any_behavior"] = prepared[available_behavior].notna().any(axis=1)
        prepared["has_all_behavior"] = prepared[available_behavior].notna().all(axis=1)
    else:
        prepared["has_any_behavior"] = False
        prepared["has_all_behavior"] = False

    prepared["has_environment_or_behavior"] = prepared["has_temperature_humidity"] | prepared["has_any_behavior"]

    return prepared.sort_values(KEY_COLUMNS).reset_index(drop=True)


def nearest_original_timestamp_context(
    corrected_only: pd.DataFrame,
    original: pd.DataFrame,
) -> pd.DataFrame:
    """Add context for corrected-only timestamps.

    For each corrected-only timestamp, the script checks whether the same animal-day
    existed in the original dataset and computes the nearest original timestamp for
    the same animal.
    """
    if corrected_only.empty:
        return corrected_only.copy()

    original_days = original[DAY_KEY_COLUMNS].drop_duplicates()
    context = corrected_only.merge(
        original_days.assign(original_day_exists=True),
        on=DAY_KEY_COLUMNS,
        how="left",
    )
    context["original_day_exists"] = context["original_day_exists"].fillna(False)

    rows = []
    original_by_animal = {
        animal_id: group["data_hora"].sort_values().to_numpy()
        for animal_id, group in original.groupby("brinco", observed=False, sort=False)
    }

    for _, row in context.iterrows():
        animal_id = row["brinco"]
        timestamp = row["data_hora"]
        times = original_by_animal.get(animal_id)

        nearest_time = pd.NaT
        nearest_gap_h = np.nan

        if times is not None and len(times) > 0:
            values = pd.to_datetime(times)
            idx = np.searchsorted(values, timestamp)
            candidates = []
            if idx > 0:
                candidates.append(values[idx - 1])
            if idx < len(values):
                candidates.append(values[idx])
            if candidates:
                nearest_time = min(candidates, key=lambda x: abs((timestamp - x).total_seconds()))
                nearest_gap_h = abs((timestamp - nearest_time).total_seconds()) / 3600.0

        rows.append(
            {
                **row.to_dict(),
                "nearest_original_timestamp_same_animal": nearest_time,
                "nearest_original_gap_h": nearest_gap_h,
            }
        )

    return pd.DataFrame(rows)


def build_matched_transition_summary(matched: pd.DataFrame) -> pd.DataFrame:
    """Summarize availability transitions for matched timestamps."""
    original_env = matched["has_temperature_humidity_original"].fillna(False)
    corrected_env = matched["has_temperature_humidity_corrected"].fillna(False)
    original_behavior = matched["has_any_behavior_original"].fillna(False)

    transition = np.select(
        [
            original_env & corrected_env,
            ~original_env & corrected_env,
            original_env & ~corrected_env,
            ~original_env & ~corrected_env,
        ],
        [
            "env_present_in_both",
            "env_missing_original_present_corrected",
            "env_present_original_missing_corrected",
            "env_missing_in_both",
        ],
        default="unknown",
    )

    existing_record_with_behavior_became_env_valid = (~original_env) & corrected_env & original_behavior

    summary = pd.Series(transition).value_counts().rename_axis("transition").reset_index(name="n")
    summary["fraction"] = summary["n"] / summary["n"].sum()

    extra = pd.DataFrame(
        [
            {
                "transition": "original_record_had_behavior_but_no_env_and_corrected_added_env",
                "n": int(existing_record_with_behavior_became_env_valid.sum()),
                "fraction": float(existing_record_with_behavior_became_env_valid.mean()) if len(matched) else 0.0,
            }
        ]
    )

    return pd.concat([summary, extra], ignore_index=True)


def build_by_animal_summary(
    original: pd.DataFrame,
    corrected: pd.DataFrame,
    matched: pd.DataFrame,
    corrected_only_context: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-animal timestamp and environment availability summary."""
    original_group = original.groupby("brinco", observed=False)
    corrected_group = corrected.groupby("brinco", observed=False)
    matched_group = matched.groupby("brinco", observed=False)
    corrected_only_group = corrected_only_context.groupby("brinco", observed=False)

    animals = sorted(set(original["brinco"]).union(set(corrected["brinco"])))
    rows = []

    for animal_id in animals:
        original_g = original_group.get_group(animal_id) if animal_id in original_group.groups else pd.DataFrame()
        corrected_g = corrected_group.get_group(animal_id) if animal_id in corrected_group.groups else pd.DataFrame()
        matched_g = matched_group.get_group(animal_id) if animal_id in matched_group.groups else pd.DataFrame()
        corrected_only_g = corrected_only_group.get_group(animal_id) if animal_id in corrected_only_group.groups else pd.DataFrame()

        original_env = int(original_g["has_temperature_humidity"].sum()) if not original_g.empty else 0
        corrected_env = int(corrected_g["has_temperature_humidity"].sum()) if not corrected_g.empty else 0
        matched_original_missing_corrected_present = 0
        matched_original_missing_behavior_corrected_present = 0

        if not matched_g.empty:
            original_missing = ~matched_g["has_temperature_humidity_original"].fillna(False)
            corrected_present = matched_g["has_temperature_humidity_corrected"].fillna(False)
            original_behavior = matched_g["has_any_behavior_original"].fillna(False)
            matched_original_missing_corrected_present = int((original_missing & corrected_present).sum())
            matched_original_missing_behavior_corrected_present = int((original_missing & corrected_present & original_behavior).sum())

        rows.append(
            {
                "brinco": animal_id,
                "original_timestamps": int(len(original_g)),
                "corrected_timestamps": int(len(corrected_g)),
                "matched_timestamps": int(len(matched_g)),
                "corrected_only_timestamps": int(len(corrected_only_g)),
                "corrected_only_same_original_day": int(corrected_only_g["original_day_exists"].sum()) if not corrected_only_g.empty else 0,
                "corrected_only_new_original_day": int((~corrected_only_g["original_day_exists"]).sum()) if not corrected_only_g.empty else 0,
                "original_env_complete_timestamps": original_env,
                "corrected_env_complete_timestamps": corrected_env,
                "gain_env_complete_timestamps": corrected_env - original_env,
                "matched_original_env_missing_corrected_present": matched_original_missing_corrected_present,
                "matched_original_had_behavior_no_env_corrected_env_present": matched_original_missing_behavior_corrected_present,
            }
        )

    return pd.DataFrame(rows).sort_values("gain_env_complete_timestamps", ascending=False)


def build_by_day_summary(original: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    """Build per animal-day summary of timestamp availability."""
    original_day = (
        original.groupby(DAY_KEY_COLUMNS, observed=False)
        .agg(
            original_timestamps=("data_hora", "size"),
            original_env_complete=("has_temperature_humidity", "sum"),
            original_any_behavior=("has_any_behavior", "sum"),
        )
        .reset_index()
    )

    corrected_day = (
        corrected.groupby(DAY_KEY_COLUMNS, observed=False)
        .agg(
            corrected_timestamps=("data_hora", "size"),
            corrected_env_complete=("has_temperature_humidity", "sum"),
            corrected_any_behavior=("has_any_behavior", "sum"),
        )
        .reset_index()
    )

    merged = original_day.merge(corrected_day, on=DAY_KEY_COLUMNS, how="outer").fillna(0)
    numeric_cols = [
        "original_timestamps",
        "original_env_complete",
        "original_any_behavior",
        "corrected_timestamps",
        "corrected_env_complete",
        "corrected_any_behavior",
    ]
    for column in numeric_cols:
        merged[column] = merged[column].astype(int)

    merged["timestamp_gain"] = merged["corrected_timestamps"] - merged["original_timestamps"]
    merged["env_complete_gain"] = merged["corrected_env_complete"] - merged["original_env_complete"]
    merged["day_existed_in_original"] = merged["original_timestamps"] > 0
    merged["day_existed_in_corrected"] = merged["corrected_timestamps"] > 0

    return merged.sort_values(["env_complete_gain", "timestamp_gain"], ascending=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit whether corrected environmental values filled existing timestamps or introduced new animal timestamps."
    )
    parser.add_argument("--original", required=True, type=Path, help="Original dataset CSV/Parquet.")
    parser.add_argument("--corrected", required=True, type=Path, help="Corrected dataset CSV/Parquet.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--temp-col", default=DEFAULT_TEMP_COL, help="Temperature column name.")
    parser.add_argument("--rh-col", default=DEFAULT_RH_COL, help="Relative humidity column name.")
    parser.add_argument(
        "--behavior-col",
        action="append",
        default=None,
        help="Behavioral column used to detect that a record existed even without T/RH. May be repeated.",
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Format for detailed outputs.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )
    return parser


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
        force=True,
    )

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    behavior_cols = args.behavior_col or DEFAULT_BEHAVIOR_COLS

    LOGGER.info("Reading original dataset: %s", args.original)
    original = prepare_dataset(
        read_table(args.original),
        "original",
        temp_col=args.temp_col,
        rh_col=args.rh_col,
        behavior_cols=behavior_cols,
    )

    LOGGER.info("Reading corrected dataset: %s", args.corrected)
    corrected = prepare_dataset(
        read_table(args.corrected),
        "corrected",
        temp_col=args.temp_col,
        rh_col=args.rh_col,
        behavior_cols=behavior_cols,
    )

    matched = original.merge(
        corrected,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_original", "_corrected"),
    )

    original_keys = original[KEY_COLUMNS].drop_duplicates()
    corrected_keys = corrected[KEY_COLUMNS].drop_duplicates()

    corrected_only = corrected.merge(original_keys, on=KEY_COLUMNS, how="left", indicator=True)
    corrected_only = corrected_only[corrected_only["_merge"] == "left_only"].drop(columns="_merge")
    original_only = original.merge(corrected_keys, on=KEY_COLUMNS, how="left", indicator=True)
    original_only = original_only[original_only["_merge"] == "left_only"].drop(columns="_merge")

    corrected_only_context = nearest_original_timestamp_context(corrected_only, original)

    transition_summary = build_matched_transition_summary(matched)
    by_animal = build_by_animal_summary(original, corrected, matched, corrected_only_context)
    by_day = build_by_day_summary(original, corrected)

    summary = {
        "original_rows": int(len(original)),
        "corrected_rows": int(len(corrected)),
        "matched_timestamps": int(len(matched)),
        "original_only_timestamps": int(len(original_only)),
        "corrected_only_timestamps": int(len(corrected_only)),
        "original_env_complete_timestamps": int(original["has_temperature_humidity"].sum()),
        "corrected_env_complete_timestamps": int(corrected["has_temperature_humidity"].sum()),
        "matched_original_env_complete_timestamps": int(matched["has_temperature_humidity_original"].sum()),
        "matched_corrected_env_complete_timestamps": int(matched["has_temperature_humidity_corrected"].sum()),
        "matched_original_env_missing_corrected_present": int((~matched["has_temperature_humidity_original"].fillna(False) & matched["has_temperature_humidity_corrected"].fillna(False)).sum()),
        "matched_original_had_behavior_no_env_corrected_env_present": int((~matched["has_temperature_humidity_original"].fillna(False) & matched["has_temperature_humidity_corrected"].fillna(False) & matched["has_any_behavior_original"].fillna(False)).sum()),
        "corrected_only_same_original_day": int(corrected_only_context["original_day_exists"].sum()) if not corrected_only_context.empty else 0,
        "corrected_only_new_original_day": int((~corrected_only_context["original_day_exists"]).sum()) if not corrected_only_context.empty else 0,
        "original_animals": int(original["brinco"].nunique(dropna=True)),
        "corrected_animals": int(corrected["brinco"].nunique(dropna=True)),
    }

    with (output_dir / "timestamp_environment_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    transition_summary.to_csv(output_dir / "matched_environment_availability_transitions.csv", index=False)
    by_animal.to_csv(output_dir / "timestamp_environment_by_animal.csv", index=False)
    by_day.to_csv(output_dir / "timestamp_environment_by_animal_day.csv", index=False)

    suffix = args.output_format
    save_table(corrected_only_context, output_dir / f"corrected_only_timestamp_context.{suffix}")
    save_table(original_only, output_dir / f"original_only_timestamp_context.{suffix}")
    save_table(matched, output_dir / f"matched_timestamp_environment_flags.{suffix}")

    LOGGER.info("Original rows: %s", f"{len(original):,}")
    LOGGER.info("Corrected rows: %s", f"{len(corrected):,}")
    LOGGER.info("Matched timestamps: %s", f"{len(matched):,}")
    LOGGER.info("Corrected-only timestamps: %s", f"{len(corrected_only):,}")
    LOGGER.info(
        "Existing original timestamps that had behavior but no T/RH and gained T/RH after correction: %s",
        f"{summary['matched_original_had_behavior_no_env_corrected_env_present']:,}",
    )
    LOGGER.info("Wrote timestamp/environment audit outputs to: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
