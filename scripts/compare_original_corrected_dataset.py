#!/usr/bin/env python3
"""Compare original and corrected monitoring datasets.

This script helps separate two effects that can change the selected accumulated
heat-load window:

1. Changes in the numerical environmental values after correction.
2. Changes in dataset composition, coverage, timestamps or valid rows.

It aligns two datasets by ``brinco + data_hora`` and generates:

- coverage summaries;
- paired environmental differences;
- matched subsets for running the thermal pipeline under comparable conditions;
- key-only subsets to test the effect of dataset composition.

Example
-------

python scripts/compare_original_corrected_dataset.py \
  --original "dataset/raw/monitoramento_1293_completo(in).csv" \
  --corrected dataset/processado/monitoramento_saude_unificado.parquet \
  --output-dir outputs_comparacao_original_corrigido
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("compare_original_corrected_dataset")

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
    "temperatura_compost2": "temperatura_compost_2",
    "temperatura_compost_2": "temperatura_compost_2",
    "humidade_compost2": "humidade_compost_2",
    "humidade_compost_2": "humidade_compost_2",
    "umidade_compost_2": "humidade_compost_2",
    "thi_compost_2": "thi_compost2",
    "thi_compost2": "thi_compost2",
}

DEFAULT_COMPARE_COLUMNS = [
    "temperatura_compost_1",
    "humidade_compost_1",
    "thi_compost1",
    "temperatura_compost_2",
    "humidade_compost_2",
    "thi_compost2",
    "ofegacao_hora",
]

THERMAL_PIPELINE_COLUMNS = [
    "brinco",
    "data_hora",
    "status_saude",
    "ruminacao_hora",
    "atividade_hora",
    "ocio_hora",
    "ofegacao_hora",
    "ruminacao_acumulado",
    "atividade_acumulado",
    "ocio_acumulado",
    "ofegacao_acumulado",
    "temperatura_compost_1",
    "humidade_compost_1",
    "thi_compost1",
    "temperatura_compost_2",
    "humidade_compost_2",
    "thi_compost2",
]


def normalize_col(name: str) -> str:
    """Normalize column names to a simple lowercase form."""
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
    """Normalize column names and apply known aliases."""
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
    """Normalize, validate and type-cast a dataset."""
    prepared = normalize_columns(df)

    missing = [column for column in KEY_COLUMNS if column not in prepared.columns]
    if missing:
        raise ValueError(f"Missing key columns in {name}: {missing}")

    prepared["data_hora"] = pd.to_datetime(prepared["data_hora"], errors="coerce")
    prepared["brinco"] = prepared["brinco"].astype(str).str.strip()

    before = len(prepared)
    prepared = prepared.dropna(subset=KEY_COLUMNS)
    after = len(prepared)

    if before != after:
        LOGGER.info("%s: removed %s rows with invalid keys.", name, before - after)

    duplicated = int(prepared.duplicated(KEY_COLUMNS).sum())
    if duplicated:
        LOGGER.warning("%s: found %s duplicated brinco + data_hora keys.", name, duplicated)
        prepared = prepared.sort_values(KEY_COLUMNS).drop_duplicates(KEY_COLUMNS, keep="last")
        LOGGER.warning("%s: duplicated keys were collapsed keeping the last record.", name)

    return prepared.sort_values(KEY_COLUMNS).reset_index(drop=True)


def numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    """Return compact numeric summary for a series."""
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


def build_coverage_summary(
    original: pd.DataFrame,
    corrected: pd.DataFrame,
    matched: pd.DataFrame,
) -> dict[str, object]:
    """Summarize key coverage and overlap."""
    original_keys = original[KEY_COLUMNS].drop_duplicates()
    corrected_keys = corrected[KEY_COLUMNS].drop_duplicates()

    n_original = len(original_keys)
    n_corrected = len(corrected_keys)
    n_overlap = len(matched)

    original_animals = original["brinco"].nunique(dropna=True)
    corrected_animals = corrected["brinco"].nunique(dropna=True)
    overlap_animals = matched["brinco"].nunique(dropna=True) if not matched.empty else 0

    return {
        "original_rows": int(len(original)),
        "corrected_rows": int(len(corrected)),
        "original_unique_keys": int(n_original),
        "corrected_unique_keys": int(n_corrected),
        "overlap_unique_keys": int(n_overlap),
        "original_only_keys": int(n_original - n_overlap),
        "corrected_only_keys": int(n_corrected - n_overlap),
        "overlap_fraction_of_original": float(n_overlap / n_original) if n_original else None,
        "overlap_fraction_of_corrected": float(n_overlap / n_corrected) if n_corrected else None,
        "original_animals": int(original_animals),
        "corrected_animals": int(corrected_animals),
        "overlap_animals": int(overlap_animals),
        "original_time_min": str(original["data_hora"].min()) if not original.empty else None,
        "original_time_max": str(original["data_hora"].max()) if not original.empty else None,
        "corrected_time_min": str(corrected["data_hora"].min()) if not corrected.empty else None,
        "corrected_time_max": str(corrected["data_hora"].max()) if not corrected.empty else None,
    }


def compare_columns(
    matched: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Compare original and corrected values for selected columns."""
    rows = []

    for column in columns:
        original_col = f"{column}_original"
        corrected_col = f"{column}_corrected"

        if original_col not in matched.columns or corrected_col not in matched.columns:
            rows.append(
                {
                    "column": column,
                    "available": False,
                    "reason": "missing in original or corrected dataset",
                }
            )
            continue

        original_values = pd.to_numeric(matched[original_col], errors="coerce")
        corrected_values = pd.to_numeric(matched[corrected_col], errors="coerce")
        valid = pd.concat([original_values, corrected_values], axis=1).dropna()

        if valid.empty:
            rows.append(
                {
                    "column": column,
                    "available": True,
                    "n_paired": 0,
                    "reason": "no numeric paired values",
                }
            )
            continue

        delta = valid.iloc[:, 1] - valid.iloc[:, 0]
        abs_delta = delta.abs()
        changed = abs_delta > 1e-12

        rows.append(
            {
                "column": column,
                "available": True,
                "n_paired": int(len(valid)),
                "n_changed": int(changed.sum()),
                "changed_fraction": float(changed.mean()),
                "mean_original": float(valid.iloc[:, 0].mean()),
                "mean_corrected": float(valid.iloc[:, 1].mean()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "std_delta": float(delta.std()),
                "min_delta": float(delta.min()),
                "max_delta": float(delta.max()),
                "mean_abs_delta": float(abs_delta.mean()),
                "p95_abs_delta": float(abs_delta.quantile(0.95)),
            }
        )

    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save CSV or Parquet based on extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def select_pipeline_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep columns expected by the thermal pipeline when available."""
    cols = [column for column in THERMAL_PIPELINE_COLUMNS if column in df.columns]
    return df.loc[:, cols].copy()


def build_matched_datasets(
    original: pd.DataFrame,
    corrected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return matched original, matched corrected and comparison table."""
    original_keyed = original.set_index(KEY_COLUMNS, drop=False)
    corrected_keyed = corrected.set_index(KEY_COLUMNS, drop=False)

    common_index = original_keyed.index.intersection(corrected_keyed.index)

    original_matched = original_keyed.loc[common_index].reset_index(drop=True)
    corrected_matched = corrected_keyed.loc[common_index].reset_index(drop=True)

    comparison = original_matched.merge(
        corrected_matched,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_original", "_corrected"),
    )

    return original_matched, corrected_matched, comparison


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compare original and corrected monitoring datasets by brinco + data_hora."
    )
    parser.add_argument("--original", required=True, type=Path, help="Original dataset (.csv or .parquet).")
    parser.add_argument("--corrected", required=True, type=Path, help="Corrected dataset (.csv or .parquet).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for comparison files.")
    parser.add_argument(
        "--compare-column",
        action="append",
        default=None,
        help="Column to compare. May be repeated. Defaults to environmental and panting columns.",
    )
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Format for matched datasets. Summary tables are always CSV/JSON.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )
    return parser


def main() -> int:
    """Run command-line comparison."""
    parser = build_parser()
    args = parser.parse_args()

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

    LOGGER.info("Building matched datasets.")
    original_matched, corrected_matched, matched_comparison = build_matched_datasets(
        original,
        corrected,
    )

    compare_columns_list = args.compare_column or DEFAULT_COMPARE_COLUMNS

    coverage_summary = build_coverage_summary(original, corrected, original_matched)
    comparison_summary = compare_columns(matched_comparison, compare_columns_list)

    LOGGER.info("Original rows: %s", f"{coverage_summary['original_rows']:,}")
    LOGGER.info("Corrected rows: %s", f"{coverage_summary['corrected_rows']:,}")
    LOGGER.info("Overlapping keys: %s", f"{coverage_summary['overlap_unique_keys']:,}")

    with (output_dir / "coverage_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(coverage_summary, handle, indent=2, ensure_ascii=False)

    comparison_summary.to_csv(output_dir / "value_difference_summary.csv", index=False)

    suffix = args.output_format
    save_table(
        select_pipeline_columns(original_matched),
        output_dir / f"original_matched_for_pipeline.{suffix}",
    )
    save_table(
        select_pipeline_columns(corrected_matched),
        output_dir / f"corrected_matched_for_pipeline.{suffix}",
    )
    save_table(
        matched_comparison,
        output_dir / f"matched_comparison_wide.{suffix}",
    )

    original_only = original.merge(corrected[KEY_COLUMNS], on=KEY_COLUMNS, how="left", indicator=True)
    original_only = original_only[original_only["_merge"] == "left_only"].drop(columns="_merge")
    corrected_only = corrected.merge(original[KEY_COLUMNS], on=KEY_COLUMNS, how="left", indicator=True)
    corrected_only = corrected_only[corrected_only["_merge"] == "left_only"].drop(columns="_merge")

    save_table(select_pipeline_columns(original_only), output_dir / f"original_only.{suffix}")
    save_table(select_pipeline_columns(corrected_only), output_dir / f"corrected_only.{suffix}")

    LOGGER.info("Wrote comparison outputs to: %s", output_dir)
    LOGGER.info("Use original_matched_for_pipeline and corrected_matched_for_pipeline to rerun comparable thermal analyses.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
