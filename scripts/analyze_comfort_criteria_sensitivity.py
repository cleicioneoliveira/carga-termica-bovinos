#!/usr/bin/env python3
"""Evaluate sensitivity of the empirical comfort definition.

The reference definition used in the dissertation combines, per animal:

- CTA at or below the individual 25th percentile;
- panting at or below the individual 25th percentile;
- at least three consecutive hourly records.

This script repeats the classification for combinations of individual
percentiles and minimum persistence durations. It reports how the number of
records, animals, continuous blocks and environmental ranges change relative
to the reference P25 + 3 h definition.

The analysis is internal and descriptive. It does not validate a physiological
or universal comfort zone.

Example
-------
python scripts/analyze_comfort_criteria_sensitivity.py \
  --dataset dataset/processado/monitoramento_saude_cta.parquet \
  --output-dir resultados_dissertacao/sensibilidade_conforto \
  --cta-window 19 \
  --percentiles 0.20 0.25 0.30 \
  --durations 2 3 4
"""

from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("comfort_criteria_sensitivity")


def normalize_col(name: object) -> str:
    """Normalize a column name to lower-case ASCII snake case."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return text.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet dataset."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported dataset format: {path}. Use .parquet or .csv.")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Wrote %s", path)


def scenario_label(percentile: float, duration: int) -> str:
    return f"p{int(round(percentile * 100)):02d}_{duration}h"


def humidity_ratio(
    temperature_c: pd.Series,
    relative_humidity_pct: pd.Series,
    pressure_kpa: float,
) -> pd.Series:
    """Calculate humidity ratio in kg kg-1 of dry air."""
    temperature = pd.to_numeric(temperature_c, errors="coerce")
    relative_humidity = pd.to_numeric(relative_humidity_pct, errors="coerce")
    saturation_pressure = 0.6108 * np.exp(
        (17.27 * temperature) / (temperature + 237.3)
    )
    partial_pressure = (relative_humidity / 100.0) * saturation_pressure
    denominator = pressure_kpa - partial_pressure
    result = 0.622 * partial_pressure / denominator
    return result.where((relative_humidity >= 0) & (relative_humidity <= 100) & (denominator > 0))


def build_quantile_flags(
    ordered: pd.DataFrame,
    animal_col: str,
    cta_col: str,
    panting_col: str,
    percentile: float,
) -> pd.DataFrame:
    """Add individual thresholds and a comfort flag for one percentile."""
    result = ordered.copy()
    cta_threshold = result.groupby(animal_col, observed=False)[cta_col].transform(
        lambda values: pd.to_numeric(values, errors="coerce").quantile(percentile)
    )
    panting_threshold = result.groupby(animal_col, observed=False)[panting_col].transform(
        lambda values: pd.to_numeric(values, errors="coerce").quantile(percentile)
    )

    result["cta_threshold"] = cta_threshold
    result["panting_threshold"] = panting_threshold
    result["comfort_flag"] = (
        result[cta_col].notna()
        & result[panting_col].notna()
        & result["cta_threshold"].notna()
        & result["panting_threshold"].notna()
        & (result[cta_col] <= result["cta_threshold"])
        & (result[panting_col] <= result["panting_threshold"])
    )
    return result


def assign_continuous_blocks(
    flagged: pd.DataFrame,
    animal_col: str,
    time_col: str,
    expected_interval_minutes: int,
) -> pd.DataFrame:
    """Split blocks on comfort changes or temporal discontinuities."""
    result = flagged.copy()
    expected_delta = pd.Timedelta(int(expected_interval_minutes), unit="min")

    flag_change = result.groupby(animal_col, observed=False)["comfort_flag"].transform(
        lambda values: values.ne(values.shift()).fillna(True)
    )
    temporal_break = result.groupby(animal_col, observed=False)[time_col].diff().ne(
        expected_delta
    )
    new_block = (flag_change | temporal_break).astype(int)
    result["block_id"] = new_block.groupby(result[animal_col], observed=False).cumsum()
    return result


def summarize_numeric(series: pd.Series, prefix: str) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_min": np.nan,
            f"{prefix}_p05": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_p05": float(values.quantile(0.05)),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_p95": float(values.quantile(0.95)),
        f"{prefix}_max": float(values.max()),
    }


def select_valid_comfort(
    blocked: pd.DataFrame,
    animal_col: str,
    duration_records: int,
    expected_interval_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return selected records and one-row-per-block metadata."""
    block_summary = (
        blocked.groupby([animal_col, "block_id"], observed=False)
        .agg(
            comfort_flag=("comfort_flag", "first"),
            n_records=("comfort_flag", "size"),
            start_time=("data_hora_normalized", "min"),
            end_time=("data_hora_normalized", "max"),
        )
        .reset_index()
    )
    block_summary["duration_h"] = (
        block_summary["n_records"] * expected_interval_minutes / 60.0
    )
    valid_blocks = block_summary[
        block_summary["comfort_flag"].fillna(False)
        & (block_summary["n_records"] >= duration_records)
    ].copy()

    if valid_blocks.empty:
        return blocked.iloc[0:0].copy(), valid_blocks

    selected = blocked.merge(
        valid_blocks[[animal_col, "block_id", "duration_h"]],
        on=[animal_col, "block_id"],
        how="inner",
    )
    return selected, valid_blocks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate sensitivity of empirical comfort criteria."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--animal-col", default="brinco")
    parser.add_argument("--time-col", default="data_hora")
    parser.add_argument("--panting-col", default="ofegacao_hora")
    parser.add_argument("--temperature-col", default="temperatura")
    parser.add_argument("--humidity-col", default="umidade")
    parser.add_argument("--cta-window", type=int, default=19)
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[0.20, 0.25, 0.30],
    )
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Minimum persistence in hours.",
    )
    parser.add_argument("--reference-percentile", type=float, default=0.25)
    parser.add_argument("--reference-duration", type=int, default=3)
    parser.add_argument("--expected-interval-minutes", type=int, default=60)
    parser.add_argument(
        "--pressure-kpa",
        type=float,
        default=101.325,
        help="Atmospheric pressure used only to calculate humidity ratio.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
        force=True,
    )

    if args.cta_window <= 0:
        raise ValueError("--cta-window must be greater than zero.")
    if args.expected_interval_minutes <= 0:
        raise ValueError("--expected-interval-minutes must be greater than zero.")
    if args.pressure_kpa <= 0:
        raise ValueError("--pressure-kpa must be greater than zero.")
    if not args.percentiles or any(value <= 0 or value >= 1 for value in args.percentiles):
        raise ValueError("All --percentiles must be between 0 and 1.")
    if not args.durations or any(value <= 0 for value in args.durations):
        raise ValueError("All --durations must be positive integers.")

    percentiles = sorted(set([*args.percentiles, args.reference_percentile]))
    durations = sorted(set([*args.durations, args.reference_duration]))

    df = read_table(args.dataset)
    df = df.rename(columns={column: normalize_col(column) for column in df.columns})

    animal_col = normalize_col(args.animal_col)
    time_col = normalize_col(args.time_col)
    panting_col = normalize_col(args.panting_col)
    temperature_col = normalize_col(args.temperature_col)
    humidity_col = normalize_col(args.humidity_col)
    cta_col = f"cta_{args.cta_window}h"

    required = [animal_col, time_col, panting_col, temperature_col, humidity_col, cta_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ordered = df[required].copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col], errors="coerce")
    for column in [panting_col, temperature_col, humidity_col, cta_col]:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    before_cleaning = len(ordered)
    ordered = ordered.dropna(subset=required).sort_values(
        [animal_col, time_col]
    ).reset_index(drop=True)

    LOGGER.info(
        "Removed %s rows with incomplete required data.",
        f"{before_cleaning - len(ordered):,}",
    )
    ordered = ordered.rename(columns={time_col: "data_hora_normalized"})
    ordered["humidity_ratio_kg_kg"] = humidity_ratio(
        ordered[temperature_col], ordered[humidity_col], args.pressure_kpa
    )

    duration_records = {
        duration: int(np.ceil(duration * 60 / args.expected_interval_minutes))
        for duration in durations
    }

    scenario_records: dict[str, pd.DataFrame] = {}
    scenario_blocks: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    LOGGER.info("Dataset rows: %s", f"{len(ordered):,}")
    LOGGER.info("CTA column: %s", cta_col)
    LOGGER.info("Percentiles: %s", ", ".join(f"{value:.2f}" for value in percentiles))
    LOGGER.info("Durations: %s h", ", ".join(str(value) for value in durations))

    for percentile in percentiles:
        flagged = build_quantile_flags(
            ordered=ordered,
            animal_col=animal_col,
            cta_col=cta_col,
            panting_col=panting_col,
            percentile=percentile,
        )
        blocked = assign_continuous_blocks(
            flagged=flagged,
            animal_col=animal_col,
            time_col="data_hora_normalized",
            expected_interval_minutes=args.expected_interval_minutes,
        )

        for duration in durations:
            label = scenario_label(percentile, duration)
            selected, blocks = select_valid_comfort(
                blocked=blocked,
                animal_col=animal_col,
                duration_records=duration_records[duration],
                expected_interval_minutes=args.expected_interval_minutes,
            )
            scenario_records[label] = selected

            blocks = blocks.copy()
            blocks.insert(0, "scenario", label)
            blocks.insert(1, "percentile", percentile)
            blocks.insert(2, "min_duration_h", duration)
            scenario_blocks.append(blocks)

            row: dict[str, object] = {
                "scenario": label,
                "percentile": percentile,
                "min_duration_h": duration,
                "n_records": int(len(selected)),
                "pct_dataset_records": float(100 * len(selected) / len(ordered)) if len(ordered) else np.nan,
                "n_animals": int(selected[animal_col].nunique()),
                "n_blocks": int(len(blocks)),
                "mean_block_duration_h": float(blocks["duration_h"].mean()) if not blocks.empty else np.nan,
                "median_block_duration_h": float(blocks["duration_h"].median()) if not blocks.empty else np.nan,
                "max_block_duration_h": float(blocks["duration_h"].max()) if not blocks.empty else np.nan,
            }
            row.update(summarize_numeric(selected[temperature_col], "temperature_c"))
            row.update(summarize_numeric(selected[humidity_col], "relative_humidity_pct"))
            row.update(summarize_numeric(selected["humidity_ratio_kg_kg"], "humidity_ratio_kg_kg"))
            row.update(summarize_numeric(selected[cta_col], cta_col))
            row.update(summarize_numeric(selected[panting_col], panting_col))
            summary_rows.append(row)

            LOGGER.info(
                "%s: records=%s animals=%s blocks=%s",
                label,
                f"{len(selected):,}",
                selected[animal_col].nunique(),
                len(blocks),
            )

    reference_label = scenario_label(
        args.reference_percentile, args.reference_duration
    )
    if reference_label not in scenario_records:
        raise RuntimeError(f"Reference scenario {reference_label} was not generated.")

    def record_keys(frame: pd.DataFrame) -> pd.MultiIndex:
        if frame.empty:
            return pd.MultiIndex.from_arrays([[], []], names=[animal_col, "data_hora_normalized"])
        return pd.MultiIndex.from_frame(
            frame[[animal_col, "data_hora_normalized"]].drop_duplicates()
        )

    reference_keys = record_keys(scenario_records[reference_label])
    overlap_rows: list[dict[str, object]] = []
    for label, selected in scenario_records.items():
        keys = record_keys(selected)
        intersection = reference_keys.intersection(keys)
        union = reference_keys.union(keys)
        overlap_rows.append(
            {
                "reference_scenario": reference_label,
                "scenario": label,
                "reference_records": int(len(reference_keys)),
                "scenario_records": int(len(keys)),
                "intersection_records": int(len(intersection)),
                "union_records": int(len(union)),
                "jaccard": float(len(intersection) / len(union)) if len(union) else np.nan,
                "reference_coverage_pct": float(100 * len(intersection) / len(reference_keys)) if len(reference_keys) else np.nan,
                "scenario_shared_pct": float(100 * len(intersection) / len(keys)) if len(keys) else np.nan,
            }
        )

    summary = pd.DataFrame.from_records(summary_rows).sort_values(
        ["percentile", "min_duration_h"]
    )
    overlap = pd.DataFrame.from_records(overlap_rows).sort_values("scenario")
    all_blocks = pd.concat(scenario_blocks, ignore_index=True) if scenario_blocks else pd.DataFrame()

    reference_records = scenario_records[reference_label][
        [
            animal_col,
            "data_hora_normalized",
            cta_col,
            panting_col,
            temperature_col,
            humidity_col,
            "humidity_ratio_kg_kg",
            "cta_threshold",
            "panting_threshold",
            "block_id",
            "duration_h",
        ]
    ].copy()
    reference_records = reference_records.rename(
        columns={"data_hora_normalized": normalize_col(args.time_col)}
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary, args.output_dir / "sensibilidade_criterios_resumo.csv")
    write_csv(overlap, args.output_dir / "sobreposicao_com_referencia.csv")
    write_csv(all_blocks, args.output_dir / "blocos_sensibilidade.csv")
    write_csv(
        reference_records,
        args.output_dir / "registros_referencia_p25_3h.csv",
    )

    LOGGER.info("Reference scenario: %s", reference_label)
    LOGGER.info("Sensitivity analysis completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
