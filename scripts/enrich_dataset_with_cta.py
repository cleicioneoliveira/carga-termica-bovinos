#!/usr/bin/env python3
"""Enrich the final monitoring-health dataset with accumulated heat load.

This script is designed to run after the upstream pipeline stages:

1. environment_correction
2. status_timeline_reconstructor
3. merge_monitoramento_saude

It reads the final integrated hourly dataset, computes ITU, heat excess and
accumulated thermal load (CTA) for one or more time windows, and writes an
enriched dataset ready for the dissertation results stage.

The input is expected to contain the final columns produced by
merge_monitoramento_saude, especially:

- brinco
- data_hora
- temperatura_compost_1
- humidade_compost_1
- thi_compost1
- temperatura_compost_2
- humidade_compost_2
- thi_compost2
- ofegacao_hora

By default, compost 1 is used as the environmental source, matching the current
thermal-comfort pipeline behavior. Use --compost 2 to use compost 2 instead.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("enrich_dataset_with_cta")


COMPOST_COLUMNS = {
    1: {
        "temperatura": "temperatura_compost_1",
        "umidade": "humidade_compost_1",
        "itu_source": "thi_compost1",
    },
    2: {
        "temperatura": "temperatura_compost_2",
        "umidade": "humidade_compost_2",
        "itu_source": "thi_compost2",
    },
}


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet table."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported input format: {path}. Use .parquet or .csv.")


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write CSV or Parquet table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {path}. Use .parquet or .csv.")


def calculate_itu(temp_c: pd.Series, humidity_pct: pd.Series) -> pd.Series:
    """Calculate ITU using the same formula adopted by the thermal pipeline."""
    temp_f = 1.8 * temp_c.astype(float) + 32.0
    rh = humidity_pct.astype(float) / 100.0
    return temp_f - (0.55 - 0.55 * rh) * (temp_f - 58.0)


def validate_input(df: pd.DataFrame, compost: int) -> None:
    """Validate the minimum columns required for enrichment."""
    if compost not in COMPOST_COLUMNS:
        raise ValueError("compost must be 1 or 2")

    cols = COMPOST_COLUMNS[compost]
    required = [
        "brinco",
        "data_hora",
        cols["temperatura"],
        cols["umidade"],
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_segments(
    df: pd.DataFrame,
    max_gap_hours: int,
) -> pd.DataFrame:
    """Create continuous temporal segments per animal.

    CTA is reset after gaps greater than ``max_gap_hours``. This prevents rolling
    accumulation from crossing long missing periods in the monitoring series.
    """
    out = df.sort_values(["brinco", "data_hora"]).copy()
    gap = pd.Timedelta(hours=max_gap_hours)

    out["cta_time_diff"] = out.groupby("brinco", observed=False)["data_hora"].diff()
    out["cta_new_segment"] = out["cta_time_diff"].isna() | (out["cta_time_diff"] > gap)
    out["cta_segment_id"] = out.groupby("brinco", observed=False)["cta_new_segment"].cumsum()

    LOGGER.info(
        "CTA segments by animal: min=%s median=%s max=%s",
        out.groupby("brinco", observed=False)["cta_segment_id"].nunique().min(),
        out.groupby("brinco", observed=False)["cta_segment_id"].nunique().median(),
        out.groupby("brinco", observed=False)["cta_segment_id"].nunique().max(),
    )
    return out


def enrich_with_cta(
    df: pd.DataFrame,
    compost: int,
    windows: Iterable[int],
    threshold: float,
    max_gap_hours: int,
    prefer_existing_itu: bool,
) -> pd.DataFrame:
    """Add operational environment, ITU, heat excess and CTA columns."""
    validate_input(df, compost)

    cols = COMPOST_COLUMNS[compost]
    out = df.copy()
    out["data_hora"] = pd.to_datetime(out["data_hora"], errors="coerce")
    out["brinco"] = out["brinco"].astype(str)
    out["temperatura"] = pd.to_numeric(out[cols["temperatura"]], errors="coerce")
    out["umidade"] = pd.to_numeric(out[cols["umidade"]], errors="coerce")

    itu_source = cols["itu_source"]
    if prefer_existing_itu and itu_source in out.columns:
        LOGGER.info("Using existing ITU column: %s", itu_source)
        out["itu"] = pd.to_numeric(out[itu_source], errors="coerce")
    else:
        LOGGER.info("Recomputing ITU from temperature and humidity.")
        out["itu"] = calculate_itu(out["temperatura"], out["umidade"])

    out["heat_excess"] = np.maximum(out["itu"] - threshold, 0.0)
    out = out.dropna(subset=["brinco", "data_hora", "temperatura", "umidade", "itu"])
    out = add_segments(out, max_gap_hours=max_gap_hours)

    for window in windows:
        if window <= 0:
            raise ValueError(f"Window must be positive. Received: {window}")

        col = f"cta_{window}h"
        LOGGER.info("Computing %s", col)
        out[col] = (
            out.groupby(["brinco", "cta_segment_id"], observed=False)["heat_excess"]
            .transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
        )

    out["cta_compost_origem"] = compost
    out["cta_threshold_itu"] = threshold

    return out.sort_values(["brinco", "data_hora"]).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich final monitoring-health dataset with CTA columns."
    )
    parser.add_argument("--input", required=True, type=Path, help="Final unified dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Output enriched dataset.")
    parser.add_argument("--output-csv", default=None, type=Path, help="Optional CSV copy.")
    parser.add_argument("--compost", type=int, choices=[1, 2], default=1, help="Compost source to use.")
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[6, 9, 12, 15, 18, 24],
        help="CTA windows in hours.",
    )
    parser.add_argument("--threshold", type=float, default=72.0, help="ITU threshold.")
    parser.add_argument(
        "--max-gap-hours",
        type=int,
        default=1,
        help="Reset CTA after gaps greater than this number of hours.",
    )
    parser.add_argument(
        "--recompute-itu",
        action="store_true",
        help="Recompute ITU instead of using existing thi_compost columns.",
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

    df = read_table(args.input)
    enriched = enrich_with_cta(
        df=df,
        compost=args.compost,
        windows=args.windows,
        threshold=args.threshold,
        max_gap_hours=args.max_gap_hours,
        prefer_existing_itu=not args.recompute_itu,
    )
    write_table(enriched, args.output)
    if args.output_csv is not None:
        write_table(enriched, args.output_csv)

    LOGGER.info("Wrote enriched dataset with %s rows to %s", len(enriched), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
