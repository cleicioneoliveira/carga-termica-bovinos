#!/usr/bin/env python3
"""Analyze CTA window association with panting by strata.

This script evaluates, for each stratum, which accumulated thermal load window
has the strongest association with panting. It is intended to complement the
main thermal-comfort pipeline after the final CTA-enriched dataset has been
created.

Examples
--------
python scripts/analyze_windows_by_strata.py \
  --dataset ../dataset/processado/monitoramento_saude_cta.parquet \
  --output-dir ../resultados_dissertacao/janelas_estratificadas \
  --group-col status_saude \
  --windows 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24

python scripts/analyze_windows_by_strata.py \
  --dataset ../dataset/processado/monitoramento_saude_cta.parquet \
  --output-dir ../resultados_dissertacao/janelas_estratificadas \
  --group-col lote_historico
"""

from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger("analyze_windows_by_strata")


def normalize_col(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return text.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def normalize_group_value(value: object) -> str:
    if pd.isna(value):
        return "SEM_INFORMACAO"
    text = str(value).strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("(", "").replace(")", "")
    text = "_".join(text.upper().split())
    return text or "SEM_INFORMACAO"


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file format: {path}. Use .parquet or .csv.")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Wrote %s", path)


def infer_windows(df: pd.DataFrame) -> list[int]:
    windows: list[int] = []
    for col in df.columns:
        if col.startswith("cta_") and col.endswith("h"):
            try:
                windows.append(int(col.replace("cta_", "").replace("h", "")))
            except ValueError:
                continue
    return sorted(set(windows))


def corr_by_animal(group: pd.DataFrame, window: int, panting_col: str, min_records_per_animal: int) -> pd.Series:
    cta_col = f"cta_{window}h"
    rows: list[tuple[object, float]] = []
    for animal, animal_df in group.groupby("brinco", observed=False):
        sub = animal_df[[cta_col, panting_col]].dropna()
        if len(sub) < min_records_per_animal:
            continue
        if sub[cta_col].nunique() < 2 or sub[panting_col].nunique() < 2:
            continue
        corr = sub[cta_col].corr(sub[panting_col])
        if pd.notna(corr):
            rows.append((animal, float(corr)))
    return pd.Series(dict(rows), dtype=float)


def summarize_window_correlations(
    df: pd.DataFrame,
    group_col: str,
    group_value: object,
    windows: Iterable[int],
    panting_col: str,
    min_records_per_animal: int,
    min_animals: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for window in windows:
        cta_col = f"cta_{window}h"
        if cta_col not in df.columns:
            LOGGER.warning("Skipping %s: missing column.", cta_col)
            continue
        animal_corr = corr_by_animal(df, window, panting_col, min_records_per_animal)
        n_animals = int(animal_corr.count())
        if n_animals < min_animals:
            records.append(
                {
                    group_col: group_value,
                    "window_h": window,
                    "mean_corr": np.nan,
                    "median_corr": np.nan,
                    "positives": np.nan,
                    "negatives": np.nan,
                    "p_ttest": np.nan,
                    "p_wilcoxon": np.nan,
                    "n_animals": n_animals,
                    "status": "insufficient_animals",
                }
            )
            continue

        values = animal_corr.dropna()
        positives = int((values > 0).sum())
        negatives = int((values < 0).sum())
        p_ttest = stats.ttest_1samp(values, popmean=0.0, nan_policy="omit").pvalue
        try:
            p_wilcoxon = stats.wilcoxon(values).pvalue
        except ValueError:
            p_wilcoxon = np.nan

        records.append(
            {
                group_col: group_value,
                "window_h": window,
                "mean_corr": float(values.mean()),
                "median_corr": float(values.median()),
                "positives": positives,
                "negatives": negatives,
                "p_ttest": float(p_ttest),
                "p_wilcoxon": float(p_wilcoxon) if pd.notna(p_wilcoxon) else np.nan,
                "n_animals": n_animals,
                "status": "ok",
            }
        )

    return pd.DataFrame.from_records(records)


def analyze_group_column(
    df: pd.DataFrame,
    group_col: str,
    windows: Iterable[int],
    panting_col: str,
    output_dir: Path,
    min_records_per_animal: int,
    min_animals: int,
    min_records_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_col not in df.columns:
        raise ValueError(f"Group column not found: {group_col}")

    all_results: list[pd.DataFrame] = []
    best_rows: list[dict[str, object]] = []

    for value, group in df.groupby(group_col, dropna=False, observed=False):
        if len(group) < min_records_per_group:
            LOGGER.info("Skipping %s=%s: only %s records.", group_col, value, len(group))
            continue

        result = summarize_window_correlations(
            group,
            group_col=group_col,
            group_value=value,
            windows=windows,
            panting_col=panting_col,
            min_records_per_animal=min_records_per_animal,
            min_animals=min_animals,
        )
        if result.empty:
            continue
        all_results.append(result)

        ok = result[result["status"] == "ok"].copy()
        if ok.empty:
            continue
        best = ok.loc[ok["mean_corr"].idxmax()].to_dict()
        best["n_registros_grupo"] = int(len(group))
        best["n_animais_grupo"] = int(group["brinco"].nunique()) if "brinco" in group.columns else np.nan
        best_rows.append(best)

    if all_results:
        detail = pd.concat(all_results, ignore_index=True)
    else:
        detail = pd.DataFrame()
    best_table = pd.DataFrame(best_rows)

    group_slug = normalize_group_value(group_col)
    write_csv(detail, output_dir / f"janelas_por_{group_slug}_detalhe.csv")
    write_csv(best_table, output_dir / f"janelas_por_{group_slug}_melhor.csv")
    return detail, best_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze best CTA windows by strata.")
    parser.add_argument("--dataset", required=True, type=Path, help="CTA-enriched dataset.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--group-col", action="append", required=True, help="Grouping column. May be repeated.")
    parser.add_argument("--panting-col", default="ofegacao_hora", help="Panting/response column.")
    parser.add_argument("--windows", nargs="+", type=int, default=None, help="CTA windows in hours.")
    parser.add_argument("--min-records-per-animal", type=int, default=24, help="Minimum records per animal within stratum.")
    parser.add_argument("--min-animals", type=int, default=5, help="Minimum animals required to summarize a stratum-window.")
    parser.add_argument("--min-records-per-group", type=int, default=1000, help="Minimum records required to analyze a stratum.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s", force=True)

    df = read_table(args.dataset)
    df = df.rename(columns={col: normalize_col(col) for col in df.columns})
    if "data_hora" in df.columns:
        df["data_hora"] = pd.to_datetime(df["data_hora"], errors="coerce")
    if args.panting_col not in df.columns:
        raise ValueError(f"Panting column not found: {args.panting_col}")

    windows = args.windows or infer_windows(df)
    if not windows:
        raise ValueError("No CTA windows found. Provide --windows or include cta_*h columns in dataset.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_all: list[pd.DataFrame] = []
    for group_col in args.group_col:
        group_col = normalize_col(group_col)
        LOGGER.info("Analyzing windows by %s", group_col)
        _, best = analyze_group_column(
            df,
            group_col=group_col,
            windows=windows,
            panting_col=args.panting_col,
            output_dir=args.output_dir,
            min_records_per_animal=args.min_records_per_animal,
            min_animals=args.min_animals,
            min_records_per_group=args.min_records_per_group,
        )
        if not best.empty:
            best = best.copy()
            best.insert(0, "group_col", group_col)
            best_all.append(best)

    if best_all:
        write_csv(pd.concat(best_all, ignore_index=True), args.output_dir / "janelas_estratos_melhores_resumo.csv")
    else:
        write_csv(pd.DataFrame(), args.output_dir / "janelas_estratos_melhores_resumo.csv")

    LOGGER.info("Finished window-by-strata analysis. Output directory: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
