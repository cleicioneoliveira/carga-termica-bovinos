#!/usr/bin/env python3
"""Internally validate CTA-window selection using animal-level holdout folds.

The full-sample analysis selected the CTA window with the highest mean
animal-specific correlation with panting. This script reduces circularity by
repeating that selection in training animals and evaluating the selected
window in animals not used for selection.

The procedure is associative and internal. It is not external validation,
does not establish causality and does not define universal physiological
thresholds.

Example
-------
python scripts/validate_cta_window_holdout.py \
  --dataset dataset/processado/monitoramento_saude_cta.parquet \
  --output-dir resultados_dissertacao/validacao_interna_janela \
  --windows 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
  --folds 5 \
  --seed 20260805 \
  --fixed-window 19 \
  --min-records-per-animal 50
"""

from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger("validate_cta_window_holdout")


def normalize_col(name: object) -> str:
    """Normalize a column name to lower-case ASCII snake case."""
    text = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return (
        text.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet dataset."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported dataset format: {path}. Use .parquet or .csv.")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV file and log its location."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Wrote %s", path)


def one_sample_tests(values: pd.Series) -> tuple[float, float]:
    """Test whether paired deltas differ from zero."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return np.nan, np.nan

    p_ttest = float(
        stats.ttest_1samp(clean, popmean=0.0, nan_policy="omit").pvalue
    )
    try:
        p_wilcoxon = float(stats.wilcoxon(clean).pvalue)
    except ValueError:
        p_wilcoxon = np.nan
    return p_ttest, p_wilcoxon


def prepare_complete_animal_data(
    df: pd.DataFrame,
    *,
    animal_col: str,
    panting_col: str,
    cta_cols: list[str],
    min_records_per_animal: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain animals with complete common records and variable panting.

    The same animal-hour observations are used for every CTA window, avoiding
    changes in the analytical sample across candidate windows.
    """
    required = [animal_col, panting_col, *cta_cols]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    working = df[required].copy()
    for column in [panting_col, *cta_cols]:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=[animal_col])

    retained_groups: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []

    for animal, group in working.groupby(animal_col, observed=False, sort=False):
        common = group.dropna(subset=[panting_col, *cta_cols]).copy()
        panting_nunique = int(common[panting_col].nunique(dropna=True))
        retained = len(common) >= min_records_per_animal and panting_nunique >= 2
        metadata_rows.append(
            {
                animal_col: animal,
                "n_common_records": int(len(common)),
                "panting_nunique": panting_nunique,
                "retained": retained,
            }
        )
        if retained:
            retained_groups.append(common)

    if not retained_groups:
        raise ValueError(
            "No animals met the complete-case, minimum-record and "
            "panting-variability criteria."
        )

    retained_data = pd.concat(retained_groups, ignore_index=True)
    metadata = pd.DataFrame.from_records(metadata_rows)
    return retained_data, metadata


def assign_balanced_folds(
    animals: pd.Series | np.ndarray | list[object],
    *,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Assign animals to deterministic, approximately balanced folds."""
    unique_animals = pd.Series(animals).drop_duplicates().tolist()
    if n_folds < 2:
        raise ValueError("--folds must be at least 2.")
    if len(unique_animals) < n_folds:
        raise ValueError(
            f"Cannot create {n_folds} folds from only {len(unique_animals)} animals."
        )

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unique_animals, dtype=object)
    rng.shuffle(shuffled)

    assignments = pd.DataFrame(
        {
            "animal": shuffled,
            "fold": np.arange(len(shuffled), dtype=int) % n_folds,
        }
    )
    return assignments


def correlations_by_animal_window(
    df: pd.DataFrame,
    *,
    animal_col: str,
    panting_col: str,
    windows: list[int],
    method: str,
) -> pd.DataFrame:
    """Calculate one correlation per animal and CTA window."""
    rows: list[dict[str, object]] = []

    for animal, group in df.groupby(animal_col, observed=False, sort=False):
        panting = pd.to_numeric(group[panting_col], errors="coerce")
        for window in windows:
            cta_col = f"cta_{window}h"
            cta = pd.to_numeric(group[cta_col], errors="coerce")
            pair = pd.DataFrame({"cta": cta, "panting": panting}).dropna()

            if pair["cta"].nunique() < 2 or pair["panting"].nunique() < 2:
                corr = np.nan
            else:
                corr = pair["cta"].corr(pair["panting"], method=method)

            rows.append(
                {
                    animal_col: animal,
                    "window_h": window,
                    "n_records": int(len(pair)),
                    "correlation": (
                        float(corr)
                        if pd.notna(corr) and np.isfinite(corr)
                        else np.nan
                    ),
                }
            )

    result = pd.DataFrame.from_records(rows)
    return result.dropna(subset=["correlation"]).reset_index(drop=True)


def summarize_windows(
    correlations: pd.DataFrame,
    *,
    split: str,
    fold: int,
) -> pd.DataFrame:
    """Summarize per-animal correlations for every candidate window."""
    if correlations.empty:
        return pd.DataFrame(
            columns=[
                "fold",
                "split",
                "window_h",
                "mean_corr",
                "median_corr",
                "std_corr",
                "positives",
                "negatives",
                "zeros",
                "n_animals",
                "n_records_total",
            ]
        )

    summary = (
        correlations.groupby("window_h", observed=False)
        .agg(
            mean_corr=("correlation", "mean"),
            median_corr=("correlation", "median"),
            std_corr=("correlation", "std"),
            positives=("correlation", lambda values: int((values > 0).sum())),
            negatives=("correlation", lambda values: int((values < 0).sum())),
            zeros=("correlation", lambda values: int((values == 0).sum())),
            n_animals=("correlation", "count"),
            n_records_total=("n_records", "sum"),
        )
        .reset_index()
    )
    summary.insert(0, "split", split)
    summary.insert(0, "fold", fold)
    return summary


def choose_best_window(
    summary: pd.DataFrame,
    *,
    criterion: str,
) -> int:
    """Choose the window with the highest requested training statistic."""
    if criterion not in {"mean_corr", "median_corr"}:
        raise ValueError("--criterion must be mean_corr or median_corr.")

    valid = summary.dropna(subset=[criterion]).copy()
    if valid.empty:
        raise ValueError("No valid window summary is available for selection.")

    best = valid.sort_values(
        [criterion, "window_h"],
        ascending=[False, True],
    ).iloc[0]
    return int(best["window_h"])


def get_window_row(summary: pd.DataFrame, window: int) -> pd.Series:
    """Return the unique summary row for a window."""
    rows = summary.loc[summary["window_h"] == window]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one summary row for window {window}, found {len(rows)}."
        )
    return rows.iloc[0]


def window_rank(summary: pd.DataFrame, window: int, criterion: str) -> int:
    """Return one-based descending rank of a candidate window."""
    ranked = summary.sort_values(
        [criterion, "window_h"],
        ascending=[False, True],
    ).reset_index(drop=True)
    positions = ranked.index[ranked["window_h"] == window].tolist()
    if len(positions) != 1:
        raise ValueError(f"Window {window} is unavailable for ranking.")
    return int(positions[0] + 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate CTA-window selection with animal-level K-fold holdout."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--animal-col", default="brinco")
    parser.add_argument("--panting-col", default="ofegacao_hora")
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(range(1, 25)),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--fixed-window", type=int, default=19)
    parser.add_argument("--min-records-per-animal", type=int, default=50)
    parser.add_argument(
        "--criterion",
        choices=["mean_corr", "median_corr"],
        default="mean_corr",
    )
    parser.add_argument(
        "--method",
        choices=["pearson", "spearman"],
        default="pearson",
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

    windows = sorted(set(int(window) for window in args.windows))
    if not windows or any(window <= 0 for window in windows):
        raise ValueError("All --windows must be positive integers.")
    if args.fixed_window not in windows:
        raise ValueError("--fixed-window must be included in --windows.")
    if args.min_records_per_animal < 2:
        raise ValueError("--min-records-per-animal must be at least 2.")

    df = read_table(args.dataset)
    df = df.rename(columns={column: normalize_col(column) for column in df.columns})

    animal_col = normalize_col(args.animal_col)
    panting_col = normalize_col(args.panting_col)
    cta_cols = [f"cta_{window}h" for window in windows]

    retained_data, eligibility = prepare_complete_animal_data(
        df,
        animal_col=animal_col,
        panting_col=panting_col,
        cta_cols=cta_cols,
        min_records_per_animal=args.min_records_per_animal,
    )
    retained_animals = retained_data[animal_col].drop_duplicates()

    fold_assignments = assign_balanced_folds(
        retained_animals,
        n_folds=args.folds,
        seed=args.seed,
    ).rename(columns={"animal": animal_col})

    LOGGER.info("Dataset rows: %s", f"{len(df):,}")
    LOGGER.info("Retained common-case rows: %s", f"{len(retained_data):,}")
    LOGGER.info("Retained animals: %s", f"{len(retained_animals):,}")
    LOGGER.info("Folds: %s", args.folds)
    LOGGER.info("Candidate windows: %s", ", ".join(map(str, windows)))
    LOGGER.info("Correlation method: %s", args.method)
    LOGGER.info("Selection criterion: %s", args.criterion)

    all_correlations = correlations_by_animal_window(
        retained_data,
        animal_col=animal_col,
        panting_col=panting_col,
        windows=windows,
        method=args.method,
    )
    all_correlations = all_correlations.merge(
        fold_assignments,
        on=animal_col,
        how="inner",
        validate="many_to_one",
    )

    window_summaries: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    holdout_animal_rows: list[pd.DataFrame] = []

    for fold in range(args.folds):
        train_corr = all_correlations.loc[all_correlations["fold"] != fold].copy()
        holdout_corr = all_correlations.loc[all_correlations["fold"] == fold].copy()

        train_summary = summarize_windows(train_corr, split="train", fold=fold)
        holdout_summary = summarize_windows(
            holdout_corr, split="holdout", fold=fold
        )
        window_summaries.extend([train_summary, holdout_summary])

        selected_window = choose_best_window(
            train_summary,
            criterion=args.criterion,
        )
        train_selected = get_window_row(train_summary, selected_window)
        holdout_selected = get_window_row(holdout_summary, selected_window)
        holdout_fixed = get_window_row(holdout_summary, args.fixed_window)

        holdout_selected_animals = holdout_corr.loc[
            holdout_corr["window_h"] == selected_window,
            [animal_col, "fold", "correlation", "n_records"],
        ].rename(
            columns={
                "correlation": "corr_selected_window",
                "n_records": "n_records_selected_window",
            }
        )
        holdout_selected_animals["selected_window_h"] = selected_window

        holdout_fixed_animals = holdout_corr.loc[
            holdout_corr["window_h"] == args.fixed_window,
            [animal_col, "fold", "correlation", "n_records"],
        ].rename(
            columns={
                "correlation": "corr_fixed_window",
                "n_records": "n_records_fixed_window",
            }
        )

        animal_comparison = holdout_selected_animals.merge(
            holdout_fixed_animals,
            on=[animal_col, "fold"],
            how="inner",
            validate="one_to_one",
        )
        animal_comparison["fixed_window_h"] = args.fixed_window
        animal_comparison["delta_selected_minus_fixed"] = (
            animal_comparison["corr_selected_window"]
            - animal_comparison["corr_fixed_window"]
        )
        holdout_animal_rows.append(animal_comparison)

        fold_rows.append(
            {
                "fold": fold,
                "n_train_animals": int(train_corr[animal_col].nunique()),
                "n_holdout_animals": int(holdout_corr[animal_col].nunique()),
                "selected_window_h": selected_window,
                "selected_window_distance_from_fixed": abs(
                    selected_window - args.fixed_window
                ),
                "fixed_window_h": args.fixed_window,
                "train_selected_mean_corr": float(
                    train_selected["mean_corr"]
                ),
                "train_selected_median_corr": float(
                    train_selected["median_corr"]
                ),
                "holdout_selected_mean_corr": float(
                    holdout_selected["mean_corr"]
                ),
                "holdout_selected_median_corr": float(
                    holdout_selected["median_corr"]
                ),
                "holdout_fixed_mean_corr": float(
                    holdout_fixed["mean_corr"]
                ),
                "holdout_fixed_median_corr": float(
                    holdout_fixed["median_corr"]
                ),
                "holdout_mean_delta_selected_minus_fixed": float(
                    holdout_selected["mean_corr"] - holdout_fixed["mean_corr"]
                ),
                "fixed_window_train_rank": window_rank(
                    train_summary,
                    args.fixed_window,
                    args.criterion,
                ),
                "fixed_window_holdout_rank": window_rank(
                    holdout_summary,
                    args.fixed_window,
                    args.criterion,
                ),
            }
        )

        LOGGER.info(
            (
                "Fold %s: selected=%sh; train mean=%.6f; "
                "holdout selected mean=%.6f; holdout fixed %sh mean=%.6f"
            ),
            fold,
            selected_window,
            train_selected["mean_corr"],
            holdout_selected["mean_corr"],
            args.fixed_window,
            holdout_fixed["mean_corr"],
        )

    fold_summary = pd.DataFrame.from_records(fold_rows)
    fold_window_results = pd.concat(window_summaries, ignore_index=True)
    holdout_by_animal = pd.concat(holdout_animal_rows, ignore_index=True)

    p_ttest, p_wilcoxon = one_sample_tests(
        holdout_by_animal["delta_selected_minus_fixed"]
    )

    selected_counts = (
        fold_summary["selected_window_h"]
        .value_counts()
        .sort_index()
        .rename_axis("selected_window_h")
        .reset_index(name="n_folds")
    )

    validation_summary = pd.DataFrame(
        [
            {
                "method": args.method,
                "criterion": args.criterion,
                "n_folds": args.folds,
                "seed": args.seed,
                "n_animals": int(len(retained_animals)),
                "n_common_records": int(len(retained_data)),
                "fixed_window_h": args.fixed_window,
                "median_selected_window_h": float(
                    fold_summary["selected_window_h"].median()
                ),
                "min_selected_window_h": int(
                    fold_summary["selected_window_h"].min()
                ),
                "max_selected_window_h": int(
                    fold_summary["selected_window_h"].max()
                ),
                "folds_selecting_fixed_window": int(
                    (fold_summary["selected_window_h"] == args.fixed_window).sum()
                ),
                "folds_selecting_18_to_20h": int(
                    fold_summary["selected_window_h"].between(18, 20).sum()
                ),
                "mean_holdout_selected_corr": float(
                    holdout_by_animal["corr_selected_window"].mean()
                ),
                "median_holdout_selected_corr": float(
                    holdout_by_animal["corr_selected_window"].median()
                ),
                "mean_holdout_fixed_corr": float(
                    holdout_by_animal["corr_fixed_window"].mean()
                ),
                "median_holdout_fixed_corr": float(
                    holdout_by_animal["corr_fixed_window"].median()
                ),
                "mean_delta_selected_minus_fixed": float(
                    holdout_by_animal[
                        "delta_selected_minus_fixed"
                    ].mean()
                ),
                "median_delta_selected_minus_fixed": float(
                    holdout_by_animal[
                        "delta_selected_minus_fixed"
                    ].median()
                ),
                "selected_higher_than_fixed_animals": int(
                    (
                        holdout_by_animal[
                            "delta_selected_minus_fixed"
                        ] > 0
                    ).sum()
                ),
                "fixed_higher_than_selected_animals": int(
                    (
                        holdout_by_animal[
                            "delta_selected_minus_fixed"
                        ] < 0
                    ).sum()
                ),
                "ties": int(
                    (
                        holdout_by_animal[
                            "delta_selected_minus_fixed"
                        ] == 0
                    ).sum()
                ),
                "p_paired_t_delta": p_ttest,
                "p_wilcoxon_delta": p_wilcoxon,
            }
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(eligibility, args.output_dir / "animal_eligibility.csv")
    write_csv(fold_assignments, args.output_dir / "fold_assignments.csv")
    write_csv(
        fold_window_results,
        args.output_dir / "fold_window_results.csv",
    )
    write_csv(fold_summary, args.output_dir / "fold_summary.csv")
    write_csv(
        holdout_by_animal,
        args.output_dir / "holdout_correlations_by_animal.csv",
    )
    write_csv(
        selected_counts,
        args.output_dir / "selected_window_frequency.csv",
    )
    write_csv(
        validation_summary,
        args.output_dir / "validation_summary.csv",
    )

    LOGGER.info(
        "Selected windows by fold: %s",
        ", ".join(map(str, fold_summary["selected_window_h"].tolist())),
    )
    LOGGER.info(
        "Fixed %sh holdout mean correlation: %.6f",
        args.fixed_window,
        validation_summary.iloc[0]["mean_holdout_fixed_corr"],
    )
    LOGGER.info("Internal holdout validation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
