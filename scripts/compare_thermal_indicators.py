#!/usr/bin/env python3
"""Compare ITU, heat excess and CTA association with panting.

The analysis is performed per animal using complete hourly records shared by all
selected indicators and the panting response. It complements the CTA-window
analysis by asking whether CTA at the selected window has a stronger internal
association with panting than the instantaneous thermal indicators available in
the same dataset.

This is an associative, in-sample comparison. It is not a predictive or external
validation of any thermal indicator.

Example
-------
python scripts/compare_thermal_indicators.py \
  --dataset dataset/processado/monitoramento_saude_cta.parquet \
  --output-dir resultados_dissertacao/comparacao_indicadores \
  --cta-window 19 \
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

LOGGER = logging.getLogger("compare_thermal_indicators")


def normalize_col(name: object) -> str:
    """Normalize a column name to ASCII snake case."""
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
    """Write a CSV file and report its location."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Wrote %s", path)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    """Return a finite correlation or NaN when a variable has no variation."""
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return np.nan
    value = x.corr(y, method=method)
    return float(value) if pd.notna(value) and np.isfinite(value) else np.nan


def correlations_by_animal(
    df: pd.DataFrame,
    animal_col: str,
    panting_col: str,
    indicator_cols: list[str],
    method: str,
    min_records_per_animal: int,
) -> pd.DataFrame:
    """Calculate indicator-panting correlations on common complete cases."""
    required = [animal_col, panting_col, *indicator_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    records: list[dict[str, object]] = []
    for animal, group in df.groupby(animal_col, observed=False):
        common = group[[panting_col, *indicator_cols]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(common) < min_records_per_animal:
            continue
        if common[panting_col].nunique() < 2:
            continue

        row: dict[str, object] = {
            animal_col: animal,
            "n_common_records": int(len(common)),
        }
        for indicator in indicator_cols:
            row[f"corr_{indicator}"] = safe_corr(common[indicator], common[panting_col], method)
        records.append(row)

    result = pd.DataFrame.from_records(records)
    corr_cols = [f"corr_{col}" for col in indicator_cols]
    if not result.empty:
        result = result.dropna(subset=corr_cols, how="any").reset_index(drop=True)
    return result


def one_sample_tests(values: pd.Series) -> tuple[float, float]:
    """Test whether a distribution of per-animal values differs from zero."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return np.nan, np.nan
    p_ttest = float(stats.ttest_1samp(clean, popmean=0.0, nan_policy="omit").pvalue)
    try:
        p_wilcoxon = float(stats.wilcoxon(clean).pvalue)
    except ValueError:
        p_wilcoxon = np.nan
    return p_ttest, p_wilcoxon


def summarize_indicators(per_animal: pd.DataFrame, indicator_cols: list[str]) -> pd.DataFrame:
    """Summarize per-animal correlations for each thermal indicator."""
    rows: list[dict[str, object]] = []
    for indicator in indicator_cols:
        corr_col = f"corr_{indicator}"
        values = pd.to_numeric(per_animal[corr_col], errors="coerce").dropna()
        p_ttest, p_wilcoxon = one_sample_tests(values)
        rows.append(
            {
                "indicator": indicator,
                "mean_corr": float(values.mean()),
                "median_corr": float(values.median()),
                "std_corr": float(values.std(ddof=1)),
                "positives": int((values > 0).sum()),
                "negatives": int((values < 0).sum()),
                "zeros": int((values == 0).sum()),
                "n_animals": int(values.count()),
                "p_ttest_vs_zero": p_ttest,
                "p_wilcoxon_vs_zero": p_wilcoxon,
            }
        )
    return pd.DataFrame.from_records(rows)


def fisher_z(values: pd.Series) -> pd.Series:
    """Apply a numerically safe Fisher r-to-z transformation."""
    clean = pd.to_numeric(values, errors="coerce").clip(-0.999999, 0.999999)
    return np.arctanh(clean)


def compare_pairs(
    per_animal: pd.DataFrame,
    reference_indicator: str,
    comparison_indicators: list[str],
) -> pd.DataFrame:
    """Compare the reference indicator with alternatives using paired tests."""
    rows: list[dict[str, object]] = []
    ref_col = f"corr_{reference_indicator}"

    for alternative in comparison_indicators:
        alt_col = f"corr_{alternative}"
        pair = per_animal[[ref_col, alt_col]].apply(pd.to_numeric, errors="coerce").dropna()
        delta_r = pair[ref_col] - pair[alt_col]
        delta_z = fisher_z(pair[ref_col]) - fisher_z(pair[alt_col])

        p_t_r, p_w_r = one_sample_tests(delta_r)
        p_t_z, p_w_z = one_sample_tests(delta_z)

        rows.append(
            {
                "reference": reference_indicator,
                "alternative": alternative,
                "n_animals": int(len(pair)),
                "mean_reference_corr": float(pair[ref_col].mean()),
                "mean_alternative_corr": float(pair[alt_col].mean()),
                "mean_delta_r": float(delta_r.mean()),
                "median_delta_r": float(delta_r.median()),
                "reference_higher": int((delta_r > 0).sum()),
                "alternative_higher": int((delta_r < 0).sum()),
                "ties": int((delta_r == 0).sum()),
                "p_paired_t_raw_r": p_t_r,
                "p_wilcoxon_raw_r": p_w_r,
                "mean_delta_fisher_z": float(delta_z.mean()),
                "p_paired_t_fisher_z": p_t_z,
                "p_wilcoxon_fisher_z": p_w_z,
            }
        )

    return pd.DataFrame.from_records(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ITU, heat excess and CTA correlations with panting per animal."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="CTA-enriched CSV or Parquet dataset.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for output CSV files.")
    parser.add_argument("--animal-col", default="brinco", help="Animal identifier column.")
    parser.add_argument("--panting-col", default="ofegacao_hora", help="Panting response column.")
    parser.add_argument("--itu-col", default="itu", help="Instantaneous ITU column.")
    parser.add_argument("--heat-excess-col", default="heat_excess", help="Instantaneous heat-excess column.")
    parser.add_argument("--cta-window", type=int, default=19, help="CTA window in hours.")
    parser.add_argument("--min-records-per-animal", type=int, default=50)
    parser.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
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
    if args.min_records_per_animal < 2:
        raise ValueError("--min-records-per-animal must be at least 2.")

    df = read_table(args.dataset)
    df = df.rename(columns={col: normalize_col(col) for col in df.columns})

    animal_col = normalize_col(args.animal_col)
    panting_col = normalize_col(args.panting_col)
    itu_col = normalize_col(args.itu_col)
    heat_excess_col = normalize_col(args.heat_excess_col)
    cta_col = f"cta_{args.cta_window}h"
    indicator_cols = [itu_col, heat_excess_col, cta_col]

    LOGGER.info("Dataset rows: %s", f"{len(df):,}")
    LOGGER.info("Indicators: %s", ", ".join(indicator_cols))
    LOGGER.info("Correlation method: %s", args.method)

    per_animal = correlations_by_animal(
        df=df,
        animal_col=animal_col,
        panting_col=panting_col,
        indicator_cols=indicator_cols,
        method=args.method,
        min_records_per_animal=args.min_records_per_animal,
    )
    if per_animal.empty:
        raise ValueError("No animals met the complete-case and variability criteria.")

    summary = summarize_indicators(per_animal, indicator_cols)
    paired = compare_pairs(
        per_animal,
        reference_indicator=cta_col,
        comparison_indicators=[itu_col, heat_excess_col],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(per_animal, args.output_dir / "correlacoes_indicadores_por_animal.csv")
    write_csv(summary, args.output_dir / "resumo_indicadores.csv")
    write_csv(paired, args.output_dir / "comparacoes_pareadas.csv")

    LOGGER.info("Animals retained: %s", f"{len(per_animal):,}")
    for row in summary.itertuples(index=False):
        LOGGER.info(
            "%s: mean=%.6f median=%.6f positives=%s negatives=%s n=%s",
            row.indicator,
            row.mean_corr,
            row.median_corr,
            row.positives,
            row.negatives,
            row.n_animals,
        )
    for row in paired.itertuples(index=False):
        LOGGER.info(
            "%s vs %s: mean delta r=%.6f; reference higher in %s/%s animals",
            row.reference,
            row.alternative,
            row.mean_delta_r,
            row.reference_higher,
            row.n_animals,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
