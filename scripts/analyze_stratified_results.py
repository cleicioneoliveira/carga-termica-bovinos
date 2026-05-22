#!/usr/bin/env python3
"""Generate stratified result tables from the CTA-enriched dataset.

This script is intended to support the dissertation results chapter after the
final dataset has been enriched with lot history, compost source, breed
composition and accumulated thermal load (CTA).

It generates descriptive summaries stratified by:

- compost source
- historical lot
- breed composition
- health status
- age class, when a date-of-birth or age column is available

It also writes an audit table for health status coverage and transitions.
"""

from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("analyze_stratified_results")


DEFAULT_VARIABLES = [
    "temperatura",
    "umidade",
    "itu",
    "heat_excess",
    "cta_19h",
    "ofegacao_hora",
    "ruminacao_hora",
    "atividade_hora",
    "ocio_hora",
]


def normalize_col(name: object) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return text.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Wrote %s", path)


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "data_hora" in out.columns:
        out["data_hora"] = pd.to_datetime(out["data_hora"], errors="coerce")
    return out


def find_age_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "idade",
        "idade_meses",
        "idade_dias",
        "data_nascimento",
        "nascimento",
        "data_de_nascimento",
        "dt_nascimento",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def add_age_fields(df: pd.DataFrame, reference_date: str | None = None) -> pd.DataFrame:
    """Add age in months and age class when possible.

    If no age-related column is available, the dataset is returned unchanged.
    """
    out = df.copy()
    age_col = find_age_column(out)
    if age_col is None:
        LOGGER.warning("No age or birth-date column found. Age stratification will be skipped.")
        return out

    ref = pd.to_datetime(reference_date) if reference_date else out["data_hora"]

    if age_col in {"data_nascimento", "nascimento", "data_de_nascimento", "dt_nascimento"}:
        birth = pd.to_datetime(out[age_col], errors="coerce")
        out["idade_meses_estim"] = ((ref - birth).dt.days / 30.4375).astype(float)
    elif age_col == "idade_dias":
        out["idade_meses_estim"] = pd.to_numeric(out[age_col], errors="coerce") / 30.4375
    elif age_col == "idade_meses":
        out["idade_meses_estim"] = pd.to_numeric(out[age_col], errors="coerce")
    else:
        # Generic idade column. If values look like years, convert to months;
        # otherwise keep as months.
        age = pd.to_numeric(out[age_col], errors="coerce")
        if age.dropna().median() < 20:
            out["idade_meses_estim"] = age * 12.0
        else:
            out["idade_meses_estim"] = age

    bins = [-np.inf, 24, 36, 60, 96, np.inf]
    labels = ["<=24 meses", "25-36 meses", "37-60 meses", "61-96 meses", ">96 meses"]
    out["classe_idade"] = pd.cut(out["idade_meses_estim"], bins=bins, labels=labels, right=True)
    return out


def descriptive_by_group(df: pd.DataFrame, group_col: str, variables: Iterable[str]) -> pd.DataFrame:
    variables = [var for var in variables if var in df.columns]
    if group_col not in df.columns:
        LOGGER.warning("Skipping group %s: column not found.", group_col)
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    grouped = df.groupby(group_col, dropna=False, observed=False)

    for group_value, group in grouped:
        base = {
            group_col: group_value,
            "n_registros": len(group),
            "n_animais": group["brinco"].nunique() if "brinco" in group.columns else np.nan,
            "inicio": group["data_hora"].min() if "data_hora" in group.columns else pd.NaT,
            "fim": group["data_hora"].max() if "data_hora" in group.columns else pd.NaT,
        }
        for var in variables:
            series = pd.to_numeric(group[var], errors="coerce")
            base[f"{var}_media"] = series.mean()
            base[f"{var}_dp"] = series.std()
            base[f"{var}_p25"] = series.quantile(0.25)
            base[f"{var}_mediana"] = series.quantile(0.50)
            base[f"{var}_p75"] = series.quantile(0.75)
            base[f"{var}_min"] = series.min()
            base[f"{var}_max"] = series.max()
        rows.append(base)

    return pd.DataFrame(rows)


def health_status_audit(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Generate health-status audit tables."""
    if "status_saude" not in df.columns:
        LOGGER.warning("status_saude column not found. Health audit skipped.")
        return {}

    out: dict[str, pd.DataFrame] = {}
    total = len(df)

    coverage = (
        df["status_saude"]
        .value_counts(dropna=False)
        .rename_axis("status_saude")
        .reset_index(name="n_registros")
    )
    coverage["percentual"] = 100.0 * coverage["n_registros"] / total
    out["status_saude_frequencia"] = coverage

    if "brinco" in df.columns:
        animal_coverage = (
            df.groupby("brinco", dropna=False, observed=False)
            .agg(
                n_registros=("data_hora", "size"),
                inicio=("data_hora", "min"),
                fim=("data_hora", "max"),
                n_status=("status_saude", lambda x: x.dropna().nunique()),
                prop_sem_status=("status_saude", lambda x: x.isna().mean()),
                status_mais_frequente=("status_saude", lambda x: x.value_counts(dropna=True).index[0] if x.notna().any() else pd.NA),
            )
            .reset_index()
            .sort_values(["prop_sem_status", "n_registros"], ascending=[False, False])
        )
        out["status_saude_por_animal"] = animal_coverage

        transitions = df.sort_values(["brinco", "data_hora"]).copy()
        transitions["status_anterior"] = transitions.groupby("brinco", observed=False)["status_saude"].shift()
        transitions["mudou_status"] = transitions["status_saude"].ne(transitions["status_anterior"]) & transitions["status_anterior"].notna()
        trans_table = (
            transitions.loc[transitions["mudou_status"], ["brinco", "data_hora", "status_anterior", "status_saude"]]
            .reset_index(drop=True)
        )
        out["status_saude_transicoes"] = trans_table

        trans_count = (
            trans_table.groupby(["status_anterior", "status_saude"], dropna=False, observed=False)
            .size()
            .reset_index(name="n_transicoes")
            .sort_values("n_transicoes", ascending=False)
        )
        out["status_saude_transicoes_resumo"] = trans_count

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate stratified dissertation result tables.")
    parser.add_argument("--dataset", required=True, type=Path, help="CTA-enriched dataset.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for CSV outputs.")
    parser.add_argument("--reference-date", default=None, help="Optional fixed date for age calculation.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s", force=True)

    df = read_table(args.dataset)
    df = df.rename(columns={col: normalize_col(col) for col in df.columns})
    df = ensure_datetime(df)
    df = add_age_fields(df, reference_date=args.reference_date)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    group_map = {
        "cta_compost_origem": "resumo_por_compost.csv",
        "lote_historico": "resumo_por_lote.csv",
        "lote_norm": "resumo_por_lote_normalizado.csv",
        "composicao_racial": "resumo_por_composicao_racial.csv",
        "status_saude": "resumo_por_status_saude.csv",
        "classe_idade": "resumo_por_classe_idade.csv",
    }

    for group_col, filename in group_map.items():
        table = descriptive_by_group(df, group_col, DEFAULT_VARIABLES)
        if not table.empty:
            write_csv(table, args.output_dir / filename)

    audit = health_status_audit(df)
    for name, table in audit.items():
        write_csv(table, args.output_dir / f"{name}.csv")

    # General short summary useful for terminal inspection.
    summary = {
        "n_registros": len(df),
        "n_animais": df["brinco"].nunique() if "brinco" in df.columns else np.nan,
        "inicio": df["data_hora"].min() if "data_hora" in df.columns else pd.NaT,
        "fim": df["data_hora"].max() if "data_hora" in df.columns else pd.NaT,
    }
    summary_df = pd.DataFrame([summary])
    write_csv(summary_df, args.output_dir / "resumo_geral_dataset.csv")

    LOGGER.info("Finished stratified analyses. Output directory: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
