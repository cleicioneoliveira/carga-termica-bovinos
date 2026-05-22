#!/usr/bin/env python3
"""Enrich the final monitoring-health dataset with accumulated heat load.

This script runs after the upstream stages:

1. environment_correction
2. status_timeline_reconstructor
3. merge_monitoramento_saude

It reads the final integrated hourly dataset, optionally attaches animal
metadata and lot history, chooses the environmental source for each record and
computes ITU, heat excess and accumulated thermal load (CTA).

When a lot history is supplied, the compost source can be assigned dynamically:

- compost 1: LOTE 05, LOTE 06, LOTE 07
- compost 2: LOTE 03, LOTE 04, LOTE POS PARTO, LOTE TRATAMENTO

Records whose lot cannot be mapped to a compost may either be kept with missing
environmental values, dropped, or assigned to a fallback compost.
"""

from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("enrich_dataset_with_cta")


COMPOST_COLUMNS = {
    1: {"temperatura": "temperatura_compost_1", "umidade": "humidade_compost_1", "itu_source": "thi_compost1"},
    2: {"temperatura": "temperatura_compost_2", "umidade": "humidade_compost_2", "itu_source": "thi_compost2"},
}

DEFAULT_LOT_COMPOST_MAP = {
    "LOTE 05": 1,
    "LOTE 06": 1,
    "LOTE 07": 1,
    "LOTE 03": 2,
    "LOTE 04": 2,
    "LOTE POS PARTO": 2,
    "LOTE POS PARTO 1": 2,
    "LOTE TRATAMENTO": 2,
    "LOTE TRATAMENTO 2": 2,
}


def normalize_text(value: object) -> str:
    """Normalize text for robust matching across accents and encoding artifacts."""
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()
    replacements = {
        "Pﺃ±S": "POS", "Pﺃ┬S": "POS", "PÓS": "POS", "PÒS": "POS", "PÔS": "POS",
        "PRﺃ┬": "PRE", "PRÉ": "PRE", "PRÈ": "PRE", "PRÊ": "PRE",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.replace("(", " ").replace(")", " ")
    normalized = " ".join(normalized.split())
    return normalized


def normalize_brinco(value: object) -> str:
    """Normalize animal identifier used for joins."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().upper().split())


def normalize_col(name: object) -> str:
    """Normalize column names to ASCII snake case."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return text.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_").replace("º", "").replace("°", "")


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV, Parquet or Excel table."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format: {path}. Use .parquet, .csv, .xlsx or .xls.")


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


def validate_environment_columns(df: pd.DataFrame) -> None:
    """Validate environmental columns for both composts."""
    required = ["brinco", "data_hora"]
    for compost_cols in COMPOST_COLUMNS.values():
        required.extend([compost_cols["temperatura"], compost_cols["umidade"]])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def attach_animal_metadata(df: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    """Attach static animal metadata such as breed composition and registered lot."""
    if path is None:
        return df

    LOGGER.info("Attaching animal metadata: %s", path)
    metadata = read_table(path)
    metadata = metadata.rename(columns={col: normalize_col(col) for col in metadata.columns})

    if "brinco" not in metadata.columns:
        raise ValueError("Animal metadata must contain a 'Brinco' column.")

    metadata["brinco"] = metadata["brinco"].map(normalize_brinco)
    rename = {
        "lote": "lote_cadastro",
        "composicao_racial": "composicao_racial",
        "id_coleira": "id_coleira",
        "status_reprodutivo": "status_reprodutivo",
        "status_lactacao": "status_lactacao",
    }
    keep = ["brinco"] + [col for col in rename if col in metadata.columns]
    metadata = metadata.loc[:, keep].rename(columns=rename)
    metadata = metadata.drop_duplicates(subset=["brinco"], keep="first")

    out = df.copy()
    out["brinco"] = out["brinco"].map(normalize_brinco)
    out = out.merge(metadata, on="brinco", how="left")
    if "composicao_racial" in out.columns:
        matched = out["composicao_racial"].notna().sum()
        LOGGER.info("Animal metadata matched rows: %s/%s", f"{matched:,}", f"{len(out):,}")
    return out


def prepare_lot_history(path: Path) -> pd.DataFrame:
    """Read and validate historical lot intervals."""
    lots = read_table(path)
    lots = lots.rename(columns={col: normalize_col(col) for col in lots.columns})

    required = ["brinco", "lote", "data_entrada_lote", "data_saida_lote"]
    missing = [col for col in required if col not in lots.columns]
    if missing:
        raise ValueError(f"Lot history missing required columns: {missing}")

    lots = lots.loc[:, required].copy()
    lots["brinco"] = lots["brinco"].map(normalize_brinco)
    lots["lote"] = lots["lote"].astype(str)
    lots["lote_norm"] = lots["lote"].map(normalize_text)
    lots["data_entrada_lote"] = pd.to_datetime(lots["data_entrada_lote"], errors="coerce")
    lots["data_saida_lote"] = pd.to_datetime(lots["data_saida_lote"], errors="coerce")
    lots = lots.dropna(subset=["brinco", "data_entrada_lote"]).copy()
    lots = lots.sort_values(["brinco", "data_entrada_lote", "data_saida_lote"])
    LOGGER.info("Loaded lot history with %s intervals and %s animals.", f"{len(lots):,}", f"{lots['brinco'].nunique():,}")
    return lots


def attach_lot_history(df: pd.DataFrame, lot_history_path: Path | None) -> pd.DataFrame:
    """Attach the lot valid at each animal-hour record."""
    if lot_history_path is None:
        return df

    LOGGER.info("Attaching lot history: %s", lot_history_path)
    lots = prepare_lot_history(lot_history_path)

    left = df.copy()
    left["brinco"] = left["brinco"].map(normalize_brinco)
    left["data_hora"] = pd.to_datetime(left["data_hora"], errors="coerce")
    left["_row_id"] = np.arange(len(left))

    merged_parts: list[pd.DataFrame] = []
    for brinco, animal_df in left.groupby("brinco", observed=False, sort=False):
        animal_lots = lots[lots["brinco"] == brinco]
        if animal_lots.empty:
            temp = animal_df.copy()
            temp["lote_historico"] = pd.NA
            temp["lote_norm"] = pd.NA
            temp["data_entrada_lote"] = pd.NaT
            temp["data_saida_lote"] = pd.NaT
            merged_parts.append(temp)
            continue

        animal_df = animal_df.sort_values("data_hora")
        animal_lots = animal_lots.sort_values("data_entrada_lote")
        temp = pd.merge_asof(
            animal_df,
            animal_lots[["data_entrada_lote", "data_saida_lote", "lote", "lote_norm"]],
            left_on="data_hora",
            right_on="data_entrada_lote",
            direction="backward",
        )
        valid = temp["data_saida_lote"].isna() | (temp["data_hora"] < temp["data_saida_lote"])
        temp.loc[~valid, ["lote", "lote_norm", "data_entrada_lote", "data_saida_lote"]] = pd.NA
        temp = temp.rename(columns={"lote": "lote_historico"})
        merged_parts.append(temp)

    out = pd.concat(merged_parts, ignore_index=True)
    out = out.sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)
    matched = out["lote_historico"].notna().sum()
    LOGGER.info("Lot history matched rows: %s/%s", f"{matched:,}", f"{len(out):,}")
    LOGGER.info("Lot history distribution: %s", out["lote_historico"].value_counts(dropna=False).to_dict())
    return out


def map_lot_to_compost(lote_norm: pd.Series) -> pd.Series:
    """Map normalized lot names to compost identifiers."""
    return lote_norm.map(DEFAULT_LOT_COMPOST_MAP)


def choose_environment_columns(df: pd.DataFrame, compost: int, prefer_existing_itu: bool, unmapped_lot_policy: str) -> pd.DataFrame:
    """Select temperature, humidity and ITU columns from the appropriate compost."""
    if compost not in COMPOST_COLUMNS:
        raise ValueError("compost must be 1 or 2")
    if unmapped_lot_policy not in {"fallback", "missing", "drop"}:
        raise ValueError("unmapped_lot_policy must be fallback, missing or drop")

    out = df.copy()
    if "lote_norm" in out.columns:
        out["cta_compost_origem"] = map_lot_to_compost(out["lote_norm"])
        out["cta_compost_mapeado_por_lote"] = out["cta_compost_origem"].notna()
        n_unmapped = int(out["cta_compost_origem"].isna().sum())
        if n_unmapped:
            LOGGER.warning("Rows with unmapped lot: %s", f"{n_unmapped:,}")
        if unmapped_lot_policy == "fallback":
            out["cta_compost_origem"] = out["cta_compost_origem"].fillna(compost)
        elif unmapped_lot_policy == "drop":
            out = out.dropna(subset=["cta_compost_origem"]).copy()
    else:
        out["cta_compost_origem"] = compost
        out["cta_compost_mapeado_por_lote"] = False

    out["cta_compost_origem"] = pd.to_numeric(out["cta_compost_origem"], errors="coerce")
    out["temperatura"] = np.nan
    out["umidade"] = np.nan
    out["itu"] = np.nan

    for compost_id, cols in COMPOST_COLUMNS.items():
        mask = out["cta_compost_origem"] == compost_id
        if not mask.any():
            continue
        out.loc[mask, "temperatura"] = pd.to_numeric(out.loc[mask, cols["temperatura"]], errors="coerce")
        out.loc[mask, "umidade"] = pd.to_numeric(out.loc[mask, cols["umidade"]], errors="coerce")
        if prefer_existing_itu and cols["itu_source"] in out.columns:
            out.loc[mask, "itu"] = pd.to_numeric(out.loc[mask, cols["itu_source"]], errors="coerce")
        else:
            out.loc[mask, "itu"] = calculate_itu(out.loc[mask, "temperatura"], out.loc[mask, "umidade"])

    LOGGER.info("CTA compost source distribution: %s", out["cta_compost_origem"].value_counts(dropna=False).to_dict())
    return out


def add_segments(df: pd.DataFrame, max_gap_hours: int) -> pd.DataFrame:
    """Create continuous temporal segments per animal and compost source."""
    out = df.sort_values(["brinco", "data_hora"]).copy()
    gap = pd.Timedelta(hours=max_gap_hours)
    out["cta_time_diff"] = out.groupby("brinco", observed=False)["data_hora"].diff()
    out["cta_compost_diff"] = out.groupby("brinco", observed=False)["cta_compost_origem"].diff().fillna(0)
    out["cta_new_segment"] = out["cta_time_diff"].isna() | (out["cta_time_diff"] > gap) | (out["cta_compost_diff"] != 0)
    out["cta_segment_id"] = out.groupby("brinco", observed=False)["cta_new_segment"].cumsum()
    segments = out.groupby("brinco", observed=False)["cta_segment_id"].nunique()
    LOGGER.info("CTA segments by animal: min=%s median=%s max=%s", segments.min(), segments.median(), segments.max())
    return out.drop(columns=["cta_compost_diff"])


def enrich_with_cta(
    df: pd.DataFrame,
    compost: int,
    windows: Iterable[int],
    threshold: float,
    max_gap_hours: int,
    prefer_existing_itu: bool,
    animal_metadata: Path | None = None,
    lot_history: Path | None = None,
    unmapped_lot_policy: str = "fallback",
) -> pd.DataFrame:
    """Add metadata, operational environment, ITU, heat excess and CTA columns."""
    validate_environment_columns(df)
    out = df.copy()
    out["data_hora"] = pd.to_datetime(out["data_hora"], errors="coerce")
    out["brinco"] = out["brinco"].map(normalize_brinco)
    out = attach_animal_metadata(out, animal_metadata)
    out = attach_lot_history(out, lot_history)
    out = choose_environment_columns(out, compost=compost, prefer_existing_itu=prefer_existing_itu, unmapped_lot_policy=unmapped_lot_policy)
    out["heat_excess"] = np.maximum(out["itu"] - threshold, 0.0)
    out = out.dropna(subset=["brinco", "data_hora", "temperatura", "umidade", "itu"]).copy()
    out = add_segments(out, max_gap_hours=max_gap_hours)

    for window in windows:
        if window <= 0:
            raise ValueError(f"Window must be positive. Received: {window}")
        col = f"cta_{window}h"
        LOGGER.info("Computing %s", col)
        out[col] = out.groupby(["brinco", "cta_segment_id"], observed=False)["heat_excess"].transform(
            lambda s, w=window: s.rolling(w, min_periods=1).sum()
        )

    out["cta_threshold_itu"] = threshold
    return out.sort_values(["brinco", "data_hora"]).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich final monitoring-health dataset with CTA columns.")
    parser.add_argument("--input", required=True, type=Path, help="Final unified dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Output enriched dataset.")
    parser.add_argument("--output-csv", default=None, type=Path, help="Optional CSV copy.")
    parser.add_argument("--compost", type=int, choices=[1, 2], default=1, help="Fallback compost source.")
    parser.add_argument("--animal-metadata", default=None, type=Path, help="Animal registration file with breed and current lot.")
    parser.add_argument("--lot-history", default=None, type=Path, help="Historical lot interval file.")
    parser.add_argument("--unmapped-lot-policy", choices=["fallback", "missing", "drop"], default="fallback", help="How to handle rows whose lot is not mapped to compost 1 or 2.")
    parser.add_argument("--windows", nargs="+", type=int, default=[6, 9, 12, 15, 18, 24], help="CTA windows in hours.")
    parser.add_argument("--threshold", type=float, default=72.0, help="ITU threshold.")
    parser.add_argument("--max-gap-hours", type=int, default=1, help="Reset CTA after gaps greater than this number of hours.")
    parser.add_argument("--recompute-itu", action="store_true", help="Recompute ITU instead of using existing thi_compost columns.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s", force=True)
    df = read_table(args.input)
    enriched = enrich_with_cta(
        df=df,
        compost=args.compost,
        windows=args.windows,
        threshold=args.threshold,
        max_gap_hours=args.max_gap_hours,
        prefer_existing_itu=not args.recompute_itu,
        animal_metadata=args.animal_metadata,
        lot_history=args.lot_history,
        unmapped_lot_policy=args.unmapped_lot_policy,
    )
    write_table(enriched, args.output)
    if args.output_csv is not None:
        write_table(enriched, args.output_csv)
    LOGGER.info("Wrote enriched dataset with %s rows to %s", f"{len(enriched):,}", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
