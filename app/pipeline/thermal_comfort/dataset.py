from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .columns import Column, REQUIRED_INPUT_COLUMNS, STANDARDIZATION_MAP
from .metrics import add_thi_and_heat_excess


logger = logging.getLogger(__name__)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas esperadas do dataset para a nomenclatura interna padrão."""
    return df.rename(columns=STANDARDIZATION_MAP)


def convert_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Converte tipos, valida colunas obrigatórias, remove inválidos e ordena registros."""
    cleaned = df.copy()

    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in cleaned.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    cleaned[Column.TEMPERATURA] = pd.to_numeric(
        cleaned[Column.TEMPERATURA], errors="coerce"
    )
    cleaned[Column.UMIDADE] = pd.to_numeric(
        cleaned[Column.UMIDADE], errors="coerce"
    )
    cleaned[Column.OFEGACAO] = pd.to_numeric(
        cleaned[Column.OFEGACAO], errors="coerce"
    )
    cleaned[Column.DATA_HORA] = pd.to_datetime(
        cleaned[Column.DATA_HORA], errors="coerce"
    )

    before = len(cleaned)
    cleaned = cleaned.dropna(
        subset=[
            Column.ANIMAL_ID,
            Column.DATA_HORA,
            Column.TEMPERATURA,
            Column.UMIDADE,
            Column.OFEGACAO,
        ]
    )
    after = len(cleaned)

    if before != after:
        logger.info("Removed %s invalid rows during cleaning.", before - after)

    cleaned = cleaned.sort_values(
        [Column.ANIMAL_ID, Column.DATA_HORA]
    ).reset_index(drop=True)

    return cleaned


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Carrega dataset parquet."""
    return pd.read_parquet(dataset_path)


def load_and_prepare(dataset_path: str | Path, thi_threshold: float) -> pd.DataFrame:
    """Carrega, padroniza, limpa e calcula métricas térmicas básicas."""
    logger.info("Loading dataset: %s", dataset_path)
    df = load_dataset(dataset_path)

    logger.info("Standardizing columns.")
    df = standardize_columns(df)

    logger.info("Cleaning dataset.")
    df = convert_and_clean(df)

    logger.info("Computing THI and thermal excess.")
    df = add_thi_and_heat_excess(df, thi_threshold=thi_threshold)

    return df
