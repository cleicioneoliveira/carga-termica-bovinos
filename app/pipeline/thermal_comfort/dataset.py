from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .columns import COLUMN_ALIASES, Column, REQUIRED_INPUT_COLUMNS, STANDARDIZATION_MAP
from .metrics import add_thi_and_heat_excess


logger = logging.getLogger(__name__)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known source columns to the canonical internal schema.

    The function first applies broad aliases such as ``timestamp -> data_hora``
    and ``animal_id -> brinco``. It then applies the thermal-analysis mapping,
    for example ``temperatura_compost_1 -> temperatura`` and
    ``ofegacao_hora -> ofegacao``.
    """
    rename_map: dict[str, str] = {}

    for source, target in COLUMN_ALIASES.items():
        if source in df.columns and target not in df.columns:
            rename_map[source] = target

    for source, target in STANDARDIZATION_MAP.items():
        if source in df.columns and target not in df.columns:
            rename_map[source] = target

    if rename_map:
        logger.info("Renaming columns using canonical schema: %s", rename_map)

    return df.rename(columns=rename_map)


def _convert_series_to_integer(series: pd.Series, method: str) -> pd.Series:
    """Convert a numeric Series to integer-like float values using a method."""
    method = method.lower()

    if method == "round":
        converted = np.rint(series)
    elif method == "floor":
        converted = np.floor(series)
    elif method == "ceil":
        converted = np.ceil(series)
    elif method == "trunc":
        converted = np.trunc(series)
    else:
        raise ValueError(
            "Invalid integer conversion method. Use one of: "
            "round, floor, ceil, trunc."
        )

    return pd.Series(converted, index=series.index)


def apply_thermal_input_preprocessing(
    df: pd.DataFrame,
    preprocessing_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply optional experimental preprocessing to thermal inputs.

    This is intended for sensitivity analysis only. It allows comparing the
    corrected floating-point environmental dataset against an integer-valued
    version similar to the original monitoring data.
    """
    preprocessing_cfg = preprocessing_cfg or {}
    convert_to_integer = bool(
        preprocessing_cfg.get("convert_temperature_humidity_to_integer", False)
    )

    if not convert_to_integer:
        return df

    method = str(preprocessing_cfg.get("integer_method", "round")).lower()
    converted = df.copy()

    logger.info(
        "Converting thermal inputs to integer-like values using method '%s'.",
        method,
    )

    for column in (Column.TEMPERATURA, Column.UMIDADE):
        if column not in converted.columns:
            raise ValueError(
                f"Cannot convert thermal input to integer: missing column {column!r}."
            )
        converted[column] = _convert_series_to_integer(converted[column], method)

    return converted


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
    """Load a dataset from Parquet or CSV based on file extension."""
    path = Path(dataset_path).expanduser()
    suffix = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    raise ValueError(
        "Unsupported dataset format. Use .parquet or .csv. "
        f"Received: {path}"
    )


def load_and_prepare(
    dataset_path: str | Path,
    thi_threshold: float,
    preprocessing_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Carrega, padroniza, limpa e calcula métricas térmicas básicas."""
    logger.info("Loading dataset: %s", dataset_path)
    df = load_dataset(dataset_path)

    logger.info("Standardizing columns.")
    df = standardize_columns(df)

    logger.info("Cleaning dataset.")
    df = convert_and_clean(df)

    df = apply_thermal_input_preprocessing(df, preprocessing_cfg)

    logger.info("Computing THI and thermal excess.")
    df = add_thi_and_heat_excess(df, thi_threshold=thi_threshold)

    return df
