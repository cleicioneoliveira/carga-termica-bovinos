from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .columns import Column
from .constants import DEFAULT_THI_THRESHOLD
from .ITU import calculate_itu


logger = logging.getLogger(__name__)


def calculate_specific_humidity(
    temperatura_c: pd.Series | np.ndarray,
    umidade_relativa: pd.Series | np.ndarray,
    pressure_kpa: float = 101.325,
) -> np.ndarray:
    """Calcula umidade específica (kg/kg) a partir de temperatura e umidade relativa."""
    t = np.asarray(temperatura_c, dtype=float)
    rh = np.asarray(umidade_relativa, dtype=float)

    es = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
    e = (rh / 100.0) * es
    r = 0.622 * e / (pressure_kpa - e)
    q = r / (1.0 + r)

    return q


def calcular_dpv(temp: float | np.ndarray, ur: float | np.ndarray) -> float | np.ndarray:
    """Calcula o Déficit de Pressão de Vapor (DPV) em kPa."""
    es = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    ea = es * (ur / 100.0)
    dpv = es - ea
    return dpv


def add_thi_and_heat_excess(
    df: pd.DataFrame,
    thi_threshold: float = DEFAULT_THI_THRESHOLD,
) -> pd.DataFrame:
    """Adiciona colunas de THI e excesso térmico."""
    enriched = df.copy()

    logger.info("Using THI threshold: %s", thi_threshold)
    enriched[Column.THI] = calculate_itu(
        enriched[Column.TEMPERATURA],
        enriched[Column.UMIDADE],
    )
    enriched[Column.HEAT_EXCESS] = np.maximum(
        0,
        enriched[Column.THI] - thi_threshold,
    )

    return enriched


def _records_per_hour(input_frequency_minutes: int) -> int:
    """Return the number of records expected in one hour."""
    if input_frequency_minutes <= 0:
        raise ValueError("input_frequency_minutes must be greater than zero.")

    if 60 % input_frequency_minutes != 0:
        raise ValueError(
            "input_frequency_minutes must divide 60 exactly. "
            f"Received: {input_frequency_minutes}"
        )

    return int(60 / input_frequency_minutes)


def add_heat_load(
    df: pd.DataFrame,
    window: int,
    *,
    input_frequency_minutes: int = 60,
    weighted_by_time: bool = False,
) -> pd.DataFrame:
    """Calcula carga térmica acumulada em janela móvel por animal.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with ``heat_excess`` already computed.
    window : int
        Accumulation window expressed in hours.
    input_frequency_minutes : int, optional
        Time represented by each environmental record. The default, 60,
        preserves the historical hourly behavior.
    weighted_by_time : bool, optional
        When false, the function preserves the original behavior and sums
        ``heat_excess`` over the last ``window`` records. When true, it treats
        ``window`` as hours, converts it to records and computes
        ``sum(heat_excess * delta_t_hours)``. For 5-minute data,
        ``delta_t_hours = 5 / 60`` and a 15 h window uses 180 records.
    """
    enriched = df.copy()
    heat_col = f"heat_load_{window}h"

    if not weighted_by_time:
        logger.info(
            "Computing heat load using legacy record-based rolling sum: %sh.",
            window,
        )
        enriched[heat_col] = (
            enriched.groupby(Column.ANIMAL_ID, observed=False)[Column.HEAT_EXCESS]
            .transform(lambda series: series.rolling(window, min_periods=1).sum())
        )
        return enriched

    records_per_hour = _records_per_hour(input_frequency_minutes)
    window_records = int(window * records_per_hour)
    delta_t_hours = input_frequency_minutes / 60.0

    logger.info(
        "Computing time-weighted heat load: %sh, %s min records, %s records, dt=%s h.",
        window,
        input_frequency_minutes,
        window_records,
        delta_t_hours,
    )

    def rolling_weighted_sum(series: pd.Series) -> pd.Series:
        return (series * delta_t_hours).rolling(window_records, min_periods=1).sum()

    enriched[heat_col] = (
        enriched.groupby(Column.ANIMAL_ID, observed=False)[Column.HEAT_EXCESS]
        .transform(rolling_weighted_sum)
    )

    return enriched
