from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import Column
from .constants import DEFAULT_THI_THRESHOLD
from .ITU import calculate_itu


def calculate_specific_humidity(
    temperatura_c: pd.Series | np.ndarray,
    umidade_relativa: pd.Series | np.ndarray,
    pressure_kpa: float = 101.325,
) -> np.ndarray:
    """
    Calcula umidade específica (kg/kg) a partir de temperatura e umidade relativa.
    """
    t = np.asarray(temperatura_c, dtype=float)
    rh = np.asarray(umidade_relativa, dtype=float)

    es = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
    e = (rh / 100.0) * es
    r = 0.622 * e / (pressure_kpa - e)
    q = r / (1.0 + r)

    return q


def calcular_dpv(temp: float | np.ndarray, ur: float | np.ndarray) -> float | np.ndarray:
    """
    Calcula o Déficit de Pressão de Vapor (DPV) em kPa.
    """
    es = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    ea = es * (ur / 100.0)
    dpv = es - ea
    return dpv


def add_thi_and_heat_excess(
    df: pd.DataFrame,
    thi_threshold: float = DEFAULT_THI_THRESHOLD,
) -> pd.DataFrame:
    """
    Adiciona colunas de THI e excesso térmico.
    """
    enriched = df.copy()

    print(f"THI Threshold : {thi_threshold}")
    enriched[Column.THI] = calculate_itu(
        enriched[Column.TEMPERATURA],
        enriched[Column.UMIDADE],
    )
    enriched[Column.HEAT_EXCESS] = np.maximum(
        0,
        enriched[Column.THI] - thi_threshold,
    )

    return enriched


def add_heat_load(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calcula carga térmica acumulada em janela móvel por animal.
    """
    enriched = df.copy()
    heat_col = f"heat_load_{window}h"

    enriched[heat_col] = (
        enriched.groupby(Column.ANIMAL_ID, observed=False)[Column.HEAT_EXCESS]
        .transform(lambda series: series.rolling(window, min_periods=1).sum())
    )

    return enriched
