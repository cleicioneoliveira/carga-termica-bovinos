from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import Column
from .constants import DEFAULT_MIN_DURATION


def define_comfort(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Define registros de conforto com base no percentil 25 de carga térmica e ofegação.
    """
    enriched = df.copy()
    heat_col = f"heat_load_{window}h"

    if heat_col not in enriched.columns:
        raise ValueError(f"Coluna {heat_col} não encontrada.")

    enriched[Column.HEAT_P25] = (
        enriched.groupby(Column.ANIMAL_ID, observed=False)[heat_col]
        .transform(
            lambda series: (
                series.dropna().quantile(0.25) if len(series.dropna()) > 0 else np.nan
            )
        )
    )

    enriched[Column.PANT_P25] = (
        enriched.groupby(Column.ANIMAL_ID, observed=False)[Column.OFEGACAO]
        .transform(
            lambda series: (
                series.dropna().quantile(0.25) if len(series.dropna()) > 0 else np.nan
            )
        )
    )

    enriched[Column.COMFORT_FLAG] = (
        (enriched[heat_col] <= enriched[Column.HEAT_P25])
        & (enriched[Column.OFEGACAO] <= enriched[Column.PANT_P25])
    )

    return enriched


def extract_comfort_periods(
    df: pd.DataFrame,
    min_duration: int = DEFAULT_MIN_DURATION,
    expected_interval_minutes: int = 60,
) -> pd.DataFrame:
    """
    Extrai blocos temporalmente contínuos de conforto por animal.

    Um novo bloco é iniciado quando a classificação de conforto muda ou quando
    o intervalo entre dois registros sucessivos difere da resolução temporal
    esperada. ``min_duration`` é expresso em número de registros; a API converte
    previamente horas em registros quando a entrada possui resolução sub-horária.
    """
    if expected_interval_minutes <= 0:
        raise ValueError("expected_interval_minutes deve ser maior que zero.")

    ordered = df.sort_values([Column.ANIMAL_ID, Column.DATA_HORA]).copy()
    ordered[Column.DATA_HORA] = pd.to_datetime(
        ordered[Column.DATA_HORA], errors="coerce"
    )

    comfort_change = ordered.groupby(
        Column.ANIMAL_ID, observed=False
    )[Column.COMFORT_FLAG].transform(
        lambda series: series.ne(series.shift()).fillna(True)
    )

    expected_delta = pd.Timedelta(minutes=expected_interval_minutes)
    temporal_break = ordered.groupby(
        Column.ANIMAL_ID, observed=False
    )[Column.DATA_HORA].diff().ne(expected_delta)

    change = (comfort_change | temporal_break).astype(int)

    ordered[Column.BLOCK] = change.groupby(
        ordered[Column.ANIMAL_ID], observed=False
    ).cumsum()

    block_info = (
        ordered.groupby([Column.ANIMAL_ID, Column.BLOCK], observed=False)
        .agg(
            comfort_flag_first=(Column.COMFORT_FLAG, "first"),
            block_duration_records=(Column.COMFORT_FLAG, "size"),
        )
        .reset_index()
    )
    block_info[Column.BLOCK_DURATION_H] = (
        block_info["block_duration_records"] * expected_interval_minutes / 60.0
    )

    valid_blocks = block_info[
        block_info["comfort_flag_first"].fillna(False)
        & (block_info["block_duration_records"] >= min_duration)
    ][[Column.ANIMAL_ID, Column.BLOCK, Column.BLOCK_DURATION_H]]

    if valid_blocks.empty:
        return pd.DataFrame(columns=list(ordered.columns) + [Column.BLOCK_DURATION_H])

    result = ordered.merge(
        valid_blocks,
        on=[Column.ANIMAL_ID, Column.BLOCK],
        how="inner",
    )
    return result.reset_index(drop=True)
