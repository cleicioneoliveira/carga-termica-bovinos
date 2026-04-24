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
) -> pd.DataFrame:
    """
    Extrai blocos contínuos de conforto por animal.
    """
    ordered = df.sort_values([Column.ANIMAL_ID, Column.DATA_HORA]).copy()

    change = (
        ordered.groupby(Column.ANIMAL_ID, observed=False)[Column.COMFORT_FLAG]
        .transform(lambda series: series.ne(series.shift()).fillna(True))
        .astype(int)
    )

    ordered[Column.BLOCK] = change.groupby(
        ordered[Column.ANIMAL_ID], observed=False
    ).cumsum()

    block_info = (
        ordered.groupby([Column.ANIMAL_ID, Column.BLOCK], observed=False)
        .agg(
            comfort_flag_first=(Column.COMFORT_FLAG, "first"),
            block_duration_h=(Column.COMFORT_FLAG, "size"),
        )
        .reset_index()
    )

    valid_blocks = block_info[
        block_info["comfort_flag_first"].fillna(False)
        & (block_info["block_duration_h"] >= min_duration)
    ][[Column.ANIMAL_ID, Column.BLOCK, "block_duration_h"]]

    if valid_blocks.empty:
        return pd.DataFrame(columns=list(ordered.columns) + [Column.BLOCK_DURATION_H])

    result = ordered.merge(
        valid_blocks,
        on=[Column.ANIMAL_ID, Column.BLOCK],
        how="inner",
    )
    return result.reset_index(drop=True)
