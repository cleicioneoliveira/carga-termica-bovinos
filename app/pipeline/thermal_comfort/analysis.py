from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

from .metrics import add_heat_load
from .columns import Column


logger = logging.getLogger(__name__)


def find_series_max_point(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> tuple[float, float]:
    """Return the x/y coordinates of the maximum value in a series."""
    if df.empty:
        raise ValueError("O DataFrame está vazio.")

    valid_series = df[y_col].dropna()
    if valid_series.empty:
        raise ValueError(f"A coluna '{y_col}' não possui valores válidos.")

    idx = valid_series.idxmax()
    x_at_max = float(df.loc[idx, x_col])
    y_max = float(df.loc[idx, y_col])

    return x_at_max, y_max


def find_zero_crossing(x: pd.Series, y: pd.Series) -> Optional[float]:
    """Return the first y=0 crossing using linear interpolation."""
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)

    if len(x_values) != len(y_values):
        raise ValueError("x e y devem ter o mesmo comprimento.")

    if len(x_values) < 2:
        return None

    for index in range(len(y_values) - 1):
        y0 = y_values[index]
        y1 = y_values[index + 1]
        x0 = x_values[index]
        x1 = x_values[index + 1]

        if y0 == 0:
            return float(x0)

        if y0 * y1 < 0:
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))

        if y1 == 0:
            return float(x1)

    return None


def find_consensus_negative_end(
    x: pd.Series,
    y1: pd.Series,
    y2: pd.Series,
) -> Optional[float]:
    """Return the end of the phase where both series are negative."""
    x_values = x.to_numpy(dtype=float)
    y1_values = y1.to_numpy(dtype=float)
    y2_values = y2.to_numpy(dtype=float)

    if not (len(x_values) == len(y1_values) == len(y2_values)):
        raise ValueError("x, y1 e y2 devem ter o mesmo comprimento.")

    if len(x_values) < 2:
        return None

    both_negative = (y1_values < 0) & (y2_values < 0)

    if not both_negative.any():
        return None

    last_negative_idx = np.where(both_negative)[0][-1]

    if last_negative_idx == len(x_values) - 1:
        return float(x_values[last_negative_idx])

    x0 = x_values[last_negative_idx]
    x1 = x_values[last_negative_idx + 1]

    y1_0 = y1_values[last_negative_idx]
    y1_1 = y1_values[last_negative_idx + 1]

    y2_0 = y2_values[last_negative_idx]
    y2_1 = y2_values[last_negative_idx + 1]

    crossings: list[float] = []

    if y1_0 < 0 <= y1_1:
        crossings.append(x0 - y1_0 * (x1 - x0) / (y1_1 - y1_0))
    elif y1_0 == 0:
        crossings.append(x0)

    if y2_0 < 0 <= y2_1:
        crossings.append(x0 - y2_0 * (x1 - x0) / (y2_1 - y2_0))
    elif y2_0 == 0:
        crossings.append(x0)

    if crossings:
        return float(min(crossings))

    return float(x0)


def analyze_per_animal(df: pd.DataFrame, heat_col: str) -> np.ndarray:
    """Calculate heat-load/panting correlation by animal."""
    corrs: list[float] = []

    for _, group in df.groupby(Column.ANIMAL_ID, observed=False, sort=False):
        group = group.dropna(subset=[heat_col, Column.OFEGACAO])

        if len(group) < 50:
            continue

        corr = group[[heat_col, Column.OFEGACAO]].corr().iloc[0, 1]
        if not np.isnan(corr):
            corrs.append(float(corr))

    return np.array(corrs, dtype=float)


def compute_significance(corr_values: np.ndarray) -> tuple[float, float]:
    """Run one-sample t-test and Wilcoxon test against zero."""
    corr_values = corr_values[~np.isnan(corr_values)]

    if len(corr_values) == 0:
        return np.nan, np.nan

    try:
        _, p_t = ttest_1samp(corr_values, 0)
    except Exception:
        p_t = np.nan

    try:
        _, p_w = wilcoxon(corr_values)
    except Exception:
        p_w = np.nan

    return p_t, p_w


def _summarize_window(window: int, corr_values: np.ndarray) -> dict[str, float | int]:
    """Build the aggregate metrics row for one candidate window."""
    corr_values = corr_values[~np.isnan(corr_values)]

    mean_corr = np.nanmean(corr_values) if len(corr_values) > 0 else np.nan
    median_corr = np.nanmedian(corr_values) if len(corr_values) > 0 else np.nan
    positives = int(np.sum(corr_values > 0)) if len(corr_values) > 0 else 0
    negatives = int(np.sum(corr_values < 0)) if len(corr_values) > 0 else 0
    p_t, p_w = compute_significance(corr_values)

    return {
        "window_h": window,
        "mean_corr": mean_corr,
        "median_corr": median_corr,
        "positives": positives,
        "negatives": negatives,
        "p_ttest": p_t,
        "p_wilcoxon": p_w,
        "n_animals": int(len(corr_values)),
    }


def run_window_analysis(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Evaluate candidate windows and return aggregate metrics.

    The scientific result is unchanged: for each candidate window, heat load is
    accumulated per animal and correlated with panting. The implementation now
    validates the window list, logs progress through Python logging and keeps
    the result-building logic isolated for easier testing.
    """
    unique_windows = sorted({int(window) for window in windows})
    if not unique_windows:
        raise ValueError("A lista de janelas não pode estar vazia.")

    if any(window < 1 for window in unique_windows):
        raise ValueError("Todas as janelas devem ser maiores ou iguais a 1.")

    results: list[dict[str, float | int]] = []

    for window in unique_windows:
        logger.info("Testing accumulation window: %sh", window)

        temp_df = add_heat_load(df, window)
        heat_col = f"heat_load_{window}h"
        corr_values = analyze_per_animal(temp_df, heat_col)
        results.append(_summarize_window(window, corr_values))

    return pd.DataFrame(results)


def choose_best_window(
    df_results: pd.DataFrame,
    criterion: str = "mean_corr",
) -> int:
    """Select the best window according to the requested criterion."""
    valid_criteria = {"mean_corr", "median_corr"}
    if criterion not in valid_criteria:
        raise ValueError(f"Critério inválido: {criterion}. Use um de {valid_criteria}")

    temp = df_results.dropna(subset=[criterion]).copy()
    if temp.empty:
        raise ValueError("Não foi possível escolher a melhor janela: resultados vazios.")

    best_row = temp.sort_values(by=criterion, ascending=False).iloc[0]
    return int(best_row["window_h"])
