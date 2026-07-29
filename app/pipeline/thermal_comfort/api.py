from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis import (
    choose_best_window,
    run_window_analysis,
)
from .comfort import define_comfort, extract_comfort_periods
from .dataset import load_and_prepare
from .metrics import add_heat_load
from .outputs import ensure_output_dir, save_best_window, save_dataframe_csv
from .plotting import plot_psychrometric, plot_window_results_academic


logger = logging.getLogger(__name__)


def load_and_prepare_dataset(
    dataset_path: str | Path,
    thi_threshold: float,
    preprocessing_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    API pública para carga e preparação do dataset.
    Assinatura preservada para compatibilidade com o framework externo.
    """
    return load_and_prepare(
        dataset_path=dataset_path,
        thi_threshold=thi_threshold,
        preprocessing_cfg=preprocessing_cfg,
    )


def _duration_records(min_duration: int, time_resolution_cfg: dict[str, Any] | None) -> int:
    """Convert minimum comfort duration in hours to records when needed."""
    time_resolution_cfg = time_resolution_cfg or {}
    weighted_by_time = bool(time_resolution_cfg.get("weighted_by_time", False))
    input_frequency_minutes = int(time_resolution_cfg.get("input_frequency_minutes", 60))

    if not weighted_by_time:
        return int(min_duration)

    records_per_hour = int(60 / input_frequency_minutes)
    return int(min_duration * records_per_hour)


def run_manual_mode(
    df: pd.DataFrame,
    window: int,
    min_duration: int,
    output_dir: str | Path,
    show_plots: bool = False,
    time_resolution_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    API pública para execução em modo manual.
    Assinatura preservada para compatibilidade com o framework externo.
    """
    logger.info("Manual mode: using fixed window of %sh", window)

    time_resolution_cfg = time_resolution_cfg or {}
    input_frequency_minutes = int(time_resolution_cfg.get("input_frequency_minutes", 60))
    weighted_by_time = bool(time_resolution_cfg.get("weighted_by_time", False))
    min_duration_records = _duration_records(min_duration, time_resolution_cfg)

    output_path = ensure_output_dir(output_dir)

    df_window = add_heat_load(
        df,
        window,
        input_frequency_minutes=input_frequency_minutes,
        weighted_by_time=weighted_by_time,
    )
    df_comfort = define_comfort(df_window, window)
    df_periods = extract_comfort_periods(
        df_comfort,
        min_duration=min_duration_records,
        expected_interval_minutes=input_frequency_minutes,
    )

    plot_psychrometric(df_periods, output_path, show_plot=show_plots)
    save_dataframe_csv(df_periods, output_path / "dados_conforto_psicrometrico.csv")

    return df_window, df_periods


def run_auto_mode(
    df: pd.DataFrame,
    windows: list[int],
    criterion: str,
    min_duration: int,
    output_dir: str | Path,
    show_plots: bool = False,
    time_resolution_cfg: dict[str, Any] | None = None,
) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    API pública para execução em modo automático.
    Assinatura preservada para compatibilidade com o framework externo.
    """
    logger.info("Automatic mode: searching for the best accumulation window.")

    time_resolution_cfg = time_resolution_cfg or {}
    input_frequency_minutes = int(time_resolution_cfg.get("input_frequency_minutes", 60))
    weighted_by_time = bool(time_resolution_cfg.get("weighted_by_time", False))
    min_duration_records = _duration_records(min_duration, time_resolution_cfg)

    output_path = ensure_output_dir(output_dir)

    df_results = run_window_analysis(
        df,
        windows,
        input_frequency_minutes=input_frequency_minutes,
        weighted_by_time=weighted_by_time,
    )
    save_dataframe_csv(df_results, output_path / "resultados_janelas.csv")

    plot_window_results_academic(df_results, output_path, show_plot=show_plots)

    best_window = choose_best_window(df_results, criterion=criterion)
    logger.info("Best window selected: %sh using criterion '%s'", best_window, criterion)

    save_best_window(output_path, best_window, criterion)

    df_window = add_heat_load(
        df,
        best_window,
        input_frequency_minutes=input_frequency_minutes,
        weighted_by_time=weighted_by_time,
    )
    df_comfort = define_comfort(df_window, best_window)
    df_periods = extract_comfort_periods(
        df_comfort,
        min_duration=min_duration_records,
        expected_interval_minutes=input_frequency_minutes,
    )

    plot_psychrometric(df_periods, output_path, show_plot=show_plots)
    save_dataframe_csv(df_periods, output_path / "dados_conforto_psicrometrico.csv")

    return best_window, df_results, df_window, df_periods
