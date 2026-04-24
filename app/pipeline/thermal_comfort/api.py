from __future__ import annotations

from pathlib import Path

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


def load_and_prepare_dataset(
    dataset_path: str | Path,
    thi_threshold: float,
) -> pd.DataFrame:
    """
    API pública para carga e preparação do dataset.
    Assinatura preservada para compatibilidade com o framework externo.
    """
    return load_and_prepare(dataset_path=dataset_path, thi_threshold=thi_threshold)


def run_manual_mode(
    df: pd.DataFrame,
    window: int,
    min_duration: int,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    API pública para execução em modo manual.
    Assinatura preservada para compatibilidade com o framework externo.
    """
    print(f"[INFO] Modo manual: usando janela fixa de {window}h")

    output_path = ensure_output_dir(output_dir)

    df_window = add_heat_load(df, window)
    df_comfort = define_comfort(df_window, window)
    df_periods = extract_comfort_periods(df_comfort, min_duration=min_duration)

    plot_psychrometric(df_periods, output_path)
    save_dataframe_csv(df_periods, output_path / "dados_conforto_psicrometrico.csv")

    return df_window, df_periods


def run_auto_mode(
    df: pd.DataFrame,
    windows: list[int],
    criterion: str,
    min_duration: int,
    output_dir: str | Path,
) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    API pública para execução em modo automático.
    Assinatura preservada para compatibilidade com o framework externo.
    """
    print("[INFO] Modo automático: procurando melhor janela...")

    output_path = ensure_output_dir(output_dir)

    df_results = run_window_analysis(df, windows)
    save_dataframe_csv(df_results, output_path / "resultados_janelas.csv")

    plot_window_results_academic(df_results, output_path)

    best_window = choose_best_window(df_results, criterion=criterion)
    print(f"[INFO] Melhor janela escolhida: {best_window}h (critério: {criterion})")

    save_best_window(output_path, best_window, criterion)

    df_window = add_heat_load(df, best_window)
    df_comfort = define_comfort(df_window, best_window)
    df_periods = extract_comfort_periods(df_comfort, min_duration=min_duration)

    plot_psychrometric(df_periods, output_path)
    save_dataframe_csv(df_periods, output_path / "dados_conforto_psicrometrico.csv")

    return best_window, df_results, df_window, df_periods
