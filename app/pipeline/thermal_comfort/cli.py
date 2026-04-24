from __future__ import annotations

import argparse

from .api import load_and_prepare_dataset, run_auto_mode, run_manual_mode
from .constants import (
    DEFAULT_MIN_DURATION,
    DEFAULT_THI_THRESHOLD,
    DEFAULT_WINDOW,
    DEFAULT_WINDOWS,
)


def build_parser() -> argparse.ArgumentParser:
    """
    Constrói parser do modo standalone.
    """
    parser = argparse.ArgumentParser(
        description="Análise de carga térmica e extração de períodos de conforto."
    )

    parser.add_argument("--dataset", required=True, help="Caminho para o arquivo parquet.")
    parser.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="manual",
        help="Modo de execução.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Janela usada no modo manual.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
        help="Lista de janelas testadas no modo auto.",
    )
    parser.add_argument(
        "--criterion",
        choices=["mean_corr", "median_corr"],
        default="mean_corr",
        help="Critério para escolher a melhor janela.",
    )
    parser.add_argument(
        "--thi-threshold",
        type=float,
        default=DEFAULT_THI_THRESHOLD,
        help="Limiar de THI.",
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=DEFAULT_MIN_DURATION,
        help="Duração mínima do bloco de conforto.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_conforto",
        help="Diretório de saída.",
    )

    return parser


def main() -> None:
    """
    Ponto de entrada do modo standalone.
    """
    parser = build_parser()
    args = parser.parse_args()

    df = load_and_prepare_dataset(
        dataset_path=args.dataset,
        thi_threshold=args.thi_threshold,
    )

    if args.mode == "manual":
        _, df_periods = run_manual_mode(
            df=df,
            window=args.window,
            min_duration=args.min_duration,
            output_dir=args.output_dir,
        )
        print(f"[INFO] Registros de conforto extraídos: {len(df_periods):,}")
        print("[INFO] Processo finalizado com sucesso.")
        return

    best_window, df_results, _, df_periods = run_auto_mode(
        df=df,
        windows=args.windows,
        criterion=args.criterion,
        min_duration=args.min_duration,
        output_dir=args.output_dir,
    )
    print(f"[INFO] Melhor janela: {best_window}h")
    print("[INFO] Resultados das janelas:")
    print(df_results)
    print(f"[INFO] Registros de conforto extraídos: {len(df_periods):,}")
    print("[INFO] Processo finalizado com sucesso.")


if __name__ == "__main__":
    main()
