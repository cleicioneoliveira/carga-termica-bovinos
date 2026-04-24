from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_output_dir(output_dir: str | Path) -> Path:
    """
    Garante a existência do diretório de saída.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Salva DataFrame em CSV.
    """
    df.to_csv(output_path, index=False)


def save_best_window(output_dir: Path, best_window: int, criterion: str) -> None:
    """
    Salva metadados da melhor janela em JSON.
    """
    with open(output_dir / "best_window.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_window": best_window,
                "criterion": criterion,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
