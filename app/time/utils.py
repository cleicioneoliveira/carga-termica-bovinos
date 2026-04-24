from __future__ import annotations

import pandas as pd


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Ensure a column is datetime and sorted.

    Parameters
    ----------
    df : pd.DataFrame
    col : str

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.sort_values(col)


def floor_time(df: pd.DataFrame, col: str, freq: str = "h") -> pd.DataFrame:
    """
    Floor datetime column to given frequency.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
    freq : str

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df[col] = df[col].dt.floor(freq)
    return df
