from __future__ import annotations

from typing import Literal, Optional
import pandas as pd

from .utils import ensure_datetime, floor_time


MergeDirection = Literal["backward", "forward", "nearest"]


def merge_time_series(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    left_time: str,
    right_time: str,
    tolerance: str | pd.Timedelta = "1h",
    direction: MergeDirection = "backward",
    floor: Optional[str] = "h",
    suffixes: tuple[str, str] = ("_left", "_right"),
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Merge two time-series DataFrames using nearest timestamp matching.
    """

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------
    if left_time not in df_left.columns:
        raise KeyError(f"{left_time} not found in df_left")

    if right_time not in df_right.columns:
        raise KeyError(f"{right_time} not found in df_right")

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------
    left = ensure_datetime(df_left, left_time)
    right = ensure_datetime(df_right, right_time)

    # --------------------------------------------------
    # DROP NaT (ESSENCIAL)
    # --------------------------------------------------
    if dropna:
        n_left_before = len(left)
        n_right_before = len(right)

        left = left.dropna(subset=[left_time])
        right = right.dropna(subset=[right_time])

        # debug opcional
        if n_left_before != len(left):
            print(f"[INFO] Dropped {n_left_before - len(left)} rows from left (NaT in {left_time})")

        if n_right_before != len(right):
            print(f"[INFO] Dropped {n_right_before - len(right)} rows from right (NaT in {right_time})")

    if floor is not None:
        left = floor_time(left, left_time, floor)
        right = floor_time(right, right_time, floor)

    # --------------------------------------------------
    # Convert tolerance
    # --------------------------------------------------
    if isinstance(tolerance, str):
        tolerance = pd.Timedelta(tolerance)

    # --------------------------------------------------
    # Merge
    # --------------------------------------------------
    merged = pd.merge_asof(
        left,
        right,
        left_on=left_time,
        right_on=right_time,
        direction=direction,
        tolerance=tolerance,
        suffixes=suffixes,
    )

    return merged
