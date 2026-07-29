from __future__ import annotations

import pandas as pd

from app.pipeline.thermal_comfort.columns import Column
from app.pipeline.thermal_comfort.comfort import extract_comfort_periods


def _frame(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            Column.ANIMAL_ID: ["A1"] * len(timestamps),
            Column.DATA_HORA: pd.to_datetime(timestamps),
            Column.COMFORT_FLAG: [True] * len(timestamps),
        }
    )


def test_hourly_gap_splits_comfort_block() -> None:
    df = _frame(
        [
            "2025-01-01 00:00:00",
            "2025-01-01 01:00:00",
            "2025-01-01 05:00:00",
            "2025-01-01 06:00:00",
        ]
    )

    result = extract_comfort_periods(
        df,
        min_duration=3,
        expected_interval_minutes=60,
    )

    assert result.empty


def test_hourly_continuous_run_is_kept() -> None:
    df = _frame(
        [
            "2025-01-01 00:00:00",
            "2025-01-01 01:00:00",
            "2025-01-01 02:00:00",
            "2025-01-01 06:00:00",
        ]
    )

    result = extract_comfort_periods(
        df,
        min_duration=3,
        expected_interval_minutes=60,
    )

    assert len(result) == 3
    assert result[Column.BLOCK_DURATION_H].eq(3.0).all()


def test_five_minute_resolution_uses_expected_interval() -> None:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=36, freq="5min")
    df = _frame([str(value) for value in timestamps])

    result = extract_comfort_periods(
        df,
        min_duration=36,
        expected_interval_minutes=5,
    )

    assert len(result) == 36
    assert result[Column.BLOCK_DURATION_H].eq(3.0).all()
