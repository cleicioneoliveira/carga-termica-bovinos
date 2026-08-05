from __future__ import annotations

import pandas as pd

from scripts.analyze_comfort_criteria_sensitivity import (
    assign_continuous_blocks,
    select_valid_comfort,
)


def test_gap_splits_comfort_block() -> None:
    frame = pd.DataFrame(
        {
            "brinco": [1, 1, 1, 1, 1],
            "data_hora_normalized": pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                    "2026-01-01 05:00",
                    "2026-01-01 06:00",
                ]
            ),
            "comfort_flag": [True, True, True, True, True],
        }
    )

    blocked = assign_continuous_blocks(
        frame,
        animal_col="brinco",
        time_col="data_hora_normalized",
        expected_interval_minutes=60,
    )
    selected, blocks = select_valid_comfort(
        blocked,
        animal_col="brinco",
        duration_records=3,
        expected_interval_minutes=60,
    )

    assert len(blocks) == 1
    assert len(selected) == 3
    assert selected["data_hora_normalized"].max() == pd.Timestamp("2026-01-01 02:00")


def test_duration_threshold_keeps_hourly_sequence() -> None:
    frame = pd.DataFrame(
        {
            "brinco": [1, 1, 1, 1],
            "data_hora_normalized": pd.date_range(
                "2026-01-01 00:00", periods=4, freq="h"
            ),
            "comfort_flag": [True, True, True, True],
        }
    )

    blocked = assign_continuous_blocks(
        frame,
        animal_col="brinco",
        time_col="data_hora_normalized",
        expected_interval_minutes=60,
    )
    selected, blocks = select_valid_comfort(
        blocked,
        animal_col="brinco",
        duration_records=4,
        expected_interval_minutes=60,
    )

    assert len(blocks) == 1
    assert len(selected) == 4
    assert blocks.iloc[0]["duration_h"] == 4.0
