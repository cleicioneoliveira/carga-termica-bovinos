from __future__ import annotations

import pandas as pd

from scripts.validate_cta_window_holdout import (
    assign_balanced_folds,
    choose_best_window,
    correlations_by_animal_window,
    prepare_complete_animal_data,
)


def test_balanced_folds_are_deterministic_and_complete() -> None:
    animals = pd.Series([f"A{i}" for i in range(12)])

    first = assign_balanced_folds(animals, n_folds=5, seed=42)
    second = assign_balanced_folds(animals, n_folds=5, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert first["animal"].nunique() == 12
    assert sorted(first.groupby("fold").size().tolist()) == [2, 2, 2, 3, 3]


def test_complete_case_preparation_uses_same_records_for_all_windows() -> None:
    frame = pd.DataFrame(
        {
            "brinco": ["A", "A", "A", "B", "B", "B"],
            "ofegacao_hora": [0, 1, 2, 0, None, 2],
            "cta_1h": [1, 2, 3, 1, 2, 3],
            "cta_2h": [2, 3, 4, 2, 3, None],
        }
    )

    retained, metadata = prepare_complete_animal_data(
        frame,
        animal_col="brinco",
        panting_col="ofegacao_hora",
        cta_cols=["cta_1h", "cta_2h"],
        min_records_per_animal=2,
    )

    assert set(retained["brinco"]) == {"A"}
    assert len(retained) == 3
    assert bool(metadata.loc[metadata["brinco"] == "A", "retained"].iloc[0])
    assert not bool(metadata.loc[metadata["brinco"] == "B", "retained"].iloc[0])


def test_training_selection_does_not_use_holdout_summary() -> None:
    training_summary = pd.DataFrame(
        {
            "window_h": [18, 19, 20],
            "mean_corr": [0.10, 0.12, 0.11],
            "median_corr": [0.09, 0.10, 0.08],
        }
    )

    assert choose_best_window(training_summary, criterion="mean_corr") == 19
    assert choose_best_window(training_summary, criterion="median_corr") == 19


def test_correlations_are_calculated_per_animal_and_window() -> None:
    frame = pd.DataFrame(
        {
            "brinco": ["A"] * 4 + ["B"] * 4,
            "ofegacao_hora": [0, 1, 2, 3, 3, 2, 1, 0],
            "cta_1h": [0, 1, 2, 3, 0, 1, 2, 3],
            "cta_2h": [3, 2, 1, 0, 3, 2, 1, 0],
        }
    )

    result = correlations_by_animal_window(
        frame,
        animal_col="brinco",
        panting_col="ofegacao_hora",
        windows=[1, 2],
        method="pearson",
    )

    indexed = result.set_index(["brinco", "window_h"])["correlation"]
    assert indexed.loc[("A", 1)] == 1.0
    assert indexed.loc[("A", 2)] == -1.0
    assert indexed.loc[("B", 1)] == -1.0
    assert indexed.loc[("B", 2)] == 1.0
