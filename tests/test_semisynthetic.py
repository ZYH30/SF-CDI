import numpy as np

from scripts.run_semisynthetic import (
    TRUE_PAIRS,
    generate_semisynthetic,
    selection_metrics,
)


def test_semisynthetic_generator_has_known_lags_and_shifted_shortcut():
    background = np.random.default_rng(3).normal(size=(4, 300, 6))
    series = generate_semisynthetic(background, seed=11)
    assert series.features.shape == (1200, 9)
    assert set(np.unique(series.split_ids)) == {0, 1, 2}
    train = series.split_ids == 0
    validation = series.split_ids == 1
    assert np.corrcoef(series.features[train, -1], series.target[train])[0, 1] > 0.99
    assert (
        np.corrcoef(series.features[validation, -1], series.target[validation])[0, 1]
        < -0.99
    )


def test_selection_metrics_reward_exact_pairs_and_flag_shortcut():
    exact = selection_metrics(set(TRUE_PAIRS))
    assert exact["pair_precision"] == 1.0
    assert exact["pair_recall"] == 1.0
    assert exact["exact_lag_accuracy"] == 1.0
    assert not exact["shortcut_selected"]

    contaminated = selection_metrics({("D1", 2), ("shortcut_proxy", 0)})
    assert contaminated["pair_precision"] == 0.5
    assert contaminated["shortcut_selected"]
