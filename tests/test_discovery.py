from copy import deepcopy

import numpy as np

from sfcdi.discovery import (
    _ridge_residuals_forward,
    benjamini_hochberg,
    discover_lagged_drivers,
    manifest_content_sha256,
)


def test_segment_crossfit_excludes_fragments_that_cannot_support_all_blocks():
    rng = np.random.default_rng(9)
    conditioning = rng.normal(size=(550, 3))
    outcomes = rng.normal(size=(550, 2))
    groups = np.concatenate(
        [np.zeros(500, dtype=np.int64), np.ones(50, dtype=np.int64)]
    )

    residuals, evaluated_rows, fold_ids = _ridge_residuals_forward(
        conditioning,
        outcomes,
        n_blocks=5,
        ridge_alpha=1.0,
        groups=groups,
    )

    assert residuals.shape == (400, 2)
    assert np.all(evaluated_rows < 500)
    assert set(fold_ids) == {1, 2, 3, 4}


def test_benjamini_hochberg_is_monotone_in_rank():
    p_values = np.array([0.04, 0.001, 0.02, 0.9])
    q_values = benjamini_hochberg(p_values)
    order = np.argsort(p_values)
    assert np.all(np.diff(q_values[order]) >= -1e-12)
    assert np.all((q_values >= 0) & (q_values <= 1))


def test_forward_discovery_recovers_known_lagged_driver():
    rng = np.random.default_rng(17)
    n = 5000
    driver = rng.normal(size=n)
    distractor = rng.normal(size=n)
    target = np.zeros(n)
    for time in range(5, n - 1):
        target[time + 1] = (
            0.55 * target[time] + 0.8 * driver[time - 3] + rng.normal(scale=0.2)
        )
    features = np.column_stack([driver, distractor])
    manifest = discover_lagged_drivers(
        features,
        target,
        regimes=np.zeros(n, dtype=np.int64),
        segments=np.zeros(n, dtype=np.int64),
        feature_names=["known_driver", "noise"],
        candidate_lags=[0, 1, 2, 3, 4],
        conditioning_target_lags=[0, 1],
        max_samples=n,
        n_blocks=5,
        minimum_effect=0.05,
        minimum_stability=0.5,
        maximum_pairs=2,
        maximum_lags_per_feature=1,
        minimum_regime_samples=100,
    )
    first = manifest["selected_pairs"][0]
    assert manifest["schema_version"] == "1.2"
    assert "no unobserved confounding" in manifest["claim_boundary"]
    assert first["feature"] == "known_driver"
    assert first["lag"] == 3
    assert first["passes_inference"]

    later_copy = deepcopy(manifest)
    later_copy["created_utc"] = "2099-01-01T00:00:00+00:00"
    assert manifest_content_sha256(later_copy) == manifest["manifest_sha256"]
