import torch
from torch.utils.data import TensorDataset

from sfcdi.stress import (
    PerturbedWindowDataset,
    unique_manifest_feature_indices,
)
from scripts.evaluate_stress import manifest_model_signature


def make_dataset():
    features = torch.arange(2 * 10 * 4, dtype=torch.float32).reshape(2, 10, 4)
    history = torch.zeros(2, 10, 1)
    future = torch.zeros(2, 3)
    regimes = torch.zeros(2, dtype=torch.long)
    anchors = torch.tensor([20, 40], dtype=torch.long)
    return TensorDataset(features, history, future, regimes, anchors)


def test_manifest_feature_indices_preserve_rank_and_remove_duplicates():
    manifest = {
        "selected_pairs": [
            {"feature_index": 2},
            {"feature_index": 0},
            {"feature_index": 2},
        ]
    }
    assert unique_manifest_feature_indices(manifest) == [2, 0]


def test_stress_perturbations_are_deterministic_and_driver_scoped():
    base = make_dataset()
    gaussian = PerturbedWindowDataset(base, [1, 3], "gaussian_0.10")
    assert torch.equal(gaussian[0][0], gaussian[0][0])
    assert torch.equal(gaussian[0][0][:, 0], base[0][0][:, 0])
    assert not torch.equal(gaussian[0][0][:, 1], base[0][0][:, 1])

    removed = PerturbedWindowDataset(base, [1, 3], "all_drivers_removed")[0][0]
    assert torch.count_nonzero(removed[:, 1]) == 0
    assert torch.count_nonzero(removed[:, 3]) == 0
    assert torch.equal(removed[:, 0], base[0][0][:, 0])


def test_manifest_signature_ignores_only_nonfunctional_metadata():
    manifest = {
        "schema_version": "1.1",
        "created_utc": "old",
        "manifest_sha256": "old-hash",
        "regime_ids": [0, 1],
        "selected_pairs": [
            {
                "feature_index": 2,
                "feature": "driver",
                "lag": 3,
                "available_regimes": [0, 1],
                "governance": "stable_core",
            }
        ],
    }
    revised = dict(
        manifest,
        schema_version="1.2",
        created_utc="new",
        manifest_sha256="new-hash",
    )
    assert manifest_model_signature(manifest) == manifest_model_signature(revised)
    changed = dict(revised)
    changed["selected_pairs"] = [dict(revised["selected_pairs"][0], lag=4)]
    assert manifest_model_signature(manifest) != manifest_model_signature(changed)
