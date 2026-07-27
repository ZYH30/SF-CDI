"""Deterministic inference-time perturbations for frozen mechanical models."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch.utils.data import Dataset


STRESS_CONDITIONS = (
    "clean",
    "gaussian_0.10",
    "block_missing_0.20",
    "top_driver_removed",
    "bottom_driver_removed",
    "all_drivers_removed",
)


def unique_manifest_feature_indices(manifest: dict) -> list[int]:
    """Return selected feature indices in manifest rank order without duplicates."""

    indices: list[int] = []
    seen: set[int] = set()
    for pair in manifest["selected_pairs"]:
        feature_index = int(pair["feature_index"])
        if feature_index not in seen:
            seen.add(feature_index)
            indices.append(feature_index)
    if not indices:
        raise ValueError("Stress evaluation requires at least one selected driver")
    return indices


class PerturbedWindowDataset(Dataset):
    """Wrap a window dataset with anchor-deterministic driver perturbations."""

    def __init__(
        self,
        base: Dataset,
        selected_feature_indices: Sequence[int],
        condition: str,
        perturbation_seed: int = 20260723,
    ) -> None:
        if condition not in STRESS_CONDITIONS:
            raise ValueError(f"Unknown stress condition: {condition}")
        selected = list(dict.fromkeys(int(index) for index in selected_feature_indices))
        if not selected:
            raise ValueError("selected_feature_indices cannot be empty")
        self.base = base
        self.selected = selected
        self.condition = condition
        self.perturbation_seed = int(perturbation_seed)

    def __len__(self) -> int:
        return len(self.base)

    def _anchor_seed(self, anchor: int) -> int:
        # Keep the seed in the range accepted by torch.Generator.manual_seed.
        return (self.perturbation_seed + int(anchor) * 1_000_003) % (2**63 - 1)

    def __getitem__(self, item: int):
        features, target_history, target_future, regime, anchor = self.base[item]
        features = features.clone()
        anchor_value = int(anchor)
        selected = torch.as_tensor(self.selected, dtype=torch.long)

        if self.condition == "gaussian_0.10":
            generator = torch.Generator().manual_seed(self._anchor_seed(anchor_value))
            noise = torch.randn(
                (features.shape[0], len(self.selected)),
                generator=generator,
                dtype=features.dtype,
            )
            features[:, selected] += 0.10 * noise
        elif self.condition == "block_missing_0.20":
            block_length = max(1, int(math.ceil(0.20 * features.shape[0])))
            if block_length >= features.shape[0]:
                raise ValueError("Missing block must be shorter than the history")
            possible_starts = features.shape[0] - block_length
            start = 1 + self._anchor_seed(anchor_value) % possible_starts
            stop = start + block_length
            features[start:stop, selected] = features[start - 1, selected]
        elif self.condition == "top_driver_removed":
            features[:, self.selected[0]] = 0.0
        elif self.condition == "bottom_driver_removed":
            features[:, self.selected[-1]] = 0.0
        elif self.condition == "all_drivers_removed":
            features[:, selected] = 0.0

        return features, target_history, target_future, regime, anchor
