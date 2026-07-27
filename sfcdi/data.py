"""Dataset loading and leakage-safe window construction.

The module deliberately keeps split assignment, segment boundaries, and regime
metadata outside the model inputs.  Preprocessing statistics are fitted only on
the training indices supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


METROPT3_DIGITAL_COLUMNS = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

METROPT3_DEFAULT_FEATURES = [
    "TP2",
    # TP3 is an almost exact duplicate of Reservoirs (r > 0.99999) and is
    # excluded from the forecasting inputs.
    "H1",
    "DV_pressure",
    "Oil_temperature",
    "Motor_current",
    *METROPT3_DIGITAL_COLUMNS,
]


@dataclass(frozen=True)
class Standardizer:
    """Column-wise standardization statistics fitted on training data only."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        values = np.asarray(values, dtype=np.float64)
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=np.asarray(mean), scale=np.asarray(scale))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values) - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.scale + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


@dataclass
class MechanicalSeries:
    """One ordered multivariate series plus non-input boundary metadata."""

    timestamps: np.ndarray
    features: np.ndarray
    target: np.ndarray
    regimes: np.ndarray
    segments: np.ndarray
    feature_names: list[str]
    target_name: str
    regime_names: list[str]
    source_path: str
    dataset_name: str = "mechanical_series"
    predefined_split_ids: np.ndarray | None = None
    trial_ids: np.ndarray | None = None
    split_protocol: str | None = None
    evaluation_regimes: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.target)
        fields = (self.timestamps, self.features, self.regimes, self.segments)
        if any(len(field) != n for field in fields):
            raise ValueError("All MechanicalSeries arrays must have equal length")
        if self.features.ndim != 2:
            raise ValueError("features must have shape [time, variables]")
        if self.features.shape[1] != len(self.feature_names):
            raise ValueError("feature_names do not match the feature matrix")
        if self.predefined_split_ids is not None and len(self.predefined_split_ids) != n:
            raise ValueError("predefined_split_ids must match the series length")
        if self.trial_ids is not None and len(self.trial_ids) != n:
            raise ValueError("trial_ids must match the series length")
        if self.evaluation_regimes is not None and len(self.evaluation_regimes) != n:
            raise ValueError("evaluation_regimes must match the series length")


def _metropt3_regime(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build an origin-observable operating-state label.

    The two dominant electrical states are kept separate.  All transition and
    unusual combinations are pooled to avoid pretending that very rare binary
    combinations are independently estimable regimes.
    """

    comp = frame["COMP"].to_numpy(dtype=np.int8)
    discharge = frame["DV_eletric"].to_numpy(dtype=np.int8)
    regimes = np.full(len(frame), 2, dtype=np.int64)
    regimes[(comp == 1) & (discharge == 0)] = 0
    regimes[(comp == 0) & (discharge == 1)] = 1
    names = ["off_or_unloaded", "loaded", "transition_or_other"]
    return regimes, names


def load_metropt3(
    csv_path: str | Path,
    target: str = "Reservoirs",
    feature_names: Sequence[str] | None = None,
    max_gap_seconds: float = 30.0,
) -> MechanicalSeries:
    """Load the official MetroPT-3 CSV and recover physical discontinuities."""

    csv_path = Path(csv_path)
    requested = list(feature_names or METROPT3_DEFAULT_FEATURES)
    if target in requested:
        raise ValueError(f"Target {target!r} must not appear in feature_names")
    required = ["timestamp", target, "COMP", "DV_eletric", *requested]
    required = list(dict.fromkeys(required))
    frame = pd.read_csv(csv_path, usecols=required)
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("MetroPT-3 timestamps are not monotonically increasing")
    if timestamps.duplicated().any():
        raise ValueError("MetroPT-3 contains duplicate timestamps")

    delta_seconds = timestamps.diff().dt.total_seconds().to_numpy()
    boundary = np.zeros(len(frame), dtype=bool)
    boundary[0] = True
    boundary[1:] = delta_seconds[1:] > max_gap_seconds
    segments = np.cumsum(boundary, dtype=np.int64) - 1
    regimes, regime_names = _metropt3_regime(frame)

    return MechanicalSeries(
        timestamps=timestamps.to_numpy(dtype="datetime64[ns]"),
        features=frame[requested].to_numpy(dtype=np.float32),
        target=frame[target].to_numpy(dtype=np.float32),
        regimes=regimes,
        segments=segments,
        feature_names=requested,
        target_name=target,
        regime_names=regime_names,
        source_path=str(csv_path.resolve()),
        dataset_name="MetroPT-3",
    )


def load_nist_ur5(
    npz_path: str | Path, protocol: str = "trial_replicate"
) -> MechanicalSeries:
    """Load the prepared NIST UR5 trial-replicate forecasting artifact."""

    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as artifact:
        feature_names = [str(value) for value in artifact["feature_names"].tolist()]
        regime_names = [str(value) for value in artifact["regime_names"].tolist()]
        target_name = str(artifact["target_name"].item())
        if protocol == "trial_replicate":
            regimes = artifact["regimes"].astype(np.int64, copy=False)
            split_ids = artifact["split_ids"]
            split_protocol = "trial_replicate_1_train_2_validation_3_test"
        elif protocol == "cold_start_ood":
            regimes = artifact["regimes_cold_start_ood"].astype(np.int64, copy=False)
            split_ids = artifact["split_ids_cold_start_ood"]
            split_protocol = (
                "normal_start_train_cold_half_speed_validation_cold_full_speed_test"
            )
        else:
            raise ValueError(f"Unknown NIST protocol: {protocol}")
        original_regimes = artifact["regimes"].astype(np.int64, copy=False)
        return MechanicalSeries(
            timestamps=artifact["timestamps"],
            features=artifact["features"].astype(np.float32, copy=False),
            target=artifact["target"].astype(np.float32, copy=False),
            regimes=regimes,
            segments=artifact["segments"].astype(np.int64, copy=False),
            feature_names=feature_names,
            target_name=target_name,
            regime_names=regime_names,
            source_path=str(npz_path.resolve()),
            dataset_name="NIST UR5 controller trials",
            predefined_split_ids=split_ids.astype(np.int8, copy=False),
            trial_ids=artifact["trial_ids"].astype(np.int64, copy=False),
            split_protocol=split_protocol,
            evaluation_regimes=original_regimes,
        )


def chronological_split_ids(
    n_rows: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> np.ndarray:
    """Assign rows to train=0, validation=1, test=2 in chronological order."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie in (0, 1)")
    if not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("validation_fraction leaves no test partition")
    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))
    split_ids = np.full(n_rows, 2, dtype=np.int8)
    split_ids[:train_end] = 0
    split_ids[train_end:validation_end] = 1
    return split_ids


def valid_window_anchors(
    split_ids: np.ndarray,
    segments: np.ndarray,
    split: int,
    sequence_length: int,
    prediction_length: int,
) -> np.ndarray:
    """Return history-end indices whose complete window stays in one partition.

    An anchor ``t`` denotes history ``[t-L+1, ..., t]`` and labels
    ``[t+1, ..., t+H]``.  Both the split identifier and physical segment must be
    identical at the two ends, which prevents windows from crossing data gaps.
    """

    n = len(split_ids)
    if len(segments) != n:
        raise ValueError("split_ids and segments have different lengths")
    anchors = np.arange(sequence_length - 1, n - prediction_length, dtype=np.int64)
    start = anchors - sequence_length + 1
    end = anchors + prediction_length
    valid = (
        (split_ids[start] == split)
        & (split_ids[end] == split)
        & (segments[start] == segments[end])
    )
    return anchors[valid]


def deterministic_subsample(indices: np.ndarray, maximum: int | None) -> np.ndarray:
    """Time-stratified deterministic subsampling without random cherry-picking."""

    indices = np.asarray(indices, dtype=np.int64)
    if maximum is None or maximum <= 0 or len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=maximum, dtype=np.int64)
    return indices[positions]


class SeriesWindowDataset(Dataset):
    """Lazy windows over a standardized base series."""

    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        regimes: np.ndarray,
        anchors: Iterable[int],
        sequence_length: int,
        prediction_length: int,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32).reshape(-1)
        self.regimes = np.asarray(regimes, dtype=np.int64)
        self.anchors = np.asarray(list(anchors), dtype=np.int64)
        self.sequence_length = int(sequence_length)
        self.prediction_length = int(prediction_length)

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, item: int):
        anchor = int(self.anchors[item])
        begin = anchor - self.sequence_length + 1
        end = anchor + 1
        future_end = end + self.prediction_length
        return (
            torch.from_numpy(self.features[begin:end]),
            torch.from_numpy(self.target[begin:end, None]),
            torch.from_numpy(self.target[end:future_end]),
            torch.tensor(self.regimes[anchor], dtype=torch.long),
            torch.tensor(anchor, dtype=torch.long),
        )


def target_lag_matrix(target_history: torch.Tensor, lags: Sequence[int]) -> torch.Tensor:
    """Gather target values at non-negative lags from a batched history."""

    if target_history.ndim == 3:
        target_history = target_history[..., 0]
    length = target_history.shape[1]
    positions = torch.as_tensor(
        [length - 1 - int(lag) for lag in lags],
        device=target_history.device,
        dtype=torch.long,
    )
    if torch.any(positions < 0):
        raise ValueError("A requested target lag exceeds the history length")
    return target_history.index_select(1, positions)
