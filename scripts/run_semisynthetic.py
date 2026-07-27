#!/usr/bin/env python3
"""Known-graph mechanism validation with Metro-derived sensor innovations."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from sfcdi.data import load_metropt3, valid_window_anchors
from sfcdi.discovery import discover_lagged_drivers
from sfcdi.training import regression_metrics


CANDIDATE_LAGS = (0, 1, 2, 4, 6, 8, 12)
TARGET_LAGS = (0, 1, 2, 6, 12)
TRUE_PAIRS = (("D1", 2), ("D2", 6), ("D3", 12))
FEATURE_NAMES = ["D1", "D2", "D3", "N1", "N2", "N3", "N4", "N5", "shortcut_proxy"]


@dataclass(frozen=True)
class SemiSyntheticSeries:
    features: np.ndarray
    target: np.ndarray
    segments: np.ndarray
    regimes: np.ndarray
    split_ids: np.ndarray


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metro-csv",
        type=Path,
        default=Path("dataset/metropt3/raw/MetroPT3(AirCompressor).csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20260730, 20260740)))
    parser.add_argument("--segment-count", type=int, default=8)
    parser.add_argument("--rows-per-segment", type=int, default=5000)
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _block_permute(values: np.ndarray, rng: np.random.Generator, block_size: int = 100):
    blocks = [values[start : start + block_size] for start in range(0, len(values), block_size)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[index] for index in order])


def _empirical_ar_channel(
    source: np.ndarray, rng: np.random.Generator, coefficient: float = 0.35
) -> np.ndarray:
    innovations = np.diff(source.astype(np.float64), prepend=float(source[0]))
    scale = np.std(innovations)
    if scale < 1e-8:
        innovations = rng.normal(size=len(source))
    else:
        innovations = (innovations - np.mean(innovations)) / scale
    innovations = _block_permute(innovations, rng)
    output = np.zeros(len(source), dtype=np.float64)
    for index in range(1, len(source)):
        output[index] = coefficient * output[index - 1] + innovations[index]
    output -= np.mean(output)
    output /= max(np.std(output), 1e-8)
    return output


def generate_semisynthetic(
    background: np.ndarray, seed: int
) -> SemiSyntheticSeries:
    """Generate a known graph while retaining empirical Metro innovations.

    ``background`` has shape [segments, rows, >=6].  The first five channels
    provide sensor innovations and the sixth provides target-noise innovations.
    """

    background = np.asarray(background, dtype=np.float64)
    if background.ndim != 3 or background.shape[2] < 6:
        raise ValueError("background must have shape [segments, rows, >=6]")
    if background.shape[1] < 100:
        raise ValueError("semi-synthetic segments are too short")
    rng = np.random.default_rng(seed)
    segment_count, rows_per_segment, _ = background.shape
    all_features = []
    all_targets = []
    all_segments = []
    all_regimes = []
    all_splits = []

    for segment in range(segment_count):
        candidate = np.column_stack(
            [
                _empirical_ar_channel(
                    background[segment, :, channel % 5], rng
                )
                for channel in range(8)
            ]
        )
        empirical_noise = np.diff(
            background[segment, :, 5], prepend=background[segment, 0, 5]
        ).astype(np.float64)
        empirical_noise = _block_permute(empirical_noise, rng)
        empirical_noise -= np.mean(empirical_noise)
        empirical_noise /= max(np.std(empirical_noise), 1e-8)

        regime = segment % 2
        target = np.zeros(rows_per_segment, dtype=np.float64)
        target[:13] = 0.15 * empirical_noise[:13]
        # At forecast origin t, Y[t+1] has direct parents D1[t-2],
        # D2[t-6], and D3[t-12], matching the manifest lag convention.
        for next_index in range(13, rows_per_segment):
            origin = next_index - 1
            target[next_index] = (
                0.65 * target[origin]
                - 0.12 * target[origin - 1]
                + 0.55 * candidate[origin - 2, 0]
                - 0.45 * candidate[origin - 6, 1]
                + 0.35 * candidate[origin - 12, 2]
                + 0.08 * (2 * regime - 1)
                + 0.20 * empirical_noise[next_index]
            )

        train_end = int(0.60 * rows_per_segment)
        validation_end = int(0.80 * rows_per_segment)
        split_ids = np.full(rows_per_segment, 2, dtype=np.int8)
        split_ids[:train_end] = 0
        split_ids[train_end:validation_end] = 1
        shortcut_sign = np.where(split_ids == 0, 1.0, -1.0)
        shortcut = shortcut_sign * target + 0.05 * rng.normal(size=rows_per_segment)
        features = np.column_stack([candidate, shortcut])

        all_features.append(features.astype(np.float32))
        all_targets.append(target.astype(np.float32))
        all_segments.append(np.full(rows_per_segment, segment, dtype=np.int64))
        all_regimes.append(np.full(rows_per_segment, regime, dtype=np.int64))
        all_splits.append(split_ids)

    return SemiSyntheticSeries(
        features=np.concatenate(all_features),
        target=np.concatenate(all_targets),
        segments=np.concatenate(all_segments),
        regimes=np.concatenate(all_regimes),
        split_ids=np.concatenate(all_splits),
    )


def selected_pair_set(manifest: dict) -> set[tuple[str, int]]:
    return {
        (str(pair["feature"]), int(pair["lag"]))
        for pair in manifest["selected_pairs"]
    }


def selection_metrics(selected: set[tuple[str, int]]) -> dict:
    truth = set(TRUE_PAIRS)
    selected_variables = {feature for feature, _ in selected}
    true_variables = {feature for feature, _ in truth}
    true_positive = len(selected & truth)
    pair_precision = true_positive / len(selected) if selected else 0.0
    pair_recall = true_positive / len(truth)
    variable_true_positive = len(selected_variables & true_variables)
    variable_precision = (
        variable_true_positive / len(selected_variables) if selected_variables else 0.0
    )
    variable_recall = variable_true_positive / len(true_variables)
    return {
        "selected_pairs": [f"{feature}@{lag}" for feature, lag in sorted(selected)],
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f1": (
            2 * pair_precision * pair_recall / (pair_precision + pair_recall)
            if pair_precision + pair_recall > 0
            else 0.0
        ),
        "variable_precision": variable_precision,
        "variable_recall": variable_recall,
        "exact_lag_accuracy": pair_recall,
        "shortcut_selected": any(feature == "shortcut_proxy" for feature, _ in selected),
    }


def marginal_top_pairs(series: SemiSyntheticSeries, maximum_pairs: int = 3):
    train_rows = np.flatnonzero(series.split_ids == 0)
    features = series.features[train_rows]
    target = series.target[train_rows]
    segments = series.segments[train_rows]
    maximum_lag = max(CANDIDATE_LAGS)
    origins = np.arange(maximum_lag, len(target) - 1, dtype=np.int64)
    origins = origins[segments[origins - maximum_lag] == segments[origins + 1]]
    records = []
    for feature_index, feature in enumerate(FEATURE_NAMES):
        for lag in CANDIDATE_LAGS:
            left = features[origins - lag, feature_index]
            right = target[origins + 1]
            correlation = float(np.corrcoef(left, right)[0, 1])
            records.append((abs(correlation), feature, lag))
    selected = []
    used_features = set()
    for _, feature, lag in sorted(records, reverse=True):
        if feature in used_features:
            continue
        selected.append((feature, lag))
        used_features.add(feature)
        if len(selected) == maximum_pairs:
            break
    return set(selected)


def discover(series: SemiSyntheticSeries, candidate_lags) -> dict:
    train_rows = np.flatnonzero(series.split_ids == 0)
    return discover_lagged_drivers(
        features=series.features[train_rows],
        target=series.target[train_rows],
        regimes=series.regimes[train_rows],
        segments=series.segments[train_rows],
        feature_names=FEATURE_NAMES,
        candidate_lags=candidate_lags,
        conditioning_target_lags=TARGET_LAGS,
        max_samples=30000,
        n_blocks=5,
        ridge_alpha=1.0,
        hac_bandwidth=12,
        fdr_level=0.05,
        minimum_effect=0.02,
        minimum_stability=0.50,
        maximum_pairs=3,
        maximum_lags_per_feature=1,
        minimum_regime_samples=200,
        crossfit_within_segments=True,
        screening_horizon=1,
        train_start="semi_synthetic_train_start",
        train_end="semi_synthetic_train_end",
    )


def pair_design(series: SemiSyntheticSeries, anchors: np.ndarray, pairs):
    name_to_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    target_part = np.column_stack(
        [series.target[anchors - lag] for lag in TARGET_LAGS]
    )
    driver_part = (
        np.column_stack(
            [
                series.features[anchors - lag, name_to_index[feature]]
                for feature, lag in sorted(pairs)
            ]
        )
        if pairs
        else np.empty((len(anchors), 0), dtype=np.float32)
    )
    return np.column_stack([target_part, driver_part])


def future_targets(series: SemiSyntheticSeries, anchors: np.ndarray, horizon: int = 6):
    return np.column_stack(
        [series.target[anchors + step] for step in range(1, horizon + 1)]
    )


def ridge_forecast_comparison(
    series: SemiSyntheticSeries,
    gcm_pairs: set[tuple[str, int]],
    marginal_pairs: set[tuple[str, int]],
) -> dict:
    train_anchors = valid_window_anchors(
        series.split_ids, series.segments, 0, max(CANDIDATE_LAGS) + 1, 6
    )
    validation_anchors = valid_window_anchors(
        series.split_ids, series.segments, 1, max(CANDIDATE_LAGS) + 1, 6
    )
    all_pairs = {
        (feature, lag) for feature in FEATURE_NAMES for lag in CANDIDATE_LAGS
    }
    method_pairs = {
        "target_only": set(),
        "all_variables_all_lags": all_pairs,
        "marginal_top3": marginal_pairs,
        "gcm_selected": gcm_pairs,
        "oracle_drivers": set(TRUE_PAIRS),
    }
    train_truth = future_targets(series, train_anchors)
    validation_truth = future_targets(series, validation_anchors)
    results = {}
    for method, pairs in method_pairs.items():
        train_design = pair_design(series, train_anchors, pairs)
        validation_design = pair_design(series, validation_anchors, pairs)
        scaler = StandardScaler().fit(train_design)
        estimator = Ridge(alpha=1.0).fit(
            scaler.transform(train_design), train_truth
        )
        prediction = estimator.predict(scaler.transform(validation_design))
        results[method] = {
            **regression_metrics(validation_truth, prediction),
            "input_pairs": [f"{feature}@{lag}" for feature, lag in sorted(pairs)],
        }
    return results


def mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    arguments = parse_arguments()
    output_path = arguments.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    metro_path = (REPOSITORY_ROOT / arguments.metro_csv).resolve()
    metro = load_metropt3(metro_path)
    segment_ids, counts = np.unique(metro.segments, return_counts=True)
    eligible = [
        int(segment)
        for segment, count in sorted(
            zip(segment_ids, counts), key=lambda item: (-item[1], item[0])
        )
        if count >= arguments.rows_per_segment
    ][: arguments.segment_count]
    if len(eligible) < arguments.segment_count:
        raise ValueError("Not enough Metro segments satisfy rows-per-segment")
    background_parts = []
    for segment in eligible:
        rows = np.flatnonzero(metro.segments == segment)[: arguments.rows_per_segment]
        background_parts.append(
            np.column_stack([metro.features[rows, :5], metro.target[rows]])
        )
    background = np.stack(background_parts)

    seed_records = []
    first_series = None
    selection_frequencies = {
        "proposed_lagged_gcm": Counter(),
        "contemporaneous_gcm": Counter(),
        "marginal_top3": Counter(),
    }
    for seed in arguments.seeds:
        series = generate_semisynthetic(background, seed)
        proposed_manifest = discover(series, CANDIDATE_LAGS)
        contemporaneous_manifest = discover(series, [0])
        proposed_pairs = selected_pair_set(proposed_manifest)
        contemporaneous_pairs = selected_pair_set(contemporaneous_manifest)
        marginal_pairs = marginal_top_pairs(series)
        methods = {
            "proposed_lagged_gcm": proposed_pairs,
            "contemporaneous_gcm": contemporaneous_pairs,
            "marginal_top3": marginal_pairs,
        }
        for method, pairs in methods.items():
            selection_frequencies[method].update(pairs)
        seed_records.append(
            {
                "seed": seed,
                "selection": {
                    method: selection_metrics(pairs) for method, pairs in methods.items()
                },
                "forecast": ridge_forecast_comparison(
                    series, proposed_pairs, marginal_pairs
                ),
                "proposed_manifest_sha256": proposed_manifest["manifest_sha256"],
            }
        )
        if first_series is None:
            first_series = series

    selection_summary = {}
    for method in selection_frequencies:
        selection_summary[method] = {
            metric: mean_sd(
                [record["selection"][method][metric] for record in seed_records]
            )
            for metric in (
                "pair_precision",
                "pair_recall",
                "pair_f1",
                "variable_precision",
                "variable_recall",
                "exact_lag_accuracy",
            )
        }
        selection_summary[method]["shortcut_selection_rate"] = float(
            np.mean(
                [record["selection"][method]["shortcut_selected"] for record in seed_records]
            )
        )
        selection_summary[method]["pair_selection_frequency"] = {
            f"{feature}@{lag}": count / len(seed_records)
            for (feature, lag), count in sorted(selection_frequencies[method].items())
        }

    forecast_methods = list(seed_records[0]["forecast"])
    forecast_summary = {
        method: {
            "mae": mean_sd(
                [record["forecast"][method]["mae"] for record in seed_records]
            ),
            "rmse": mean_sd(
                [record["forecast"][method]["rmse"] for record in seed_records]
            ),
        }
        for method in forecast_methods
    }
    record = {
        "schema_version": "1.0",
        "experiment": "metro_background_known_graph_semisynthetic",
        "source": {"path": str(metro_path), "sha256": file_sha256(metro_path)},
        "source_segments": eligible,
        "rows_per_segment": arguments.rows_per_segment,
        "seeds": arguments.seeds,
        "feature_names": FEATURE_NAMES,
        "true_pairs": [f"{feature}@{lag}" for feature, lag in TRUE_PAIRS],
        "candidate_lags": list(CANDIDATE_LAGS),
        "target_lags": list(TARGET_LAGS),
        "dgp": {
            "forecast_origin_equation": (
                "Y[t+1]=0.65Y[t]-0.12Y[t-1]+0.55D1[t-2]-0.45D2[t-6]"
                "+0.35D3[t-12]+0.08regime+0.20epsilon"
            ),
            "driver_ar_coefficient": 0.35,
            "shortcut_train": "P[t]=+Y[t]+N(0,0.05^2)",
            "shortcut_validation_test": "P[t]=-Y[t]+N(0,0.05^2)",
        },
        "selection_summary": selection_summary,
        "forecast_summary": forecast_summary,
        "runs": seed_records,
        "interpretation_boundary": (
            "The shortcut proxy is an intentional negative-control violation used to "
            "test target-history conditioning; causal consistency claims apply to the "
            "assumption-satisfied direct-driver part of the DGP."
        ),
    }
    output_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        background=background.astype(np.float32),
        first_seed_features=first_series.features,
        first_seed_target=first_series.target,
        first_seed_segments=first_series.segments,
        first_seed_regimes=first_series.regimes,
        first_seed_split_ids=first_series.split_ids,
        forecast_mae=np.asarray(
            [
                [record["forecast"][method]["mae"] for method in forecast_methods]
                for record in seed_records
            ],
            dtype=np.float64,
        ),
        forecast_methods=np.asarray(forecast_methods),
    )
    print(json.dumps({"selection": selection_summary, "forecast": forecast_summary}, indent=2))


if __name__ == "__main__":
    main()
