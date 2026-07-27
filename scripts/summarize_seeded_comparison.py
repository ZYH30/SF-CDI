"""Aggregate paired multi-seed forecasts at seed and physical-trial levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument(
        "--dataset-artifact",
        type=Path,
        help="NPZ used to recover trial IDs when prediction files omit them",
    )
    parser.add_argument(
        "--temporal-blocks",
        type=int,
        help="Use this many contiguous evaluation blocks instead of physical trial IDs",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mean_std(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "values": values.tolist(),
    }


def hierarchical_paired_bootstrap(
    difference_by_seed_trial: np.ndarray,
    repeats: int,
    seed: int,
    second_axis: str = "physical_trial",
) -> dict:
    """Resample initialization seeds and physical trials as two cluster axes."""

    values = np.asarray(difference_by_seed_trial, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("hierarchical bootstrap needs at least 2 seeds and 2 trials")
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    generator = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        seed_indices = generator.integers(0, values.shape[0], size=values.shape[0])
        trial_indices = generator.integers(0, values.shape[1], size=values.shape[1])
        estimates[index] = values[np.ix_(seed_indices, trial_indices)].mean()
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "difference": float(values.mean()),
        "ci95": [float(lower), float(upper)],
        "bootstrap_repeats": int(repeats),
        "resampling_units": ["initialization_seed", second_axis],
        "interpretation": "negative means the candidate has lower MAE",
    }


def _load_run(
    run_directory: Path,
    model_name: str,
    fallback_trial_ids: np.ndarray | None,
    temporal_blocks: int | None,
) -> dict:
    run_directory = run_directory.resolve()
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run_directory / "results.json").read_text(encoding="utf-8"))
    prediction_path = run_directory / f"{model_name}_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as artifact:
        prediction = artifact["prediction"].astype(np.float64)
        truth = artifact["truth"].astype(np.float64)
        anchors = artifact["anchors"].astype(np.int64)
        if temporal_blocks is not None:
            if temporal_blocks < 2 or len(anchors) < temporal_blocks:
                raise ValueError("temporal_blocks must be between 2 and window count")
            cluster_ids = (
                np.arange(len(anchors), dtype=np.int64) * temporal_blocks // len(anchors)
            )
        elif "trial_ids" in artifact.files:
            trial_ids = artifact["trial_ids"].astype(np.int64)
            cluster_ids = trial_ids
        elif fallback_trial_ids is not None:
            trial_ids = fallback_trial_ids[anchors]
            cluster_ids = trial_ids
        else:
            raise ValueError(
                f"{prediction_path} has no trial_ids and no fallback artifact was supplied"
            )
    return {
        "run_directory": str(run_directory),
        "seed": int(config["seed"]),
        "config_hash": result["config_hash"],
        "manifest_sha256": result["manifest_sha256"],
        "prediction": prediction,
        "truth": truth,
        "anchors": anchors,
        "cluster_ids": cluster_ids,
    }


def _index_runs(
    directories: list[Path],
    model_name: str,
    fallback_trial_ids: np.ndarray | None,
    temporal_blocks: int | None,
) -> dict[int, dict]:
    indexed = {}
    for directory in directories:
        run = _load_run(
            directory, model_name, fallback_trial_ids, temporal_blocks
        )
        if run["seed"] in indexed:
            raise ValueError(f"duplicate seed {run['seed']} for {model_name}")
        indexed[run["seed"]] = run
    return indexed


def summarize(arguments: argparse.Namespace) -> dict:
    fallback_trial_ids = None
    if arguments.dataset_artifact is not None:
        with np.load(arguments.dataset_artifact, allow_pickle=False) as dataset:
            fallback_trial_ids = dataset["trial_ids"].astype(np.int64)

    candidate = _index_runs(
        arguments.candidate_runs,
        arguments.candidate_model,
        fallback_trial_ids,
        arguments.temporal_blocks,
    )
    baseline = _index_runs(
        arguments.baseline_runs,
        arguments.baseline_model,
        fallback_trial_ids,
        arguments.temporal_blocks,
    )
    if set(candidate) != set(baseline):
        raise ValueError("candidate and baseline seeds do not match")
    seeds = sorted(candidate)

    candidate_mae = []
    baseline_mae = []
    candidate_rmse = []
    baseline_rmse = []
    horizon_differences = []
    trial_differences = []
    trial_ids_reference = None
    paired_runs = []

    for seed in seeds:
        current = candidate[seed]
        reference = baseline[seed]
        for key in ("truth", "anchors", "cluster_ids"):
            if not np.array_equal(current[key], reference[key]):
                raise ValueError(f"seed {seed} has mismatched {key}")
        truth = current["truth"]
        candidate_error = current["prediction"] - truth
        baseline_error = reference["prediction"] - truth
        current_candidate_mae = float(np.mean(np.abs(candidate_error)))
        current_baseline_mae = float(np.mean(np.abs(baseline_error)))
        current_candidate_rmse = float(np.sqrt(np.mean(candidate_error**2)))
        current_baseline_rmse = float(np.sqrt(np.mean(baseline_error**2)))
        candidate_mae.append(current_candidate_mae)
        baseline_mae.append(current_baseline_mae)
        candidate_rmse.append(current_candidate_rmse)
        baseline_rmse.append(current_baseline_rmse)
        horizon_differences.append(
            np.mean(np.abs(candidate_error), axis=0)
            - np.mean(np.abs(baseline_error), axis=0)
        )

        trial_ids = np.unique(current["cluster_ids"])
        if trial_ids_reference is None:
            trial_ids_reference = trial_ids
        elif not np.array_equal(trial_ids_reference, trial_ids):
            raise ValueError("physical trial IDs differ across seeds")
        seed_trial_differences = []
        for trial_id in trial_ids:
            mask = current["cluster_ids"] == trial_id
            seed_trial_differences.append(
                float(
                    np.mean(np.abs(candidate_error[mask]))
                    - np.mean(np.abs(baseline_error[mask]))
                )
            )
        trial_differences.append(seed_trial_differences)
        paired_runs.append(
            {
                "seed": seed,
                "candidate_run": current["run_directory"],
                "baseline_run": reference["run_directory"],
                "candidate_mae": current_candidate_mae,
                "baseline_mae": current_baseline_mae,
                "mae_difference": current_candidate_mae - current_baseline_mae,
                "candidate_rmse": current_candidate_rmse,
                "baseline_rmse": current_baseline_rmse,
                "rmse_difference": current_candidate_rmse - current_baseline_rmse,
            }
        )

    candidate_mae = np.asarray(candidate_mae)
    baseline_mae = np.asarray(baseline_mae)
    candidate_rmse = np.asarray(candidate_rmse)
    baseline_rmse = np.asarray(baseline_rmse)
    horizon_differences = np.asarray(horizon_differences)
    trial_differences = np.asarray(trial_differences)
    mean_horizon_difference = horizon_differences.mean(axis=0)

    cluster_axis = (
        "contiguous_evaluation_block"
        if arguments.temporal_blocks
        else "physical_trial"
    )
    limitations = (
        [
            "Contiguous temporal blocks reduce window-level pseudoreplication "
            "but are not independent devices.",
            "Initialization seeds quantify optimization variability, not new "
            "physical systems.",
            "The hierarchical bootstrap is a clustered uncertainty summary, "
            "not proof of independence.",
        ]
        if arguments.temporal_blocks
        else [
            "Only three physical trials are available per held-out NIST regime.",
            "Initialization seeds quantify optimization variability, not new "
            "physical systems.",
            "The hierarchical bootstrap is a clustered uncertainty summary, "
            "not proof of independence.",
        ]
    )
    return {
        "candidate_model": arguments.candidate_model,
        "baseline_model": arguments.baseline_model,
        "seeds": seeds,
        "cluster_axis": cluster_axis,
        "cluster_ids": trial_ids_reference.tolist(),
        "candidate_mae": _mean_std(candidate_mae),
        "baseline_mae": _mean_std(baseline_mae),
        "paired_mae_difference": _mean_std(candidate_mae - baseline_mae),
        "candidate_rmse": _mean_std(candidate_rmse),
        "baseline_rmse": _mean_std(baseline_rmse),
        "paired_rmse_difference": _mean_std(candidate_rmse - baseline_rmse),
        "candidate_seed_wins_mae": int(np.sum(candidate_mae < baseline_mae)),
        "mean_mae_difference_by_horizon": mean_horizon_difference.tolist(),
        "candidate_horizon_wins": int(np.sum(mean_horizon_difference < 0)),
        "mae_difference_by_seed_cluster": trial_differences.tolist(),
        "hierarchical_paired_bootstrap": hierarchical_paired_bootstrap(
            trial_differences,
            arguments.bootstrap_repeats,
            arguments.seed,
            second_axis=cluster_axis,
        ),
        "paired_runs": paired_runs,
        "limitations": limitations,
    }


def main() -> None:
    arguments = parse_arguments()
    summary = summarize(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
