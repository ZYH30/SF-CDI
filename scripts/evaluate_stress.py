#!/usr/bin/env python3
"""Replay frozen checkpoints under a minimal, preregistered stress matrix."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from sfcdi.data import (
    SeriesWindowDataset,
    Standardizer,
    chronological_split_ids,
    deterministic_subsample,
    load_metropt3,
    load_nist_ur5,
    valid_window_anchors,
)
from sfcdi.models import (
    model_from_manifest,
    separated_itransformer_from_manifest,
)
from sfcdi.stress import (
    PerturbedWindowDataset,
    STRESS_CONDITIONS,
    unique_manifest_feature_indices,
)
from sfcdi.training import predict_torch, regression_metrics


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--perturbation-seed", type=int, default=20260723)
    parser.add_argument(
        "--conditions", nargs="+", choices=STRESS_CONDITIONS, default=STRESS_CONDITIONS
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_model_signature(manifest: dict) -> str:
    """Hash only the manifest fields that determine model inputs and gates."""

    structure = {
        "regime_ids": manifest["regime_ids"],
        "selected_pairs": [
            {
                "feature_index": pair["feature_index"],
                "feature": pair["feature"],
                "lag": pair["lag"],
                "available_regimes": pair["available_regimes"],
                "governance": pair["governance"],
            }
            for pair in manifest["selected_pairs"]
        ],
    }
    payload = json.dumps(
        structure, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def standardizer_from_record(record: dict) -> Standardizer:
    return Standardizer(
        mean=np.asarray(record["mean"], dtype=np.float64),
        scale=np.asarray(record["scale"], dtype=np.float64),
    )


def load_series_and_split(config: dict):
    dataset = config["dataset"]
    if dataset.get("kind", "metropt3") == "metropt3":
        series = load_metropt3(
            REPOSITORY_ROOT / dataset["csv_path"],
            target=dataset["target"],
            feature_names=dataset.get("feature_names"),
            max_gap_seconds=float(dataset["max_gap_seconds"]),
        )
        split_ids = chronological_split_ids(
            len(series.target),
            train_fraction=float(config["split"]["train_fraction"]),
            validation_fraction=float(config["split"]["validation_fraction"]),
        )
    elif dataset["kind"] == "nist_ur5":
        series = load_nist_ur5(
            REPOSITORY_ROOT / dataset["processed_path"],
            protocol=dataset.get("protocol", "trial_replicate"),
        )
        split_ids = series.predefined_split_ids.copy()
    else:
        raise ValueError(f"Unsupported dataset kind: {dataset['kind']}")
    return series, split_ids


def build_frozen_model(model_name: str, config: dict, manifest: dict):
    model_config = config["model"]
    sequence_length = int(config["window"]["sequence_length"])
    prediction_length = int(config["window"]["prediction_length"])
    if model_name != "sf_cdi":
        raise ValueError(f"Unsupported frozen stress model: {model_name}")
    if config["dataset"]["kind"] == "nist_ur5":
        return separated_itransformer_from_manifest(
            manifest,
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            target_probe_lags=model_config["target_probe_lags"],
            d_model=int(model_config["d_model"]),
            heads=int(model_config["heads"]),
            layers=int(model_config["layers"]),
            d_ff=int(model_config["d_ff"]),
            dropout=float(model_config["dropout"]),
        )
    if config["dataset"]["kind"] == "metropt3":
        return model_from_manifest(
            manifest,
            prediction_length=prediction_length,
            target_probe_lags=model_config["target_probe_lags"],
            hidden_size=int(model_config["hidden_size"]),
            target_hidden_size=int(model_config["target_hidden_size"]),
            dropout=float(model_config["dropout"]),
            use_regime_gating=True,
        )
    raise ValueError(f"Unsupported dataset kind: {config['dataset']['kind']}")


def load_frozen_state_dict(model: torch.nn.Module, state: dict) -> tuple[list[str], list[str]]:
    """Load every trainable parameter and registered buffer strictly."""

    model.load_state_dict(state, strict=True)
    return [], []


def sample_standard_deviation(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def per_group_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    names: list[str] | None = None,
) -> dict:
    result = {}
    for group in np.unique(groups):
        mask = groups == group
        group_id = int(group)
        record = {
            "n_windows": int(mask.sum()),
            **regression_metrics(truth[mask], prediction[mask]),
        }
        if names is not None and 0 <= group_id < len(names):
            record["name"] = names[group_id]
        result[str(group_id)] = record
    return result


def main() -> None:
    arguments = parse_arguments()
    run_dirs = [path.resolve() for path in arguments.run_dirs]
    if len(run_dirs) < 2:
        raise ValueError("Stress summaries require at least two frozen checkpoints")
    output_path = arguments.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)

    wants_cuda = any(load_json(path / "config.json").get("device") == "cuda" for path in run_dirs)
    device = torch.device("cuda" if wants_cuda and torch.cuda.is_available() else "cpu")
    if wants_cuda and device.type != "cuda":
        raise RuntimeError("CUDA checkpoints requested but CUDA is unavailable")

    run_results = []
    prediction_artifact: dict[str, np.ndarray] = {}
    reference_truth = None
    reference_anchors = None
    dataset_name = None
    evaluation_split_name = None
    model_signature_hash = None
    source_manifest_hashes: list[str] = []

    for run_index, run_dir in enumerate(run_dirs):
        config = load_json(run_dir / "config.json")
        result_record = load_json(run_dir / "results.json")
        manifest = load_json(run_dir / "driver_manifest.json")
        preprocessing = load_json(run_dir / "preprocessing.json")
        if arguments.model not in result_record["results"]:
            raise KeyError(f"{arguments.model} missing from {run_dir}")
        current_signature = manifest_model_signature(manifest)
        source_manifest_hashes.append(manifest["manifest_sha256"])
        if model_signature_hash is None:
            model_signature_hash = current_signature
        elif current_signature != model_signature_hash:
            raise ValueError(
                "Stress runs do not share the same driver/lag/gating structure"
            )

        series, split_ids = load_series_and_split(config)
        dataset_name = series.dataset_name if dataset_name is None else dataset_name
        if series.dataset_name != dataset_name:
            raise ValueError("Stress run directories mix datasets")
        evaluation_split = config.get("evaluation", {}).get("split", "test")
        evaluation_split_name = (
            evaluation_split if evaluation_split_name is None else evaluation_split_name
        )
        if evaluation_split != evaluation_split_name:
            raise ValueError("Stress run directories mix evaluation splits")
        split_number = {"validation": 1, "test": 2}[evaluation_split]

        feature_scaler = standardizer_from_record(preprocessing["feature_scaler"])
        target_scaler = standardizer_from_record(preprocessing["target_scaler"])
        features_z = feature_scaler.transform(series.features)
        target_z = target_scaler.transform(series.target[:, None]).reshape(-1)
        sequence_length = int(config["window"]["sequence_length"])
        prediction_length = int(config["window"]["prediction_length"])
        anchors = deterministic_subsample(
            valid_window_anchors(
                split_ids,
                series.segments,
                split_number,
                sequence_length,
                prediction_length,
            ),
            config["window"][f"max_{evaluation_split}_windows"],
        )
        base_dataset = SeriesWindowDataset(
            features_z,
            target_z,
            series.regimes,
            anchors,
            sequence_length,
            prediction_length,
        )
        selected_indices = unique_manifest_feature_indices(manifest)
        model = build_frozen_model(arguments.model, config, manifest).to(device)
        checkpoint_path = run_dir / f"{arguments.model}_checkpoint.pt"
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        missing_keys, unexpected_keys = load_frozen_state_dict(model, state)
        model.eval()

        saved_predictions = np.load(
            run_dir / f"{arguments.model}_predictions.npz", allow_pickle=False
        )
        saved_prediction = saved_predictions["prediction"]
        saved_truth = saved_predictions["truth"]
        saved_anchors = saved_predictions["anchors"]
        if not np.array_equal(saved_anchors, anchors):
            raise RuntimeError("Reconstructed anchors differ from the frozen run")

        metric_regimes = (
            series.evaluation_regimes
            if series.evaluation_regimes is not None
            else series.regimes
        )[anchors]
        trial_ids = series.trial_ids[anchors] if series.trial_ids is not None else None
        run_condition_results = {}
        clean_mae = None
        for condition in arguments.conditions:
            dataset = PerturbedWindowDataset(
                base_dataset,
                selected_indices,
                condition,
                perturbation_seed=arguments.perturbation_seed,
            )
            loader = DataLoader(
                dataset,
                batch_size=int(config["training"]["batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            prediction, truth, _, predicted_anchors = predict_torch(
                model, loader, device, target_scaler
            )
            if not np.array_equal(predicted_anchors, anchors):
                raise RuntimeError("Stress prediction order changed")
            metrics = regression_metrics(truth, prediction)
            if condition == "clean":
                max_prediction_difference = float(
                    np.max(np.abs(prediction - saved_prediction))
                )
                max_truth_difference = float(np.max(np.abs(truth - saved_truth)))
                if max_prediction_difference > 2e-6 or max_truth_difference > 2e-6:
                    raise RuntimeError(
                        "Clean checkpoint replay does not reproduce the frozen predictions"
                    )
                clean_mae = metrics["mae"]
            if clean_mae is None:
                raise RuntimeError("The clean condition must be evaluated first")
            run_condition_results[condition] = {
                "metrics": metrics,
                "mae_change_from_clean": float(metrics["mae"] - clean_mae),
                "mae_change_percent": float(
                    100.0 * (metrics["mae"] - clean_mae) / clean_mae
                ),
                "per_regime": per_group_metrics(
                    truth, prediction, metric_regimes, series.regime_names
                ),
                "per_trial": (
                    per_group_metrics(truth, prediction, trial_ids)
                    if trial_ids is not None
                    else None
                ),
            }
            prediction_artifact[f"run{run_index}_{condition}"] = prediction.astype(
                np.float32
            )

        if "clean" in arguments.conditions:
            run_condition_results["clean"]["replay_max_abs_prediction_difference"] = (
                max_prediction_difference
            )
            run_condition_results["clean"]["replay_max_abs_truth_difference"] = (
                max_truth_difference
            )
        reference_truth = truth if reference_truth is None else reference_truth
        reference_anchors = anchors if reference_anchors is None else reference_anchors
        if not np.allclose(truth, reference_truth) or not np.array_equal(
            anchors, reference_anchors
        ):
            raise RuntimeError("Frozen runs do not share the same evaluation samples")
        run_results.append(
            {
                "run_id": result_record["run_id"],
                "seed": int(config["seed"]),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "checkpoint_compatibility": {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                },
                "selected_feature_indices": selected_indices,
                "selected_feature_names": [
                    series.feature_names[index] for index in selected_indices
                ],
                "top_driver": series.feature_names[selected_indices[0]],
                "bottom_driver": series.feature_names[selected_indices[-1]],
                "conditions": run_condition_results,
            }
        )

    summary = {}
    for condition in arguments.conditions:
        maes = [run["conditions"][condition]["metrics"]["mae"] for run in run_results]
        rmses = [run["conditions"][condition]["metrics"]["rmse"] for run in run_results]
        changes = [
            run["conditions"][condition]["mae_change_percent"] for run in run_results
        ]
        summary[condition] = {
            "mae_mean": float(np.mean(maes)),
            "mae_sample_sd": sample_standard_deviation(maes),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_sample_sd": sample_standard_deviation(rmses),
            "mae_change_percent_mean": float(np.mean(changes)),
            "mae_change_percent_sample_sd": sample_standard_deviation(changes),
            "runs_worse_than_clean": int(sum(value > 0 for value in changes)),
        }

    record = {
        "schema_version": "1.0",
        "dataset": dataset_name,
        "evaluation_split": evaluation_split_name,
        "model": arguments.model,
        "manifest_model_signature_sha256": model_signature_hash,
        "source_manifest_sha256s": source_manifest_hashes,
        "perturbation_seed": arguments.perturbation_seed,
        "conditions": list(arguments.conditions),
        "n_checkpoints": len(run_results),
        "summary": summary,
        "runs": run_results,
        "interpretation_boundary": (
            "Inference-time stress evaluation of frozen checkpoints; no retraining or "
            "post-stress model selection was performed."
        ),
    }
    output_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        truth=reference_truth.astype(np.float32),
        anchors=reference_anchors,
        **prediction_artifact,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
