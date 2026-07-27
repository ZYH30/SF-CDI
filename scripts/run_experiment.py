#!/usr/bin/env python3
"""Run the paper-facing SF-CDI experiments and reported baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import pandas as pd
import scipy
import sklearn
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
from sfcdi.discovery import discover_lagged_drivers, save_manifest
from sfcdi.models import (
    LaggedDualPathNetwork,
    RepositoryDLinear,
    RepositoryITransformer,
    RepositoryTimesNet,
    SeparatedCausalITransformer,
    TargetOnlyLSTM,
    model_from_manifest,
    separated_itransformer_from_manifest,
)
from sfcdi.training import (
    blocked_cross_fitted_shortcut_probe,
    independent_shortcut_probe,
    paired_mae_bootstrap,
    parameter_count,
    predict_torch,
    regression_metrics,
    set_reproducible_seed,
    train_torch_model,
)


MODEL_CHOICES = (
    "sf_cdi",
    "sf_cdi_no_control",
    "sf_cdi_joint",
    "sf_cdi_self_supervised",
    "sf_cdi_closed_form",
    "target_lstm",
    "itransformer",
    "dlinear",
    "timesnet",
)
DUAL_PATH_TYPES = (LaggedDualPathNetwork, SeparatedCausalITransformer)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce SF-CDI and baseline experiments from a JSON config."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Override the device recorded in the configuration.",
    )
    parser.add_argument(
        "--rediscover",
        action="store_true",
        help="Run train-only causal discovery instead of loading the frozen manifest.",
    )
    return parser.parse_args()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def command_output(arguments: list[str]) -> str:
    result = subprocess.run(
        arguments,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip()


def file_sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def environment_record(device: torch.device) -> dict:
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_mib": properties.total_memory / 2**20,
        }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
        },
        "device": str(device),
        "gpu": gpu,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "command": " ".join(sys.argv),
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def model_setting(config: dict, model_name: str, key: str):
    plural = {
        "batch_size": "model_batch_sizes",
        "epochs": "model_epochs",
        "patience": "model_patiences",
    }[key]
    return config["training"].get(plural, {}).get(
        model_name, config["training"][key]
    )


def make_loaders(
    train_dataset,
    validation_dataset,
    evaluation_dataset,
    config: dict,
    model_name: str,
):
    generator = torch.Generator().manual_seed(int(config["seed"]))
    common = {
        "batch_size": int(model_setting(config, model_name, "batch_size")),
        "num_workers": int(config["training"].get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        DataLoader(validation_dataset, shuffle=False, **common),
        DataLoader(evaluation_dataset, shuffle=False, **common),
        DataLoader(train_dataset, shuffle=False, **common),
    )


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


def load_series(config: dict):
    dataset = config["dataset"]
    kind = dataset["kind"]
    if kind == "metropt3":
        data_path = (REPOSITORY_ROOT / dataset["csv_path"]).resolve()
        series = load_metropt3(
            data_path,
            target=dataset["target"],
            feature_names=dataset.get("feature_names"),
            max_gap_seconds=float(dataset["max_gap_seconds"]),
        )
        split_ids = chronological_split_ids(
            len(series.target),
            train_fraction=float(config["split"]["train_fraction"]),
            validation_fraction=float(config["split"]["validation_fraction"]),
        )
    elif kind == "nist_ur5":
        data_path = (REPOSITORY_ROOT / dataset["processed_path"]).resolve()
        series = load_nist_ur5(
            data_path, protocol=dataset.get("protocol", "cold_start_ood")
        )
        split_ids = series.predefined_split_ids.copy()
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")
    return series, split_ids, data_path


def validate_frozen_manifest(manifest: dict, feature_names: list[str]) -> None:
    if not manifest.get("selected_pairs"):
        raise ValueError("The frozen driver manifest has no selected pairs")
    for pair in manifest["selected_pairs"]:
        index = int(pair["feature_index"])
        if not 0 <= index < len(feature_names):
            raise ValueError(f"Manifest feature index is out of range: {index}")
        if feature_names[index] != pair["feature"]:
            raise ValueError(
                "Manifest feature mapping does not match the loaded dataset: "
                f"{pair['feature']} != {feature_names[index]}"
            )


def obtain_manifest(
    config: dict,
    series,
    split_ids: np.ndarray,
    data_path: Path,
    rediscover: bool,
) -> tuple[dict, str]:
    manifest_path = config["discovery"].get("manifest_path")
    if manifest_path and not rediscover:
        resolved = (REPOSITORY_ROOT / manifest_path).resolve()
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
        validate_frozen_manifest(manifest, series.feature_names)
        return manifest, f"frozen:{resolved.relative_to(REPOSITORY_ROOT)}"

    train_rows = np.flatnonzero(split_ids == 0)
    settings = config["discovery"]
    manifest = discover_lagged_drivers(
        features=series.features[train_rows],
        target=series.target[train_rows],
        regimes=series.regimes[train_rows],
        segments=series.segments[train_rows],
        feature_names=series.feature_names,
        candidate_lags=settings["candidate_lags"],
        conditioning_target_lags=settings["conditioning_target_lags"],
        max_samples=int(settings["max_samples"]),
        n_blocks=int(settings["n_blocks"]),
        ridge_alpha=float(settings["ridge_alpha"]),
        hac_bandwidth=int(settings["hac_bandwidth"]),
        fdr_level=float(settings["fdr_level"]),
        minimum_effect=float(settings["minimum_effect"]),
        minimum_stability=float(settings["minimum_stability"]),
        maximum_pairs=int(settings["maximum_pairs"]),
        maximum_lags_per_feature=int(settings["maximum_lags_per_feature"]),
        minimum_regime_samples=int(settings["minimum_regime_samples"]),
        crossfit_within_segments=bool(settings["crossfit_within_segments"]),
        screening_horizon=int(settings["screening_horizon"]),
        source_path=data_path,
        train_start=str(series.timestamps[train_rows[0]]),
        train_end=str(series.timestamps[train_rows[-1]]),
    )
    return manifest, "train_only_discovery"


def build_model(
    model_name: str,
    dataset_kind: str,
    manifest: dict,
    feature_count: int,
    sequence_length: int,
    prediction_length: int,
    config: dict,
):
    settings = config["model"]
    schedule = "joint_minimax"
    adversarial_weight = 0.0
    covariance_weight = 0.0

    if model_name == "target_lstm":
        if dataset_kind != "metropt3":
            raise ValueError("The target-only LSTM is reported for MetroPT-3")
        model = TargetOnlyLSTM(
            prediction_length,
            hidden_size=int(settings["target_hidden_size"]),
            dropout=float(settings["dropout"]),
        )
    elif model_name == "itransformer":
        model = RepositoryITransformer(
            feature_count,
            sequence_length,
            prediction_length,
            d_model=int(settings["d_model"]),
            heads=int(settings["heads"]),
            layers=int(settings["layers"]),
            d_ff=int(settings["d_ff"]),
            dropout=float(settings["dropout"]),
        )
    elif model_name == "dlinear":
        model = RepositoryDLinear(
            feature_count,
            sequence_length,
            prediction_length,
            moving_average=int(settings["dlinear_kernel"]),
        )
    elif model_name == "timesnet":
        model = RepositoryTimesNet(
            feature_count,
            sequence_length,
            prediction_length,
            d_model=int(settings["timesnet_d_model"]),
            d_ff=int(settings["timesnet_d_ff"]),
            layers=int(settings["timesnet_layers"]),
            top_k=int(settings["timesnet_top_k"]),
            num_kernels=int(settings["timesnet_num_kernels"]),
            dropout=float(settings["dropout"]),
        )
    elif model_name in {
        "sf_cdi",
        "sf_cdi_no_control",
        "sf_cdi_joint",
        "sf_cdi_self_supervised",
        "sf_cdi_closed_form",
    }:
        if model_name in {
            "sf_cdi_no_control",
            "sf_cdi_joint",
        } and dataset_kind != "metropt3":
            raise ValueError(
                "The reported control and optimizer-boundary ablations "
                "are defined for MetroPT-3"
            )
        if model_name in {
            "sf_cdi_self_supervised",
            "sf_cdi_closed_form",
        } and dataset_kind != "nist_ur5":
            raise ValueError(
                "The reported strict variants are defined for NIST UR5"
            )
        if dataset_kind == "nist_ur5":
            closed_form = model_name == "sf_cdi_closed_form"
            self_supervised = model_name == "sf_cdi_self_supervised"
            model = separated_itransformer_from_manifest(
                manifest,
                sequence_length=sequence_length,
                prediction_length=prediction_length,
                target_probe_lags=settings["target_probe_lags"],
                d_model=int(settings["d_model"]),
                heads=int(settings["heads"]),
                layers=int(settings["layers"]),
                d_ff=int(settings["d_ff"]),
                dropout=float(settings["dropout"]),
                full_token_control=closed_form or self_supervised,
                driver_reconstruction=closed_form or self_supervised,
                closed_form_residualization=closed_form,
            )
        else:
            model = model_from_manifest(
                manifest,
                prediction_length=prediction_length,
                target_probe_lags=settings["target_probe_lags"],
                hidden_size=int(settings["hidden_size"]),
                target_hidden_size=int(settings["target_hidden_size"]),
                dropout=float(settings["dropout"]),
                use_regime_gating=True,
            )

        if model_name == "sf_cdi_closed_form":
            schedule = "closed_form_innovation_static_freeze"
        elif model_name == "sf_cdi_self_supervised":
            schedule = "self_supervised_pretrain_static_freeze"
            adversarial_weight = float(config["adversarial"]["weight"])
            covariance_weight = float(config["shortcut"]["covariance_weight"])
        elif model_name == "sf_cdi_joint":
            schedule = "joint_minimax"
            adversarial_weight = float(config["adversarial"]["weight"])
            covariance_weight = float(config["shortcut"]["covariance_weight"])
        else:
            schedule = "minimax_pretrain_static_freeze"
            if model_name == "sf_cdi":
                adversarial_weight = float(config["adversarial"]["weight"])
                covariance_weight = float(config["shortcut"]["covariance_weight"])
    else:
        raise ValueError(f"Unsupported trainable model: {model_name}")

    return model, schedule, adversarial_weight, covariance_weight


def main() -> None:
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if arguments.seed is not None:
        config["seed"] = arguments.seed
    if arguments.device is not None:
        config["device"] = arguments.device
    models = arguments.models or config["models"]
    if not models:
        raise ValueError("At least one model is required")

    config_hash = sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    set_reproducible_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))

    dataset_kind = config["dataset"]["kind"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = arguments.run_id or f"{dataset_kind}-{timestamp}-{config_hash}"
    output_root = REPOSITORY_ROOT / config["output_root"] / run_id
    output_root.mkdir(parents=True, exist_ok=False)
    dump_json(output_root / "config.json", config)
    dump_json(output_root / "environment.json", environment_record(device))

    series, split_ids, data_path = load_series(config)
    manifest, manifest_origin = obtain_manifest(
        config, series, split_ids, data_path, arguments.rediscover
    )
    save_manifest(manifest, output_root / "driver_manifest.json")

    train_rows = np.flatnonzero(split_ids == 0)
    feature_scaler = Standardizer.fit(series.features[train_rows])
    target_scaler = Standardizer.fit(series.target[train_rows, None])
    features_z = feature_scaler.transform(series.features)
    target_z = target_scaler.transform(series.target[:, None]).reshape(-1)
    dump_json(
        output_root / "preprocessing.json",
        {
            "fit_partition": "train_only",
            "source_sha256": file_sha256(data_path),
            "feature_names": series.feature_names,
            "feature_scaler": feature_scaler.to_dict(),
            "target": series.target_name,
            "target_scaler": target_scaler.to_dict(),
            "forbidden_or_excluded_inputs": config["dataset"].get(
                "forbidden_or_excluded_inputs", []
            ),
        },
    )

    sequence_length = int(config["window"]["sequence_length"])
    prediction_length = int(config["window"]["prediction_length"])
    split_names = ("train", "validation", "test")
    anchors = {
        name: deterministic_subsample(
            valid_window_anchors(
                split_ids,
                series.segments,
                split_id,
                sequence_length,
                prediction_length,
            ),
            config["window"][f"max_{name}_windows"],
        )
        for split_id, name in enumerate(split_names)
    }
    datasets = {
        name: SeriesWindowDataset(
            features_z,
            target_z,
            series.regimes,
            split_anchors,
            sequence_length,
            prediction_length,
        )
        for name, split_anchors in anchors.items()
    }
    evaluation_split = config["evaluation"]["split"]
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation.split must be validation or test")
    evaluation_anchors = anchors[evaluation_split]
    evaluation_regimes = (
        series.evaluation_regimes
        if series.evaluation_regimes is not None
        else series.regimes
    )[evaluation_anchors]
    model_evaluation_regimes = series.regimes[evaluation_anchors]
    evaluation_trials = (
        series.trial_ids[evaluation_anchors]
        if series.trial_ids is not None
        else None
    )
    dump_json(
        output_root / "split_manifest.json",
        {
            "protocol": series.split_protocol
            or "chronological_60_20_20_gap_safe",
            "evaluation_split": evaluation_split,
            "row_counts": {
                name: int(np.sum(split_ids == split_id))
                for split_id, name in enumerate(split_names)
            },
            "window_counts": {name: int(len(value)) for name, value in anchors.items()},
            "segment_count": int(series.segments.max() + 1),
            "manifest_origin": manifest_origin,
        },
    )

    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    truth_reference = None
    for model_name in models:
        print(f"START model={model_name}", flush=True)
        set_reproducible_seed(int(config["seed"]))
        train_loader, validation_loader, evaluation_loader, probe_train_loader = (
            make_loaders(
                datasets["train"],
                datasets["validation"],
                datasets[evaluation_split],
                config,
                model_name,
            )
        )

        model, schedule, adversarial_weight, covariance_weight = build_model(
            model_name,
            dataset_kind,
            manifest,
            len(series.feature_names),
            sequence_length,
            prediction_length,
            config,
        )
        is_dual_path = isinstance(model, DUAL_PATH_TYPES)
        optimization = config["optimization"]
        training = train_torch_model(
            model,
            train_loader,
            validation_loader,
            device,
            epochs=int(model_setting(config, model_name, "epochs")),
            learning_rate=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            patience=int(model_setting(config, model_name, "patience")),
            adversarial_weight=adversarial_weight,
            shortcut_covariance_weight=covariance_weight,
            adversary_learning_rate=float(config["adversarial"]["learning_rate"]),
            adversary_steps=int(config["adversarial"]["steps"]),
            adversarial_warmup_epochs=int(config["adversarial"]["warmup_epochs"]),
            gradient_clip=float(config["training"]["gradient_clip"]),
            main_steps=int(optimization["main_steps"]) if is_dual_path else 1,
            optimization_mode=schedule,
            driver_pretrain_epochs=int(optimization["driver_pretrain_epochs"]),
            driver_encoder_pretrain_epochs=int(
                optimization["driver_encoder_pretrain_epochs"]
            ),
            driver_reconstruction_weight=float(
                optimization["driver_reconstruction_weight"]
            ),
            closed_form_residualizer_ridge=float(
                optimization["closed_form_residualizer_ridge"]
            ),
        )
        prediction, truth, predicted_regimes, predicted_anchors = predict_torch(
            model, evaluation_loader, device, target_scaler
        )
        if not np.array_equal(predicted_anchors, evaluation_anchors):
            raise RuntimeError("Prediction order differs from the split manifest")
        if not np.array_equal(predicted_regimes, model_evaluation_regimes):
            raise RuntimeError("Prediction regimes differ from the split manifest")
        torch.save(model.state_dict(), output_root / f"{model_name}_checkpoint.pt")
        dump_json(output_root / f"{model_name}_history.json", training.history)
        model_result = {
            "training": {
                "schedule": schedule,
                "epochs_completed": training.epochs_completed,
                "best_validation_mse_standardized": training.best_validation_mse,
                "elapsed_seconds": training.elapsed_seconds,
                "peak_gpu_memory_mib": training.peak_gpu_memory_mib,
                "batch_size": int(
                    model_setting(config, model_name, "batch_size")
                ),
                "optimization_audit": training.optimization_audit,
            },
            "parameters": parameter_count(model),
        }
        if is_dual_path:
            model_result["shortcut_probe"] = independent_shortcut_probe(
                model, probe_train_loader, evaluation_loader, device
            )
            model_result["shortcut_probe_evaluation_crossfit"] = (
                blocked_cross_fitted_shortcut_probe(
                    model, evaluation_loader, device
                )
            )

        model_result["metrics"] = regression_metrics(truth, prediction)
        model_result["per_regime"] = per_group_metrics(
            truth, prediction, evaluation_regimes, series.regime_names
        )
        if evaluation_trials is not None:
            model_result["per_trial"] = per_group_metrics(
                truth, prediction, evaluation_trials
            )
        results[model_name] = model_result
        predictions[model_name] = prediction
        if truth_reference is None:
            truth_reference = truth
        elif not np.allclose(truth, truth_reference):
            raise RuntimeError("Compared models do not share identical labels")
        np.savez_compressed(
            output_root / f"{model_name}_predictions.npz",
            prediction=prediction.astype(np.float32),
            truth=truth.astype(np.float32),
            anchors=evaluation_anchors,
            regimes=evaluation_regimes,
            trial_ids=(
                evaluation_trials
                if evaluation_trials is not None
                else np.full(len(evaluation_anchors), -1, dtype=np.int64)
            ),
        )
        dump_json(output_root / "results.partial.json", results)
        metrics = model_result["metrics"]
        print(
            f"DONE model={model_name} MAE={metrics['mae']:.6f} "
            f"RMSE={metrics['rmse']:.6f}",
            flush=True,
        )

    comparisons = {}
    if "sf_cdi" in predictions:
        for other_name, other_prediction in predictions.items():
            if other_name != "sf_cdi":
                comparisons[f"sf_cdi_vs_{other_name}"] = paired_mae_bootstrap(
                    truth_reference,
                    predictions["sf_cdi"],
                    other_prediction,
                    seed=int(config["seed"]),
                )
    dump_json(
        output_root / "results.json",
        {
            "run_id": run_id,
            "config_hash": config_hash,
            "status": "completed",
            "dataset": series.dataset_name,
            "evaluation_split": evaluation_split,
            "target": series.target_name,
            "models_in_order": models,
            "manifest_sha256": manifest["manifest_sha256"],
            "results": results,
            "paired_comparisons": comparisons,
        },
    )
    print(f"COMPLETE output={output_root}", flush=True)


if __name__ == "__main__":
    main()
