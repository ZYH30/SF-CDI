"""Training, evaluation, baselines, and frozen shortcut probes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import torch
from scipy.stats import bootstrap
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader

from .data import Standardizer, target_lag_matrix
from .models import LaggedDualPathNetwork, SeparatedCausalITransformer


DUAL_PATH_TYPES = (LaggedDualPathNetwork, SeparatedCausalITransformer)
TRAINING_MODES = {
    "joint_minimax",
    "minimax_pretrain_static_freeze",
    "self_supervised_pretrain_static_freeze",
    "closed_form_innovation_static_freeze",
}


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - truth
    mae_by_horizon = np.mean(np.abs(error), axis=0)
    rmse_by_horizon = np.sqrt(np.mean(error**2, axis=0))
    denominator = np.abs(truth) + np.abs(prediction)
    smape_terms = np.divide(
        2.0 * np.abs(error),
        denominator,
        out=np.zeros_like(error),
        where=denominator > 1e-8,
    )
    smape_by_horizon = 100.0 * np.mean(smape_terms, axis=0)
    return {
        "mae": float(np.mean(mae_by_horizon)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "smape_percent": float(np.mean(smape_by_horizon)),
        "mae_by_horizon": mae_by_horizon.tolist(),
        "rmse_by_horizon": rmse_by_horizon.tolist(),
        "smape_percent_by_horizon": smape_by_horizon.tolist(),
        "n_windows": int(len(truth)),
    }


def paired_mae_bootstrap(
    truth: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    seed: int = 0,
) -> dict:
    """Paired window bootstrap for diagnostic MAE(A)-MAE(B) intervals."""

    loss_difference = np.mean(
        np.abs(prediction_a - truth) - np.abs(prediction_b - truth), axis=1
    )
    if len(loss_difference) < 2:
        return {"difference": float(np.mean(loss_difference)), "ci95": [None, None]}
    result = bootstrap(
        (loss_difference,),
        statistic=np.mean,
        paired=True,
        vectorized=False,
        n_resamples=2_000,
        confidence_level=0.95,
        random_state=np.random.default_rng(seed),
        method="basic",
    )
    return {
        "difference": float(np.mean(loss_difference)),
        "ci95": [
            float(result.confidence_interval.low),
            float(result.confidence_interval.high),
        ],
        "interpretation": "negative means model A has lower MAE",
    }


def _move_batch(batch, device: torch.device):
    features, target_history, target_future, regimes, anchors = batch
    return (
        features.to(device, non_blocking=True),
        target_history.to(device, non_blocking=True),
        target_future.to(device, non_blocking=True),
        regimes.to(device, non_blocking=True),
        anchors,
    )


def standardized_cross_covariance_loss(
    representation: torch.Tensor, target_history_lags: torch.Tensor
) -> torch.Tensor:
    """Squared cross-correlation used as a probe-aligned shortcut penalty."""

    representation = representation.flatten(start_dim=1)
    representation = representation - representation.mean(dim=0, keepdim=True)
    target_history_lags = target_history_lags - target_history_lags.mean(
        dim=0, keepdim=True
    )
    representation = representation / (
        representation.std(dim=0, unbiased=False, keepdim=True) + 1e-5
    )
    target_history_lags = target_history_lags / (
        target_history_lags.std(dim=0, unbiased=False, keepdim=True) + 1e-5
    )
    cross_correlation = representation.transpose(0, 1) @ target_history_lags
    cross_correlation = cross_correlation / max(representation.shape[0], 1)
    return torch.mean(cross_correlation.square())


def minimax_main_objective(
    prediction_loss: torch.Tensor,
    shortcut_loss: torch.Tensor | None,
    confusion_loss: torch.Tensor | None,
    shortcut_weight: float,
    adversarial_weight: float,
) -> torch.Tensor:
    """Compose prediction, covariance, and reversed adversarial terms."""

    objective = prediction_loss
    if shortcut_loss is not None and shortcut_weight > 0:
        objective = objective + shortcut_weight * shortcut_loss
    if confusion_loss is not None and adversarial_weight > 0:
        objective = objective - adversarial_weight * confusion_loss
    return objective


def _set_requires_grad(parameters, enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def _clear_gradients(parameters) -> None:
    for parameter in parameters:
        parameter.grad = None


def _gradient_norm(parameters) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(torch.sum(parameter.grad.detach() ** 2))
    return float(np.sqrt(squared))


@torch.no_grad()
def validation_mse(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    squared_error = 0.0
    count = 0
    for batch in loader:
        features, target_history, target_future, regimes, _ = _move_batch(
            batch, device
        )
        output = model(features, target_history, regimes)
        prediction = output[0] if isinstance(output, tuple) else output
        squared_error += torch.sum((prediction - target_future) ** 2).item()
        count += target_future.numel()
    return squared_error / max(count, 1)


@torch.no_grad()
def fit_closed_form_driver_residualizer(
    model: SeparatedCausalITransformer,
    loader: DataLoader,
    device: torch.device,
    ridge: float,
) -> dict:
    """Fit a training-only target-history projection by normal equations."""

    if not model.closed_form_residualization or model.target_to_driver is None:
        raise ValueError("model has no closed-form driver residualizer")
    if ridge < 0:
        raise ValueError("closed-form residualizer ridge cannot be negative")
    was_training = model.training
    model.eval()
    controls = len(model.driver_residualizer_lags)
    output_size = model.driver_token_count * model.driver_token_size
    xtx = torch.zeros((controls + 1, controls + 1), dtype=torch.float64)
    xtz = torch.zeros((controls + 1, output_size), dtype=torch.float64)
    samples = 0
    for batch in loader:
        features, target_history, _, regimes, _ = _move_batch(batch, device)
        control = model.target_residualizer_input(target_history).double().cpu()
        raw_tokens = (
            model.raw_driver_control_tokens(features, regimes)
            .flatten(start_dim=1)
            .double()
            .cpu()
        )
        design = torch.cat(
            (control, torch.ones((len(control), 1), dtype=torch.float64)), dim=1
        )
        xtx.add_(design.transpose(0, 1) @ design)
        xtz.add_(design.transpose(0, 1) @ raw_tokens)
        samples += len(control)
    if samples < controls + 1:
        raise ValueError("insufficient samples for closed-form residualization")
    xtx_mean = xtx / samples
    xtz_mean = xtz / samples
    penalty = torch.eye(controls + 1, dtype=torch.float64) * ridge
    penalty[-1, -1] = 0.0
    coefficients = torch.linalg.solve(xtx_mean + penalty, xtz_mean)
    model.target_to_driver.weight.copy_(
        coefficients[:-1].transpose(0, 1).to(
            device=model.target_to_driver.weight.device,
            dtype=model.target_to_driver.weight.dtype,
        )
    )
    model.target_to_driver.bias.copy_(
        coefficients[-1].to(
            device=model.target_to_driver.bias.device,
            dtype=model.target_to_driver.bias.dtype,
        )
    )
    residual = xtz_mean - xtx_mean @ coefficients
    if was_training:
        model.train()
    return {
        "method": "train_only_ridge_normal_equations",
        "samples": int(samples),
        "control_lags": model.driver_residualizer_lags.tolist(),
        "ridge": float(ridge),
        "mean_abs_control_normal_equation_residual": float(
            residual[:-1].abs().mean()
        ),
        "max_abs_control_normal_equation_residual": float(
            residual[:-1].abs().max()
        ),
        "coefficient_l2": float(torch.linalg.vector_norm(coefficients[:-1])),
    }


@dataclass
class TrainResult:
    history: list[dict]
    best_validation_mse: float
    epochs_completed: int
    elapsed_seconds: float
    peak_gpu_memory_mib: float
    optimization_audit: dict


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    adversarial_weight: float = 0.0,
    shortcut_covariance_weight: float = 0.0,
    adversary_learning_rate: float = 1e-3,
    adversary_steps: int = 1,
    adversarial_warmup_epochs: int = 1,
    gradient_clip: float = 1.0,
    main_steps: int = 1,
    optimization_mode: str = "joint_minimax",
    driver_pretrain_epochs: int = 0,
    driver_reconstruction_weight: float = 1.0,
    driver_encoder_pretrain_epochs: int = 0,
    closed_form_residualizer_ridge: float = 1e-3,
) -> TrainResult:
    """Train a baseline, primary static-freeze model, or reported strict variant.

    ``joint_minimax`` trains all non-adversary parameters together and serves
    as the optimizer-boundary ablation. ``minimax_pretrain_static_freeze``
    learns driver utility and shortcut control jointly before freezing the
    complete driver representation. ``self_supervised_pretrain_static_freeze``
    blocks future-target gradients from the driver encoder throughout and
    trains that encoder by reconstruction plus shortcut control.
    ``closed_form_innovation_static_freeze``
    prevents future-target gradients from reaching the driver encoder from the
    first epoch, learns it by driver reconstruction, fits a training-only
    closed-form target-history projection, and then freezes it.
    """

    if optimization_mode not in TRAINING_MODES:
        raise ValueError(f"Unknown optimization mode: {optimization_mode}")
    model.to(device)
    is_dual_path = isinstance(model, DUAL_PATH_TYPES)
    if not is_dual_path and optimization_mode != "joint_minimax":
        raise ValueError("Static-freeze modes require a dual-path model")
    if main_steps < 1 or adversary_steps < 0:
        raise ValueError("main_steps must be positive and adversary_steps non-negative")
    if optimization_mode in {
        "minimax_pretrain_static_freeze",
        "self_supervised_pretrain_static_freeze",
    } and (
        driver_pretrain_epochs <= adversarial_warmup_epochs
    ):
        raise ValueError(
            "driver_pretrain_epochs must exceed adversarial_warmup_epochs"
        )
    if optimization_mode == "self_supervised_pretrain_static_freeze":
        if (
            not isinstance(model, SeparatedCausalITransformer)
            or model.driver_reconstruction is None
        ):
            raise ValueError(
                "self-supervised mode requires a separated driver "
                "reconstruction model"
            )
        if driver_reconstruction_weight <= 0:
            raise ValueError("driver reconstruction weight must be positive")
    if optimization_mode == "closed_form_innovation_static_freeze":
        if (
            not isinstance(model, SeparatedCausalITransformer)
            or not model.closed_form_residualization
            or model.driver_reconstruction is None
        ):
            raise ValueError(
                "closed-form mode requires reconstruction and residualization"
            )
        if driver_encoder_pretrain_epochs < 1:
            raise ValueError("closed-form mode requires encoder pretraining")
    if closed_form_residualizer_ridge < 0:
        raise ValueError("closed-form residualizer ridge cannot be negative")

    main_parameters = (
        model.main_parameters() if is_dual_path else list(model.parameters())
    )
    main_optimizer = torch.optim.AdamW(
        main_parameters, lr=learning_rate, weight_decay=weight_decay
    )
    driver_parameters = (
        list(model.driver_representation_parameters()) if is_dual_path else []
    )
    forecast_parameters = (
        list(model.forecast_given_driver_parameters()) if is_dual_path else []
    )
    forecast_optimizer = None
    driver_base_parameters = []
    driver_base_optimizer = None
    driver_frozen_parameters = []
    driver_pretraining_parameters = []
    driver_pretraining_optimizer = None
    if optimization_mode in {
        "minimax_pretrain_static_freeze",
        "self_supervised_pretrain_static_freeze",
        "closed_form_innovation_static_freeze",
    }:
        forecast_optimizer = torch.optim.AdamW(
            forecast_parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if optimization_mode == "closed_form_innovation_static_freeze":
        driver_base_parameters = list(model.driver_base_pretraining_parameters())
        driver_frozen_parameters = list(model.driver_pretraining_parameters())
        driver_base_optimizer = torch.optim.AdamW(
            driver_base_parameters, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimization_mode == "self_supervised_pretrain_static_freeze":
        driver_pretraining_parameters = list(
            model.driver_pretraining_parameters()
        )
        driver_pretraining_optimizer = torch.optim.AdamW(
            driver_pretraining_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    adversary_optimizer = None
    if is_dual_path and adversarial_weight > 0:
        adversary_optimizer = torch.optim.AdamW(
            model.adversary.parameters(),
            lr=adversary_learning_rate,
            weight_decay=weight_decay,
        )

    criterion = nn.MSELoss()
    best_validation = float("inf")
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict] = []
    closed_form_audit = None
    start_time = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        model.train()
        active_adversarial_weight = (
            adversarial_weight if epoch >= adversarial_warmup_epochs else 0.0
        )
        if optimization_mode == "joint_minimax":
            phase = "joint_minimax"
        elif optimization_mode == "minimax_pretrain_static_freeze":
            phase = (
                "driver_minimax_pretrain"
                if epoch < driver_pretrain_epochs
                else "static_frozen_forecast"
            )
            if epoch == driver_pretrain_epochs:
                _clear_gradients(driver_parameters)
                _set_requires_grad(driver_parameters, False)
                best_validation = float("inf")
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
        elif optimization_mode == "self_supervised_pretrain_static_freeze":
            phase = (
                "driver_self_supervised_pretrain"
                if epoch < driver_pretrain_epochs
                else "static_frozen_forecast"
            )
            if epoch == driver_pretrain_epochs:
                _clear_gradients(driver_pretraining_parameters)
                _set_requires_grad(driver_pretraining_parameters, False)
                best_validation = float("inf")
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
        else:
            phase = (
                "driver_autoencoder_pretrain"
                if epoch < driver_encoder_pretrain_epochs
                else "static_frozen_forecast"
            )
            if epoch == driver_encoder_pretrain_epochs:
                closed_form_audit = fit_closed_form_driver_residualizer(
                    model,
                    train_loader,
                    device,
                    ridge=closed_form_residualizer_ridge,
                )
                _clear_gradients(driver_frozen_parameters)
                _set_requires_grad(driver_frozen_parameters, False)
                best_validation = float("inf")
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0

        prediction_losses: list[float] = []
        adversary_losses: list[float] = []
        shortcut_losses: list[float] = []
        objective_losses: list[float] = []
        driver_objective_losses: list[float] = []
        reconstruction_losses: list[float] = []
        driver_gradient_norms: list[float] = []
        forecast_to_driver_gradient_norms: list[float] = []
        forecast_gradient_norms: list[float] = []
        adversary_gradient_norms: list[float] = []

        for batch in train_loader:
            features, target_history, target_future, regimes, _ = _move_batch(
                batch, device
            )
            probe_target = (
                target_lag_matrix(
                    target_history, model.target_probe_lags.tolist()
                )
                if is_dual_path
                and (
                    shortcut_covariance_weight > 0
                    or adversary_optimizer is not None
                )
                else None
            )

            if phase in {"joint_minimax", "driver_minimax_pretrain"}:
                for _ in range(main_steps):
                    main_optimizer.zero_grad(set_to_none=True)
                    if is_dual_path:
                        prediction, driver_latent = model(
                            features, target_history, regimes
                        )
                    else:
                        prediction = model(features, target_history, regimes)
                        driver_latent = None
                    prediction_loss = criterion(prediction, target_future)
                    shortcut_loss = (
                        standardized_cross_covariance_loss(
                            driver_latent, probe_target
                        )
                        if is_dual_path and shortcut_covariance_weight > 0
                        else None
                    )
                    if shortcut_loss is not None:
                        shortcut_losses.append(float(shortcut_loss.detach()))
                    if (
                        adversary_optimizer is not None
                        and active_adversarial_weight > 0
                    ):
                        _set_requires_grad(model.adversary.parameters(), False)
                        confusion_loss = criterion(
                            model.adversary_prediction(driver_latent),
                            probe_target,
                        )
                    else:
                        confusion_loss = None
                    objective = minimax_main_objective(
                        prediction_loss,
                        shortcut_loss,
                        confusion_loss,
                        shortcut_covariance_weight,
                        active_adversarial_weight,
                    )
                    objective.backward()
                    if is_dual_path:
                        norm = _gradient_norm(driver_parameters)
                        driver_gradient_norms.append(norm)
                        forecast_to_driver_gradient_norms.append(norm)
                        forecast_gradient_norms.append(
                            _gradient_norm(forecast_parameters)
                        )
                    torch.nn.utils.clip_grad_norm_(main_parameters, gradient_clip)
                    main_optimizer.step()
                    if adversary_optimizer is not None:
                        _set_requires_grad(model.adversary.parameters(), True)
                    prediction_losses.append(float(prediction_loss.detach()))
                    objective_losses.append(float(objective.detach()))

            elif phase == "driver_self_supervised_pretrain":
                _clear_gradients(driver_pretraining_parameters)
                for _ in range(main_steps):
                    forecast_optimizer.zero_grad(set_to_none=True)
                    prediction, _ = model(
                        features,
                        target_history,
                        regimes,
                        detach_driver=True,
                    )
                    prediction_loss = criterion(prediction, target_future)
                    prediction_loss.backward()
                    norm = _gradient_norm(driver_parameters)
                    driver_gradient_norms.append(norm)
                    forecast_to_driver_gradient_norms.append(norm)
                    forecast_gradient_norms.append(
                        _gradient_norm(forecast_parameters)
                    )
                    torch.nn.utils.clip_grad_norm_(
                        forecast_parameters, gradient_clip
                    )
                    forecast_optimizer.step()
                    prediction_losses.append(float(prediction_loss.detach()))
                    objective_losses.append(float(prediction_loss.detach()))

                driver_pretraining_optimizer.zero_grad(set_to_none=True)
                (
                    driver_latent,
                    reconstruction,
                    reconstruction_target,
                    reconstruction_availability,
                ) = model.driver_pretraining_outputs(
                    features, regimes, target_history
                )
                squared_error = (reconstruction - reconstruction_target).square()
                mask = reconstruction_availability[:, :, None]
                reconstruction_loss = (squared_error * mask).sum() / (
                    mask.sum().clamp_min(1.0) * squared_error.shape[-1]
                )
                shortcut_loss = (
                    standardized_cross_covariance_loss(
                        driver_latent, probe_target
                    )
                    if shortcut_covariance_weight > 0
                    else None
                )
                if shortcut_loss is not None:
                    shortcut_losses.append(float(shortcut_loss.detach()))
                if (
                    adversary_optimizer is not None
                    and active_adversarial_weight > 0
                ):
                    _set_requires_grad(model.adversary.parameters(), False)
                    confusion_loss = criterion(
                        model.adversary_prediction(driver_latent),
                        probe_target,
                    )
                else:
                    confusion_loss = None
                driver_objective = minimax_main_objective(
                    driver_reconstruction_weight * reconstruction_loss,
                    shortcut_loss,
                    confusion_loss,
                    shortcut_covariance_weight,
                    active_adversarial_weight,
                )
                driver_objective.backward()
                driver_gradient_norms.append(
                    _gradient_norm(driver_pretraining_parameters)
                )
                torch.nn.utils.clip_grad_norm_(
                    driver_pretraining_parameters, gradient_clip
                )
                driver_pretraining_optimizer.step()
                if adversary_optimizer is not None:
                    _set_requires_grad(model.adversary.parameters(), True)
                reconstruction_losses.append(float(reconstruction_loss.detach()))
                driver_objective_losses.append(float(driver_objective.detach()))

            elif phase == "driver_autoencoder_pretrain":
                _clear_gradients(driver_frozen_parameters)
                for _ in range(main_steps):
                    forecast_optimizer.zero_grad(set_to_none=True)
                    prediction, _ = model(
                        features,
                        target_history,
                        regimes,
                        detach_driver=True,
                    )
                    prediction_loss = criterion(prediction, target_future)
                    prediction_loss.backward()
                    norm = _gradient_norm(driver_parameters)
                    driver_gradient_norms.append(norm)
                    forecast_to_driver_gradient_norms.append(norm)
                    forecast_gradient_norms.append(
                        _gradient_norm(forecast_parameters)
                    )
                    torch.nn.utils.clip_grad_norm_(
                        forecast_parameters, gradient_clip
                    )
                    forecast_optimizer.step()
                    prediction_losses.append(float(prediction_loss.detach()))
                    objective_losses.append(float(prediction_loss.detach()))

                driver_base_optimizer.zero_grad(set_to_none=True)
                (
                    _,
                    reconstruction,
                    reconstruction_target,
                    reconstruction_availability,
                ) = model.driver_pretraining_outputs(
                    features, regimes, target_history
                )
                squared_error = (reconstruction - reconstruction_target).square()
                mask = reconstruction_availability[:, :, None]
                reconstruction_loss = (squared_error * mask).sum() / (
                    mask.sum().clamp_min(1.0) * squared_error.shape[-1]
                )
                driver_objective = (
                    driver_reconstruction_weight * reconstruction_loss
                )
                driver_objective.backward()
                driver_gradient_norms.append(
                    _gradient_norm(driver_base_parameters)
                )
                torch.nn.utils.clip_grad_norm_(
                    driver_base_parameters, gradient_clip
                )
                driver_base_optimizer.step()
                reconstruction_losses.append(float(reconstruction_loss.detach()))
                driver_objective_losses.append(float(driver_objective.detach()))

            else:
                for _ in range(main_steps):
                    forecast_optimizer.zero_grad(set_to_none=True)
                    prediction, _ = model(
                        features,
                        target_history,
                        regimes,
                        detach_driver=True,
                    )
                    prediction_loss = criterion(prediction, target_future)
                    prediction_loss.backward()
                    norm = _gradient_norm(driver_parameters)
                    driver_gradient_norms.append(norm)
                    forecast_to_driver_gradient_norms.append(norm)
                    forecast_gradient_norms.append(
                        _gradient_norm(forecast_parameters)
                    )
                    torch.nn.utils.clip_grad_norm_(
                        forecast_parameters, gradient_clip
                    )
                    forecast_optimizer.step()
                    prediction_losses.append(float(prediction_loss.detach()))
                    objective_losses.append(float(prediction_loss.detach()))

            run_adversary = (
                adversary_optimizer is not None
                and phase
                in {
                    "joint_minimax",
                    "driver_minimax_pretrain",
                    "driver_self_supervised_pretrain",
                }
                and active_adversarial_weight > 0
            )
            if run_adversary:
                for _ in range(adversary_steps):
                    adversary_optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        detached_latent = model.encode_driver(
                            features, regimes, target_history
                        )
                    recovered = model.adversary_prediction(
                        detached_latent.detach()
                    )
                    adversary_loss = criterion(recovered, probe_target)
                    adversary_loss.backward()
                    adversary_gradient_norms.append(
                        _gradient_norm(model.adversary.parameters())
                    )
                    torch.nn.utils.clip_grad_norm_(
                        model.adversary.parameters(), gradient_clip
                    )
                    adversary_optimizer.step()
                    adversary_losses.append(float(adversary_loss.detach()))

        current_validation = validation_mse(
            model, validation_loader, device
        )
        history.append(
            {
                "epoch": epoch + 1,
                "optimization_phase": phase,
                "prediction_mse": float(np.mean(prediction_losses)),
                "adversary_mse": (
                    float(np.mean(adversary_losses))
                    if adversary_losses
                    else None
                ),
                "shortcut_cross_correlation": (
                    float(np.mean(shortcut_losses))
                    if shortcut_losses
                    else None
                ),
                "objective": float(np.mean(objective_losses)),
                "driver_objective": (
                    float(np.mean(driver_objective_losses))
                    if driver_objective_losses
                    else None
                ),
                "driver_reconstruction_mse": (
                    float(np.mean(reconstruction_losses))
                    if reconstruction_losses
                    else None
                ),
                "validation_mse": current_validation,
                "active_adversarial_weight": active_adversarial_weight,
                "mean_driver_gradient_norm": (
                    float(np.mean(driver_gradient_norms))
                    if driver_gradient_norms
                    else None
                ),
                "mean_forecast_to_driver_gradient_norm": (
                    float(np.mean(forecast_to_driver_gradient_norms))
                    if forecast_to_driver_gradient_norms
                    else None
                ),
                "mean_forecast_gradient_norm": (
                    float(np.mean(forecast_gradient_norms))
                    if forecast_gradient_norms
                    else None
                ),
                "mean_adversary_gradient_norm": (
                    float(np.mean(adversary_gradient_norms))
                    if adversary_gradient_norms
                    else None
                ),
            }
        )
        if current_validation < best_validation - 1e-8:
            best_validation = current_validation
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            can_stop = (
                optimization_mode == "joint_minimax"
                or (
                    optimization_mode == "minimax_pretrain_static_freeze"
                    and epoch >= driver_pretrain_epochs
                )
                or (
                    optimization_mode
                    == "self_supervised_pretrain_static_freeze"
                    and epoch >= driver_pretrain_epochs
                )
                or (
                    optimization_mode
                    == "closed_form_innovation_static_freeze"
                    and epoch >= driver_encoder_pretrain_epochs
                )
            )
            if can_stop and stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    elapsed = perf_counter() - start_time
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else 0.0
    )
    return TrainResult(
        history=history,
        best_validation_mse=best_validation,
        epochs_completed=len(history),
        elapsed_seconds=elapsed,
        peak_gpu_memory_mib=peak_memory,
        optimization_audit={
            "mode": optimization_mode,
            "main_steps": int(main_steps),
            "adversary_steps": int(adversary_steps),
            "adversarial_warmup_epochs": int(adversarial_warmup_epochs),
            "driver_pretrain_epochs": int(driver_pretrain_epochs),
            "driver_encoder_pretrain_epochs": int(
                driver_encoder_pretrain_epochs
            ),
            "driver_reconstruction_weight": float(
                driver_reconstruction_weight
            ),
            "closed_form_residualizer_ridge": float(
                closed_form_residualizer_ridge
            ),
            "closed_form_residualizer_fit": closed_form_audit,
            "parameter_counts": {
                "main": int(
                    sum(parameter.numel() for parameter in main_parameters)
                ),
                "driver_representation": int(
                    sum(parameter.numel() for parameter in driver_parameters)
                ),
                "forecast_given_driver": int(
                    sum(parameter.numel() for parameter in forecast_parameters)
                ),
                "driver_pretraining": int(
                    sum(
                        parameter.numel()
                        for parameter in driver_pretraining_parameters
                    )
                ),
                "adversary": (
                    int(
                        sum(
                            parameter.numel()
                            for parameter in model.adversary.parameters()
                        )
                    )
                    if is_dual_path
                    else 0
                ),
            },
        },
    )


@torch.no_grad()
def predict_torch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scaler: Standardizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    regimes_all: list[np.ndarray] = []
    anchors_all: list[np.ndarray] = []
    for batch in loader:
        features, target_history, target_future, regimes, anchors = _move_batch(
            batch, device
        )
        output = model(features, target_history, regimes)
        prediction = output[0] if isinstance(output, tuple) else output
        predictions.append(prediction.cpu().numpy())
        truths.append(target_future.cpu().numpy())
        regimes_all.append(regimes.cpu().numpy())
        anchors_all.append(anchors.numpy())
    prediction_z = np.concatenate(predictions)
    truth_z = np.concatenate(truths)
    return (
        target_scaler.inverse_transform(prediction_z),
        target_scaler.inverse_transform(truth_z),
        np.concatenate(regimes_all),
        np.concatenate(anchors_all),
    )


@torch.no_grad()
def _driver_representations(
    model: LaggedDualPathNetwork | SeparatedCausalITransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    representations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        features, target_history, _, regimes, _ = _move_batch(batch, device)
        representations.append(
            model.encode_driver(
                features, regimes, target_history
            ).cpu().numpy()
        )
        targets.append(
            target_lag_matrix(
                target_history, model.target_probe_lags.tolist()
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(representations), np.concatenate(targets)


def independent_shortcut_probe(
    model: LaggedDualPathNetwork | SeparatedCausalITransformer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    alpha: float = 1.0,
) -> dict:
    """Fit a fresh Ridge probe after the forecasting model is frozen."""

    train_z, train_y = _driver_representations(model, train_loader, device)
    test_z, test_y = _driver_representations(model, test_loader, device)
    probe = Ridge(alpha=alpha).fit(train_z, train_y)
    prediction = probe.predict(test_z)
    per_lag = r2_score(test_y, prediction, multioutput="raw_values")
    return {
        "probe": "post_training_frozen_ridge",
        "alpha": alpha,
        "target_lags": model.target_probe_lags.tolist(),
        "r2_by_lag": np.asarray(per_lag).tolist(),
        "mean_r2": float(np.mean(per_lag)),
        "train_windows": int(len(train_z)),
        "test_windows": int(len(test_z)),
    }


def blocked_cross_fitted_shortcut_probe(
    model: LaggedDualPathNetwork | SeparatedCausalITransformer,
    loader: DataLoader,
    device: torch.device,
    alpha: float = 1.0,
    n_blocks: int = 5,
) -> dict:
    """Measure within-split recoverability with contiguous held-out blocks."""

    representation, target = _driver_representations(model, loader, device)
    if n_blocks < 2 or len(representation) < n_blocks:
        raise ValueError("blocked probe requires at least two non-empty blocks")
    edges = np.linspace(0, len(representation), n_blocks + 1, dtype=int)
    prediction = np.empty_like(target)
    all_indices = np.arange(len(representation))
    block_sizes: list[int] = []
    for block in range(n_blocks):
        start, stop = int(edges[block]), int(edges[block + 1])
        test_indices = all_indices[start:stop]
        train_indices = np.concatenate(
            (all_indices[:start], all_indices[stop:])
        )
        if not len(test_indices) or not len(train_indices):
            raise ValueError("blocked probe produced an empty train or test fold")
        estimator = Ridge(alpha=alpha).fit(
            representation[train_indices], target[train_indices]
        )
        prediction[test_indices] = estimator.predict(
            representation[test_indices]
        )
        block_sizes.append(int(len(test_indices)))
    per_lag = r2_score(target, prediction, multioutput="raw_values")
    return {
        "probe": "post_training_frozen_ridge_blocked_crossfit",
        "alpha": alpha,
        "n_blocks": int(n_blocks),
        "block_sizes": block_sizes,
        "target_lags": model.target_probe_lags.tolist(),
        "r2_by_lag": np.asarray(per_lag).tolist(),
        "mean_r2": float(np.mean(per_lag)),
        "windows": int(len(representation)),
        "interpretation": (
            "within-split recoverability diagnostic; not an independence test"
        ),
    }


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
