import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sfcdi.models import (
    RepositoryDLinear,
    RepositoryITransformer,
    RepositoryTimesNet,
    model_from_manifest,
    separated_itransformer_from_manifest,
)
from sfcdi.training import train_torch_model


def manifest():
    return {
        "regime_ids": [0, 1],
        "selected_pairs": [
            {
                "feature": "driver_a",
                "feature_index": 0,
                "lag": 1,
                "available_regimes": [0, 1],
                "governance": "stable_core",
            },
            {
                "feature": "driver_b",
                "feature_index": 2,
                "lag": 2,
                "available_regimes": [0, 1],
                "governance": "stable_core",
            },
        ],
    }


def tiny_loader(samples=16, sequence_length=8, prediction_length=3):
    generator = torch.Generator().manual_seed(5)
    features = torch.randn(samples, sequence_length, 3, generator=generator)
    history = torch.randn(samples, sequence_length, 1, generator=generator)
    future = history[:, -1] + 0.1 * torch.randn(
        samples, prediction_length, generator=generator
    )
    regimes = torch.arange(samples) % 2
    anchors = torch.arange(samples)
    return DataLoader(
        TensorDataset(features, history, future, regimes, anchors),
        batch_size=8,
        shuffle=False,
    )


def test_dataset_specific_sf_cdi_shapes():
    metro = model_from_manifest(
        manifest(),
        prediction_length=3,
        target_probe_lags=[0, 1],
        hidden_size=8,
        target_hidden_size=8,
        dropout=0.0,
        use_regime_gating=True,
    )
    nist = separated_itransformer_from_manifest(
        manifest(),
        sequence_length=8,
        prediction_length=3,
        target_probe_lags=[0, 1],
        d_model=8,
        heads=2,
        layers=1,
        d_ff=16,
        dropout=0.0,
    )
    batch = next(iter(tiny_loader()))
    for model in (metro, nist):
        prediction, driver = model(batch[0], batch[1], batch[3])
        assert prediction.shape == (8, 3)
        assert driver.shape[0] == 8


def test_static_phase_has_zero_driver_gradient():
    model = model_from_manifest(
        manifest(),
        prediction_length=3,
        target_probe_lags=[0, 1],
        hidden_size=8,
        target_hidden_size=8,
        dropout=0.0,
        use_regime_gating=True,
    )
    loader = tiny_loader()
    result = train_torch_model(
        model,
        loader,
        loader,
        torch.device("cpu"),
        epochs=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=10,
        adversarial_weight=0.05,
        shortcut_covariance_weight=0.05,
        adversary_learning_rate=1e-3,
        adversary_steps=1,
        adversarial_warmup_epochs=1,
        gradient_clip=1.0,
        main_steps=1,
        optimization_mode="minimax_pretrain_static_freeze",
        driver_pretrain_epochs=2,
    )
    assert [row["optimization_phase"] for row in result.history] == [
        "driver_minimax_pretrain",
        "driver_minimax_pretrain",
        "static_frozen_forecast",
    ]
    assert result.history[-1]["mean_driver_gradient_norm"] == 0.0
    assert result.history[-1]["mean_forecast_gradient_norm"] > 0.0


def test_closed_form_variant_fits_then_freezes_driver_path():
    model = separated_itransformer_from_manifest(
        manifest(),
        sequence_length=8,
        prediction_length=3,
        target_probe_lags=[0, 1],
        d_model=8,
        heads=2,
        layers=1,
        d_ff=16,
        dropout=0.0,
        full_token_control=True,
        driver_reconstruction=True,
        closed_form_residualization=True,
    )
    loader = tiny_loader()
    result = train_torch_model(
        model,
        loader,
        loader,
        torch.device("cpu"),
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=10,
        gradient_clip=1.0,
        main_steps=1,
        optimization_mode="closed_form_innovation_static_freeze",
        driver_encoder_pretrain_epochs=1,
        driver_reconstruction_weight=1.0,
        closed_form_residualizer_ridge=1e-3,
    )
    assert [row["optimization_phase"] for row in result.history] == [
        "driver_autoencoder_pretrain",
        "static_frozen_forecast",
    ]
    assert result.history[-1]["mean_driver_gradient_norm"] == 0.0
    assert result.optimization_audit["closed_form_residualizer_fit"] is not None


def test_self_supervised_variant_never_backpropagates_forecast_to_driver():
    model = separated_itransformer_from_manifest(
        manifest(),
        sequence_length=8,
        prediction_length=3,
        target_probe_lags=[0, 1],
        d_model=8,
        heads=2,
        layers=1,
        d_ff=16,
        dropout=0.0,
        full_token_control=True,
        driver_reconstruction=True,
    )
    loader = tiny_loader()
    result = train_torch_model(
        model,
        loader,
        loader,
        torch.device("cpu"),
        epochs=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=10,
        adversarial_weight=0.05,
        shortcut_covariance_weight=0.05,
        adversary_learning_rate=1e-3,
        adversary_steps=1,
        adversarial_warmup_epochs=1,
        gradient_clip=1.0,
        main_steps=1,
        optimization_mode="self_supervised_pretrain_static_freeze",
        driver_pretrain_epochs=2,
        driver_reconstruction_weight=1.0,
    )
    assert [row["optimization_phase"] for row in result.history] == [
        "driver_self_supervised_pretrain",
        "driver_self_supervised_pretrain",
        "static_frozen_forecast",
    ]
    assert all(
        row["mean_forecast_to_driver_gradient_norm"] == 0.0
        for row in result.history
    )
    assert result.history[-1]["mean_driver_gradient_norm"] == 0.0


def test_reported_tslib_baseline_adapters_return_target_forecasts():
    batch = next(iter(tiny_loader(samples=4)))
    models = [
        RepositoryDLinear(3, 8, 3, moving_average=3),
        RepositoryITransformer(
            3, 8, 3, d_model=8, heads=2, layers=1, d_ff=16
        ),
        RepositoryTimesNet(
            3,
            8,
            3,
            d_model=8,
            d_ff=8,
            layers=1,
            top_k=2,
            num_kernels=2,
            dropout=0.0,
        ),
    ]
    for model in models:
        prediction = model(batch[0], batch[1], batch[3])
        assert prediction.shape == (4, 3)
        assert np.isfinite(prediction.detach().numpy()).all()
