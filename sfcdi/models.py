"""Compact forecasting models sized for a 2 GiB GTX 1050."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import torch
from torch import nn


class TargetOnlyLSTM(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            1,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, prediction_length),
        )

    def forward(self, x, target_history, regimes=None):
        _, (hidden, _) = self.encoder(target_history)
        delta = self.head(hidden[-1])
        return target_history[:, -1, 0:1] + delta


class RepositoryDLinear(nn.Module):
    """Adapter for the repository-native multivariate DLinear implementation."""

    def __init__(
        self,
        feature_count: int,
        sequence_length: int,
        prediction_length: int,
        moving_average: int = 25,
    ) -> None:
        super().__init__()
        from .vendor.tslib.models.DLinear import Model as DLinearModel

        variable_count = feature_count + 1
        config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=sequence_length,
            pred_len=prediction_length,
            enc_in=variable_count,
            moving_avg=moving_average,
        )
        self.backbone = DLinearModel(config, individual=False)

    def forward(self, x, target_history, regimes=None):
        combined = torch.cat([x, target_history], dim=-1)
        forecast = self.backbone(combined, None, None, None)
        return forecast[..., -1]


class RepositoryTimesNet(nn.Module):
    """Adapter for the repository-native TimesNet forecasting implementation."""

    def __init__(
        self,
        feature_count: int,
        sequence_length: int,
        prediction_length: int,
        d_model: int = 32,
        d_ff: int = 32,
        layers: int = 1,
        top_k: int = 3,
        num_kernels: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from .vendor.tslib.models.TimesNet import Model as TimesNetModel

        variable_count = feature_count + 1
        config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=sequence_length,
            label_len=max(1, sequence_length // 2),
            pred_len=prediction_length,
            enc_in=variable_count,
            c_out=variable_count,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=layers,
            top_k=top_k,
            num_kernels=num_kernels,
            embed="timeF",
            freq="s",
            dropout=dropout,
        )
        self.backbone = TimesNetModel(config)

    def forward(self, x, target_history, regimes=None):
        combined = torch.cat([x, target_history], dim=-1)
        forecast = self.backbone(combined, None, None, None)
        return forecast[..., -1]


class RepositoryITransformer(nn.Module):
    """Adapter for the repository-native iTransformer implementation."""

    def __init__(
        self,
        feature_count: int,
        sequence_length: int,
        prediction_length: int,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from .vendor.tslib.models.iTransformer import Model as ITransformerModel

        config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=sequence_length,
            pred_len=prediction_length,
            enc_in=feature_count + 1,
            d_model=d_model,
            n_heads=heads,
            e_layers=layers,
            d_ff=d_ff,
            factor=1,
            dropout=dropout,
            activation="gelu",
            embed="timeF",
            freq="s",
        )
        self.backbone = ITransformerModel(config)

    def forward(self, x, target_history, regimes=None):
        combined = torch.cat([x, target_history], dim=-1)
        forecast = self.backbone(combined, None, None, None)
        return forecast[..., -1]


class SeparatedCausalITransformer(nn.Module):
    """iTransformer fusion with a separately optimizable driver embedding.

    Driver and target histories use separate inverted embeddings.  The driver
    tokens enter the repository-native iTransformer encoder only after their
    pooled representation has been exposed to the shortcut adversary.  This
    permits forecast-time detachment of the driver embedding while continuing
    to train the target embedding, fusion encoder, and forecast projection.
    """

    def __init__(
        self,
        pair_feature_indices: Sequence[int],
        pair_lags: Sequence[int],
        target_probe_lags: Sequence[int],
        regime_availability: torch.Tensor,
        core_pair_mask: Sequence[bool],
        sequence_length: int,
        prediction_length: int,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        use_regime_gating: bool = True,
        full_token_control: bool = False,
        driver_reconstruction: bool = False,
        closed_form_residualization: bool = False,
        residualizer_lags: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        from .vendor.tslib.layers.Embed import DataEmbedding_inverted
        from .vendor.tslib.layers.SelfAttention_Family import (
            AttentionLayer,
            FullAttention,
        )
        from .vendor.tslib.layers.Transformer_EncDec import Encoder, EncoderLayer

        if len(pair_feature_indices) != len(pair_lags) or not pair_lags:
            raise ValueError("separated iTransformer requires non-empty aligned pairs")
        availability = torch.as_tensor(regime_availability, dtype=torch.float32)
        if availability.ndim != 2 or availability.shape[1] != len(pair_lags):
            raise ValueError("regime_availability must have shape [regimes, pairs]")
        self.register_buffer(
            "pair_feature_indices",
            torch.as_tensor(pair_feature_indices, dtype=torch.long),
        )
        self.register_buffer("pair_lags", torch.as_tensor(pair_lags, dtype=torch.long))
        self.register_buffer("regime_availability", availability)
        self.register_buffer(
            "core_pair_mask", torch.as_tensor(core_pair_mask, dtype=torch.float32)
        )
        self.register_buffer(
            "target_probe_lags", torch.as_tensor(target_probe_lags, dtype=torch.long)
        )
        self.register_buffer(
            "driver_residualizer_lags",
            torch.as_tensor(
                residualizer_lags
                if residualizer_lags is not None
                else target_probe_lags,
                dtype=torch.long,
            ),
        )
        self.use_regime_gating = bool(use_regime_gating)
        self.full_token_control = bool(full_token_control)
        self.closed_form_residualization = bool(closed_form_residualization)
        self.gate_logits = nn.Parameter(
            torch.full((availability.shape[0], len(pair_lags)), 2.0)
        )
        self.driver_embedding = DataEmbedding_inverted(
            sequence_length, d_model, "timeF", "s", dropout
        )
        self.target_embedding = DataEmbedding_inverted(
            sequence_length, d_model, "timeF", "s", dropout
        )
        self.fusion_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor=1,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, prediction_length)
        control_size = d_model * len(pair_lags) if self.full_token_control else d_model
        self.adversary = nn.Sequential(
            nn.LayerNorm(control_size),
            nn.Linear(control_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(target_probe_lags)),
        )
        self.driver_reconstruction = (
            nn.Linear(d_model, sequence_length) if driver_reconstruction else None
        )
        self.target_to_driver = (
            nn.Linear(len(self.driver_residualizer_lags), len(pair_lags) * d_model)
            if self.closed_form_residualization
            else None
        )
        if self.target_to_driver is not None and self.closed_form_residualization:
            nn.init.zeros_(self.target_to_driver.weight)
            nn.init.zeros_(self.target_to_driver.bias)
        self.driver_token_count = len(pair_lags)
        self.driver_token_size = d_model

    @staticmethod
    def _normalize(sequence: torch.Tensor):
        mean = sequence.mean(dim=1, keepdim=True).detach()
        scale = sequence.var(dim=1, unbiased=False, keepdim=True).add(1e-5).sqrt()
        return (sequence - mean) / scale, mean, scale

    def pair_gate(self, regimes: torch.Tensor) -> torch.Tensor:
        batch_size = regimes.shape[0]
        if not self.use_regime_gating:
            return torch.ones(
                (batch_size, self.pair_lags.numel()),
                device=regimes.device,
                dtype=torch.float32,
            )
        known = (regimes >= 0) & (regimes < self.regime_availability.shape[0])
        safe_regimes = regimes.clamp(0, self.regime_availability.shape[0] - 1)
        learned = torch.sigmoid(self.gate_logits[safe_regimes])
        gate = learned * self.regime_availability[safe_regimes]
        learned_all = torch.sigmoid(self.gate_logits) * self.regime_availability
        availability_count = self.regime_availability.sum(dim=0).clamp_min(1.0)
        mean_known_gate = learned_all.sum(dim=0) / availability_count
        unknown_gate = (mean_known_gate * self.core_pair_mask).unsqueeze(0).expand(
            batch_size, -1
        )
        return torch.where(known[:, None], gate, unknown_gate)

    def _raw_driver_tokens(
        self, x: torch.Tensor, regimes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = x[:, :, self.pair_feature_indices]
        normalized, _, _ = self._normalize(selected)
        gate = self.pair_gate(regimes)
        tokens = self.driver_embedding(normalized, None) * gate[:, :, None]
        return tokens, gate, normalized

    def _driver_tokens(
        self,
        x: torch.Tensor,
        regimes: torch.Tensor,
        target_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, gate, _ = self._raw_driver_tokens(x, regimes)
        if self.target_to_driver is not None:
            if target_history is None:
                raise ValueError(
                    "target history is required for driver innovation residualization"
                )
            shortcut = self.target_to_driver(
                self.target_residualizer_input(target_history)
            ).view(
                len(tokens), self.driver_token_count, self.driver_token_size
            )
            tokens = tokens - shortcut * gate[:, :, None]
        return tokens, gate

    def target_residualizer_input(self, target_history: torch.Tensor) -> torch.Tensor:
        normalized_target, _, _ = self._normalize(target_history)
        positions = (
            normalized_target.shape[1] - 1 - self.driver_residualizer_lags
        ).clamp_min(0)
        return normalized_target[:, positions, 0]

    def raw_driver_control_tokens(
        self, x: torch.Tensor, regimes: torch.Tensor
    ) -> torch.Tensor:
        tokens, _, _ = self._raw_driver_tokens(x, regimes)
        return tokens

    @staticmethod
    def _pool_driver(tokens: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        weights = gate[:, :, None]
        # ``tokens`` have already been multiplied by the availability gate in
        # ``_driver_tokens``.  Applying ``weights`` again would square the gate
        # and unintentionally over-suppress uncertain regime-specific drivers.
        return tokens.sum(dim=1) / weights.sum(dim=1).clamp_min(1e-5)

    def _control_representation(
        self, tokens: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        if self.full_token_control:
            # The forecaster consumes every driver token.  Flattening therefore
            # exposes the exact forecast-visible representation to the
            # adversary/covariance objective instead of controlling only a
            # pooled summary that could hide token-wise target-history leakage.
            return tokens.flatten(start_dim=1)
        return self._pool_driver(tokens, gate)

    def encode_driver(
        self,
        x: torch.Tensor,
        regimes: torch.Tensor,
        target_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens, gate = self._driver_tokens(x, regimes, target_history)
        return self._control_representation(tokens, gate)

    def driver_pretraining_outputs(
        self,
        x: torch.Tensor,
        regimes: torch.Tensor,
        target_history: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return full driver control state and a denoising reconstruction task.

        This self-supervised task preserves driver information without using
        future targets, so forecasting gradients never need to update the
        driver representation in the strict two-stage schedule.
        """

        if self.driver_reconstruction is None:
            raise RuntimeError("driver reconstruction head is not enabled")
        raw_tokens, gate, normalized = self._raw_driver_tokens(x, regimes)
        innovation_tokens = raw_tokens
        if self.target_to_driver is not None:
            if target_history is None:
                raise ValueError(
                    "target history is required for driver innovation residualization"
                )
            shortcut = self.target_to_driver(
                self.target_residualizer_input(target_history)
            ).view(
                len(raw_tokens), self.driver_token_count, self.driver_token_size
            )
            shortcut = shortcut * gate[:, :, None]
            innovation_tokens = raw_tokens - shortcut
        reconstruction = self.driver_reconstruction(raw_tokens)
        reconstruction_target = normalized.transpose(1, 2)
        availability = (gate > 0).to(reconstruction.dtype)
        return (
            self._control_representation(innovation_tokens, gate),
            reconstruction,
            reconstruction_target,
            availability,
        )

    def forward(self, x, target_history, regimes, detach_driver: bool = False):
        driver_tokens, gate = self._driver_tokens(x, regimes, target_history)
        driver_latent = self._control_representation(driver_tokens, gate)
        if detach_driver:
            driver_tokens = driver_tokens.detach()
            driver_latent = driver_latent.detach()
        normalized_target, target_mean, target_scale = self._normalize(target_history)
        target_token = self.target_embedding(normalized_target, None)
        fused, _ = self.fusion_encoder(
            torch.cat((driver_tokens, target_token), dim=1), attn_mask=None
        )
        standardized_forecast = self.projection(fused[:, -1, :])
        prediction = (
            standardized_forecast * target_scale[:, 0, 0:1]
            + target_mean[:, 0, 0:1]
        )
        return prediction, driver_latent

    def adversary_prediction(self, driver_latent: torch.Tensor) -> torch.Tensor:
        return self.adversary(driver_latent)

    def main_parameters(self):
        adversary_ids = {id(parameter) for parameter in self.adversary.parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in adversary_ids]

    def driver_representation_parameters(self):
        parameters = list(self.driver_encoder_parameters())
        if self.target_to_driver is not None:
            parameters.extend(self.target_to_driver.parameters())
        return parameters

    def driver_encoder_parameters(self):
        return [self.gate_logits, *self.driver_embedding.parameters()]

    def driver_base_pretraining_parameters(self):
        parameters = list(self.driver_encoder_parameters())
        if self.driver_reconstruction is not None:
            parameters.extend(self.driver_reconstruction.parameters())
        return parameters

    def forecast_given_driver_parameters(self):
        return [
            *self.target_embedding.parameters(),
            *self.fusion_encoder.parameters(),
            *self.projection.parameters(),
        ]

    def driver_pretraining_parameters(self):
        parameters = list(self.driver_representation_parameters())
        if self.driver_reconstruction is not None:
            parameters.extend(self.driver_reconstruction.parameters())
        return parameters


class LaggedDualPathNetwork(nn.Module):
    """SF-CDI late fusion with lag governance and a shortcut adversary.

    Driver and target-history paths remain separate until the forecast head.
    The driver input is a versioned set of variable-lag pairs rather than an
    untracked channel list.  A fixed deployment-availability mask is composed
    with a learned regime gate.
    """

    def __init__(
        self,
        pair_feature_indices: Sequence[int],
        pair_lags: Sequence[int],
        prediction_length: int,
        target_probe_lags: Sequence[int],
        regime_availability: torch.Tensor,
        core_pair_mask: Sequence[bool],
        hidden_size: int = 32,
        target_hidden_size: int = 32,
        dropout: float = 0.1,
        use_regime_gating: bool = True,
    ) -> None:
        super().__init__()
        if len(pair_feature_indices) != len(pair_lags):
            raise ValueError("pair feature indices and lags must have equal length")
        if not pair_lags:
            raise ValueError("At least one driver pair is required")
        availability = torch.as_tensor(regime_availability, dtype=torch.float32)
        if availability.ndim != 2 or availability.shape[1] != len(pair_lags):
            raise ValueError("regime_availability must have shape [regimes, pairs]")
        self.register_buffer(
            "pair_feature_indices",
            torch.as_tensor(pair_feature_indices, dtype=torch.long),
        )
        self.register_buffer("pair_lags", torch.as_tensor(pair_lags, dtype=torch.long))
        self.register_buffer("regime_availability", availability)
        self.register_buffer(
            "core_pair_mask", torch.as_tensor(core_pair_mask, dtype=torch.float32)
        )
        self.register_buffer(
            "target_probe_lags", torch.as_tensor(target_probe_lags, dtype=torch.long)
        )
        self.use_regime_gating = bool(use_regime_gating)
        pair_count = len(pair_lags)
        regime_count = availability.shape[0]
        self.gate_logits = nn.Parameter(torch.full((regime_count, pair_count), 2.0))
        self.driver_encoder = nn.LSTM(
            pair_count, hidden_size, batch_first=True, num_layers=1
        )
        self.target_encoder = nn.LSTM(
            1, target_hidden_size, batch_first=True, num_layers=1
        )
        self.target_head = nn.Sequential(
            nn.LayerNorm(target_hidden_size),
            nn.Linear(target_hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, prediction_length),
        )
        self.driver_head = nn.Sequential(
            nn.LayerNorm(hidden_size + target_hidden_size),
            nn.Linear(hidden_size + target_hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, prediction_length),
        )
        # Conditional innovation begins as a target-only forecast.
        nn.init.zeros_(self.driver_head[-1].weight)
        nn.init.zeros_(self.driver_head[-1].bias)
        self.adversary = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, len(target_probe_lags)),
        )

    @property
    def pair_count(self) -> int:
        return int(self.pair_lags.numel())

    def pair_gate(self, regimes: torch.Tensor) -> torch.Tensor:
        batch_size = regimes.shape[0]
        if not self.use_regime_gating:
            return torch.ones(
                (batch_size, self.pair_count), device=regimes.device, dtype=torch.float32
            )
        known = (regimes >= 0) & (regimes < self.regime_availability.shape[0])
        safe_regimes = regimes.clamp(0, self.regime_availability.shape[0] - 1)
        learned = torch.sigmoid(self.gate_logits[safe_regimes])
        available = self.regime_availability[safe_regimes]
        gate = learned * available
        # An unseen deployment regime receives stable-core pairs only.  If a
        # screening run has no inferred core, the caller may explicitly mark
        # screening pairs as core for the within-regime exploratory protocol.
        learned_all = torch.sigmoid(self.gate_logits) * self.regime_availability
        availability_count = self.regime_availability.sum(dim=0).clamp_min(1.0)
        mean_known_gate = learned_all.sum(dim=0) / availability_count
        unknown_gate = (mean_known_gate * self.core_pair_mask).unsqueeze(0).expand(
            batch_size, -1
        )
        return torch.where(known[:, None], gate, unknown_gate)

    def gather_lagged_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Align each selected channel by its manifest lag over the full history."""

        length = x.shape[1]
        time = torch.arange(length, device=x.device, dtype=torch.long)[:, None]
        positions = (time - self.pair_lags[None, :]).clamp_min(0)
        return x[:, positions, self.pair_feature_indices[None, :]]

    def encode_driver(
        self,
        x: torch.Tensor,
        regimes: torch.Tensor,
        target_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate = self.pair_gate(regimes)
        sequence = self.gather_lagged_sequence(x)
        sequence = sequence * gate[:, None, :]
        _, (hidden, _) = self.driver_encoder(sequence)
        return hidden[-1]

    def encode_target(self, target_history: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.target_encoder(target_history)
        return hidden[-1]

    def forward(self, x, target_history, regimes, detach_driver: bool = False):
        driver_latent = self.encode_driver(x, regimes)
        if detach_driver:
            driver_latent = driver_latent.detach()
        target_latent = self.encode_target(target_history)
        conditional_driver = torch.cat([driver_latent, target_latent], dim=-1)
        delta = self.target_head(target_latent) + self.driver_head(
            conditional_driver
        )
        prediction = target_history[:, -1, 0:1] + delta
        return prediction, driver_latent

    def target_only_prediction(self, target_history: torch.Tensor) -> torch.Tensor:
        target_latent = self.encode_target(target_history)
        return target_history[:, -1, 0:1] + self.target_head(target_latent)

    def adversary_prediction(self, driver_latent: torch.Tensor) -> torch.Tensor:
        return self.adversary(driver_latent)

    def main_parameters(self):
        adversary_ids = {id(parameter) for parameter in self.adversary.parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in adversary_ids]

    def driver_representation_parameters(self):
        return [self.gate_logits, *self.driver_encoder.parameters()]

    def forecast_given_driver_parameters(self):
        """Parameters allowed to move when the driver representation is frozen."""

        parameters = list(self.target_encoder.parameters())
        parameters.extend(self.target_head.parameters())
        parameters.extend(self.driver_head.parameters())
        return parameters


def model_from_manifest(
    manifest: dict,
    prediction_length: int,
    target_probe_lags: Sequence[int],
    hidden_size: int,
    target_hidden_size: int,
    dropout: float,
    use_regime_gating: bool,
) -> LaggedDualPathNetwork:
    pairs = manifest["selected_pairs"]
    regime_ids = manifest["regime_ids"]
    if regime_ids != list(range(len(regime_ids))):
        raise ValueError("Current model expects contiguous regime IDs starting at zero")
    availability = torch.zeros((len(regime_ids), len(pairs)), dtype=torch.float32)
    core_mask: list[bool] = []
    for column, pair in enumerate(pairs):
        for regime in pair["available_regimes"]:
            availability[int(regime), column] = 1.0
        core_mask.append(pair["governance"] == "stable_core")
    return LaggedDualPathNetwork(
        pair_feature_indices=[pair["feature_index"] for pair in pairs],
        pair_lags=[pair["lag"] for pair in pairs],
        prediction_length=prediction_length,
        target_probe_lags=target_probe_lags,
        regime_availability=availability,
        core_pair_mask=core_mask,
        hidden_size=hidden_size,
        target_hidden_size=target_hidden_size,
        dropout=dropout,
        use_regime_gating=use_regime_gating,
    )


def separated_itransformer_from_manifest(
    manifest: dict,
    sequence_length: int,
    prediction_length: int,
    target_probe_lags: Sequence[int],
    d_model: int,
    heads: int,
    layers: int,
    d_ff: int,
    dropout: float,
    full_token_control: bool = False,
    driver_reconstruction: bool = False,
    closed_form_residualization: bool = False,
    residualizer_lags: Sequence[int] | None = None,
) -> SeparatedCausalITransformer:
    pairs = manifest["selected_pairs"]
    regime_ids = manifest["regime_ids"]
    if regime_ids != list(range(len(regime_ids))):
        raise ValueError("Current model expects contiguous regime IDs starting at zero")
    availability = torch.zeros((len(regime_ids), len(pairs)), dtype=torch.float32)
    core_mask: list[bool] = []
    for column, pair in enumerate(pairs):
        for regime in pair["available_regimes"]:
            availability[int(regime), column] = 1.0
        core_mask.append(pair["governance"] == "stable_core")
    return SeparatedCausalITransformer(
        pair_feature_indices=[pair["feature_index"] for pair in pairs],
        pair_lags=[pair["lag"] for pair in pairs],
        target_probe_lags=target_probe_lags,
        regime_availability=availability,
        core_pair_mask=core_mask,
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        d_model=d_model,
        heads=heads,
        layers=layers,
        d_ff=d_ff,
        dropout=dropout,
        use_regime_gating=True,
        full_token_control=full_token_control,
        driver_reconstruction=driver_reconstruction,
        closed_form_residualization=closed_form_residualization,
        residualizer_lags=residualizer_lags,
    )
