"""Time-valid lagged causal-driver discovery.

Selected variable--lag pairs are interpreted as causal drivers under the
explicit assumptions recorded in each manifest: no unobserved confounding,
no descendants of the target among candidates, an adequate conditioning set,
and sufficiently powerful conditional-independence tests. The implementation
uses training data only, forward-blocked cross-fitting, a
heteroskedasticity/autocorrelation-consistent statistic, Benjamini--Hochberg
correction, and time-block stability gates. It does not estimate intervention
effect sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import json
import math

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Candidate:
    feature_index: int
    feature: str
    lag: int


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values."""

    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    m = len(p_values)
    if m == 0:
        return p_values.copy()
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _sha256_file(path: str | Path, chunk_size: int = 2**20) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_content_sha256(manifest: dict) -> str:
    """Hash scientific manifest content while excluding run-time metadata."""

    content = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_utc", "manifest_sha256"}
    }
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256(canonical).hexdigest()


def _ridge_residuals_forward(
    conditioning: np.ndarray,
    outcomes: np.ndarray,
    n_blocks: int,
    ridge_alpha: float,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residualize outcomes using prior time blocks only.

    Block zero is a warm-up block and is never evaluated.  Every reported
    residual is therefore produced by a model fitted strictly on earlier rows.
    """

    n = len(conditioning)
    if n_blocks < 3:
        raise ValueError("At least three blocks are required")
    if n < n_blocks * 20:
        raise ValueError("Too few observations for blocked cross-fitting")
    if groups is None:
        group_rows = [np.arange(n, dtype=np.int64)]
    else:
        groups = np.asarray(groups)
        all_group_rows = [
            np.flatnonzero(groups == group) for group in np.unique(groups)
        ]
        # Industrial logs often contain many short fragments around acquisition
        # gaps.  Such fragments cannot support a five-block forward fit and are
        # excluded from discovery instead of forcing the entire dataset back to
        # a global cross-fit that could bridge physical discontinuities.
        group_rows = [rows for rows in all_group_rows if len(rows) >= n_blocks * 20]
        if not group_rows:
            raise ValueError(
                "No cross-fit group has at least 20 rows per block"
            )
    residual_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    fold_parts: list[np.ndarray] = []
    for fold in range(1, n_blocks):
        train_indices = []
        evaluation_indices = []
        for rows in group_rows:
            boundaries = np.linspace(0, len(rows), n_blocks + 1, dtype=np.int64)
            train_indices.append(rows[: boundaries[fold]])
            evaluation_indices.append(rows[boundaries[fold] : boundaries[fold + 1]])
        train_index = np.sort(np.concatenate(train_indices))
        evaluation_index = np.sort(np.concatenate(evaluation_indices))
        scaler = StandardScaler().fit(conditioning[train_index])
        z_train = scaler.transform(conditioning[train_index])
        z_eval = scaler.transform(conditioning[evaluation_index])
        estimator = Ridge(alpha=ridge_alpha, fit_intercept=True)
        estimator.fit(z_train, outcomes[train_index])
        predicted = estimator.predict(z_eval)
        residual_parts.append(outcomes[evaluation_index] - predicted)
        index_parts.append(evaluation_index)
        fold_parts.append(np.full(len(evaluation_index), fold, dtype=np.int64))
    return (
        np.concatenate(residual_parts, axis=0),
        np.concatenate(index_parts),
        np.concatenate(fold_parts),
    )


def _hac_mean_test(
    values: np.ndarray,
    groups: np.ndarray,
    bandwidth: int,
) -> tuple[float, float, float]:
    """Two-sided HAC test that the ordered product mean equals zero."""

    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    n = len(values)
    mean = float(np.mean(values))
    centered = values - mean
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for lag in range(1, min(bandwidth, n - 1) + 1):
        same_group = groups[lag:] == groups[:-lag]
        if not np.any(same_group):
            continue
        covariance = float(
            np.dot(centered[lag:][same_group], centered[:-lag][same_group]) / n
        )
        weight = 1.0 - lag / (bandwidth + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, gamma0 / max(n, 1), 1e-16)
    standard_error = math.sqrt(long_run_variance / n)
    statistic = mean / standard_error
    p_value = float(2.0 * norm.sf(abs(statistic)))
    return statistic, p_value, standard_error


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _valid_origins(
    segments: np.ndarray,
    maximum_lag: int,
    future_horizon: int = 1,
) -> np.ndarray:
    origins = np.arange(maximum_lag, len(segments) - future_horizon, dtype=np.int64)
    valid = segments[origins - maximum_lag] == segments[origins + future_horizon]
    return origins[valid]


def discover_lagged_drivers(
    features: np.ndarray,
    target: np.ndarray,
    regimes: np.ndarray,
    segments: np.ndarray,
    feature_names: Sequence[str],
    candidate_lags: Sequence[int],
    conditioning_target_lags: Sequence[int] = (0, 1, 2, 6, 12),
    max_samples: int = 50_000,
    n_blocks: int = 5,
    ridge_alpha: float = 1.0,
    hac_bandwidth: int = 12,
    fdr_level: float = 0.05,
    minimum_effect: float = 0.015,
    minimum_stability: float = 0.60,
    maximum_pairs: int = 12,
    maximum_lags_per_feature: int = 2,
    minimum_regime_samples: int = 200,
    crossfit_within_segments: bool = False,
    screening_horizon: int = 1,
    source_path: str | Path | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
) -> dict:
    """Discover a bounded, auditable set of lagged predictive-driver pairs."""

    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    regimes = np.asarray(regimes, dtype=np.int64)
    segments = np.asarray(segments, dtype=np.int64)
    candidate_lags = sorted({int(lag) for lag in candidate_lags})
    conditioning_target_lags = sorted({int(lag) for lag in conditioning_target_lags})
    if not candidate_lags or min(candidate_lags) < 0:
        raise ValueError("candidate_lags must contain non-negative integers")
    if features.shape != (len(target), len(feature_names)):
        raise ValueError("features, target, and feature_names have inconsistent shapes")

    maximum_lag = max([*candidate_lags, *conditioning_target_lags])
    screening_horizon = int(screening_horizon)
    if screening_horizon < 1:
        raise ValueError("screening_horizon must be positive")
    origins = _valid_origins(segments, maximum_lag, screening_horizon)
    if len(origins) > max_samples:
        positions = np.linspace(0, len(origins) - 1, max_samples, dtype=np.int64)
        origins = origins[positions]

    candidates = [
        Candidate(feature_index=j, feature=name, lag=lag)
        for j, name in enumerate(feature_names)
        for lag in candidate_lags
    ]
    candidate_matrix = np.column_stack(
        [features[origins - candidate.lag, candidate.feature_index] for candidate in candidates]
    )
    target_future = np.mean(
        np.column_stack(
            [target[origins + step] for step in range(1, screening_horizon + 1)]
        ),
        axis=1,
        keepdims=True,
    )
    target_history = np.column_stack(
        [target[origins - lag] for lag in conditioning_target_lags]
    )
    unique_regimes = np.unique(regimes)
    regime_design = np.column_stack(
        [(regimes[origins] == regime).astype(np.float64) for regime in unique_regimes]
    )
    conditioning = np.column_stack([target_history, regime_design])
    outcomes = np.column_stack([candidate_matrix, target_future])

    crossfit_groups = segments[origins] if crossfit_within_segments else None
    residuals, evaluated_rows, fold_ids = _ridge_residuals_forward(
        conditioning=conditioning,
        outcomes=outcomes,
        n_blocks=n_blocks,
        ridge_alpha=ridge_alpha,
        groups=crossfit_groups,
    )
    candidate_residuals = residuals[:, :-1]
    target_residuals = residuals[:, -1]
    evaluated_origins = origins[evaluated_rows]
    evaluated_regimes = regimes[evaluated_origins]
    # A group change prevents HAC covariance terms spanning a fold, regime, or
    # physical discontinuity.
    hac_groups = (
        fold_ids.astype(np.int64) * (len(unique_regimes) + 1) * (segments.max() + 2)
        + evaluated_regimes * (segments.max() + 2)
        + segments[evaluated_origins]
    )

    records: list[dict] = []
    p_values: list[float] = []
    for column, candidate in enumerate(candidates):
        driver_residual = candidate_residuals[:, column]
        product = driver_residual * target_residuals
        statistic, p_value, standard_error = _hac_mean_test(
            product, hac_groups, bandwidth=hac_bandwidth
        )
        effect = _safe_corr(driver_residual, target_residuals)
        block_effects = [
            _safe_corr(driver_residual[fold_ids == fold], target_residuals[fold_ids == fold])
            for fold in np.unique(fold_ids)
        ]
        effect_sign = 1.0 if effect >= 0 else -1.0
        stable_blocks = [
            abs(value) >= minimum_effect and np.sign(value) == effect_sign
            for value in block_effects
        ]
        stability = float(np.mean(stable_blocks)) if stable_blocks else 0.0
        regime_effects: dict[str, dict] = {}
        strong_regimes: list[int] = []
        for regime in unique_regimes:
            mask = evaluated_regimes == regime
            count = int(mask.sum())
            regime_effect = _safe_corr(driver_residual[mask], target_residuals[mask])
            regime_effects[str(int(regime))] = {
                "n": count,
                "residual_correlation": regime_effect,
            }
            if (
                count >= minimum_regime_samples
                and abs(regime_effect) >= minimum_effect
                and np.sign(regime_effect) == effect_sign
            ):
                strong_regimes.append(int(regime))
        records.append(
            {
                "feature": candidate.feature,
                "feature_index": candidate.feature_index,
                "lag": candidate.lag,
                "effect": effect,
                "hac_statistic": statistic,
                "hac_standard_error": standard_error,
                "p_value": p_value,
                "block_effects": block_effects,
                "stability": stability,
                "regime_effects": regime_effects,
                "strong_regimes": strong_regimes,
            }
        )
        p_values.append(p_value)

    q_values = benjamini_hochberg(np.asarray(p_values))
    for record, q_value in zip(records, q_values):
        record["q_value"] = float(q_value)
        record["passes_inference"] = bool(
            q_value <= fdr_level
            and abs(record["effect"]) >= minimum_effect
            and record["stability"] >= minimum_stability
        )

    ranking = sorted(
        range(len(records)),
        key=lambda index: (
            not records[index]["passes_inference"],
            -records[index]["stability"],
            -abs(records[index]["effect"]),
            records[index]["q_value"],
        ),
    )
    selected_indices: list[int] = []
    feature_counts: dict[str, int] = {}
    for index in ranking:
        record = records[index]
        if not record["passes_inference"]:
            continue
        feature_count = feature_counts.get(record["feature"], 0)
        if feature_count >= maximum_lags_per_feature:
            continue
        if len(selected_indices) >= maximum_pairs:
            break
        selected_indices.append(index)
        feature_counts[record["feature"]] = feature_count + 1

    selected: list[dict] = []
    for index in selected_indices:
        record = dict(records[index])
        record["selection_basis"] = "fdr_effect_stability"
        strong = record["strong_regimes"]
        # A pair is a stable core only when supported in at least two observed
        # regimes.  Otherwise its deployment mask is restricted to supported
        # regimes.
        if record["passes_inference"] and len(strong) >= 2:
            record["governance"] = "stable_core"
            record["available_regimes"] = [int(value) for value in unique_regimes]
        elif record["passes_inference"] and strong:
            record["governance"] = "regime_specific"
            record["available_regimes"] = strong
        selected.append(record)

    source_hash = _sha256_file(source_path) if source_path else None
    crossfit_group_counts = (
        np.unique(crossfit_groups, return_counts=True)[1]
        if crossfit_groups is not None
        else np.asarray([len(origins)], dtype=np.int64)
    )
    minimum_crossfit_group_rows = n_blocks * 20
    retained_crossfit_groups = crossfit_group_counts >= minimum_crossfit_group_rows
    manifest = {
        "schema_version": "1.2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Selected variables are treated as causal drivers of Y under the explicit "
            "assumptions of no unobserved confounding, no descendants of Y among the "
            "candidates, an adequate conditioning set, and sufficiently powerful "
            "conditional-independence tests; this is not an intervention-effect estimate."
        ),
        "source": {"path": str(source_path) if source_path else None, "sha256": source_hash},
        "training_interval": {"start": train_start, "end": train_end},
        "sample_counts": {
            "eligible": int(
                len(_valid_origins(segments, maximum_lag, screening_horizon))
            ),
            "screened": int(len(origins)),
            "crossfit_evaluated": int(len(evaluated_rows)),
            "crossfit_groups_screened": int(len(crossfit_group_counts)),
            "crossfit_groups_retained": int(np.sum(retained_crossfit_groups)),
            "crossfit_rows_excluded_short_groups": int(
                np.sum(crossfit_group_counts[~retained_crossfit_groups])
            ),
        },
        "configuration": {
            "candidate_lags": candidate_lags,
            "conditioning_target_lags": conditioning_target_lags,
            "n_blocks": n_blocks,
            "ridge_alpha": ridge_alpha,
            "hac_bandwidth": hac_bandwidth,
            "fdr_level": fdr_level,
            "minimum_effect": minimum_effect,
            "minimum_stability": minimum_stability,
            "maximum_pairs": maximum_pairs,
            "maximum_lags_per_feature": maximum_lags_per_feature,
            "crossfit_within_segments": crossfit_within_segments,
            "minimum_crossfit_rows_per_group": minimum_crossfit_group_rows,
            "screening_horizon": screening_horizon,
            "screening_target": "mean_future_target_over_horizon",
            "selection_ranking": "inference_pass_then_stability_then_absolute_effect",
        },
        "regime_ids": [int(value) for value in unique_regimes],
        "selected_pairs": selected,
        "all_candidates": records,
    }
    manifest["manifest_sha256"] = manifest_content_sha256(manifest)
    return manifest


def save_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
