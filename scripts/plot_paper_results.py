#!/usr/bin/env python3
"""Generate the manuscript figures from the frozen public result summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results" / "paper" / "metrics.json"
OUTPUT_DIR = ROOT / "figures"
COLORS = {
    "blue": "#3465A4",
    "orange": "#E07A3F",
    "green": "#4C956C",
    "red": "#C44536",
    "gray": "#747D8C",
}


def load_results() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.55,
            "lines.linewidth": 1.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(figure: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{name}.pdf")
    figure.savefig(OUTPUT_DIR / f"{name}.png", dpi=320)
    plt.close(figure)


def relative_gain(candidate: float, baseline: float) -> float:
    return 100.0 * (baseline - candidate) / baseline


def plot_main_gains(results: dict) -> None:
    nist = results["primary_results"]["nist_ur5"]["models"]
    metro = results["primary_results"]["metropt3"]["models"]
    groups = [
        (
            "NIST UR5",
            nist["sf_cdi"],
            [
                ("iTransformer\n($n=5$)", nist["itransformer"]),
                ("DLinear\n($n=5$)", nist["dlinear"]),
                ("TimesNet\n($n=5$)", nist["timesnet"]),
            ],
        ),
        (
            "MetroPT-3",
            metro["sf_cdi_5_seed"],
            [
                ("Target LSTM\n($n=5$)", metro["target_lstm"]),
                ("iTransformer\n($n=5$)", metro["itransformer"]),
                ("DLinear\n($n=3$)", metro["dlinear"], metro["sf_cdi_3_seed"]),
                ("TimesNet\n($n=3$)", metro["timesnet"], metro["sf_cdi_3_seed"]),
            ],
        ),
    ]
    figure, axes = plt.subplots(
        1, 2, figsize=(7.05, 2.65), constrained_layout=True
    )
    for axis, (title, default_candidate, comparisons) in zip(axes, groups):
        labels = []
        mae_gains = []
        rmse_gains = []
        for comparison in comparisons:
            label, baseline = comparison[:2]
            candidate = comparison[2] if len(comparison) == 3 else default_candidate
            labels.append(label)
            mae_gains.append(
                relative_gain(
                    candidate["mae"]["mean"], baseline["mae"]["mean"]
                )
            )
            rmse_gains.append(
                relative_gain(
                    candidate["rmse"]["mean"], baseline["rmse"]["mean"]
                )
            )
        positions = np.arange(len(labels))
        width = 0.36
        mae_bars = axis.bar(
            positions - width / 2,
            mae_gains,
            width,
            label="MAE",
            color=COLORS["blue"],
        )
        rmse_bars = axis.bar(
            positions + width / 2,
            rmse_gains,
            width,
            label="RMSE",
            color=COLORS["orange"],
        )
        axis.set_title(title)
        axis.set_xticks(positions, labels)
        axis.set_ylabel("SF-CDI relative improvement (%)")
        axis.grid(axis="y", alpha=0.28)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, ncol=2, loc="upper left")
        upper = max(mae_gains + rmse_gains) * 1.19
        axis.set_ylim(0, max(upper, 3.0))
        for bars in (mae_bars, rmse_bars):
            for bar in bars:
                value = bar.get_height()
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + upper * 0.018,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                )
    save(figure, "main_relative_gains")


def plot_horizon_consistency(results: dict) -> None:
    primary = results["primary_results"]
    specifications = [
        (
            np.asarray(
                primary["nist_ur5"][
                    "sf_cdi_minus_itransformer_mae_by_horizon"
                ]
            )
            * 1e5,
            "NIST UR5: SF-CDI $-$ iTransformer",
            r"MAE difference ($\times 10^{-5}$)",
            COLORS["blue"],
        ),
        (
            np.asarray(
                primary["metropt3"][
                    "sf_cdi_minus_target_lstm_mae_by_horizon"
                ]
            )
            * 1e3,
            "MetroPT-3: SF-CDI $-$ target LSTM",
            r"MAE difference ($\times 10^{-3}$)",
            COLORS["green"],
        ),
    ]
    figure, axes = plt.subplots(
        1, 2, figsize=(7.05, 2.45), constrained_layout=True
    )
    for axis, (differences, title, ylabel, color) in zip(axes, specifications):
        horizon = np.arange(1, len(differences) + 1)
        axis.axhline(0, color="#333333", linestyle="--", linewidth=0.9)
        axis.plot(horizon, differences, marker="o", markersize=3.1, color=color)
        axis.fill_between(horizon, differences, 0, color=color, alpha=0.12)
        axis.set_title(title)
        axis.set_xlabel("Forecast horizon")
        axis.set_ylabel(ylabel)
        axis.set_xlim(1, len(differences))
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
    save(figure, "horizon_consistency")


def plot_shortcut_pareto(results: dict) -> None:
    settings = results["shortcut_control_ablation"]["settings"]
    labels = {
        "none": "No control",
        "default": "Default (0.05)",
        "strong": "Strong (0.20)",
    }
    colors = {
        "none": COLORS["red"],
        "default": COLORS["orange"],
        "strong": COLORS["blue"],
    }
    figure, axis = plt.subplots(figsize=(3.55, 2.75), constrained_layout=True)
    for key in ("none", "default", "strong"):
        record = settings[key]
        x = record["blocked_probe_r2"]
        y = record["mae"]
        axis.errorbar(
            x["mean"],
            y["mean"],
            xerr=x["sample_sd"],
            yerr=y["sample_sd"],
            fmt="o",
            markersize=6,
            capsize=3,
            color=colors[key],
        )
        axis.annotate(
            labels[key],
            (x["mean"], y["mean"]),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=7.2,
        )
    axis.set_xlabel(r"Blocked frozen-probe recoverability ($R^2$)")
    axis.set_ylabel("MAE")
    axis.set_title("MetroPT-3 utility--shortcut trade-off")
    axis.grid(alpha=0.27)
    axis.set_axisbelow(True)
    save(figure, "shortcut_utility_pareto")


def plot_stress_tests(results: dict) -> None:
    stress = results["stress_tests"]
    conditions = [
        "gaussian_0.10",
        "block_missing_0.20",
        "top_driver_removed",
        "bottom_driver_removed",
        "all_drivers_removed",
    ]
    labels = [
        "Gaussian\nnoise",
        "Block\nmissing",
        "First driver\nremoved",
        "Last driver\nremoved",
        "All drivers\nremoved",
    ]
    positions = np.arange(len(conditions))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.05, 2.8), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        [stress["nist_ur5"][key]["mae_change_percent"] for key in conditions],
        width,
        color=COLORS["blue"],
        label="NIST UR5",
    )
    axis.bar(
        positions + width / 2,
        [stress["metropt3"][key]["mae_change_percent"] for key in conditions],
        width,
        color=COLORS["green"],
        label="MetroPT-3",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("MAE change from clean replay (%)")
    axis.set_title("Frozen-model driver-input stress tests")
    axis.grid(axis="y", alpha=0.26)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    save(figure, "stress_tests")


def plot_semisynthetic(results: dict) -> None:
    experiment = results["semisynthetic"]
    forecast = experiment["forecast"]
    labels = [
        "Target\nonly",
        "All variables\nall lags",
        "Marginal\ntop-3",
        "GCM\nselected",
        "Oracle\ndrivers",
    ]
    keys = [
        "target_only",
        "all_variables_all_lags",
        "marginal_top3",
        "gcm_selected",
        "oracle_drivers",
    ]
    positions = np.arange(len(keys))
    width = 0.36
    figure, axes = plt.subplots(
        1, 2, figsize=(7.05, 2.65), constrained_layout=True
    )
    axes[0].bar(
        positions,
        [forecast[key]["mae"] for key in keys],
        color=[COLORS["gray"]] * 3 + [COLORS["blue"], COLORS["green"]],
    )
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Known-graph forecast views")
    axes[0].grid(axis="y", alpha=0.25)
    selection = experiment["lagged_gcm"]
    marginal = experiment["marginal_top3"]
    metric_labels = ["Pair F1", "Exact lag", "Shortcut selected"]
    axes[1].bar(
        positions[:3] - width / 2,
        [
            selection["pair_f1"]["mean"],
            selection["exact_lag_accuracy"]["mean"],
            selection["shortcut_selection_rate"],
        ],
        width,
        color=COLORS["blue"],
        label="Lagged GCM",
    )
    axes[1].bar(
        positions[:3] + width / 2,
        [
            marginal["pair_f1"]["mean"],
            marginal["exact_lag_accuracy"]["mean"],
            marginal["shortcut_selection_rate"],
        ],
        width,
        color=COLORS["orange"],
        label="Marginal top-3",
    )
    axes[1].set_xticks(positions[:3], metric_labels)
    axes[1].set_ylim(0, 1.12)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Driver and lag recovery")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    save(figure, "semisynthetic_mechanism")


def main() -> None:
    configure_style()
    results = load_results()
    plot_main_gains(results)
    plot_horizon_consistency(results)
    plot_shortcut_pareto(results)
    plot_stress_tests(results)
    plot_semisynthetic(results)
    print(f"Wrote figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
