# SF-CDI

[![tests](https://github.com/ZYH30/SF-CDI/actions/workflows/tests.yml/badge.svg)](https://github.com/ZYH30/SF-CDI/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official experiment code for **Static-Frozen Causal Driver Innovation
(SF-CDI)**, a multi-horizon forecasting framework for mechanical systems under
operating-condition shifts.

SF-CDI combines:

1. train-only lagged causal-driver discovery;
2. separate driver and target-history representations;
3. utility-aware minimax shortcut control;
4. an explicit static-freeze optimizer boundary; and
5. conditional driver-innovation forecasting.

The public repository contains the final paper-facing implementation,
reproduction configurations, frozen manifests, and compact result records.

## Method

$ H_D(t) $
Let \(H_D(t)\) denote histories selected by the frozen causal-driver manifest
and \(H_Y(t)\) denote the target history. SF-CDI learns separate
representations:

\[
Z_D = E_D(H_D(t)), \qquad Z_Y = E_Y(H_Y(t)).
\]

During the adjustment phase, the predictor minimizes future forecasting error
and cross-covariance while the driver encoder maximizes the error of an
adversary that reconstructs selected target-history lags:

\[
\mathcal{L}_{\mathrm{main}}
= \mathcal{L}_{\mathrm{future}}
+ \lambda_{\mathrm{cov}}\mathcal{L}_{\mathrm{cov}}
- \lambda_{\mathrm{adv}}\mathcal{L}_{\mathrm{adv}}.
\]

After adjustment, every driver-representation parameter is removed from the
forecast optimizer, its gradients are cleared, and \(Z_D\) is detached from
the future-target graph. The static phase trains only the target path and the
decoder. For the MetroPT-3 instantiation, the forecast has the explicit form

\[
\widehat{Y}^{+}
= Y_t + f_Y(Z_Y) + f_D([Z_D,Z_Y]),
\]

where the final term is the driver contribution conditional on the current
target state.

The NIST UR5 instantiation uses separated inverted-Transformer embeddings over
complete selected-driver histories. The MetroPT-3 instantiation uses a
lag-aligned driver LSTM and a separate target LSTM. The causal manifest,
minimax adjustment, static-freeze boundary, and conditional-innovation
principle are shared.

## Causal interpretation

Selected variables are interpreted as causal drivers of the target under the
following explicit assumptions:

- no unobserved confounding;
- no descendants of the target among candidate inputs;
- an adequate conditioning set that blocks non-direct paths;
- Markov/faithfulness conditions and sufficiently powerful conditional-
  independence tests; and
- discovery uses training data only.

The discovery module identifies supported driver-variable and lag pairs. It
does not estimate a continuous intervention-response curve.

## Repository layout

```text
SF-CDI/
├── configs/                  # Frozen paper configurations
├── docs/
│   ├── DATA.md               # Official sources and data preparation
│   └── REPRODUCIBILITY.md    # Evidence and evaluation boundaries
├── results/
│   ├── manifests/            # Frozen public causal-driver manifests
│   └── paper/metrics.json    # Compact paper result summary
├── scripts/
│   ├── prepare_nist_ur5.py
│   ├── run_experiment.py
│   ├── evaluate_stress.py
│   ├── run_semisynthetic.py
│   ├── summarize_seeded_comparison.py
│   ├── plot_paper_results.py
│   └── reproduce_paper.sh
├── sfcdi/
│   ├── data.py               # Split- and segment-safe datasets
│   ├── discovery.py          # Forward-blocked HAC-GCM discovery
│   ├── models.py             # SF-CDI and baseline adapters
│   ├── training.py           # Minimax and static-freeze optimization
│   ├── stress.py             # Frozen-input perturbations
│   └── vendor/tslib/         # Reported TSLib baselines
└── tests/
```

## Installation

The frozen environment uses Python 3.11 and PyTorch 2.5.1. Create an isolated
environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For a CUDA 12.1 installation matching the paper workstation:

```bash
python -m pip install \
  torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For CPU-only testing:

```bash
python -m pip install \
  torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Confirm the environment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest -q
```

## Data

Raw data are intentionally excluded from version control.

- **NIST UR5 controller trials:** download the 18 controller CSV files from
  the official NIST page and run `python scripts/prepare_nist_ur5.py`.
- **MetroPT-3:** download `MetroPT3(AirCompressor).csv` from the UCI Machine
  Learning Repository.

Expected paths, hashes, target construction, exclusions, and split protocols
are documented in [docs/DATA.md](docs/DATA.md).

## Run a primary experiment

The supplied configurations load the frozen public driver manifest by default.
This reproduces the forecast stage from the exact paper input view.

NIST UR5:

```bash
python scripts/run_experiment.py \
  --config configs/nist_ur5.json \
  --seed 20260722 \
  --run-id nist-primary-seed20260722 \
  --models sf_cdi itransformer dlinear timesnet
```

MetroPT-3:

```bash
python scripts/run_experiment.py \
  --config configs/metropt3.json \
  --seed 20260724 \
  --run-id metro-primary-seed20260724 \
  --models sf_cdi target_lstm itransformer
```

Use `--device cpu` for a functional CPU run. Full paper training is intended
for CUDA. To rerun causal discovery from training data rather than loading the
frozen manifest, add:

```text
--rediscover
```

Paper-facing model identifiers are:

| Identifier | Role |
|---|---|
| `sf_cdi` | Primary static-frozen model |
| `sf_cdi_no_control` | MetroPT-3 no-shortcut-control ablation |
| `sf_cdi_joint` | MetroPT-3 matched joint-training ablation |
| `sf_cdi_self_supervised` | NIST strict self-supervised control |
| `sf_cdi_closed_form` | NIST strict closed-form innovation variant |
| `target_lstm` | MetroPT-3 target-history reference |
| `itransformer`, `dlinear`, `timesnet` | Reported repository baselines |

Each run writes:

- effective configuration and environment;
- split and preprocessing manifests;
- the causal-driver manifest;
- training traces and static-phase gradient audits;
- checkpoints and predictions;
- horizon, operating-state, and physical-trial metrics; and
- frozen post-training shortcut probes.

## Reproduce the experiment matrix

After preparing both datasets:

```bash
bash scripts/reproduce_paper.sh
```

The script runs the minimum experiment matrix reported in the paper:

- five NIST seeds for SF-CDI, iTransformer, DLinear, and TimesNet;
- three NIST seeds for the strict closed-form variant;
- one matched NIST run for the reported strict self-supervised control;
- five MetroPT-3 seeds for SF-CDI, target-only LSTM, and iTransformer;
- three MetroPT-3 seeds for DLinear and TimesNet;
- no/default/strong shortcut-control comparisons;
- the matched joint-training optimizer-boundary ablation;
- frozen noise, missing-block, and driver-removal stress tests;
- ten-seed known-graph semi-synthetic validation;
- hierarchical paired summaries; and
- historically exposed frozen directional confirmations.

The script is intentionally exhaustive and can take several hours on a
low-memory GPU. Individual commands may be run independently.

## Paper-result snapshot

The compact result record is stored in
[`results/paper/metrics.json`](results/paper/metrics.json). Primary means are:

| Dataset | Model | MAE | RMSE |
|---|---|---:|---:|
| NIST UR5 | SF-CDI | **0.001484** | **0.002072** |
| NIST UR5 | enhanced iTransformer | 0.001505 | 0.002102 |
| NIST UR5 | DLinear | 0.001931 | 0.002562 |
| NIST UR5 | TimesNet | 0.001511 | 0.002087 |
| MetroPT-3 | SF-CDI | **0.059729** | **0.126903** |
| MetroPT-3 | target-only LSTM | 0.072592 | 0.146224 |
| MetroPT-3 | iTransformer | 0.094102 | 0.212518 |

The NIST comparison with TimesNet has a lower SF-CDI mean but its hierarchical
paired MAE interval crosses zero. The repository therefore does not claim
statistical superiority over TimesNet.

Generate all programmatic paper figures without retraining:

```bash
python scripts/plot_paper_results.py
```

## Evaluation boundary

The primary operating-condition and chronological evaluation periods
participated in model development. They are not described as pristine,
untouched tests. The later frozen confirmations were also historically exposed
and are provided only as directional checks. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Third-party code

The reported iTransformer, DLinear, and TimesNet implementations are
derived from THUML Time-Series-Library and are retained under its MIT license.
See [NOTICE](NOTICE) and
[`sfcdi/vendor/tslib/LICENSE`](sfcdi/vendor/tslib/LICENSE).

## Citation

The paper citation will be added after publication. Until then,
please cite the software metadata in [CITATION.cff](CITATION.cff) and reference
the repository URL.

## License

SF-CDI is released under the [MIT License](LICENSE). Dataset licenses and
terms remain governed by their official providers.
