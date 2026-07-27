# Reproducibility scope

## Public code identity

This repository contains the paper-facing SF-CDI implementation, the strict
closed-form and self-supervised controls, the matched joint-training ablation,
reported baselines, stress tests, semi-synthetic mechanism validation, frozen
driver manifests, and a compact machine-readable result summary. Raw datasets
and trained checkpoints are not distributed.

## Deterministic controls

- Python 3.11 and pinned scientific dependencies.
- NumPy and PyTorch seeds recorded in every run configuration.
- Deterministic cuDNN behavior with benchmark mode disabled.
- Time-stratified deterministic window subsampling.
- Train-only standardization and causal discovery.
- Configuration, environment, split, preprocessing, manifest, history,
  checkpoint, prediction, and metric artifacts written per run.
- Static-phase gradient norms recorded at every epoch.

## Public frozen manifests

The machine-specific source paths in the archival manifests were replaced by
repository-relative paths. Selected pairs and all statistical fields are
unchanged. The archival and public hashes are listed in
`results/manifests/PROVENANCE.md`.

## Evaluation boundary

The primary evaluation periods were used during model development. They are
reported as operating-condition or chronological evaluation evidence, not as
pristine untouched tests. The later frozen confirmations were also
historically exposed and are used only for directional consistency. This
limitation is encoded in the corresponding configuration purposes and in the
public result summary.
