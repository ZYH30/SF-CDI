#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"

nist_seeds=(20260722 20260723 20260724 20260725 20260726)
metro_seeds=(20260724 20260725 20260726 20260727 20260728)

for seed in "${nist_seeds[@]}"; do
  "${python_bin}" scripts/run_experiment.py \
    --config configs/nist_ur5.json \
    --seed "${seed}" \
    --run-id "nist-primary-seed${seed}" \
    --models sf_cdi itransformer dlinear timesnet
done

for seed in 20260722 20260723 20260724; do
  "${python_bin}" scripts/run_experiment.py \
    --config configs/nist_ur5.json \
    --seed "${seed}" \
    --run-id "nist-closed-form-seed${seed}" \
    --models sf_cdi_closed_form
done

"${python_bin}" scripts/run_experiment.py \
  --config configs/nist_ur5.json \
  --seed 20260722 \
  --run-id nist-self-supervised-seed20260722 \
  --models sf_cdi_self_supervised

for seed in "${metro_seeds[@]}"; do
  "${python_bin}" scripts/run_experiment.py \
    --config configs/metropt3.json \
    --seed "${seed}" \
    --run-id "metro-primary-seed${seed}" \
    --models sf_cdi target_lstm itransformer
done

for seed in 20260724 20260725 20260726; do
  "${python_bin}" scripts/run_experiment.py \
    --config configs/metropt3.json \
    --seed "${seed}" \
    --run-id "metro-paper-baselines-seed${seed}" \
    --models dlinear timesnet
done

for seed in "${metro_seeds[@]}"; do
  "${python_bin}" scripts/run_experiment.py \
    --config configs/metropt3_default_control.json \
    --seed "${seed}" \
    --run-id "metro-default-control-seed${seed}"
done

for seed in 20260724 20260725 20260726; do
  "${python_bin}" scripts/run_experiment.py \
    --config configs/metropt3.json \
    --seed "${seed}" \
    --run-id "metro-no-control-seed${seed}" \
    --models sf_cdi_no_control
done

"${python_bin}" scripts/evaluate_stress.py \
  --run-dirs \
    outputs/nist-primary-seed20260722 \
    outputs/nist-primary-seed20260723 \
    outputs/nist-primary-seed20260724 \
    outputs/nist-primary-seed20260725 \
    outputs/nist-primary-seed20260726 \
  --model sf_cdi \
  --output outputs/nist-sf-cdi-stress-5seed.json

"${python_bin}" scripts/evaluate_stress.py \
  --run-dirs \
    outputs/metro-primary-seed20260724 \
    outputs/metro-primary-seed20260725 \
    outputs/metro-primary-seed20260726 \
    outputs/metro-primary-seed20260727 \
    outputs/metro-primary-seed20260728 \
  --model sf_cdi \
  --output outputs/metro-sf-cdi-stress-5seed.json

"${python_bin}" scripts/run_experiment.py \
  --config configs/metropt3.json \
  --seed 20260724 \
  --run-id "metro-joint-boundary-seed20260724" \
  --models sf_cdi_joint

"${python_bin}" scripts/summarize_seeded_comparison.py \
  --candidate-runs \
    outputs/nist-primary-seed20260722 \
    outputs/nist-primary-seed20260723 \
    outputs/nist-primary-seed20260724 \
    outputs/nist-primary-seed20260725 \
    outputs/nist-primary-seed20260726 \
  --baseline-runs \
    outputs/nist-primary-seed20260722 \
    outputs/nist-primary-seed20260723 \
    outputs/nist-primary-seed20260724 \
    outputs/nist-primary-seed20260725 \
    outputs/nist-primary-seed20260726 \
  --candidate-model sf_cdi \
  --baseline-model itransformer \
  --dataset-artifact dataset/nist_ur5/processed/nist_ur5_trials_v2.npz \
  --bootstrap-repeats 50000 \
  --output outputs/nist-sf-cdi-vs-itransformer-5seed.json

"${python_bin}" scripts/summarize_seeded_comparison.py \
  --candidate-runs \
    outputs/nist-primary-seed20260722 \
    outputs/nist-primary-seed20260723 \
    outputs/nist-primary-seed20260724 \
    outputs/nist-primary-seed20260725 \
    outputs/nist-primary-seed20260726 \
  --baseline-runs \
    outputs/nist-primary-seed20260722 \
    outputs/nist-primary-seed20260723 \
    outputs/nist-primary-seed20260724 \
    outputs/nist-primary-seed20260725 \
    outputs/nist-primary-seed20260726 \
  --candidate-model sf_cdi \
  --baseline-model timesnet \
  --dataset-artifact dataset/nist_ur5/processed/nist_ur5_trials_v2.npz \
  --bootstrap-repeats 50000 \
  --output outputs/nist-sf-cdi-vs-timesnet-5seed.json

"${python_bin}" scripts/summarize_seeded_comparison.py \
  --candidate-runs \
    outputs/nist-primary-seed20260722 \
    outputs/nist-primary-seed20260723 \
    outputs/nist-primary-seed20260724 \
    outputs/nist-primary-seed20260725 \
    outputs/nist-primary-seed20260726 \
  --baseline-runs \
    outputs/nist-primary-seed20260722 \
    outputs/nist-primary-seed20260723 \
    outputs/nist-primary-seed20260724 \
    outputs/nist-primary-seed20260725 \
    outputs/nist-primary-seed20260726 \
  --candidate-model sf_cdi \
  --baseline-model dlinear \
  --dataset-artifact dataset/nist_ur5/processed/nist_ur5_trials_v2.npz \
  --bootstrap-repeats 50000 \
  --output outputs/nist-sf-cdi-vs-dlinear-5seed.json

"${python_bin}" scripts/summarize_seeded_comparison.py \
  --candidate-runs \
    outputs/nist-closed-form-seed20260722 \
    outputs/nist-closed-form-seed20260723 \
    outputs/nist-closed-form-seed20260724 \
  --baseline-runs \
    outputs/nist-primary-seed20260722 \
    outputs/nist-primary-seed20260723 \
    outputs/nist-primary-seed20260724 \
  --candidate-model sf_cdi_closed_form \
  --baseline-model itransformer \
  --dataset-artifact dataset/nist_ur5/processed/nist_ur5_trials_v2.npz \
  --bootstrap-repeats 50000 \
  --output outputs/nist-closed-form-vs-itransformer-3seed.json

"${python_bin}" scripts/summarize_seeded_comparison.py \
  --candidate-runs \
    outputs/metro-primary-seed20260724 \
    outputs/metro-primary-seed20260725 \
    outputs/metro-primary-seed20260726 \
    outputs/metro-primary-seed20260727 \
    outputs/metro-primary-seed20260728 \
  --baseline-runs \
    outputs/metro-primary-seed20260724 \
    outputs/metro-primary-seed20260725 \
    outputs/metro-primary-seed20260726 \
    outputs/metro-primary-seed20260727 \
    outputs/metro-primary-seed20260728 \
  --candidate-model sf_cdi \
  --baseline-model target_lstm \
  --temporal-blocks 5 \
  --bootstrap-repeats 50000 \
  --output outputs/metro-sf-cdi-vs-target-lstm-5seed.json

"${python_bin}" scripts/summarize_seeded_comparison.py \
  --candidate-runs \
    outputs/metro-primary-seed20260724 \
    outputs/metro-primary-seed20260725 \
    outputs/metro-primary-seed20260726 \
    outputs/metro-primary-seed20260727 \
    outputs/metro-primary-seed20260728 \
  --baseline-runs \
    outputs/metro-primary-seed20260724 \
    outputs/metro-primary-seed20260725 \
    outputs/metro-primary-seed20260726 \
    outputs/metro-primary-seed20260727 \
    outputs/metro-primary-seed20260728 \
  --candidate-model sf_cdi \
  --baseline-model itransformer \
  --temporal-blocks 5 \
  --bootstrap-repeats 50000 \
  --output outputs/metro-sf-cdi-vs-itransformer-5seed.json

for baseline in dlinear timesnet; do
  "${python_bin}" scripts/summarize_seeded_comparison.py \
    --candidate-runs \
      outputs/metro-primary-seed20260724 \
      outputs/metro-primary-seed20260725 \
      outputs/metro-primary-seed20260726 \
    --baseline-runs \
      outputs/metro-paper-baselines-seed20260724 \
      outputs/metro-paper-baselines-seed20260725 \
      outputs/metro-paper-baselines-seed20260726 \
    --candidate-model sf_cdi \
    --baseline-model "${baseline}" \
    --temporal-blocks 5 \
    --bootstrap-repeats 50000 \
    --output "outputs/metro-sf-cdi-vs-${baseline}-3seed.json"
done

"${python_bin}" scripts/summarize_seeded_comparison.py \
  --candidate-runs \
    outputs/metro-primary-seed20260724 \
    outputs/metro-primary-seed20260725 \
    outputs/metro-primary-seed20260726 \
    outputs/metro-primary-seed20260727 \
    outputs/metro-primary-seed20260728 \
  --baseline-runs \
    outputs/metro-default-control-seed20260724 \
    outputs/metro-default-control-seed20260725 \
    outputs/metro-default-control-seed20260726 \
    outputs/metro-default-control-seed20260727 \
    outputs/metro-default-control-seed20260728 \
  --candidate-model sf_cdi \
  --baseline-model sf_cdi \
  --temporal-blocks 5 \
  --bootstrap-repeats 50000 \
  --output outputs/metro-strong-vs-default-control-5seed.json

"${python_bin}" scripts/summarize_seeded_comparison.py \
  --candidate-runs \
    outputs/metro-primary-seed20260724 \
    outputs/metro-primary-seed20260725 \
    outputs/metro-primary-seed20260726 \
  --baseline-runs \
    outputs/metro-no-control-seed20260724 \
    outputs/metro-no-control-seed20260725 \
    outputs/metro-no-control-seed20260726 \
  --candidate-model sf_cdi \
  --baseline-model sf_cdi_no_control \
  --temporal-blocks 5 \
  --bootstrap-repeats 50000 \
  --output outputs/metro-strong-vs-no-control-3seed.json

"${python_bin}" scripts/run_experiment.py \
  --config configs/nist_ur5_confirmation.json \
  --seed 20260727 \
  --run-id "nist-exposed-confirmation-seed20260727"

"${python_bin}" scripts/run_experiment.py \
  --config configs/metropt3_confirmation.json \
  --seed 20260729 \
  --run-id "metro-exposed-confirmation-seed20260729"

"${python_bin}" scripts/run_semisynthetic.py \
  --output outputs/metro-background-semisynthetic-10seed.json

"${python_bin}" scripts/plot_paper_results.py
