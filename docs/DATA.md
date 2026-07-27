# Data preparation

The repository does not redistribute either real-world dataset. Download each
dataset from its official host and retain the original files unchanged.

## NIST UR5 controller trials

Official source:

<https://www.nist.gov/el/intelligent-systems-division-73500/degradation-measurement-robot-arm-position-accuracy>

Download the 18 controller-level CSV files covering normal and cold-start
conditions, two speeds, supported payloads, and three trials per condition.
Place the CSV files under:

```text
dataset/nist_ur5/raw/
```

Then run:

```bash
python scripts/prepare_nist_ur5.py
```

The parser:

- validates the tuple-CSV schema and strictly increasing controller time;
- excludes target and actual joint positions from model inputs because the
  forecasting target is algebraically derived from those channels;
- constructs per-timestamp six-joint tracking-error RMS;
- preserves trial, operating-condition, and cold-start split metadata; and
- writes `dataset/nist_ur5/processed/nist_ur5_trials_v2.npz`.

The artifact used in the paper has SHA-256:

```text
596b2c890e0ccdece54d78e5840e5156d3d347ba79c0051861d6575dd0888beb
```

The task uses 125 historical samples and predicts the next 25 samples. Normal
starts form the training partition; cold-start half-speed trials form the
primary operating-condition evaluation; cold-start full-speed trials are a
historically exposed directional confirmation.

## MetroPT-3

Official source and license:

<https://archive.ics.uci.edu/dataset/791/metropt+3+dataset>

The dataset is distributed by UCI under CC BY 4.0. Download
`MetroPT3(AirCompressor).csv` and place it at:

```text
dataset/metropt3/raw/MetroPT3(AirCompressor).csv
```

The CSV used in the paper has SHA-256:

```text
db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24
```

The loader removes the redundant row index, excludes `TP3` from the primary
protocol because it is a near-duplicate pressure proxy, and does not expose
future control states. Gaps longer than 30 seconds define physical segment
boundaries. The task uses 60 historical samples and predicts the next 12
`Reservoirs` pressure samples under a chronological 60/20/20 split.

## Leakage controls

For both datasets:

- causal discovery and standardization are fitted on training rows only;
- every forecasting window remains inside one split and one physical
  trial/segment;
- operating-condition labels are metadata for governance and stratified
  reporting, not unrestricted input features;
- no future-known variable is used; and
- the repository defaults to the frozen public driver manifests. Pass
  `--rediscover` to rerun train-only discovery.
