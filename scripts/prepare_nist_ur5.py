#!/usr/bin/env python3
"""Parse official NIST UR5 tuple-CSV trials into a versioned NPZ artifact."""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

REGIMES = [
    (False, "half", "1.6lb", "normal_half_speed_payload_1.6lb"),
    (False, "full", "1.6lb", "normal_full_speed_payload_1.6lb"),
    (False, "half", "4.5lb", "normal_half_speed_payload_4.5lb"),
    (False, "full", "4.5lb", "normal_full_speed_payload_4.5lb"),
    (True, "half", "4.5lb", "cold_half_speed_payload_4.5lb"),
    (True, "full", "4.5lb", "cold_full_speed_payload_4.5lb"),
]
REGIME_LOOKUP = {
    (cold, speed, payload): index
    for index, (cold, speed, payload, _) in enumerate(REGIMES)
}

FEATURE_GROUPS = [
    ("target_joint_velocity", 3),
    ("actual_joint_velocity", 4),
    ("target_joint_current", 5),
    ("actual_joint_current", 6),
    ("target_joint_acceleration", 7),
    ("target_joint_torque", 8),
    ("joint_control_current", 9),
    ("cartesian_tool", 10),
    ("tcp_force", 11),
    ("joint_temperature", 12),
]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def trial_metadata(path: Path) -> dict:
    name = path.name.lower()
    if name == "half_speed_payload_1p6lb_1.csv":
        cold, speed, payload, replicate = False, "half", "1.6lb", 1
    else:
        cold = "coldstart" in name
        speed = "full" if "fullspeed" in name else "half"
        payload = "1.6lb" if "payload16lb" in name else "4.5lb"
        match = re.search(r"lb([123])csv", name)
        if match is None:
            raise ValueError(f"Cannot recover replicate from {path.name}")
        replicate = int(match.group(1))
    key = (cold, speed, payload)
    if key not in REGIME_LOOKUP:
        raise ValueError(f"Unsupported NIST regime {key} in {path.name}")
    return {
        "path": path,
        "cold_start": cold,
        "speed": speed,
        "payload": payload,
        "replicate": replicate,
        "regime": REGIME_LOOKUP[key],
    }


def parse_trial(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: list[float] = []
    features: list[list[float]] = []
    targets: list[float] = []
    with path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = ast.literal_eval(line)
            except (ValueError, SyntaxError) as error:
                raise ValueError(f"Invalid tuple row {path.name}:{line_number}") from error
            if len(row) != 13 or any(len(group) != 6 for group in row[1:]):
                raise ValueError(f"Unexpected field shape at {path.name}:{line_number}")
            time_value = float(row[0][0])
            target_position = np.asarray(row[1], dtype=np.float64)
            actual_position = np.asarray(row[2], dtype=np.float64)
            # Position channels are deliberately excluded from features because
            # the target is algebraically derived from them.
            tracking_rmse = float(np.sqrt(np.mean((actual_position - target_position) ** 2)))
            flattened: list[float] = []
            for _, field_index in FEATURE_GROUPS:
                flattened.extend(float(value) for value in row[field_index])
            times.append(time_value)
            features.append(flattened)
            targets.append(tracking_rmse)
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def feature_names() -> list[str]:
    names = []
    for group, _ in FEATURE_GROUPS:
        suffixes = ("x", "y", "z", "rx", "ry", "rz") if group in {
            "cartesian_tool", "tcp_force"
        } else tuple(f"j{joint}" for joint in range(1, 7))
        names.extend(f"{group}_{suffix}" for suffix in suffixes)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPOSITORY_ROOT / "dataset/nist_ur5/raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "dataset/nist_ur5/processed/nist_ur5_trials_v2.npz",
    )
    arguments = parser.parse_args()
    raw_files = [
        path
        for path in arguments.raw_dir.glob("*.csv")
        if path.stat().st_size > 1_000_000
    ]
    metadata = [trial_metadata(path) for path in raw_files]
    metadata.sort(key=lambda item: (item["replicate"], item["regime"]))
    if len(metadata) != 18:
        raise RuntimeError(f"Expected 18 NIST trials, found {len(metadata)}")
    observed = {(item["replicate"], item["regime"]) for item in metadata}
    expected = {(replicate, regime) for replicate in (1, 2, 3) for regime in range(6)}
    if observed != expected:
        raise RuntimeError(f"Trial grid mismatch: missing={sorted(expected-observed)}")

    feature_parts = []
    target_parts = []
    regime_parts = []
    segment_parts = []
    split_parts = []
    trial_parts = []
    timestamp_parts = []
    trial_records = []
    synthetic_offset_seconds = 0.0
    for trial_id, item in enumerate(metadata):
        times, features, target = parse_trial(item["path"])
        relative_time = times - times[0]
        median_delta = float(np.median(np.diff(times)))
        if not np.all(np.diff(times) > 0):
            raise RuntimeError(f"Non-increasing controller time in {item['path'].name}")
        synthetic_seconds = synthetic_offset_seconds + relative_time
        timestamp_parts.append(
            (synthetic_seconds * 1e9).round().astype(np.int64).astype("datetime64[ns]")
        )
        synthetic_offset_seconds = float(synthetic_seconds[-1] + 10.0)
        n = len(target)
        feature_parts.append(features)
        target_parts.append(target)
        regime_parts.append(np.full(n, item["regime"], dtype=np.int64))
        segment_parts.append(np.full(n, trial_id, dtype=np.int64))
        split_parts.append(np.full(n, item["replicate"] - 1, dtype=np.int8))
        trial_parts.append(np.full(n, trial_id, dtype=np.int64))
        trial_records.append(
            {
                "trial_id": trial_id,
                "file": item["path"].name,
                "sha256": file_sha256(item["path"]),
                "rows": n,
                "controller_time_start": float(times[0]),
                "controller_time_end": float(times[-1]),
                "median_delta_seconds": median_delta,
                "replicate": item["replicate"],
                "regime": item["regime"],
                "regime_name": REGIMES[item["regime"]][3],
            }
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    regimes_all = np.concatenate(regime_parts)
    split_ids_cold_start_ood = np.where(
        regimes_all < 4, 0, np.where(regimes_all == 4, 1, 2)
    ).astype(np.int8)
    regimes_cold_start_ood = np.where(regimes_all < 4, regimes_all, -1).astype(
        np.int64
    )
    np.savez_compressed(
        arguments.output,
        timestamps=np.concatenate(timestamp_parts),
        features=np.concatenate(feature_parts),
        target=np.concatenate(target_parts),
        regimes=regimes_all,
        segments=np.concatenate(segment_parts),
        split_ids=np.concatenate(split_parts),
        split_ids_cold_start_ood=split_ids_cold_start_ood,
        regimes_cold_start_ood=regimes_cold_start_ood,
        trial_ids=np.concatenate(trial_parts),
        feature_names=np.asarray(feature_names()),
        regime_names=np.asarray([item[3] for item in REGIMES]),
        target_name=np.asarray("joint_position_tracking_rmse_deg"),
    )
    record = {
        "schema_version": "1.0",
        "official_source": (
            "https://www.nist.gov/el/intelligent-systems-division-73500/"
            "degradation-measurement-robot-arm-position-accuracy"
        ),
        "target_definition": (
            "Per-timestamp RMS(actual_joint_position - target_joint_position) over six joints"
        ),
        "forbidden_as_features": ["target_joint_position", "actual_joint_position"],
        "split_protocol": "replicate 1 train, replicate 2 validation, replicate 3 test",
        "additional_protocol": (
            "normal-start trials train, cold half-speed validation, cold full-speed test; "
            "cold regimes mapped to unknown (-1) for deployment gating"
        ),
        "feature_names": feature_names(),
        "trials": trial_records,
        "output": str(arguments.output.resolve()),
        "output_sha256": file_sha256(arguments.output),
    }
    metadata_path = arguments.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "shape": [
                    sum(item["rows"] for item in trial_records),
                    len(feature_names()),
                ],
                "trials": len(trial_records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
