#!/usr/bin/env python3
"""Aggregate multiple analyze_playback summaries into tables and comparison charts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Evaluation label and analyze_playback summary.json path. May be repeated.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def load_summaries(specs: list[str]) -> list[tuple[str, dict]]:
    records = []
    seen_labels = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected LABEL=PATH, received: {spec}")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Evaluation label is empty: {spec}")
        if label in seen_labels:
            raise ValueError(f"Duplicate evaluation label: {label}")
        path = Path(raw_path).expanduser().resolve()
        with path.open(encoding="utf-8") as file:
            document = json.load(file)
        seen_labels.add(label)
        records.append((label, document))
    return records


def flatten_record(label: str, document: dict) -> dict:
    metrics = document["metrics"]
    position = document["platform_position"]
    checks = document["quality_checks"]
    windows = document.get("xy_tracking_windows", {})
    initial_window = windows.get("initial_stationary_0_4p24s", {})
    tail_window = windows.get("stationary_tail_33p25_39p72s", {})
    terminal_terms = document.get("terminal_terms", [])
    return {
        "label": label,
        "logged_steps": document["logged_steps"],
        "motion_total_frames": document.get("motion_total_frames"),
        "simulated_duration_s": document["simulated_duration_s"],
        "motion_completion_fraction": document.get("motion_completion_fraction"),
        "motion_completed": checks["motion_completed"],
        "exact_motion_frame_sequence": checks.get("exact_motion_frame_sequence", False),
        "terminal_terms": ",".join(terminal_terms),
        "all_metrics_finite": checks["all_metrics_finite"],
        "no_motion_wrap": checks.get("no_motion_wrap"),
        "started_near_platform_center": checks["started_near_platform_center"],
        "reward_mean": metrics["reward"]["mean"],
        "reward_p95": metrics["reward"]["p95"],
        "anchor_pos_mean": metrics["error_anchor_pos"]["mean"],
        "anchor_pos_p95": metrics["error_anchor_pos"]["p95"],
        "anchor_pos_max": metrics["error_anchor_pos"]["max"],
        "anchor_rot_p95": metrics["error_anchor_rot"]["p95"],
        "body_pos_mean": metrics["error_body_pos"]["mean"],
        "body_pos_p95": metrics["error_body_pos"]["p95"],
        "body_pos_max": metrics["error_body_pos"]["max"],
        "joint_pos_mean": metrics["error_joint_pos"]["mean"],
        "joint_pos_p95": metrics["error_joint_pos"]["p95"],
        "joint_pos_max": metrics["error_joint_pos"]["max"],
        "max_distance_from_center_m": position["max_distance_from_center_m"],
        "xy_tracking_drift_p95_m": position.get("xy_tracking_drift_p95_m"),
        "xy_tracking_drift_max_m": position.get("xy_tracking_drift_max_m"),
        "xy_tracking_drift_last_m": position.get("xy_tracking_drift_last_m"),
        "initial_stationary_xy_max_m": initial_window.get("max_m"),
        "stationary_tail_xy_increase_m": tail_window.get("increase_m"),
        "stationary_tail_xy_slope_m_per_s": tail_window.get("linear_slope_m_per_s"),
    }


def passes_acceptance(record: dict) -> bool:
    nominal = str(record["label"]).startswith("nominal")
    xy_p95_limit = 0.20 if nominal else 0.25
    xy_max_limit = 0.35 if nominal else 0.50
    xy_last_limit = 0.25 if nominal else 0.35
    initial_stationary_limit = 0.15 if nominal else 0.25
    return bool(
        record["motion_completed"]
        and record["exact_motion_frame_sequence"]
        and record["motion_total_frames"] is not None
        and record["logged_steps"] == record["motion_total_frames"]
        and not record["terminal_terms"]
        and record["all_metrics_finite"]
        and record["no_motion_wrap"] is True
        and record["started_near_platform_center"]
        and record["max_distance_from_center_m"] < 1.0
        and record["body_pos_p95"] <= 0.08
        and record["joint_pos_p95"] <= 0.90
        and record["anchor_rot_p95"] <= 0.15
        and record["xy_tracking_drift_p95_m"] is not None
        and record["xy_tracking_drift_p95_m"] <= xy_p95_limit
        and record["xy_tracking_drift_max_m"] is not None
        and record["xy_tracking_drift_max_m"] <= xy_max_limit
        and record["xy_tracking_drift_last_m"] is not None
        and record["xy_tracking_drift_last_m"] <= xy_last_limit
        and record["initial_stationary_xy_max_m"] is not None
        and record["initial_stationary_xy_max_m"] <= initial_stationary_limit
        and record["stationary_tail_xy_increase_m"] is not None
        and record["stationary_tail_xy_increase_m"] <= 0.05
        and record["stationary_tail_xy_slope_m_per_s"] is not None
        and record["stationary_tail_xy_slope_m_per_s"] <= 0.01
    )


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def bar_chart(axis: plt.Axes, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    positions = np.arange(len(labels))
    axis.bar(positions, values, color="#2b6cb0")
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_summaries(args.summary)
    records = [flatten_record(label, document) for label, document in loaded]
    for record in records:
        record["acceptance_profile"] = (
            "nominal" if str(record["label"]).startswith("nominal") else "robust"
        )
        record["acceptance_pass"] = passes_acceptance(record)

    write_csv(args.output_dir / "evaluation_comparison.csv", records)
    comparison = {
        "criteria": {
            "nominal": {
                "xy_p95_max_m": 0.20,
                "xy_max_m": 0.35,
                "xy_last_m": 0.25,
                "initial_stationary_xy_max_m": 0.15,
            },
            "robust": {
                "xy_p95_max_m": 0.25,
                "xy_max_m": 0.50,
                "xy_last_m": 0.35,
                "initial_stationary_xy_max_m": 0.25,
            },
            "shared": {
                "platform_center_distance_safety_max_m": 1.0,
                "body_pos_p95_max_m": 0.08,
                "joint_pos_p95_max_rad_norm": 0.90,
                "anchor_rot_p95_max_rad": 0.15,
                "stationary_tail_xy_increase_max_m": 0.05,
                "stationary_tail_xy_slope_max_m_per_s": 0.01,
                "motion_wrap_count": 0,
            },
        },
        "evaluations": records,
        "passed": [record["label"] for record in records if record["acceptance_pass"]],
        "failed": [record["label"] for record in records if not record["acceptance_pass"]],
        "pass_count": sum(bool(record["acceptance_pass"]) for record in records),
        "evaluation_count": len(records),
    }
    (args.output_dir / "evaluation_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    labels = [record["label"] for record in records]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    bar_chart(
        axes[0, 0],
        labels,
        [100.0 * float(record["motion_completion_fraction"] or 0.0) for record in records],
        "Motion completion",
        "Percent",
    )
    axes[0, 0].axhline(100.0, color="#38a169", linestyle="--", linewidth=1.5)
    bar_chart(
        axes[0, 1],
        labels,
        [float(record["anchor_pos_p95"]) for record in records],
        "Anchor position error p95",
        "Error (m)",
    )
    bar_chart(
        axes[1, 0],
        labels,
        [float(record["body_pos_p95"]) for record in records],
        "Mean body position error p95",
        "Error (m)",
    )
    bar_chart(
        axes[1, 1],
        labels,
        [float(record["xy_tracking_drift_p95_m"] or 0.0) for record in records],
        "XY reference drift p95",
        "Distance (m)",
    )
    axes[1, 1].axhline(0.25, color="#c53030", linestyle="--", linewidth=1.5, label="0.25 m criterion")
    axes[1, 1].legend()
    figure.suptitle("Policy playback comparison", fontsize=16)
    figure.savefig(args.output_dir / "evaluation_comparison.png", dpi=180)
    plt.close(figure)

    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
