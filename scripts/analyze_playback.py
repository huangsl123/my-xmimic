#!/usr/bin/env python3
"""Summarize a policy playback CSV and save diagnostic plots."""

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
    parser.add_argument("--metrics_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--motion_file", type=Path)
    parser.add_argument("--playback_speed", type=float, default=1.0)
    parser.add_argument("--platform_size", type=float, default=8.0)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No playback rows found in {path}")
    return rows


def numeric_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def metric_stats(values: np.ndarray) -> dict[str, float | bool]:
    return {
        "first": float(values[0]),
        "last": float(values[-1]),
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "all_finite": bool(np.isfinite(values).all()),
    }


def tracking_window_stats(
    time_s: np.ndarray, xy_drift: np.ndarray, start_s: float, end_s: float
) -> dict[str, float | int | None]:
    """Summarize XY reference error in one motion-specific diagnostic window."""
    mask = (time_s >= start_s) & (time_s <= end_s)
    if not bool(np.any(mask)):
        return {
            "start_s": start_s,
            "end_s": end_s,
            "sample_count": 0,
            "first_m": None,
            "last_m": None,
            "increase_m": None,
            "mean_m": None,
            "p95_m": None,
            "max_m": None,
            "linear_slope_m_per_s": None,
        }
    window_time = time_s[mask]
    window_error = xy_drift[mask]
    slope = (
        float(np.polyfit(window_time, window_error, 1)[0])
        if len(window_time) >= 2 and float(np.ptp(window_time)) > 0.0
        else 0.0
    )
    return {
        "start_s": start_s,
        "end_s": end_s,
        "sample_count": int(len(window_error)),
        "first_m": float(window_error[0]),
        "last_m": float(window_error[-1]),
        "increase_m": float(window_error[-1] - window_error[0]),
        "mean_m": float(np.mean(window_error)),
        "p95_m": float(np.percentile(window_error, 95)),
        "max_m": float(np.max(window_error)),
        "linear_slope_m_per_s": slope,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_logged_numeric_finite = all(
        bool(np.isfinite(numeric_column(rows, name)).all())
        for name in rows[0]
    )
    done = numeric_column(rows, "done").astype(bool)
    motion_frame_values = numeric_column(rows, "motion_frame")
    motion_frames = motion_frame_values.astype(np.int64)
    motion_frames_are_integers = bool(
        np.equal(motion_frame_values, motion_frames.astype(np.float64)).all()
    )
    motion_frame_unit_increments = bool(
        len(motion_frames) <= 1 or np.equal(np.diff(motion_frames), 1).all()
    )
    done_indexes = np.flatnonzero(done)
    first_done_index = int(done_indexes[0]) if done_indexes.size else None
    # Isaac Lab auto-resets before command metrics are exposed for a terminal row.
    # Exclude that post-reset metric sample from trajectory/error statistics.
    stable_count = first_done_index if first_done_index is not None else len(rows)
    stable_rows = rows[:stable_count]
    if not stable_rows:
        raise ValueError("Playback terminated before a non-terminal sample was recorded")

    time_s = numeric_column(stable_rows, "simulation_time_s")
    step_dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.0
    simulated_duration_s = (float(rows[-1]["playback_step"]) + 1.0) * step_dt
    encoded_duration_s = simulated_duration_s / args.playback_speed

    termination_counts = {}
    terminal_terms = []
    for name in rows[0]:
        if not name.startswith("termination_"):
            continue
        count = int(sum(int(float(row[name])) for row in rows))
        termination_counts[name.removeprefix("termination_")] = count
        if first_done_index is not None and int(float(rows[first_done_index][name])):
            terminal_terms.append(name.removeprefix("termination_"))

    metric_names = [
        "reward",
        "error_anchor_pos",
        "error_anchor_xy",
        "error_anchor_rot",
        "error_anchor_lin_vel",
        "error_anchor_xy_vel",
        "error_anchor_ang_vel",
        "error_body_pos",
        "error_body_rot",
        "error_joint_pos",
        "error_joint_vel",
        "error_body_lin_vel",
        "error_body_ang_vel",
        "motion_wrap_count",
    ]
    metrics = {name: numeric_column(stable_rows, name) for name in metric_names if name in stable_rows[0]}

    actual_xy = np.column_stack(
        [
            numeric_column(stable_rows, "root_x_from_env_origin"),
            numeric_column(stable_rows, "root_y_from_env_origin"),
        ]
    )
    actual_radius = np.linalg.norm(actual_xy, axis=1)

    target_xy = None
    xy_drift = None
    motion_total_frames = None
    if args.motion_file is not None:
        with np.load(args.motion_file) as motion:
            motion_total_frames = int(motion["joint_pos"].shape[0])
            body_index = 0
            if "body_names" in motion:
                body_names = [str(name) for name in np.asarray(motion["body_names"]).tolist()]
                if "pelvis" in body_names:
                    body_index = body_names.index("pelvis")
            frames = numeric_column(stable_rows, "motion_frame").astype(np.int64)
            target_xy = np.asarray(motion["body_pos_w"][frames, body_index, :2], dtype=np.float64)
        xy_drift = np.linalg.norm(actual_xy - target_xy, axis=1)

    diagnostic_windows = {}
    if xy_drift is not None:
        diagnostic_windows = {
            "initial_stationary_0_4p24s": tracking_window_stats(time_s, xy_drift, 0.0, 4.24),
            "high_dynamic_22p38_25p38s": tracking_window_stats(time_s, xy_drift, 22.38, 25.38),
            "stop_transition_29p79_31p87s": tracking_window_stats(time_s, xy_drift, 29.79, 31.87),
            "stationary_tail_33p25_39p72s": tracking_window_stats(time_s, xy_drift, 33.25, 39.72),
        }

    motion_wrap_max = (
        float(np.max(metrics["motion_wrap_count"])) if "motion_wrap_count" in metrics else None
    )
    exact_motion_frame_sequence = bool(
        motion_total_frames is None
        or (
            motion_frames_are_integers
            and np.array_equal(motion_frames, np.arange(motion_total_frames, dtype=np.int64))
        )
    )
    summary = {
        "metrics_file": str(args.metrics_file.resolve()),
        "motion_file": str(args.motion_file.resolve()) if args.motion_file is not None else None,
        "logged_steps": len(rows),
        "pretermination_steps": stable_count,
        "step_dt_s": step_dt,
        "simulated_duration_s": simulated_duration_s,
        "playback_speed": args.playback_speed,
        "encoded_duration_s": encoded_duration_s,
        "first_done_step": first_done_index,
        "first_done_time_s": (
            float(rows[first_done_index]["simulation_time_s"]) if first_done_index is not None else None
        ),
        "terminal_terms": terminal_terms,
        "termination_counts": termination_counts,
        "motion_total_frames": motion_total_frames,
        "motion_completion_fraction": (
            len(rows) / motion_total_frames if motion_total_frames is not None else None
        ),
        "motion_frame_sequence": {
            "first": int(motion_frames[0]),
            "last": int(motion_frames[-1]),
            "minimum": int(np.min(motion_frames)),
            "maximum": int(np.max(motion_frames)),
            "unique_count": int(len(np.unique(motion_frames))),
            "all_integer": motion_frames_are_integers,
            "starts_at_zero": bool(motion_frames[0] == 0),
            "unit_increments": motion_frame_unit_increments,
            "exact_full_sequence": exact_motion_frame_sequence,
        },
        "metrics": {name: metric_stats(values) for name, values in metrics.items()},
        "platform_position": {
            "platform_size_m": args.platform_size,
            "start_xy_m": actual_xy[0].tolist(),
            "last_pretermination_xy_m": actual_xy[-1].tolist(),
            "max_distance_from_center_m": float(np.max(actual_radius)),
            "p95_distance_from_center_m": float(np.percentile(actual_radius, 95)),
            "max_distance_as_fraction_of_half_width": float(
                np.max(actual_radius) / (args.platform_size * 0.5)
            ),
            "target_last_xy_m": target_xy[-1].tolist() if target_xy is not None else None,
            "target_net_xy_displacement_m": (
                float(np.linalg.norm(target_xy[-1] - target_xy[0])) if target_xy is not None else None
            ),
            "xy_tracking_drift_last_m": float(xy_drift[-1]) if xy_drift is not None else None,
            "xy_tracking_drift_p95_m": (
                float(np.percentile(xy_drift, 95)) if xy_drift is not None else None
            ),
            "xy_tracking_drift_max_m": float(np.max(xy_drift)) if xy_drift is not None else None,
            "center_distance_is_safety_only": True,
        },
        "xy_tracking_windows": diagnostic_windows,
        "quality_checks": {
            # Retain the legacy key used by the aggregation script, but make
            # its contract cover every numeric CSV field (positions, frame
            # counters, done flags, and terminations as well as errors).
            "all_metrics_finite": all_logged_numeric_finite,
            "all_logged_numeric_finite": all_logged_numeric_finite,
            "exact_motion_frame_count": bool(
                motion_total_frames is None or len(rows) == motion_total_frames
            ),
            "exact_motion_frame_sequence": exact_motion_frame_sequence,
            "motion_completed": bool(
                first_done_index is None
                and exact_motion_frame_sequence
                and (motion_total_frames is None or len(rows) >= motion_total_frames)
            ),
            "no_motion_wrap": bool(motion_wrap_max == 0.0) if motion_wrap_max is not None else None,
            "started_near_platform_center": bool(actual_radius[0] < 0.05),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    failure_time = summary["first_done_time_s"]

    axes[0, 0].plot(time_s, metrics["reward"], color="#2b6cb0", linewidth=2)
    axes[0, 0].set_title("Per-step reward")
    axes[0, 0].set_xlabel("Simulation time (s)")
    axes[0, 0].grid(alpha=0.25)

    for name, label in (
        ("error_anchor_pos", "Anchor position"),
        ("error_body_pos", "Mean body position"),
        ("error_joint_pos", "Joint position norm"),
    ):
        axes[0, 1].plot(time_s, metrics[name], label=label, linewidth=2)
    axes[0, 1].set_title("Position tracking errors")
    axes[0, 1].set_xlabel("Simulation time (s)")
    axes[0, 1].set_ylabel("Error")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[0, 2].plot(time_s, metrics["error_anchor_rot"], label="Anchor rotation", linewidth=2)
    axes[0, 2].plot(time_s, metrics["error_body_rot"], label="Mean body rotation", linewidth=2)
    axes[0, 2].set_title("Orientation tracking errors")
    axes[0, 2].set_xlabel("Simulation time (s)")
    axes[0, 2].set_ylabel("Error (rad)")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.25)

    for name, label in (
        ("error_anchor_lin_vel", "Anchor linear"),
        ("error_body_lin_vel", "Mean body linear"),
        ("error_joint_vel", "Joint velocity norm"),
    ):
        axes[1, 0].plot(time_s, metrics[name], label=label, linewidth=1.8)
    axes[1, 0].set_title("Velocity tracking errors")
    axes[1, 0].set_xlabel("Simulation time (s)")
    axes[1, 0].set_ylabel("Error")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(time_s, actual_radius, label="Distance from tile center", linewidth=2)
    if xy_drift is not None:
        axes[1, 1].plot(time_s, xy_drift, label="XY reference drift", linewidth=2)
    axes[1, 1].set_title("Root position drift")
    axes[1, 1].set_xlabel("Simulation time (s)")
    axes[1, 1].set_ylabel("Distance (m)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    half_width = args.platform_size * 0.5
    axes[1, 2].add_patch(
        plt.Rectangle(
            (-half_width, -half_width),
            args.platform_size,
            args.platform_size,
            fill=False,
            linestyle="--",
            linewidth=1.5,
            color="#718096",
            label=f"{args.platform_size:g} m tile",
        )
    )
    if target_xy is not None:
        axes[1, 2].plot(target_xy[:, 0], target_xy[:, 1], color="#38a169", label="Reference")
    axes[1, 2].plot(actual_xy[:, 0], actual_xy[:, 1], color="#c53030", linewidth=2, label="Robot")
    axes[1, 2].scatter(*actual_xy[0], color="#2b6cb0", s=60, label="Start", zorder=3)
    axes[1, 2].scatter(*actual_xy[-1], color="#c53030", s=60, marker="x", label="Last stable", zorder=3)
    axes[1, 2].scatter(0.0, 0.0, color="black", s=35, marker="+", label="Tile center", zorder=3)
    axes[1, 2].set_xlim(-half_width, half_width)
    axes[1, 2].set_ylim(-half_width, half_width)
    axes[1, 2].set_aspect("equal", adjustable="box")
    axes[1, 2].set_title("Position within terrain tile")
    axes[1, 2].set_xlabel("X from environment origin (m)")
    axes[1, 2].set_ylabel("Y from environment origin (m)")
    axes[1, 2].legend(loc="upper right", fontsize=8)
    axes[1, 2].grid(alpha=0.2)

    if failure_time is not None:
        fig.suptitle(
            f"Policy playback: terminated at {failure_time:.2f} s ({', '.join(terminal_terms)})",
            fontsize=16,
        )
    else:
        fig.suptitle("Policy playback: no termination", fontsize=16)

    figure_path = args.output_dir / "playback_analysis.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print(f"Saved summary: {summary_path}")
    print(f"Saved figure:  {figure_path}")


if __name__ == "__main__":
    main()
