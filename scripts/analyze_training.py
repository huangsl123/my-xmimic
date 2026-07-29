"""Export TensorBoard scalars and create reproducible training charts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ScalarSeries = dict[str, np.ndarray]
ScalarCollection = dict[str, ScalarSeries]
CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True, help="RSL-RL run directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for CSV, JSON, and PNG outputs.")
    parser.add_argument("--num_envs", type=int, default=None, help="Environment count used by quality checks.")
    parser.add_argument(
        "--expected_iterations",
        type=int,
        default=None,
        help=(
            "Expected global iteration endpoint (last TensorBoard step + 1). "
            "For a resumed run covering steps 9 through 50008, use 50009."
        ),
    )
    return parser.parse_args()


def load_scalars(run_dir: Path) -> ScalarCollection:
    """Load all scalar samples into compact NumPy arrays."""
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    output: ScalarCollection = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        output[tag] = {
            "wall_time": np.fromiter((event.wall_time for event in events), dtype=np.float64, count=len(events)),
            "step": np.fromiter((event.step for event in events), dtype=np.int64, count=len(events)),
            "value": np.fromiter((event.value for event in events), dtype=np.float64, count=len(events)),
        }
    return output


def best_mode_for_tag(tag: str) -> str:
    """Return the optimization direction used for the summary's best value."""
    if tag == "Episode_Termination/time_out":
        return "maximum"
    lower_is_better_prefixes = (
        "Episode_Termination/",
        "Loss/",
        "Metrics/motion/error",
    )
    lower_is_better_tags = {
        "Perf/collection time",
        "Perf/learning_time",
    }
    if tag.startswith(lower_is_better_prefixes) or tag in lower_is_better_tags:
        return "minimum"
    return "maximum"


def metric_summary(
    tag: str, series: ScalarSeries
) -> dict[str, float | int | str | None]:
    values = series["value"]
    steps = series["step"]
    finite_indices = np.flatnonzero(np.isfinite(values))
    minimum_index = int(finite_indices[np.argmin(values[finite_indices])]) if len(finite_indices) else None
    maximum_index = int(finite_indices[np.argmax(values[finite_indices])]) if len(finite_indices) else None
    best_mode = best_mode_for_tag(tag)
    best_index = minimum_index if best_mode == "minimum" else maximum_index

    def tail_mean(count: int) -> float:
        return float(values[-min(count, len(values)) :].mean())

    return {
        "count": int(len(values)),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "first": float(values[0]),
        "last": float(values[-1]),
        "minimum": float(values[minimum_index]) if minimum_index is not None else None,
        "minimum_step": int(steps[minimum_index]) if minimum_index is not None else None,
        "maximum": float(values[maximum_index]) if maximum_index is not None else None,
        "maximum_step": int(steps[maximum_index]) if maximum_index is not None else None,
        "mean": float(values.mean()),
        "last_10_mean": tail_mean(10),
        "last_100_mean": tail_mean(100),
        "last_1000_mean": tail_mean(1000),
        "best_mode": best_mode,
        "best_step": int(steps[best_index]) if best_index is not None else None,
        "best_value": float(values[best_index]) if best_index is not None else None,
    }


def rolling_window_size(point_count: int) -> int:
    """Choose a useful smoothing window for both smoke and long training runs."""
    if point_count <= 2:
        return 1
    return min(point_count, min(500, max(3, round(point_count * 0.01))))


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a trailing rolling mean while tolerating non-finite samples."""
    if window <= 1:
        return values.copy()

    finite = np.isfinite(values)
    sums = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    window_sums = sums[window:] - sums[:-window]
    window_counts = counts[window:] - counts[:-window]
    output = np.full(len(window_sums), np.nan, dtype=np.float64)
    np.divide(window_sums, window_counts, out=output, where=window_counts > 0)
    return output


def plot_tags(
    axis: plt.Axes,
    scalars: ScalarCollection,
    tags: list[str],
    title: str,
    *,
    ylabel: str = "",
    minimum_step: int | None = None,
) -> None:
    plotted = False
    smoothed = False
    for tag in tags:
        series = scalars.get(tag)
        if series is None or len(series["step"]) == 0:
            continue
        steps = series["step"]
        values = series["value"]
        if minimum_step is not None:
            mask = steps >= minimum_step
            steps = steps[mask]
            values = values[mask]
        if len(steps) == 0:
            continue
        window = rolling_window_size(len(steps))
        label = tag.split("/")[-1]
        raw_alpha = 0.55 if len(steps) <= 20 else 0.2
        raw_line = axis.plot(
            steps,
            values,
            linewidth=0.8,
            alpha=raw_alpha if window > 1 else 1.0,
            label=label if window == 1 else "_nolegend_",
        )[0]
        if window > 1:
            axis.plot(
                steps[window - 1 :],
                rolling_mean(values, window),
                color=raw_line.get_color(),
                linewidth=1.8,
                label=f"{label} (mean {window})",
            )
            smoothed = True
        plotted = True
    axis.set_title(title)
    axis.set_xlabel("Iteration")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if plotted and (len(tags) > 1 or smoothed):
        axis.legend(fontsize=8)
    if not plotted:
        axis.text(0.5, 0.5, "No scalar data", ha="center", va="center", transform=axis.transAxes)


def save_training_dashboard(
    path: Path,
    scalars: ScalarCollection,
    title: str,
    *,
    minimum_step: int | None = None,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    plot_tags(
        axes[0, 0],
        scalars,
        ["Train/mean_reward"],
        "Mean reward",
        ylabel="Reward",
        minimum_step=minimum_step,
    )
    plot_tags(
        axes[0, 1],
        scalars,
        ["Train/mean_episode_length"],
        "Mean episode length",
        ylabel="Steps",
        minimum_step=minimum_step,
    )
    plot_tags(
        axes[1, 0],
        scalars,
        [
            "Episode_Termination/time_out",
            "Episode_Termination/anchor_pos",
            "Episode_Termination/anchor_ori",
            "Episode_Termination/ee_body_pos",
        ],
        "Episode terminations",
        ylabel="TensorBoard logged value",
        minimum_step=minimum_step,
    )
    plot_tags(
        axes[1, 1],
        scalars,
        [
            "Metrics/motion/error_anchor_pos",
            "Metrics/motion/error_body_pos",
            "Metrics/motion/error_joint_pos",
        ],
        "Position tracking errors",
        ylabel="Error",
        minimum_step=minimum_step,
    )
    plot_tags(
        axes[2, 0],
        scalars,
        ["Loss/value_function", "Loss/surrogate"],
        "PPO losses",
        ylabel="Loss",
        minimum_step=minimum_step,
    )
    plot_tags(
        axes[2, 1],
        scalars,
        ["Perf/total_fps"],
        "Training throughput",
        ylabel="Steps/s",
        minimum_step=minimum_step,
    )
    figure.suptitle(title, fontsize=16)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_long_csv(path: Path, scalars: ScalarCollection) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["tag", "step", "wall_time", "value"])
        for tag, series in sorted(scalars.items()):
            writer.writerows(
                (
                    tag,
                    int(step),
                    float(wall_time),
                    float(value),
                )
                for step, wall_time, value in zip(
                    series["step"],
                    series["wall_time"],
                    series["value"],
                )
            )


def checkpoint_sort_key(path: Path) -> tuple[int, int | str]:
    """Sort numbered checkpoints numerically after any unexpected model names."""
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is not None:
        return (1, int(match.group(1)))
    return (0, path.name)


def checkpoint_metadata(run_dir: Path) -> dict[str, object]:
    checkpoint_paths = sorted(run_dir.glob("model_*.pt"), key=checkpoint_sort_key)
    numbered = [
        (int(match.group(1)), path.name)
        for path in checkpoint_paths
        if (match := CHECKPOINT_PATTERN.fullmatch(path.name)) is not None
    ]
    latest_numbered = max(numbered, default=None)
    return {
        "checkpoints": [path.name for path in checkpoint_paths],
        "checkpoint_count": len(checkpoint_paths),
        "checkpoint_steps": [step for step, _ in numbered],
        "latest_checkpoint": latest_numbered[1] if latest_numbered is not None else None,
        "latest_checkpoint_step": latest_numbered[0] if latest_numbered is not None else None,
    }


def iteration_progress(
    steps: np.ndarray, expected_end_exclusive: int | None
) -> dict[str, bool | int | str | None]:
    """Describe iteration records and validate an optional global step endpoint."""
    if len(steps) == 0:
        raise ValueError("Iteration reference series is empty.")

    unique_steps = np.unique(steps)
    first_step = int(unique_steps[0])
    last_step = int(unique_steps[-1])
    record_count = int(len(steps))
    unique_record_count = int(len(unique_steps))
    observed_span = last_step - first_step + 1
    missing_in_observed_span = max(0, observed_span - unique_record_count)
    duplicate_record_count = record_count - unique_record_count

    expected_record_count = None
    expected_missing_record_count = None
    endpoint_reached = None
    records_complete = None
    complete = None
    if expected_end_exclusive is not None:
        expected_record_count = max(0, expected_end_exclusive - first_step)
        expected_steps_present = int(
            np.count_nonzero((unique_steps >= first_step) & (unique_steps < expected_end_exclusive))
        )
        expected_missing_record_count = max(0, expected_record_count - expected_steps_present)
        endpoint_reached = last_step + 1 >= expected_end_exclusive
        records_complete = expected_missing_record_count == 0
        complete = endpoint_reached and records_complete

    return {
        "expected_iterations_semantics": "global_step_end_exclusive",
        "first_iteration_step": first_step,
        "last_iteration_step": last_step,
        "observed_global_step_end_exclusive": last_step + 1,
        "observed_iteration_records": record_count,
        "observed_unique_iteration_steps": unique_record_count,
        "observed_iteration_span": observed_span,
        "duplicate_iteration_records": duplicate_record_count,
        "missing_steps_in_observed_span": missing_in_observed_span,
        "iteration_sequence_contiguous": missing_in_observed_span == 0,
        "expected_iterations": expected_end_exclusive,
        "expected_iteration_records_from_first_step": expected_record_count,
        "expected_missing_iteration_records": expected_missing_record_count,
        "expected_global_step_end_reached": endpoint_reached,
        "iteration_records_complete": records_complete,
        "iteration_count_complete": complete,
    }


def iteration_reference_series(scalars: ScalarCollection) -> tuple[str, ScalarSeries]:
    """Choose an iteration-indexed series for progress and tail calculations."""
    preferred_tag = "Train/mean_reward"
    preferred = scalars.get(preferred_tag)
    if preferred is not None and len(preferred["step"]) > 0:
        return preferred_tag, preferred

    candidates = [
        (tag, series)
        for tag, series in scalars.items()
        if not tag.endswith("/time") and len(series["step"]) > 0
    ]
    if not candidates:
        raise RuntimeError("No iteration-indexed TensorBoard scalar data found.")
    return max(candidates, key=lambda item: (int(item[1]["step"][-1]), len(item[1]["step"])))


def main() -> None:
    analysis_start = time.perf_counter()
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scalars = load_scalars(run_dir)
    if not scalars:
        raise RuntimeError(f"No TensorBoard scalar data found in {run_dir}")

    write_long_csv(output_dir / "scalars.csv", scalars)
    summaries = {
        tag: metric_summary(tag, series)
        for tag, series in scalars.items()
        if len(series["value"]) > 0
    }

    reference_tag, reference_series = iteration_reference_series(scalars)
    progress = iteration_progress(reference_series["step"], args.expected_iterations)
    finite = all(bool(np.isfinite(series["value"]).all()) for series in scalars.values())
    quality_checks: dict[str, object] = {
        "all_scalars_finite": finite,
        "num_envs": args.num_envs,
        # Retained as a compatibility alias. Its precise meaning is now explicit below.
        "observed_iterations": progress["observed_global_step_end_exclusive"],
        **progress,
    }

    episode_tag = "Train/mean_episode_length"
    ee_tag = "Episode_Termination/ee_body_pos"
    wrap_tag = "Metrics/motion/motion_wrap_count"
    if episode_tag in summaries:
        episode_length_last = float(summaries[episode_tag]["last"])
        quality_checks["mean_episode_length_last"] = episode_length_last
        quality_checks["episodes_longer_than_one_step"] = episode_length_last > 1.0
    if ee_tag in summaries:
        ee_body_pos_last = float(summaries[ee_tag]["last"])
        quality_checks["ee_body_pos_last"] = ee_body_pos_last
        quality_checks["ee_body_pos_zero_at_last"] = math.isclose(ee_body_pos_last, 0.0, abs_tol=1.0e-9)
    if wrap_tag in summaries:
        motion_wrap_max = float(summaries[wrap_tag]["maximum"])
        quality_checks["motion_wrap_count_max"] = motion_wrap_max
        quality_checks["motion_wrap_count_zero"] = math.isclose(motion_wrap_max, 0.0, abs_tol=1.0e-9)

    checkpoint_info = checkpoint_metadata(run_dir)
    onnx_names = sorted(path.name for path in run_dir.glob("*.onnx"))
    first_iteration_step = int(progress["first_iteration_step"])
    last_iteration_step = int(progress["last_iteration_step"])
    latest_checkpoint_step = checkpoint_info["latest_checkpoint_step"]
    expected_final_checkpoint_step = (
        args.expected_iterations - 1 if args.expected_iterations is not None else None
    )
    quality_checks["latest_checkpoint_step"] = latest_checkpoint_step
    quality_checks["checkpoint_matches_last_iteration"] = (
        latest_checkpoint_step == last_iteration_step if latest_checkpoint_step is not None else False
    )
    quality_checks["expected_final_checkpoint_step"] = expected_final_checkpoint_step
    quality_checks["checkpoint_reaches_expected_final_step"] = (
        latest_checkpoint_step >= expected_final_checkpoint_step
        if latest_checkpoint_step is not None and expected_final_checkpoint_step is not None
        else None
    )
    tail_first_step = max(first_iteration_step, last_iteration_step - 999)
    tail_record_count = int(np.count_nonzero(reference_series["step"] >= tail_first_step))
    scalar_point_count = sum(len(series["value"]) for series in scalars.values())
    scalar_array_bytes = sum(array.nbytes for series in scalars.values() for array in series.values())

    summary_document = {
        "analysis_schema_version": 2,
        "run_dir": str(run_dir),
        "event_files": sorted(path.name for path in run_dir.glob("events.out.tfevents.*")),
        **checkpoint_info,
        "onnx_files": onnx_names,
        "scalar_tag_count": len(scalars),
        "scalar_point_count": scalar_point_count,
        "iteration_reference_tag": reference_tag,
        "iteration_progress": progress,
        "training_tail": {
            "requested_iterations": 1000,
            "first_step": tail_first_step,
            "last_step": last_iteration_step,
            "record_count": tail_record_count,
        },
        "analysis": {
            "compact_scalar_array_bytes": scalar_array_bytes,
            "compact_scalar_array_mib": scalar_array_bytes / (1024**2),
            "overview_rolling_window": rolling_window_size(len(reference_series["step"])),
            "tail_rolling_window": rolling_window_size(tail_record_count),
        },
        "quality_checks": quality_checks,
        "metrics": summaries,
    }

    save_training_dashboard(
        output_dir / "training_overview.png",
        scalars,
        run_dir.name,
    )
    save_training_dashboard(
        output_dir / "training_tail.png",
        scalars,
        f"{run_dir.name} — last 1000 iterations ({tail_first_step}–{last_iteration_step})",
        minimum_step=tail_first_step,
    )

    reward_tags = sorted(tag for tag in scalars if tag.startswith("Episode_Reward/"))
    reward_figure, reward_axis = plt.subplots(figsize=(14, 8), constrained_layout=True)
    plot_tags(reward_axis, scalars, reward_tags, "Reward components", ylabel="Weighted reward")
    reward_figure.savefig(output_dir / "reward_components.png", dpi=180)
    plt.close(reward_figure)

    error_tags = sorted(tag for tag in scalars if tag.startswith("Metrics/motion/"))
    error_figure, error_axis = plt.subplots(figsize=(14, 8), constrained_layout=True)
    plot_tags(error_axis, scalars, error_tags, "Motion tracking errors", ylabel="Error")
    error_figure.savefig(output_dir / "tracking_errors.png", dpi=180)
    plt.close(error_figure)

    summary_document["analysis"]["runtime_seconds"] = time.perf_counter() - analysis_start
    summary_document["analysis"]["scalars_csv_bytes"] = (output_dir / "scalars.csv").stat().st_size
    (output_dir / "summary.json").write_text(
        json.dumps(summary_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[INFO] Analysis saved to: {output_dir}")
    print(json.dumps(quality_checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
