#!/usr/bin/env python3
"""Print a read-only JSON snapshot of a running RSL-RL training run."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml
from tensorboard.backend.event_processing import event_accumulator


METRIC_TAGS = {
    "reward": "Train/mean_reward",
    "episode_length": "Train/mean_episode_length",
    "anchor_pos": "Episode_Termination/anchor_pos",
    "ee_body_pos": "Episode_Termination/ee_body_pos",
}
CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True, help="RSL-RL run directory to inspect.")
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Number of most recent iterations used for metric and ETA averages (default: 100).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path that also receives the JSON snapshot.",
    )
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be greater than zero")
    return args


def load_agent_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "params" / "agent.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Agent configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in: {config_path}")
    return config


def load_scalars(run_dir: Path) -> dict[str, list[Any]]:
    accumulator = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    return {
        tag: accumulator.Scalars(tag)
        for tag in accumulator.Tags().get(event_accumulator.SCALARS, [])
    }


def recent_mean(events: list[Any], window: int) -> float | None:
    recent = events[-window:]
    values = [float(event.value) for event in recent]
    if not values or not all(math.isfinite(value) for value in values):
        return None
    return fmean(values)


def iteration_seconds(scalars: dict[str, list[Any]], window: int) -> tuple[float | None, str | None]:
    collection = {int(event.step): float(event.value) for event in scalars.get("Perf/collection time", [])}
    learning = {int(event.step): float(event.value) for event in scalars.get("Perf/learning_time", [])}
    common_steps = sorted(collection.keys() & learning.keys())
    durations = [
        collection[step] + learning[step]
        for step in common_steps[-window:]
        if math.isfinite(collection[step]) and math.isfinite(learning[step])
    ]
    if durations:
        return fmean(durations), "Perf/collection time + Perf/learning_time"

    reference = scalars.get("Train/mean_reward", [])
    if len(reference) < 2:
        reference = max(
            (events for tag, events in scalars.items() if events and not tag.endswith("/time")),
            key=len,
            default=[],
        )
    recent = reference[-(window + 1) :]
    if len(recent) >= 2:
        step_delta = int(recent[-1].step) - int(recent[0].step)
        wall_delta = float(recent[-1].wall_time) - float(recent[0].wall_time)
        if step_delta > 0 and math.isfinite(wall_delta) and wall_delta >= 0.0:
            return wall_delta / step_delta, "TensorBoard wall-time slope"
    return None, None


def format_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    rounded = max(0, round(seconds))
    days, remainder = divmod(rounded, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, secs = divmod(remainder, 60)
    prefix = f"{days}d " if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{secs:02d}"


def checkpoint_snapshot(run_dir: Path) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    for path in run_dir.glob("model_*.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            continue
        stat = path.stat()
        checkpoints.append(
            {
                "iteration": int(match.group(1)),
                "name": path.name,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
            }
        )
    return sorted(checkpoints, key=lambda item: item["iteration"])


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory not found: {run_dir}")

    agent_config = load_agent_config(run_dir)
    scalars = load_scalars(run_dir)
    if not scalars:
        raise RuntimeError(f"No TensorBoard scalar data found in: {run_dir}")

    iteration_events = scalars.get("Train/mean_reward", [])
    if not iteration_events:
        iteration_events = max(
            (events for tag, events in scalars.items() if events and not tag.endswith("/time")),
            key=len,
            default=[],
        )
    if not iteration_events:
        raise RuntimeError(f"No iteration-indexed scalar data found in: {run_dir}")

    start_step = int(iteration_events[0].step)
    current_step = max(
        int(event.step)
        for tag, events in scalars.items()
        if not tag.endswith("/time")
        for event in events
    )
    max_iterations_raw = agent_config.get("max_iterations")
    max_iterations = int(max_iterations_raw) if max_iterations_raw is not None else None
    completed_iterations = current_step - start_step + 1
    if max_iterations is not None:
        completed_iterations = max(0, min(completed_iterations, max_iterations))
        remaining_iterations = max(0, max_iterations - completed_iterations)
        planned_final_step = start_step + max_iterations - 1
        progress_fraction = completed_iterations / max_iterations if max_iterations > 0 else None
    else:
        remaining_iterations = None
        planned_final_step = None
        progress_fraction = None

    seconds_per_iteration, eta_source = iteration_seconds(scalars, args.window)
    eta_seconds = (
        remaining_iterations * seconds_per_iteration
        if remaining_iterations is not None and seconds_per_iteration is not None
        else None
    )
    snapshot_time = datetime.now().astimezone()
    completion_time = snapshot_time + timedelta(seconds=eta_seconds) if eta_seconds is not None else None

    non_finite_examples: list[dict[str, Any]] = []
    non_finite_count = 0
    scalar_value_count = 0
    for tag, events in scalars.items():
        scalar_value_count += len(events)
        for event in events:
            value = float(event.value)
            if math.isfinite(value):
                continue
            non_finite_count += 1
            if len(non_finite_examples) < 100:
                non_finite_examples.append(
                    {
                        "tag": tag,
                        "step": int(event.step),
                        "value": repr(value),
                    }
                )

    checkpoints = checkpoint_snapshot(run_dir)
    metric_means = {
        name: recent_mean(scalars.get(tag, []), args.window)
        for name, tag in METRIC_TAGS.items()
    }
    metric_counts = {
        name: min(args.window, len(scalars.get(tag, [])))
        for name, tag in METRIC_TAGS.items()
    }

    document = {
        "run_dir": str(run_dir),
        "snapshot_time": snapshot_time.isoformat(),
        "current_step": current_step,
        "start_step": start_step,
        "planned_final_step": planned_final_step,
        "max_iterations": max_iterations,
        "completed_iterations": completed_iterations,
        "remaining_iterations": remaining_iterations,
        "progress": {
            "fraction": progress_fraction,
            "percent": progress_fraction * 100.0 if progress_fraction is not None else None,
        },
        "eta": {
            "seconds_per_iteration_recent_mean": seconds_per_iteration,
            "window": args.window,
            "source": eta_source,
            "remaining_seconds": eta_seconds,
            "remaining_human": format_duration(eta_seconds),
            "estimated_completion_time": completion_time.isoformat() if completion_time is not None else None,
        },
        "metrics_recent_mean": metric_means,
        "metrics_recent_count": metric_counts,
        "all_scalars_finite": non_finite_count == 0,
        "scalar_tag_count": len(scalars),
        "scalar_value_count": scalar_value_count,
        "non_finite_scalar_count": non_finite_count,
        "non_finite_scalar_examples": non_finite_examples,
        "event_files": sorted(path.name for path in run_dir.glob("events.out.tfevents.*")),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "latest_checkpoint": checkpoints[-1] if checkpoints else None,
    }
    serialized = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
