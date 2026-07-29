#!/usr/bin/env python3
"""Validate playback video metadata and a completed FFmpeg decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--decode_progress", type=Path, required=True)
    parser.add_argument("--decode_errors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected_fps", type=float, default=25.0)
    parser.add_argument("--expected_frames", type=int, default=3972)
    parser.add_argument("--expected_width", type=int, default=1280)
    parser.add_argument("--expected_height", type=int, default=720)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_progress(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    progress_path = args.decode_progress.expanduser().resolve()
    errors_path = args.decode_errors.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sample_indices = sorted(
        {round(index * (frames - 1) / 11) for index in range(12)}
    )
    sampled_frames = []
    sampled_images = []
    for frame_index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        decoded, frame = capture.read()
        pixel_sha256 = hashlib.sha256(frame.tobytes()).hexdigest() if decoded else None
        sampled_frames.append(
            {
                "frame": frame_index,
                "decoded": bool(decoded),
                "mean": float(frame.mean()) if decoded else None,
                "std": float(frame.std()) if decoded else None,
                "minimum": int(frame.min()) if decoded else None,
                "maximum": int(frame.max()) if decoded else None,
                "pixel_sha256": pixel_sha256,
            }
        )
        if decoded:
            sampled_images.append((frame_index, frame))
    capture.release()
    duration_s = frames / fps

    adjacent_frame_mads = []
    for (first_index, first_frame), (second_index, second_frame) in zip(
        sampled_images, sampled_images[1:]
    ):
        adjacent_frame_mads.append(
            {
                "from_frame": first_index,
                "to_frame": second_index,
                "mean_absolute_difference": float(
                    abs(first_frame.astype("float32") - second_frame.astype("float32")).mean()
                ),
            }
        )
    relative_to_first_frame_mads = []
    if sampled_images:
        first_index, first_frame = sampled_images[0]
        for frame_index, frame in sampled_images[1:]:
            relative_to_first_frame_mads.append(
                {
                    "from_frame": first_index,
                    "to_frame": frame_index,
                    "mean_absolute_difference": float(
                        abs(first_frame.astype("float32") - frame.astype("float32")).mean()
                    ),
                }
            )
    sampled_pixel_hashes = {
        item["pixel_sha256"] for item in sampled_frames if item["pixel_sha256"] is not None
    }
    sampled_mad_values = [
        item["mean_absolute_difference"]
        for item in adjacent_frame_mads + relative_to_first_frame_mads
    ]
    maximum_sampled_frame_mad = max(sampled_mad_values, default=None)
    minimum_required_sampled_frame_mad = 0.5

    progress = parse_progress(progress_path)
    decoded_frames = int(progress.get("frame", "-1"))
    decode_error_bytes = errors_path.stat().st_size
    expected_duration_s = args.expected_frames / args.expected_fps
    decoded_duration_s = float(progress.get("out_time_us", "nan")) / 1_000_000.0
    duplicate_frames = int(progress.get("dup_frames", "-1"))
    dropped_frames = int(progress.get("drop_frames", "-1"))
    video_mtime_ns = video_path.stat().st_mtime_ns
    checks = {
        "fps_matches": math.isclose(fps, args.expected_fps, rel_tol=0.0, abs_tol=1.0e-6),
        "container_frame_count_matches": frames == args.expected_frames,
        "decoded_frame_count_matches": decoded_frames == args.expected_frames,
        "duration_matches": math.isclose(
            duration_s, expected_duration_s, rel_tol=0.0, abs_tol=0.01
        ),
        "decoded_pts_duration_matches": math.isclose(
            decoded_duration_s, expected_duration_s, rel_tol=0.0, abs_tol=0.01
        ),
        "resolution_matches": (width, height) == (args.expected_width, args.expected_height),
        "all_sampled_frames_decode": all(item["decoded"] for item in sampled_frames),
        "first_frame_is_not_black": bool(
            sampled_frames
            and sampled_frames[0]["decoded"]
            and sampled_frames[0]["mean"] is not None
            and sampled_frames[0]["mean"] > 1.0
            and sampled_frames[0]["std"] is not None
            and sampled_frames[0]["std"] > 1.0
        ),
        "all_sampled_frames_are_not_black": all(
            item["decoded"]
            and item["mean"] is not None
            and item["mean"] > 1.0
            and item["std"] is not None
            and item["std"] > 1.0
            for item in sampled_frames
        ),
        "sampled_frames_have_multiple_pixel_hashes": len(sampled_pixel_hashes) >= 2,
        "sampled_frame_maximum_mad_exceeds_threshold": bool(
            maximum_sampled_frame_mad is not None
            and maximum_sampled_frame_mad > minimum_required_sampled_frame_mad
        ),
        "no_duplicate_frames_reported": duplicate_frames == 0,
        "no_dropped_frames_reported": dropped_frames == 0,
        "decode_completed": progress.get("progress") == "end",
        "decode_error_log_empty": decode_error_bytes == 0,
        "decode_progress_is_newer_than_video": progress_path.stat().st_mtime_ns >= video_mtime_ns,
        "decode_error_log_is_newer_than_video": errors_path.stat().st_mtime_ns >= video_mtime_ns,
    }
    checks["all_passed"] = all(checks.values())
    document = {
        "video": str(video_path),
        "sha256": sha256(video_path),
        "size_bytes": video_path.stat().st_size,
        "fps": fps,
        "container_frame_count": frames,
        "decoded_frame_count": decoded_frames,
        "duration_s": duration_s,
        "decoded_pts_duration_s": decoded_duration_s,
        "resolution": [width, height],
        "sampled_frames": sampled_frames,
        "sampled_frame_unique_pixel_hash_count": len(sampled_pixel_hashes),
        "adjacent_sampled_frame_mads": adjacent_frame_mads,
        "relative_to_first_sampled_frame_mads": relative_to_first_frame_mads,
        "maximum_sampled_frame_mad": maximum_sampled_frame_mad,
        "decode_progress": progress,
        "decode_error_bytes": decode_error_bytes,
        "expected": {
            "fps": args.expected_fps,
            "frames": args.expected_frames,
            "duration_s": expected_duration_s,
            "resolution": [args.expected_width, args.expected_height],
            "minimum_unique_sampled_pixel_hashes": 2,
            "minimum_sampled_frame_mad_exclusive": minimum_required_sampled_frame_mad,
        },
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
