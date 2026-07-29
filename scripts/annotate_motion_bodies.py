"""Add explicit body-name metadata to legacy DexEVT motion files.

Older xMimic motion files stored body tensors without their body names. Isaac Sim
versions may merge fixed URDF links differently, so numeric body indices are not
portable. This utility creates a named copy that the tracking loader can map safely.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# Body order used by the 39-body DexEVT motion files shipped with this repository.
# The order is confirmed by the URDF tree and first-frame forward-kinematics data.
DEX_EVT_LEGACY_39_BODY_NAMES = (
    "pelvis",
    "hip_pitch_l_link",
    "hip_pitch_r_link",
    "imu_waist_link",
    "waist_yaw_link",
    "hip_roll_l_link",
    "hip_roll_r_link",
    "waist_roll_link",
    "hip_yaw_l_link",
    "hip_yaw_r_link",
    "waist_pitch_link",
    "knee_pitch_l_link",
    "knee_pitch_r_link",
    "camera_body_front_link",
    "head_yaw_link",
    "imu_head_link",
    "radar_head_link",
    "shoulder_pitch_l_link",
    "shoulder_pitch_r_link",
    "ankle_pitch_l_link",
    "ankle_pitch_r_link",
    "head_pitch_link",
    "shoulder_roll_l_link",
    "shoulder_roll_r_link",
    "ankle_roll_l_link",
    "ankle_roll_r_link",
    "camera_head_link",
    "shoulder_yaw_l_link",
    "shoulder_yaw_r_link",
    "elbow_pitch_l_link",
    "elbow_pitch_r_link",
    "elbow_yaw_l_link",
    "elbow_yaw_r_link",
    "wrist_pitch_l_link",
    "wrist_pitch_r_link",
    "wrist_roll_l_link",
    "wrist_roll_r_link",
    "left_tcp_link",
    "right_tcp_link",
)

BODY_ARRAY_KEYS = (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True, help="Legacy 39-body NPZ motion.")
    parser.add_argument("--output_file", type=Path, required=True, help="Output NPZ with body_names metadata.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_file.expanduser().resolve()
    output_path = args.output_file.expanduser().resolve()

    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --force to replace it.")

    with np.load(input_path, allow_pickle=False) as source:
        missing_keys = [key for key in BODY_ARRAY_KEYS if key not in source]
        if missing_keys:
            raise KeyError(f"Motion file is missing body arrays: {missing_keys}")

        body_counts = {key: int(source[key].shape[1]) for key in BODY_ARRAY_KEYS}
        if set(body_counts.values()) != {len(DEX_EVT_LEGACY_39_BODY_NAMES)}:
            raise ValueError(
                f"Expected 39 bodies for the legacy DexEVT layout, received {body_counts}."
            )

        output = {key: source[key].copy() for key in source.files if key != "body_names"}

    output["body_names"] = np.asarray(DEX_EVT_LEGACY_39_BODY_NAMES)
    output["body_layout"] = np.asarray("dex_evt_legacy_39_named")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)

    frames = int(output["joint_pos"].shape[0])
    fps = float(np.asarray(output["fps"]).reshape(-1)[0])
    print(f"[INFO] Saved named motion: {output_path}")
    print(f"[INFO] Frames: {frames}, FPS: {fps:g}, bodies: {len(DEX_EVT_LEGACY_39_BODY_NAMES)}")


if __name__ == "__main__":
    main()
