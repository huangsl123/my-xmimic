#!/usr/bin/env python3
"""Validate policy provenance, weights, motion constants, metadata, and ONNX inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as torch_functional
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator


MOTION_OUTPUTS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected_iteration", type=int, required=True)
    parser.add_argument("--expected_obs", type=int, default=124)
    parser.add_argument(
        "--expected_observation_profile",
        choices=("auto", "wo-state", "state-feedback"),
        default="auto",
        help="Expected ordered policy observation terms. Auto maps 124/130 inputs to the known profiles.",
    )
    parser.add_argument("--expected_actions", type=int, default=23)
    parser.add_argument("--expected_bodies", type=int, default=24)
    parser.add_argument("--expected_action_scale", type=float, default=0.25)
    parser.add_argument(
        "--expected_randomization",
        choices=("true", "false"),
        default="false",
        help="Expected play_randomization_enabled metadata (default: false).",
    )
    return parser.parse_args()


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    return [
        dimension.dim_value or dimension.dim_param or None
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def value_info(value: onnx.ValueInfoProto) -> dict[str, object]:
    tensor_type = value.type.tensor_type
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": tensor_shape(value),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def finite_csv_numbers(value: str, expected_count: int) -> bool:
    try:
        values = np.asarray([float(item) for item in csv_values(value)], dtype=np.float64)
    except ValueError:
        return False
    return len(values) == expected_count and bool(np.isfinite(values).all())


def constant_tensor_audit(model: onnx.ModelProto) -> tuple[list[dict[str, object]], bool]:
    summaries: list[dict[str, object]] = []
    all_finite = True
    for node_index, node in enumerate(model.graph.node):
        for attribute in node.attribute:
            tensors = []
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensors = [attribute.t]
            elif attribute.type == onnx.AttributeProto.TENSORS:
                tensors = list(attribute.tensors)
            for tensor_index, tensor in enumerate(tensors):
                values = numpy_helper.to_array(tensor)
                finite = bool(np.isfinite(values).all())
                all_finite &= finite
                summaries.append(
                    {
                        "node_index": node_index,
                        "node_name": node.name,
                        "node_op_type": node.op_type,
                        "attribute": attribute.name,
                        "tensor_index": tensor_index,
                        "shape": list(values.shape),
                        "dtype": str(values.dtype),
                        "all_finite": finite,
                    }
                )
    return summaries, all_finite


def motion_constant_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    constants: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        tensor_attributes = [
            attribute
            for attribute in node.attribute
            if attribute.type == onnx.AttributeProto.TENSOR
        ]
        if len(tensor_attributes) == 1:
            constants[node.output[0]] = numpy_helper.to_array(tensor_attributes[0].t)

    output: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if (
            node.op_type == "Gather"
            and len(node.input) >= 1
            and len(node.output) == 1
            and node.output[0] in MOTION_OUTPUTS
            and node.input[0] in constants
        ):
            output[node.output[0]] = constants[node.input[0]]
    return output


def array_match(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-6,
) -> tuple[bool, float | None]:
    actual = np.asarray(actual)
    expected = np.asarray(expected, dtype=actual.dtype)
    if actual.shape != expected.shape:
        return False, None
    difference = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    return bool(np.allclose(actual, expected, rtol=rtol, atol=atol)), difference


def checkpoint_actor_forward(
    observation: np.ndarray,
    model_state: dict[str, torch.Tensor],
    normalizer_state: dict[str, torch.Tensor],
) -> np.ndarray:
    value = torch.from_numpy(observation)
    value = (value - normalizer_state["_mean"]) / (normalizer_state["_std"] + 1.0e-2)
    for layer_index in (0, 2, 4):
        value = torch_functional.linear(
            value,
            model_state[f"actor.{layer_index}.weight"],
            model_state[f"actor.{layer_index}.bias"],
        )
        value = torch_functional.elu(value)
    value = torch_functional.linear(
        value,
        model_state["actor.6.weight"],
        model_state["actor.6.bias"],
    )
    return value.detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    motion_path = args.motion_file.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    model_hash = sha256(model_path)
    checkpoint_hash = sha256(checkpoint_path)
    motion_hash = sha256(motion_path)

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    initializer_arrays = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    initializers = [
        {
            "name": name,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "all_finite": bool(np.isfinite(values).all()),
        }
        for name, values in initializer_arrays.items()
    ]
    constant_tensors, all_constant_tensors_finite = constant_tensor_audit(model)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_iteration = int(checkpoint["iter"])
    actor_state = {
        name: value.detach().cpu().numpy()
        for name, value in checkpoint["model_state_dict"].items()
        if name.startswith("actor.")
    }
    actor_differences: dict[str, float | None] = {}
    actor_weights_match = True
    for name, expected in actor_state.items():
        actual = initializer_arrays.get(name)
        if actual is None:
            actor_differences[name] = None
            actor_weights_match = False
            continue
        matches, difference = array_match(actual, expected)
        actor_differences[name] = difference
        actor_weights_match &= matches

    obs_normalizer = checkpoint["obs_norm_state_dict"]
    expected_mean = obs_normalizer["_mean"].detach().cpu().numpy()
    expected_divisor = obs_normalizer["_std"].detach().cpu().numpy() + 1.0e-2
    normalizer_mean = initializer_arrays.get("normalizer._mean")
    normalizer_mean_matches = (
        normalizer_mean is not None and array_match(normalizer_mean, expected_mean)[0]
    )
    div_node_inputs = {
        input_name
        for node in model.graph.node
        if node.op_type == "Div"
        for input_name in node.input
    }
    divisor_matches = [
        name
        for name, values in initializer_arrays.items()
        if name != "normalizer._mean"
        and name in div_node_inputs
        and list(values.shape) == [1, args.expected_obs]
        and array_match(values, expected_divisor)[0]
        and bool((values > 0.0).all())
    ]

    input_info = [value_info(value) for value in model.graph.input]
    output_info = [value_info(value) for value in model.graph.output]
    actual_inputs = {item["name"]: item["shape"] for item in input_info}
    actual_outputs = {item["name"]: item["shape"] for item in output_info}
    expected_inputs = {
        "obs": [1, args.expected_obs],
        "time_step": [1, 1],
    }
    expected_outputs = {
        "actions": [1, args.expected_actions],
        "joint_pos": [1, args.expected_actions],
        "joint_vel": [1, args.expected_actions],
        "body_pos_w": [1, args.expected_bodies, 3],
        "body_quat_w": [1, args.expected_bodies, 4],
        "body_lin_vel_w": [1, args.expected_bodies, 3],
        "body_ang_vel_w": [1, args.expected_bodies, 3],
    }

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    required_metadata = {
        "run_path",
        "checkpoint_path",
        "checkpoint_sha256",
        "motion_file",
        "motion_sha256",
        "joint_names",
        "joint_stiffness",
        "joint_damping",
        "default_joint_pos",
        "command_names",
        "observation_names",
        "action_scale",
        "body_names",
        "motion_fps",
        "motion_frame_count",
        "play_randomization_enabled",
    }
    required_metadata_nonempty = all(metadata.get(name, "").strip() for name in required_metadata)
    joint_metadata_counts_match = all(
        len(csv_values(metadata.get(name, ""))) == args.expected_actions
        for name in (
            "joint_names",
            "joint_stiffness",
            "joint_damping",
            "default_joint_pos",
            "action_scale",
        )
    )
    joint_names = csv_values(metadata.get("joint_names", ""))
    joint_names_unique = len(joint_names) == len(set(joint_names)) == args.expected_actions
    joint_numeric_metadata_finite = all(
        finite_csv_numbers(metadata.get(name, ""), args.expected_actions)
        for name in (
            "joint_stiffness",
            "joint_damping",
            "default_joint_pos",
            "action_scale",
        )
    )
    try:
        action_scales = np.asarray(
            [float(item) for item in csv_values(metadata.get("action_scale", ""))],
            dtype=np.float64,
        )
    except ValueError:
        action_scales = np.asarray([], dtype=np.float64)
    action_scale_matches_expected = (
        len(action_scales) == args.expected_actions
        and bool(
            np.allclose(
                action_scales,
                args.expected_action_scale,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
    )
    tracked_body_names = csv_values(metadata.get("body_names", ""))
    body_metadata_count_matches = len(tracked_body_names) == args.expected_bodies
    body_names_unique = (
        len(tracked_body_names) == len(set(tracked_body_names)) == args.expected_bodies
    )

    with np.load(motion_path) as motion:
        motion_frames = int(motion["joint_pos"].shape[0])
        motion_fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        motion_numeric_finite = all(
            bool(np.isfinite(motion[name]).all())
            for name in motion.files
            if np.issubdtype(motion[name].dtype, np.number)
        )
        source_body_names = [str(name) for name in np.asarray(motion["body_names"]).tolist()]
        name_to_index = {name: index for index, name in enumerate(source_body_names)}
        tracked_names_resolve = (
            len(tracked_body_names) == args.expected_bodies
            and all(name in name_to_index for name in tracked_body_names)
        )
        body_indexes = [name_to_index[name] for name in tracked_body_names] if tracked_names_resolve else []
        reference_arrays = {
            "joint_pos": np.asarray(motion["joint_pos"], dtype=np.float32),
            "joint_vel": np.asarray(motion["joint_vel"], dtype=np.float32),
            "body_pos_w": np.asarray(motion["body_pos_w"][:, body_indexes], dtype=np.float32),
            "body_quat_w": np.asarray(motion["body_quat_w"][:, body_indexes], dtype=np.float32),
            "body_lin_vel_w": np.asarray(motion["body_lin_vel_w"][:, body_indexes], dtype=np.float32),
            "body_ang_vel_w": np.asarray(motion["body_ang_vel_w"][:, body_indexes], dtype=np.float32),
        } if tracked_names_resolve else {}

    embedded_motion_arrays = motion_constant_arrays(model)
    embedded_motion_matches = (
        tracked_names_resolve and set(embedded_motion_arrays) == set(MOTION_OUTPUTS)
    )
    embedded_motion_differences: dict[str, float | None] = {}
    if embedded_motion_matches:
        for name in MOTION_OUTPUTS:
            matches, difference = array_match(embedded_motion_arrays[name], reference_arrays[name])
            embedded_motion_differences[name] = difference
            embedded_motion_matches &= matches
    quaternion_norm_max_error = (
        float(np.max(np.abs(np.linalg.norm(reference_arrays["body_quat_w"], axis=-1) - 1.0)))
        if tracked_names_resolve
        else None
    )

    evaluator = ReferenceEvaluator(model)
    rng = np.random.default_rng(42)
    observation = expected_mean + obs_normalizer["_std"].detach().cpu().numpy() * rng.standard_normal(
        (1, args.expected_obs), dtype=np.float32
    )
    last_frame = motion_frames - 1
    requested_frames = (-1, 0, min(1234, last_frame), last_frame, motion_frames)
    frame_checks: list[dict[str, object]] = []
    reference_inference_all_finite = True
    motion_outputs_match = tracked_names_resolve
    for requested_frame in requested_frames:
        outputs = evaluator.run(
            None,
            {
                "obs": observation,
                "time_step": np.asarray([[requested_frame]], dtype=np.float32),
            },
        )
        expected_frame = min(max(requested_frame, 0), last_frame)
        output_by_name = {
            info.name: values for info, values in zip(model.graph.output, outputs)
        }
        output_finite = all(bool(np.isfinite(values).all()) for values in outputs)
        reference_inference_all_finite &= output_finite
        output_matches: dict[str, bool] = {}
        maximum_differences: dict[str, float | None] = {}
        if tracked_names_resolve:
            for name in MOTION_OUTPUTS:
                matches, difference = array_match(
                    output_by_name[name],
                    reference_arrays[name][expected_frame : expected_frame + 1],
                )
                output_matches[name] = matches
                maximum_differences[name] = difference
                motion_outputs_match &= matches
        frame_checks.append(
            {
                "requested_frame": requested_frame,
                "expected_clamped_frame": expected_frame,
                "all_outputs_finite": output_finite,
                "motion_outputs_match": output_matches,
                "maximum_absolute_differences": maximum_differences,
            }
        )

    actor_parity_cases: list[dict[str, object]] = []
    actor_output_parity = True
    parity_observations = (
        np.zeros((1, args.expected_obs), dtype=np.float32),
        observation,
        expected_mean
        + obs_normalizer["_std"].detach().cpu().numpy()
        * np.random.default_rng(43).standard_normal(
            (1, args.expected_obs), dtype=np.float32
        ),
    )
    for case_index, parity_observation in enumerate(parity_observations):
        onnx_outputs = evaluator.run(
            None,
            {
                "obs": parity_observation,
                "time_step": np.asarray([[0.0]], dtype=np.float32),
            },
        )
        onnx_actions = {
            info.name: values for info, values in zip(model.graph.output, onnx_outputs)
        }["actions"]
        checkpoint_actions = checkpoint_actor_forward(
            parity_observation,
            checkpoint["model_state_dict"],
            checkpoint["obs_norm_state_dict"],
        )
        matches, difference = array_match(
            onnx_actions,
            checkpoint_actions,
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        actor_output_parity &= matches
        actor_parity_cases.append(
            {
                "case_index": case_index,
                "input_kind": "zeros" if case_index == 0 else f"random_seed_{41 + case_index}",
                "all_finite": bool(
                    np.isfinite(onnx_actions).all() and np.isfinite(checkpoint_actions).all()
                ),
                "matches": matches,
                "maximum_absolute_difference": difference,
            }
        )

    expected_randomization = args.expected_randomization == "true"
    metadata_randomization_raw = metadata.get("play_randomization_enabled", "").lower()
    metadata_randomization = metadata_randomization_raw == "true"
    all_tensor_dtypes_float = all(item["dtype"] == "FLOAT" for item in input_info + output_info)
    opset_11 = (
        len(model.opset_import) == 1
        and model.opset_import[0].domain == ""
        and model.opset_import[0].version == 11
    )
    observation_profiles = {
        "wo-state": [
            "command",
            "motion_anchor_ori_b",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ],
        "state-feedback": [
            "command",
            "motion_anchor_pos_b",
            "motion_anchor_ori_b",
            "base_lin_vel",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ],
    }
    observation_profile = args.expected_observation_profile
    if observation_profile == "auto":
        observation_profile = {124: "wo-state", 130: "state-feedback"}.get(args.expected_obs)
        if observation_profile is None:
            raise ValueError(
                "Cannot infer the observation profile for "
                f"--expected_obs={args.expected_obs}; pass --expected_observation_profile explicitly."
            )
    expected_observation_names = observation_profiles[observation_profile]
    checks = {
        "onnx_checker_passed": True,
        "ir_version_is_6": model.ir_version == 6,
        "opset_is_11": opset_11,
        "inputs_match_expected": actual_inputs == expected_inputs,
        "outputs_match_expected": actual_outputs == expected_outputs,
        "input_output_dtypes_are_float": all_tensor_dtypes_float,
        "all_initializers_finite": all(item["all_finite"] for item in initializers),
        "all_constant_tensors_finite": all_constant_tensors_finite,
        "checkpoint_iteration_matches": checkpoint_iteration == args.expected_iteration,
        "checkpoint_hash_metadata_matches": metadata.get("checkpoint_sha256") == checkpoint_hash,
        "checkpoint_path_metadata_matches": (
            Path(metadata.get("checkpoint_path", "")).expanduser().resolve() == checkpoint_path
        ),
        "actor_weights_match_checkpoint": actor_weights_match,
        "actor_outputs_match_checkpoint_forward": actor_output_parity,
        "normalizer_mean_matches_checkpoint": normalizer_mean_matches,
        "normalizer_divisor_matches_checkpoint": len(divisor_matches) == 1,
        "required_metadata_nonempty": required_metadata_nonempty,
        "joint_metadata_counts_match": joint_metadata_counts_match,
        "joint_names_unique": joint_names_unique,
        "joint_numeric_metadata_finite": joint_numeric_metadata_finite,
        "action_scale_matches_expected": action_scale_matches_expected,
        "body_metadata_count_matches": body_metadata_count_matches,
        "body_names_unique": body_names_unique,
        "observation_names_match_expected": (
            csv_values(metadata.get("observation_names", "")) == expected_observation_names
        ),
        "command_names_match_expected": csv_values(metadata.get("command_names", "")) == ["motion"],
        "run_path_identifies_checkpoint": metadata.get("run_path") == str(checkpoint_path),
        "motion_hash_metadata_matches": metadata.get("motion_sha256") == motion_hash,
        "motion_path_metadata_matches": (
            Path(metadata.get("motion_file", "")).expanduser().resolve() == motion_path
        ),
        "motion_frame_count_metadata_matches": (
            metadata.get("motion_frame_count") == str(motion_frames)
        ),
        "motion_fps_metadata_matches": math.isclose(
            float(metadata.get("motion_fps", "nan")),
            motion_fps,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ),
        "tracked_body_names_resolve": tracked_names_resolve,
        "motion_file_all_numeric_finite": motion_numeric_finite,
        "embedded_motion_constants_match_source": embedded_motion_matches,
        "motion_quaternions_normalized": (
            quaternion_norm_max_error is not None and quaternion_norm_max_error < 1.0e-4
        ),
        "motion_outputs_match_source_at_boundaries": motion_outputs_match,
        "reference_inference_all_finite": reference_inference_all_finite,
        "randomization_metadata_is_boolean": metadata_randomization_raw in {"true", "false"},
        "randomization_metadata_matches": metadata_randomization == expected_randomization,
    }
    checks["all_passed"] = all(checks.values())

    document = {
        "model": str(model_path),
        "model_sha256": model_hash,
        "model_size_bytes": model_path.stat().st_size,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_iteration": checkpoint_iteration,
        "motion_file": str(motion_path),
        "motion_sha256": motion_hash,
        "motion_frames": motion_frames,
        "motion_fps": motion_fps,
        "expected_observation_profile": observation_profile,
        "expected_observation_names": expected_observation_names,
        "ir_version": model.ir_version,
        "opsets": [
            {"domain": entry.domain, "version": entry.version}
            for entry in model.opset_import
        ],
        "inputs": input_info,
        "outputs": output_info,
        "initializer_count": len(initializers),
        "initializers": initializers,
        "constant_tensor_count": len(constant_tensors),
        "constant_tensors": constant_tensors,
        "embedded_motion_maximum_absolute_differences": embedded_motion_differences,
        "motion_quaternion_norm_max_error": quaternion_norm_max_error,
        "metadata": metadata,
        "actor_maximum_absolute_differences": actor_differences,
        "normalizer_divisor_initializer": divisor_matches,
        "frame_checks": frame_checks,
        "actor_parity_cases": actor_parity_cases,
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
