#!/usr/bin/env python3
"""Create an inference checkpoint by interpolating two compatible RSL-RL policies."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_a", type=Path, required=True)
    parser.add_argument("--checkpoint_b", type=Path, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Interpolation factor: output=(1-alpha)*A + alpha*B.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_compatible_tensors(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    section: str,
) -> None:
    if state_a.keys() != state_b.keys():
        raise ValueError(f"{section} keys differ between checkpoints")
    for name in state_a:
        tensor_a = state_a[name]
        tensor_b = state_b[name]
        if not isinstance(tensor_a, torch.Tensor) or not isinstance(tensor_b, torch.Tensor):
            raise TypeError(f"{section}.{name} is not a tensor in both checkpoints")
        if tensor_a.shape != tensor_b.shape or tensor_a.dtype != tensor_b.dtype:
            raise ValueError(
                f"{section}.{name} differs: "
                f"{tensor_a.shape}/{tensor_a.dtype} vs {tensor_b.shape}/{tensor_b.dtype}"
            )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    path_a = args.checkpoint_a.expanduser().resolve()
    path_b = args.checkpoint_b.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    checkpoint_a = torch.load(path_a, map_location="cpu", weights_only=False)
    checkpoint_b = torch.load(path_b, map_location="cpu", weights_only=False)

    state_a = checkpoint_a["model_state_dict"]
    state_b = checkpoint_b["model_state_dict"]
    assert_compatible_tensors(state_a, state_b, "model_state_dict")

    output = copy.deepcopy(checkpoint_b)
    interpolated_state = {}
    for name in state_a:
        tensor_a = state_a[name]
        tensor_b = state_b[name]
        if tensor_a.is_floating_point():
            interpolated_state[name] = torch.lerp(tensor_a, tensor_b, args.alpha)
        elif torch.equal(tensor_a, tensor_b):
            interpolated_state[name] = tensor_b.clone()
        else:
            raise ValueError(f"Non-floating tensor differs: model_state_dict.{name}")
    output["model_state_dict"] = interpolated_state

    # Observation normalizers must be identical. Interpolating a network across
    # different input normalizations would not describe one continuous policy
    # path and could hide a provenance error.
    for section in ("obs_norm_state_dict", "privileged_obs_norm_state_dict"):
        normalizer_a = checkpoint_a[section]
        normalizer_b = checkpoint_b[section]
        assert_compatible_tensors(normalizer_a, normalizer_b, section)
        for name in normalizer_a:
            if not torch.equal(normalizer_a[name], normalizer_b[name]):
                raise ValueError(f"Normalizer differs: {section}.{name}")

    output["interpolation_info"] = {
        "formula": "(1-alpha)*checkpoint_a + alpha*checkpoint_b",
        "alpha": args.alpha,
        "checkpoint_a": str(path_a),
        "checkpoint_a_sha256": sha256(path_a),
        "checkpoint_a_iteration": int(checkpoint_a["iter"]),
        "checkpoint_b": str(path_b),
        "checkpoint_b_sha256": sha256(path_b),
        "checkpoint_b_iteration": int(checkpoint_b["iter"]),
        "optimizer_state_source": "checkpoint_b",
        "resume_training_recommended": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)

    summary = {
        **output["interpolation_info"],
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "model_tensor_count": len(interpolated_state),
        "all_model_tensors_finite": all(
            bool(torch.isfinite(tensor).all())
            for tensor in interpolated_state.values()
            if tensor.is_floating_point()
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
