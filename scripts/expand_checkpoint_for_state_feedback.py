#!/usr/bin/env python3
"""Expand a 124-D Wo-State checkpoint into the 130-D state-feedback layout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as functional


OLD_TERMS = (
    ("command", 46),
    ("motion_anchor_ori_b", 6),
    ("base_ang_vel", 3),
    ("joint_pos", 23),
    ("joint_vel", 23),
    ("actions", 23),
)
NEW_TERMS = (
    ("command", 46),
    ("motion_anchor_pos_b", 3),
    ("motion_anchor_ori_b", 6),
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("joint_pos", 23),
    ("joint_vel", 23),
    ("actions", 23),
)
ACTOR_FIRST_WEIGHT = "actor.0.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def term_slices(terms: tuple[tuple[str, int], ...]) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    start = 0
    for name, width in terms:
        slices[name] = slice(start, start + width)
        start += width
    return slices


def expand_last_dimension(
    value: torch.Tensor,
    old_slices: dict[str, slice],
    new_slices: dict[str, slice],
    *,
    fill: float,
) -> torch.Tensor:
    shape = list(value.shape)
    shape[-1] = sum(width for _, width in NEW_TERMS)
    expanded = torch.full(shape, fill, dtype=value.dtype, device=value.device)
    for name, old_slice in old_slices.items():
        expanded[..., new_slices[name]] = value[..., old_slice]
    return expanded


def actor_forward(
    raw_observation: torch.Tensor,
    model: dict[str, torch.Tensor],
    normalizer: dict[str, torch.Tensor],
) -> torch.Tensor:
    value = (raw_observation - normalizer["_mean"]) / (normalizer["_std"] + 0.01)
    for layer_index in (0, 2, 4):
        value = functional.elu(
            functional.linear(
                value,
                model[f"actor.{layer_index}.weight"],
                model[f"actor.{layer_index}.bias"],
            )
        )
    return functional.linear(value, model["actor.6.weight"], model["actor.6.bias"])


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    summary = args.summary.expanduser().resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    expanded = copy.deepcopy(checkpoint)

    old_slices = term_slices(OLD_TERMS)
    new_slices = term_slices(NEW_TERMS)
    old_size = sum(width for _, width in OLD_TERMS)
    new_size = sum(width for _, width in NEW_TERMS)
    if old_size != 124 or new_size != 130:
        raise AssertionError(f"Unexpected layout sizes: {old_size} -> {new_size}")

    old_model = checkpoint["model_state_dict"]
    old_first_weight = old_model[ACTOR_FIRST_WEIGHT]
    if tuple(old_first_weight.shape) != (512, old_size):
        raise ValueError(
            f"Expected {ACTOR_FIRST_WEIGHT} shape (512, {old_size}), "
            f"received {tuple(old_first_weight.shape)}"
        )
    expanded["model_state_dict"][ACTOR_FIRST_WEIGHT] = expand_last_dimension(
        old_first_weight,
        old_slices,
        new_slices,
        fill=0.0,
    )

    old_normalizer = checkpoint["obs_norm_state_dict"]
    new_normalizer = expanded["obs_norm_state_dict"]
    for name, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        if tuple(old_normalizer[name].shape) != (1, old_size):
            raise ValueError(
                f"Expected actor normalizer {name} shape (1, {old_size}), "
                f"received {tuple(old_normalizer[name].shape)}"
            )
        new_normalizer[name] = expand_last_dimension(
            old_normalizer[name],
            old_slices,
            new_slices,
            fill=fill,
        )

    expanded_optimizer_tensors = 0
    for state in expanded["optimizer_state_dict"]["state"].values():
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(old_first_weight.shape):
                state[name] = expand_last_dimension(
                    value,
                    old_slices,
                    new_slices,
                    fill=0.0,
                )
                expanded_optimizer_tensors += 1
    if expanded_optimizer_tensors != 2:
        raise ValueError(
            "Expected to expand the Adam first/second moments for one actor input layer; "
            f"expanded {expanded_optimizer_tensors} tensors"
        )

    infos = dict(expanded.get("infos") or {})
    infos["state_feedback_expansion"] = {
        "source_checkpoint": str(source),
        "old_observation_size": old_size,
        "new_observation_size": new_size,
        "inserted_terms": ["motion_anchor_pos_b", "base_lin_vel"],
        "inserted_actor_weights_initialized_to_zero": True,
        "optimizer_moments_expanded": True,
    }
    expanded["infos"] = infos

    generator = torch.Generator().manual_seed(42)
    old_raw = old_normalizer["_mean"] + old_normalizer["_std"] * torch.randn(
        (8, old_size), generator=generator
    )
    new_raw = torch.randn((8, new_size), generator=generator)
    for name, old_slice in old_slices.items():
        new_raw[:, new_slices[name]] = old_raw[:, old_slice]
    old_actions = actor_forward(old_raw, old_model, old_normalizer)
    new_actions = actor_forward(
        new_raw,
        expanded["model_state_dict"],
        expanded["obs_norm_state_dict"],
    )
    maximum_action_difference = float(torch.max(torch.abs(old_actions - new_actions)).item())
    if maximum_action_difference > 1.0e-6:
        raise ValueError(
            "Expanded actor does not preserve the source policy: "
            f"maximum action difference {maximum_action_difference:.3e}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(expanded, output)
    document = {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "checkpoint_iteration": int(expanded["iter"]),
        "old_observation_size": old_size,
        "new_observation_size": new_size,
        "old_terms": [{"name": name, "width": width} for name, width in OLD_TERMS],
        "new_terms": [{"name": name, "width": width} for name, width in NEW_TERMS],
        "inserted_terms": ["motion_anchor_pos_b", "base_lin_vel"],
        "new_actor_columns_are_zero": bool(
            torch.count_nonzero(
                expanded["model_state_dict"][ACTOR_FIRST_WEIGHT][
                    :,
                    list(range(new_slices["motion_anchor_pos_b"].start, new_slices["motion_anchor_pos_b"].stop))
                    + list(range(new_slices["base_lin_vel"].start, new_slices["base_lin_vel"].stop)),
                ]
            ).item()
            == 0
        ),
        "maximum_initial_action_difference": maximum_action_difference,
        "optimizer_moment_tensors_expanded": expanded_optimizer_tensors,
        "actor_normalizer_count": int(expanded["obs_norm_state_dict"]["count"].item()),
        "checks": {
            "checkpoint_iteration_preserved": int(expanded["iter"]) == int(checkpoint["iter"]),
            "actor_output_preserved": maximum_action_difference <= 1.0e-6,
            "actor_input_shape_is_130": tuple(
                expanded["model_state_dict"][ACTOR_FIRST_WEIGHT].shape
            )
            == (512, new_size),
            "actor_normalizer_shape_is_130": tuple(
                expanded["obs_norm_state_dict"]["_mean"].shape
            )
            == (1, new_size),
            "critic_shape_unchanged": tuple(
                expanded["model_state_dict"]["critic.0.weight"].shape
            )
            == tuple(checkpoint["model_state_dict"]["critic.0.weight"].shape),
            "optimizer_moments_expanded": expanded_optimizer_tensors == 2,
        },
    }
    document["checks"]["all_passed"] = all(document["checks"].values())
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, ensure_ascii=False, indent=2))
    if not document["checks"]["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
