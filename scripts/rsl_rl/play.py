"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import hashlib
import sys
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=0,
    help="Length of the recorded video in steps. Use 0 to record the full motion with --play_full_motion.",
)
parser.add_argument(
    "--video_folder",
    type=str,
    default=None,
    help="Optional output folder for recorded videos. Defaults to the loaded run's videos/play folder.",
)
parser.add_argument(
    "--play_full_motion",
    action="store_true",
    default=False,
    help="Start the reference motion at phase 0 and stop playback after one full trajectory.",
)
parser.add_argument(
    "--play_env_id",
    type=int,
    default=0,
    help="Environment index to monitor for --play_full_motion stopping condition.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--seed", type=int, default=42, help="Deterministic playback seed.")
parser.add_argument(
    "--keep_running",
    action="store_true",
    default=False,
    help="Prevent automatic exit after video capture or one full motion playback.",
)
parser.add_argument(
    "--playback_speed",
    type=float,
    default=1.0,
    help="Playback speed relative to real time. Examples: 1.0=real time, 0.5=half speed, 0.25=quarter speed.",
)
parser.add_argument(
    "--metrics_file",
    type=str,
    default=None,
    help="Optional CSV path for per-step playback metrics.",
)
parser.add_argument(
    "--export_folder",
    type=str,
    default=None,
    help="Optional dedicated folder for the ONNX export from this playback.",
)
parser.add_argument(
    "--show_debug_markers",
    action="store_true",
    default=False,
    help="Show command/contact debug markers. Hidden by default so the robot motion remains readable.",
)
parser.add_argument(
    "--enable_play_randomization",
    action="store_true",
    default=False,
    help="Keep training-time events and observation corruption enabled during playback.",
)
parser.add_argument(
    "--ignore_terminations",
    action="store_true",
    default=False,
    help="Disable termination conditions so a full reference timeline can be inspected after policy failure.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

# Reward terms that were recently introduced and should be logged during playbacks.
_NEW_REWARD_TERMS = ("joint_torque_l2", "joint_vel_limit", "joint_torque_limit")


class ClipToLimit(gym.ActionWrapper):
    """Clip raw actions to a fixed scalar limit before env processing."""

    def __init__(self, env, limit: float):
        super().__init__(env)
        self.limit = float(limit)

    def action(self, action):
        if isinstance(action, torch.Tensor):
            return torch.clamp(action, -self.limit, self.limit)
        return np.clip(action, -self.limit, self.limit)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_full_motion_play(vec_env: RslRlVecEnvWrapper):
    """Align the motion command with its first frame for deterministic playback."""
    base_env = getattr(vec_env, "unwrapped", vec_env)
    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None:
        return None, None
    try:
        motion_term = command_manager.get_term("motion")
    except KeyError:
        return None, None

    env_ids = torch.arange(motion_term.num_envs, device=motion_term.device, dtype=torch.long)
    motion_term.time_steps.zero_()
    motion_term.freeze_at_motion_end = True
    horizon_s = float(motion_term.motion.time_step_total) * base_env.step_dt

    # Leave one control step beyond the final motion frame. Otherwise a
    # successful full-motion playback can be mislabeled as a time-out on its
    # last frame.
    required_episode_s = horizon_s + base_env.step_dt
    # CommandTerm.compute() decrements time_left and resamples before calling
    # MotionCommand._update_command(). Keep an additional step of margin so
    # floating-point accumulation cannot resample on the final playback step.
    motion_term.time_left[env_ids] = required_episode_s + base_env.step_dt
    if required_episode_s > base_env.cfg.episode_length_s:
        base_env.cfg.episode_length_s = required_episode_s
        if hasattr(base_env, "episode_length_buf"):
            base_env.episode_length_buf.zero_()

    joint_pos = motion_term.joint_pos.clone()
    joint_vel = motion_term.joint_vel.clone()
    motion_term.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)

    root_state = torch.cat(
        [
            motion_term.body_pos_w[:, 0],
            motion_term.body_quat_w[:, 0],
            motion_term.body_lin_vel_w[:, 0],
            motion_term.body_ang_vel_w[:, 0],
        ],
        dim=-1,
    )
    motion_term.robot.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()
    base_env.scene.update(dt=base_env.physics_dt)
    motion_term.refresh_reference_alignment()
    return motion_term, int(motion_term.motion.time_step_total)


def _log_new_reward_terms(vec_env: RslRlVecEnvWrapper, env_idx: int = 0):
    """Prints the contribution of the newly added reward terms for a representative environment."""
    reward_manager = getattr(vec_env.unwrapped, "reward_manager", None)
    if reward_manager is None:
        return

    log_values = []
    for name, values in reward_manager.get_active_iterable_terms(env_idx=env_idx):
        if name in _NEW_REWARD_TERMS and len(values) > 0:
            log_values.append(f"{name}: {values[0]:.4f}")

    # if log_values:
    #     print(f"[REWARD] env {env_idx} | " + ", ".join(log_values))


def _disable_manager_terms(cfg: object) -> None:
    """Disable every public term while keeping a valid manager config object."""
    if cfg is None:
        return
    for name in vars(cfg):
        if not name.startswith("_"):
            setattr(cfg, name, None)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    if args_cli.playback_speed <= 0:
        raise ValueError("--playback_speed 必须大于0")

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if args_cli.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # Playback is an evaluation path: use nominal physics and deterministic observations
    # unless the caller explicitly asks to reproduce training-time randomization.
    if not args_cli.enable_play_randomization:
        env_cfg.observations.policy.enable_corruption = False
        _disable_manager_terms(env_cfg.events)
    if not args_cli.show_debug_markers:
        env_cfg.commands.motion.debug_vis = False
        env_cfg.scene.contact_forces.debug_vis = False
    if args_cli.ignore_terminations:
        _disable_manager_terms(env_cfg.terminations)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    motion_for_video = env.unwrapped.command_manager.get_term("motion")
    effective_video_length = args_cli.video_length
    if effective_video_length <= 0:
        effective_video_length = (
            int(motion_for_video.motion.time_step_total) if args_cli.play_full_motion else 200
        )

    # wrap for video recording
    if args_cli.video:
        # Gymnasium's OrderEnforcing wrapper rejects render() before the first
        # reset.  This reset only initializes the episode/camera; full-motion
        # playback is aligned to frame 0 again by _prepare_full_motion_play().
        env.reset()
        # Isaac's first rgb_array render may only initialize the annotator and
        # return an all-black frame. Warm the camera without advancing physics
        # so frame 0 of the recorded policy trajectory is a real image.
        camera_warmed = False
        for _ in range(3):
            warmup_frame = env.render()
            if isinstance(warmup_frame, list) and warmup_frame:
                warmup_frame = warmup_frame[-1]
            if (
                isinstance(warmup_frame, np.ndarray)
                and warmup_frame.size > 0
                and bool(np.any(warmup_frame))
            ):
                camera_warmed = True
                break
        if not camera_warmed:
            raise RuntimeError("Camera warm-up did not produce a non-black RGB frame.")
        simulation_fps = 1.0 / float(env.unwrapped.step_dt)
        video_fps = max(1, round(simulation_fps * args_cli.playback_speed))
        video_kwargs = {
            "video_folder": (
                os.path.abspath(args_cli.video_folder)
                if args_cli.video_folder is not None
                else os.path.join(log_dir, "videos", "play")
            ),
            "step_trigger": lambda step: step == 0,
            "video_length": effective_video_length,
            "fps": video_fps,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # clip actions to keep inference consistent with training limits from PPO cfg
    env = ClipToLimit(env, limit=getattr(agent_cfg, "clip_action", np.inf))

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    motion_term = None
    motion_max_steps = None
    prev_motion_step = None
    play_env_id = args_cli.play_env_id
    if args_cli.play_full_motion:
        motion_term, motion_max_steps = _prepare_full_motion_play(env)
        if motion_term is not None:
            play_env_id = max(0, min(play_env_id, motion_term.num_envs - 1))
            prev_motion_step = motion_term.time_steps[play_env_id].item()

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    # resolve observation normalizer from actor-critic (rsl_rl >= 2.0)
    actor_obs_normalizer = getattr(ppo_runner, "obs_normalizer", None)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = (
        os.path.abspath(args_cli.export_folder)
        if args_cli.export_folder is not None
        else os.path.join(os.path.dirname(resume_path), "exported")
    )

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=actor_obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    motion_path = pathlib.Path(env.unwrapped.command_manager.get_term("motion").cfg.motion_file).resolve()
    checkpoint_path = pathlib.Path(resume_path).resolve()
    attach_onnx_metadata(
        env.unwrapped,
        str(checkpoint_path),
        export_model_dir,
        extra_metadata={
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "motion_file": str(motion_path),
            "motion_sha256": _sha256(motion_path),
            "play_randomization_enabled": args_cli.enable_play_randomization,
        },
    )
    # reset environment
    obs = env.get_observations()
    # Isaac Lab 5.0 compatibility:
    # some versions return (observations, extras)
    if isinstance(obs, tuple):
        obs = obs[0]
    timestep = 0
    metrics_rows = []

    simulation_step_dt = float(env.unwrapped.step_dt)
    target_wall_step_dt = simulation_step_dt / args_cli.playback_speed

    print(
        f"[INFO] Simulation step dt: {simulation_step_dt:.4f} s | "
        f"Playback speed: {args_cli.playback_speed:.2f}x | "
        f"Target wall time per step: {target_wall_step_dt:.4f} s"
    )
    # simulate environment
    while simulation_app.is_running():
        loop_start_time = time.perf_counter()
        metric_motion_step = None
        if motion_term is not None:
            metric_motion_step = int(motion_term.time_steps[play_env_id].item())
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, _ = env.step(actions)
        _log_new_reward_terms(env)

        if args_cli.metrics_file is not None:
            base_env = env.unwrapped
            metric_term = motion_term or base_env.command_manager.get_term("motion")
            metric_env_id = max(0, min(play_env_id, metric_term.num_envs - 1))
            row = {
                "playback_step": timestep,
                "simulation_time_s": timestep * simulation_step_dt,
                "motion_frame": (
                    metric_motion_step
                    if metric_motion_step is not None
                    else int(metric_term.time_steps[metric_env_id].item())
                ),
                "reward": float(rewards[metric_env_id].item()),
                "done": int(dones[metric_env_id].item()),
            }
            for metric_name, values in metric_term.metrics.items():
                row[metric_name] = float(values[metric_env_id].item())
            for term_name, values in base_env.termination_manager.get_active_iterable_terms(metric_env_id):
                row[f"termination_{term_name}"] = int(values[0])
            root_xy = (
                metric_term.robot.data.root_pos_w[metric_env_id, :2]
                - base_env.scene.env_origins[metric_env_id, :2]
            )
            row["root_x_from_env_origin"] = float(root_xy[0].item())
            row["root_y_from_env_origin"] = float(root_xy[1].item())
            metrics_rows.append(row)

        # Synchronize simulation time with wall-clock playback speed.
        elapsed_wall_time = time.perf_counter() - loop_start_time
        remaining_wall_time = target_wall_step_dt - elapsed_wall_time
        # Headless video speed is controlled by the encoded FPS, so wall-clock
        # throttling is only useful for an interactive viewer.
        if not args_cli.headless and remaining_wall_time > 0:
            time.sleep(remaining_wall_time)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep >= effective_video_length and not args_cli.keep_running:
                break
        if args_cli.play_full_motion and motion_term is not None:
            if bool(dones[play_env_id].item()) and not args_cli.keep_running:
                break
            # ``time_steps`` advances after reward/metric computation in
            # ManagerBasedRLEnv.step(). Stopping when it first reaches the
            # last frame would skip executing and logging that frame.
            if motion_max_steps is not None and timestep >= motion_max_steps and not args_cli.keep_running:
                break
            current_step = motion_term.time_steps[play_env_id].item()
            if current_step < prev_motion_step and not args_cli.keep_running:
                break
            prev_motion_step = current_step

    if args_cli.metrics_file is not None and metrics_rows:
        metrics_path = pathlib.Path(args_cli.metrics_file).expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"[INFO] Playback metrics saved to: {metrics_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
