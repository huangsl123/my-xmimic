import whole_body_tracking.tasks.tracking.mdp as mdp

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from whole_body_tracking.robots.dex_evt import D3_ACTION_SCALE, DEX_EVT_CFG
from whole_body_tracking.tasks.tracking.config.dex_evt.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


def _configure_long_horizon_tracking(cfg: TrackingEnvCfg) -> None:
    """Enable long contiguous segments and two-scale reference-path rewards."""
    # Stage 1 defaults to 10 s. train.py may raise this to 20 s and then
    # the exact 39.72 s motion duration in later continuation stages.
    cfg.episode_length_s = 10.0
    cfg.commands.motion.sample_phase_with_episode_horizon = True

    # Keep the original 3-D fine reward, then add a wide XY basin so the
    # policy still receives a useful gradient after long-horizon drift.
    cfg.rewards.motion_global_anchor_xy_coarse = RewTerm(
        func=mdp.motion_global_anchor_xy_position_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 1.0},
    )
    cfg.rewards.motion_global_anchor_xy_vel = RewTerm(
        func=mdp.motion_global_anchor_xy_velocity_error_exp,
        weight=0.75,
        params={"command_name": "motion", "std": 0.5},
    )


class DexEVTFlatEnvConfig(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Update the robot configuration
        self.scene.robot = DEX_EVT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = D3_ACTION_SCALE

        # Set the anchor body for motion commands
        self.commands.motion.anchor_body = "pelvis"

        # Define the body names based on the URDF structure
        self.commands.motion.body_names = [
            "pelvis",
            "hip_pitch_l_link",
            "hip_roll_l_link",
            "hip_yaw_l_link",
            "knee_pitch_l_link",
            "ankle_pitch_l_link",
            "ankle_roll_l_link",
            "hip_pitch_r_link",
            "hip_roll_r_link",
            "hip_yaw_r_link",
            "knee_pitch_r_link",
            "ankle_pitch_r_link",
            "ankle_roll_r_link",
            "waist_yaw_link",
            "waist_roll_link",
            "waist_pitch_link",
            "shoulder_pitch_l_link",
            "shoulder_roll_l_link",
            "shoulder_yaw_l_link",
            "elbow_pitch_l_link",
            # "elbow_yaw_l_link",
            # "wrist_pitch_l_link",
            # "wrist_roll_l_link",
            "shoulder_pitch_r_link",
            "shoulder_roll_r_link",
            "shoulder_yaw_r_link",
            "elbow_pitch_r_link",
            # "elbow_yaw_r_link",
            # "wrist_pitch_r_link",
            # "wrist_roll_r_link"
        ]


@configclass
class DexEVTFlatWoStateEstimationEnvCfg(DexEVTFlatEnvConfig):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class DexEVTFlatWoStateLongHorizonEnvCfg(DexEVTFlatWoStateEstimationEnvCfg):
    """Wo-State refinement task for faithful long-horizon root-path tracking."""

    def __post_init__(self):
        super().__post_init__()
        _configure_long_horizon_tracking(self)


@configclass
class DexEVTFlatStateFeedbackLongHorizonEnvCfg(DexEVTFlatEnvConfig):
    """Long-horizon task with closed-loop pelvis position/velocity feedback."""

    def __post_init__(self):
        super().__post_init__()
        _configure_long_horizon_tracking(self)

@configclass
class DexEVTFlatLowFreqEnvCfg(DexEVTFlatEnvConfig):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
