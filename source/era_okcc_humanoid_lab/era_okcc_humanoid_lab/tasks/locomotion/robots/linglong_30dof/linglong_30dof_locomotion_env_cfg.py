"""Locomotion environment configuration for LingLong 2.0."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from era_okcc_humanoid_lab.robots.linglong_30dof import (
    LINGLONG_30DOF_ACTION_SCALE,
    LINGLONG_30DOF_CFG,
)

from ...locomotion_env_cfg import LocomotionEnvCfg
from . import mdp as linglong_mdp


LINGLONG_LEG_JOINT_NAMES = [
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

LINGLONG_UPPER_BODY_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class LingLong30DofLocomotionEnvCfg(LocomotionEnvCfg):
    """Keep the L7 task logic while replacing robot-specific names and geometry."""

    def __post_init__(self):
        super().__post_init__()

        self.use_identify_params = True
        self.use_high_waist_stiffness = True

        self.scene.robot = LINGLONG_30DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = LINGLONG_30DOF_ACTION_SCALE

        # The LingLong URDF has 30 revolute joints and uses base_link as pelvis.
        self.commands.loco_command.action.dim = 30
        self.commands.loco_command.anchor_body_name = "base_link"
        self.commands.loco_command.class_type = linglong_mdp.LingLongUniformVelocityCommand

        # Domain randomization targets the upper-body mass rather than an L7-only torso link.
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_pitch_link")
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_pitch_link")

        # Explicit foot order is required by the two-column gait stance mask.
        feet_cfg = SceneEntityCfg(
            "contact_forces",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        )
        self.rewards.contact_match.func = linglong_mdp.feet_contact_number
        self.rewards.contact_match.params["sensor_cfg"] = feet_cfg
        self.rewards.contact_num_match.func = linglong_mdp.feet_contact_number_sum
        self.rewards.contact_num_match.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        )

        self.rewards.track_joint_pos.params["joint_names"] = LINGLONG_LEG_JOINT_NAMES
        self.rewards.arm_waist_deviation_l1.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LINGLONG_UPPER_BODY_JOINT_NAMES
        )
        self.rewards.body_orientation_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="base_link"
        )
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=[
                ".*_ankle_roll_link",
                ".*_wrist_yaw_link",
                ".*_elbow_link",
                ".*_hip_yaw_link",
            ],
        )

        self.terminations.bad_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["base_link", "waist_yaw_link", "waist_pitch_link"],
        )
