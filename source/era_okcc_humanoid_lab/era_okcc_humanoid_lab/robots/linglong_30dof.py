"""Isaac Lab articulation configuration for the LingLong 2.0 humanoid."""

from era_okcc_humanoid_lab.assets import ASSET_DIR
from era_okcc_humanoid_lab.robots.actuator import DelayedImplicitActuatorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg


# Provisional actuator inertias.  The URDF provides effort and velocity limits,
# but it does not contain motor-side inertia, damping, or transmission data.
# These values mirror the closest actuator groups in the working L7 baseline
# and should be replaced when measured/vendor motor parameters are available.
ARMATURE_SMALL = 0.01
ARMATURE_HIP_ROLL = 0.16473
ARMATURE_HIP_YAW = 0.088
ARMATURE_HIP_PITCH_KNEE = 0.0968
ARMATURE_ANKLE = 0.0225

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


LINGLONG_30DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ASSET_DIR}/linglong_description/LingLong2.0.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # With zero joint angles the lowest foot collision is about 0.971 m
        # below base_link, leaving a small clearance before settling.
        pos=(0.0, 0.0, 0.98),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 330.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_yaw_joint": 130.0,
                ".*_knee_joint": 150.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 12.88052988,
                ".*_hip_roll_joint": 14.66076572,
                ".*_hip_yaw_joint": 14.66076572,
                ".*_knee_joint": 14.66076572,
            },
            stiffness={
                ".*_hip_pitch_joint": _stiffness(ARMATURE_HIP_PITCH_KNEE),
                ".*_hip_roll_joint": _stiffness(ARMATURE_HIP_ROLL),
                ".*_hip_yaw_joint": _stiffness(ARMATURE_HIP_YAW),
                ".*_knee_joint": _stiffness(ARMATURE_HIP_PITCH_KNEE),
            },
            damping={
                ".*_hip_pitch_joint": _damping(ARMATURE_HIP_PITCH_KNEE),
                ".*_hip_roll_joint": _damping(ARMATURE_HIP_ROLL),
                ".*_hip_yaw_joint": _damping(ARMATURE_HIP_YAW),
                ".*_knee_joint": _damping(ARMATURE_HIP_PITCH_KNEE),
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_HIP_PITCH_KNEE,
                ".*_hip_roll_joint": ARMATURE_HIP_ROLL,
                ".*_hip_yaw_joint": ARMATURE_HIP_YAW,
                ".*_knee_joint": ARMATURE_HIP_PITCH_KNEE,
            },
            min_delay=0,
            max_delay=4,
        ),
        "feet": DelayedImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=90.0,
            velocity_limit_sim=16.44100155,
            stiffness=2.0 * _stiffness(ARMATURE_ANKLE),
            damping=2.0 * _damping(ARMATURE_ANKLE),
            armature=2.0 * ARMATURE_ANKLE,
            min_delay=0,
            max_delay=4,
        ),
        "waist": DelayedImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_pitch_joint"],
            effort_limit_sim=130.0,
            velocity_limit_sim=14.66076572,
            stiffness=500.0,
            damping=6.0,
            armature=ARMATURE_SMALL,
            min_delay=0,
            max_delay=4,
        ),
        "head": DelayedImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=10.47197551,
            stiffness=_stiffness(ARMATURE_SMALL),
            damping=_damping(ARMATURE_SMALL),
            armature=ARMATURE_SMALL,
            min_delay=0,
            max_delay=4,
        ),
        "arms": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 97.0,
                ".*_shoulder_roll_joint": 97.0,
                ".*_shoulder_yaw_joint": 27.0,
                ".*_elbow_joint": 27.0,
                ".*_wrist_roll_joint": 7.0,
                ".*_wrist_pitch_joint": 7.0,
                ".*_wrist_yaw_joint": 7.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 6.28,
                ".*_shoulder_roll_joint": 6.28,
                ".*_shoulder_yaw_joint": 10.47,
                ".*_elbow_joint": 10.47,
                ".*_wrist_roll_joint": 20.94,
                ".*_wrist_pitch_joint": 20.94,
                ".*_wrist_yaw_joint": 20.94,
            },
            stiffness=_stiffness(ARMATURE_SMALL),
            damping=_damping(ARMATURE_SMALL),
            armature=ARMATURE_SMALL,
            min_delay=0,
            max_delay=4,
        ),
    },
)


LINGLONG_30DOF_ACTION_SCALE = {}
for actuator in LINGLONG_30DOF_CFG.actuators.values():
    effort = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    names = actuator.joint_names_expr
    if not isinstance(effort, dict):
        effort = {name: effort for name in names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in names}
    for name in names:
        if name in effort and name in stiffness and stiffness[name]:
            LINGLONG_30DOF_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
