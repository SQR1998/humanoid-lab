"""RSL-RL runner configuration for LingLong locomotion."""

from isaaclab.utils import configclass

from ...agents.rsl_rl_ppo_cfg import LocomotionPPORunnerCfg


@configclass
class LingLongLocomotionPPORunnerCfg(LocomotionPPORunnerCfg):
    """Use the original PPO settings while keeping LingLong logs separate."""

    experiment_name = "linglong_locomotion"
