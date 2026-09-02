"""Register the LingLong 2.0 locomotion task."""

import gymnasium as gym


gym.register(
    id="Locomotion-Flat-LingLong-30Dof-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.linglong_30dof_locomotion_env_cfg:LingLong30DofLocomotionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:LingLongLocomotionPPORunnerCfg",
    },
)
