import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Velocity-Flat-Zbot2-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Zbot2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Zbot2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Flat-Zbot2-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Zbot2FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Zbot2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-Zbot2-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={    
        "env_cfg_entry_point": rough_env_cfg.Zbot2RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Zbot2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-Zbot2-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.Zbot2RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Zbot2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-Zbot2-Eval-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.Zbot2RoughEnvCfg_EVAL,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Zbot2RoughPPORunnerCfg",
    },
)