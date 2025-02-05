"""Utility functions for environment setup and configuration."""

import os
import pickle
from datetime import datetime
import gymnasium as gym
from omni.isaac.lab_tasks.utils import parse_env_cfg, get_checkpoint_path
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
import math
def apply_play_overrides(env_cfg):
    """
    Applies common overrides to an environment config to make it more suitable
    for "play" (i.e., manual inspection / demo). Specifically:
      1) Reduces the number of environments.
      2) Decreases environment spacing.
      3) Optionally sets a shorter episode length.
      4) Disables randomization (policy corruption, external forces, pushing).
      5) Adjusts terrain to a smaller grid with no curriculum.
      6) Sets certain default command ranges.

    Args:
        env_cfg: The environment configuration object (e.g. KbotRoughEnvCfg, G1RoughEnvCfg, etc.)

    Returns:
        env_cfg: The modified environment configuration object.
    """
    # 1) Make a smaller scene for play
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5

    # 2) (Optional) shorter episode length to quickly see resets
    env_cfg.episode_length_s = 40.0

    # 3) Disable randomization / corruption
    #    e.g., random sensor noise, random pushes, etc.
    env_cfg.observations.policy.enable_corruption = False
    
    # For older versions or different structure,
    # your randomization might be under env_cfg.randomization.<something> instead.
    # Below, we assume "events" is where external forces & pushes live.
    if hasattr(env_cfg, "events"):
        env_cfg.events.base_external_force_torque = None
        env_cfg.events.push_robot = None

    # turn off terminations by base conact 
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "base_contact"):
            env_cfg.terminations.base_contact = None
    
    # 4) Smaller or flat terrain with no curriculum, if terrain info exists
    if hasattr(env_cfg.scene, "terrain"):
        # E.g., skip multi-level terrain spawns
        env_cfg.scene.terrain.max_init_terrain_level = None
        
        # If there's a terrain_generator, reduce row/col count and disable curriculum
        tg = getattr(env_cfg.scene.terrain, "terrain_generator", None)
        if tg is not None:
            tg.num_rows = 5
            tg.num_cols = 5
            tg.curriculum = False

    # # for zbot
    # env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = (-1.0, -1.0)

    # 5) Adjust command ranges for a friendlier "play" scenario
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0, -1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

    return env_cfg

def _load_config_from_pickle(pickle_path: str, base_config: dict, config_name: str) -> dict:
    """Helper function to load a config from a pickle file with fallback to base config.
    
    Args:
        pickle_path (str): Path to the pickle file
        base_config (dict): Base configuration to use as fallback
        config_name (str): Name of the config for logging purposes
    
    Returns:
        dict: Loaded configuration
    """
    if os.path.exists(pickle_path):
        print(f"[INFO] Loading {config_name} config from checkpoint: {pickle_path}")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"[WARNING] No {config_name} pickle found at {pickle_path}, using base config.")
        return base_config

def overwrite_configs(
    task_name: str, 
    base_agent_cfg: dict,
    checkpoint_dir: str, 
    num_envs: int = None, 
    use_fabric: bool = True,
    do_play_overrides: bool = True,
) -> tuple[dict, dict]:
    """Load and layer configs in the following order:
    1) Parse base configs from code (both env and agent)
    2) Overwrite with checkpoint configs (to match observation shape, etc.)
    3) Optionally re-apply "play" overrides
    
    Args:
        task_name (str): Name of the task to create
        base_agent_cfg (dict): Base agent configuration to use as fallback
        checkpoint_dir (str): Directory containing checkpoint parameters
        num_envs (int, optional): Number of environments to override. Defaults to None.
        use_fabric (bool, optional): Whether to use fabric. Defaults to True.
        do_play_overrides (bool, optional): Whether to apply play overrides. Defaults to True.
    
    Returns:
        tuple[dict, dict]: Tuple containing (env_cfg, agent_cfg)
    """
    # --- Step A: Parse base env config from code ---
    agent_cfg = base_agent_cfg
    env_cfg = parse_env_cfg(task_name, num_envs=num_envs, use_fabric=use_fabric)

    # --- Step B: If we have checkpoint configs, load them. Otherwise use the base ones ---
    params_dir = os.path.join(checkpoint_dir, "params")
    
    # Load configs from checkpoint with fallback to base configs
    env_cfg = _load_config_from_pickle(
        pickle_path=os.path.join(params_dir, "env.pkl"),
        base_config=env_cfg,
        config_name="environment"
    )
    
    agent_cfg = _load_config_from_pickle(
        pickle_path=os.path.join(params_dir, "agent.pkl"),
        base_config=base_agent_cfg,
        config_name="agent"
    )

    # --- Step C: Re-apply "play" overrides if requested ---
    if do_play_overrides:
        env_cfg = apply_play_overrides(env_cfg)
    
    return env_cfg, agent_cfg

def setup_experiment_paths(experiment_name: str, load_run: str = None, load_checkpoint: str = None) -> dict:
    """Set up experiment paths and extract checkpoint information.
    
    Args:
        experiment_name (str): Name of the experiment
        load_run (str, optional): Run to load from. Defaults to None.
        load_checkpoint (str, optional): Checkpoint to load. Defaults to None.
    
    Returns:
        dict: Dictionary containing paths and checkpoint info
    """
    # Set up logging directory
    log_root_path = os.path.join("logs", "rsl_rl", experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # Get checkpoint path
    resume_path = get_checkpoint_path(log_root_path, load_run, load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    
    # Extract checkpoint information
    checkpoint_name = os.path.basename(resume_path).replace(".pt", "")
    log_dir_timestamp = os.path.basename(log_dir)
    session_timestamp = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{checkpoint_name}"
    
    return {
        "log_root_path": log_root_path,
        "resume_path": resume_path,
        "log_dir": log_dir,
        "checkpoint_name": checkpoint_name,
        "log_dir_timestamp": log_dir_timestamp,
        "session_timestamp": session_timestamp
    }

def setup_video_recording(log_dir: str, session_timestamp: str, video_length: int) -> tuple[str, dict]:
    """Set up video recording configuration and directories.
    
    Args:
        log_dir (str): Base logging directory
        session_timestamp (str): Timestamp for the current session
        video_length (int): Length of video recording
    
    Returns:
        tuple[str, dict]: Run directory path and video configuration dictionary
    """
    plots_dir = os.path.join(log_dir, "imu_plots")
    run_dir = os.path.join(plots_dir, session_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    video_kwargs = {
        "video_folder": run_dir,
        "step_trigger": lambda step: step == 0,
        "video_length": video_length,
        "name_prefix": f"{session_timestamp}_video",
        "fps": 30,
        "disable_logger": True,
    }
    
    return run_dir, video_kwargs

def create_env(task_name: str, env_cfg, video: bool = False, video_kwargs: dict = None):
    """Create and configure the environment with optional video recording.
    
    Args:
        task_name (str): Name of the task to create
        env_cfg: Environment configuration object
        video (bool, optional): Whether to enable video recording. Defaults to False.
        video_kwargs (dict, optional): Video recording parameters. Defaults to None.
    
    Returns:
        env: Configured environment instance
    """
    # Set render mode based on video flag
    render_mode = "rgb_array" if video else None
    
    # Create base environment
    env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)
    
    # Add video wrapper if requested
    if video:
        if video_kwargs is None:
            raise ValueError("video_kwargs must be provided when video=True")
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap for RSL-RL compatibility
    env = RslRlVecEnvWrapper(env)
    
    return env

