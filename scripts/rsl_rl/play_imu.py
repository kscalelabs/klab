"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import pandas as pd
import math
import shutil
from omni.isaac.lab.app import AppLauncher
import wandb
from tqdm import tqdm

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--imu_type",
    type=str,
    choices=["quat", "euler", "projected_gravity"],
    default="projected_gravity",
    help="Type of IMU data to log. Choose from ['quat', 'euler', 'projected_gravity']."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import kbot.tasks  # noqa: F401
import zbot2.tasks  # noqa: F401

import yaml
import pickle

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx

from play_utils import (
    env_utils,
    logging_utils,
    wandb_utils,
    policy_logging_utils as policy_utils
)

from play_utils.policy_logging_utils import process_imu_data, collect_nn_data
from play_utils.testing_utils import test_joints


def main():
    """Play with RSL-RL agent."""
    # First parse the base configs
    base_agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Set up experiment paths and extract checkpoint info
    paths = env_utils.setup_experiment_paths(
        experiment_name=base_agent_cfg.experiment_name,
        load_run=base_agent_cfg.load_run,
        load_checkpoint=base_agent_cfg.load_checkpoint
    )

    # Load and layer configs: base -> checkpoint -> play overrides
    env_cfg, agent_cfg = env_utils.overwrite_configs(
        task_name=args_cli.task,
        base_agent_cfg=base_agent_cfg,
        checkpoint_dir=paths["log_dir"],
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        do_play_overrides=True
    )

    # set num_envs from cli args
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Setup video recording if enabled
    run_dir, video_kwargs = None, None
    if args_cli.video:
        run_dir, video_kwargs = env_utils.setup_video_recording(
            log_dir=paths["log_dir"],
            session_timestamp=paths["session_timestamp"],
            video_length=args_cli.video_length
        )

    # Create and configure environment
    env = env_utils.create_env(
        task_name=args_cli.task,
        env_cfg=env_cfg,
        video=args_cli.video,
        video_kwargs=video_kwargs
    )

    print(f"[INFO]: Loading model checkpoint from: {paths['resume_path']}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(paths["resume_path"])

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    
    exported_files = logging_utils.export_policy_to_onnx(
        policy_model=ppo_runner.alg.actor_critic,
        paths=paths,
        run_dir=run_dir
    )

    config_files = logging_utils.copy_config_files(paths, run_dir)

    # Lists to store data in memory
    timestamps = []
    imu_data = []
    nn_inputs = []
    nn_outputs = []

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # Get absolute path of the checkpoint
    checkpoint_path = os.path.abspath(paths["resume_path"])

    # Create config info dictionary with session timestamp
    config_info = {
        "checkpoint_path": checkpoint_path,
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "device": agent_cfg.device,
        "experiment_name": agent_cfg.experiment_name,
        "timestamp": paths["session_timestamp"],
        "imu_type": args_cli.imu_type,
        "cli_args": vars(args_cli),
    }

    # Initialize wandb
    wandb_run = wandb_utils.init_wandb(paths["log_dir"], vars(args_cli))

    # Set up progress bar
    total_steps = args_cli.video_length if args_cli.video else float('inf')
    pbar = tqdm(total=total_steps, desc="Simulating", unit="steps")

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            
            # Collect data in memory
            collect_nn_data(obs, actions, nn_inputs, nn_outputs)
            imu_vals = process_imu_data(obs, args_cli.imu_type)
            imu_data.append(imu_vals)
            timestamps.append(timestep)
            
            # step environment
            obs, _, _, _ = env.step(actions)
            timestep += 1

            # Update progress bar
            pbar.update(1)

            if timestep == args_cli.video_length:
                break

    # Close progress bar
    pbar.close()

    # Save all collected data to disk
    run_dir = logging_utils.finalize_play_data(
        timestamps=timestamps,
        imu_data=imu_data,
        imu_type=args_cli.imu_type,
        nn_inputs=nn_inputs,
        nn_outputs=nn_outputs,
        config_info=config_info,
        log_dir=paths["log_dir"],
        session_timestamp=paths["session_timestamp"]
    )

    # close the simulator
    env.close()

    # Upload videos to wandb if video recording was enabled
    if args_cli.video:
        wandb_utils.upload_videos_to_wandb(wandb_run, run_dir)

    # close wandb
    wandb_utils.finish_wandb_run(wandb_run)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
