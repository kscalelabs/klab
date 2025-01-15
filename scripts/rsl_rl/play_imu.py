"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import pandas as pd
import math

from omni.isaac.lab.app import AppLauncher

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

from play_utils import imu_utils
from play_utils import logging_utils

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx




def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # Generate a single timestamp for the entire session
    session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        # Create imu_plots directory path first
        plots_dir = os.path.join(log_dir, "imu_plots")
        run_dir = os.path.join(plots_dir, session_timestamp)
        os.makedirs(run_dir, exist_ok=True)
        
        video_kwargs = {
            "video_folder": run_dir,  # Use the same directory as IMU plots
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": f"{session_timestamp}_video",  # Use session timestamp
            "fps": 30,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # Extract checkpoint name from resume_path
    checkpoint_name = os.path.basename(resume_path).replace(".pt", "")
    onnx_path = os.path.join(export_model_dir, f"policy_{checkpoint_name}.onnx")
    print(f"[INFO] Exporting ONNX policy to: {onnx_path}")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename=f"policy_{checkpoint_name}.onnx")

    # Lists to store data for plotting
    timestamps = []
    imu_data = []

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # Get absolute path of the checkpoint
    checkpoint_path = os.path.abspath(resume_path)

    # Create config info dictionary with session timestamp
    config_info = {
        "checkpoint_path": checkpoint_path,
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "device": agent_cfg.device,
        "experiment_name": agent_cfg.experiment_name,
        "timestamp": session_timestamp,
        "imu_type": args_cli.imu_type,
        "cli_args": vars(args_cli),
    }

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            timestep += 1

            # Slice out IMU data from obs
            imu_values = imu_utils.extract_imu_values(obs, args_cli.imu_type)

            # Round, convert (if needed), and display
            imu_rounded = imu_utils.round_and_display_imu(imu_values, args_cli.imu_type, timestep)

            # Store the first environment's values for plotting
            timestamps.append(timestep)
            imu_data.append(imu_rounded)

            # Save data every 100 timesteps or at the end of video
            if timestep == args_cli.video_length:
                logging_utils.save_data(timestamps, imu_data, log_dir, config_info, session_timestamp, args_cli.imu_type)

        if args_cli.video:
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
