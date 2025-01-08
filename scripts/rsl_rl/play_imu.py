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
import gpr.tasks  # noqa: F401
import zbot2.tasks  # noqa: F401

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx

def save_data(timestamps, roll_data, pitch_data, yaw_data, log_dir, config_info, session_timestamp):
    """Helper function to save IMU data to CSV and create plots.
    
    Args:
        timestamps (list): List of timestep values
        roll_data (list): List of roll angles in radians
        pitch_data (list): List of pitch angles in radians
        yaw_data (list): List of yaw angles in radians
        log_dir (str): Base logging directory path
        config_info (dict): Dictionary containing configuration information to save
        session_timestamp (str): Timestamp for the current play session
    """
    # Create imu_plots directory in the log directory
    plots_dir = os.path.join(log_dir, "imu_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a directory for this specific run using the session timestamp
    run_dir = os.path.join(plots_dir, session_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Convert angles from radians to degrees
    roll_degrees = [math.degrees(r) for r in roll_data]
    pitch_degrees = [math.degrees(p) for p in pitch_data]
    yaw_degrees = [math.degrees(y) for y in yaw_data]
    
    # Save data to CSV
    csv_filename = f"{session_timestamp}_imu_values.csv"
    csv_path = os.path.join(run_dir, csv_filename)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'isaaclab_timestep': timestamps,
        'roll (deg)': roll_degrees,
        'pitch (deg)': pitch_degrees,
        'yaw (deg)': yaw_degrees
    })
    df.to_csv(csv_path, index=False)
    
    # Save the plot
    plot_filename = f"{session_timestamp}_imu_plot.png"
    plot_path = os.path.join(run_dir, plot_filename)
    
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, roll_degrees, label='Roll', color='red')
    plt.plot(timestamps, pitch_degrees, label='Pitch', color='green')
    plt.plot(timestamps, yaw_degrees, label='Yaw', color='blue')
    plt.xlabel('Timestep')
    plt.ylabel('Angle (degrees)')
    plt.title(f'IMU Orientation Over Time - {session_timestamp}')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    
    # Save the configuration info
    config_filename = f"{session_timestamp}_config.json"
    config_path = os.path.join(run_dir, config_filename)
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=4)
    
    print(f"\nData, plot, and config saved in '{run_dir}'")

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
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": f"{session_timestamp}_imu_video",  # Use session timestamp
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
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # Create lists to store data for plotting
    timestamps = []
    roll_data = []
    pitch_data = []
    yaw_data = []

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
        "timestamp": session_timestamp,  # Use session timestamp
        "cli_args": vars(args_cli)
    }

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            timestep += 1
            # Print timestep and split observations
            # IMU orientation is at indices 9-11 (index 3 in observation manager, with shape (3,))
            start_idx = 9  # 3 for base_lin_vel + 3 for base_ang_vel + 3 for velocity_commands
            imu_orientation = obs[..., start_idx:start_idx+3]  # kscale_imu_orientation (3,)
            print(f"\nTimestep: {timestep}")
            # Handle nested lists by rounding each value
            imu_rounded = [[round(val, 8) for val in env_vals] for env_vals in imu_orientation.tolist()]
            print(f"IMU Orientation (roll, pitch, yaw): {imu_rounded}")

            # Store data for plotting
            timestamps.append(timestep)
            roll_data.append(imu_rounded[0][0])
            pitch_data.append(imu_rounded[0][1])
            yaw_data.append(imu_rounded[0][2])
            
            # Save data every 100 timesteps
            if timestep == args_cli.video_length:
                save_data(timestamps, roll_data, pitch_data, yaw_data, log_dir, config_info, session_timestamp)
            
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
