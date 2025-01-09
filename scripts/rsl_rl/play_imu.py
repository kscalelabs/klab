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
import gpr.tasks  # noqa: F401
import zbot2.tasks  # noqa: F401

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx

def extract_imu_values(obs, imu_type, imu_start_idx=1):
    """
    Extracts the desired slice of the observation tensor given the imu_type.
    Args:
        obs (torch.Tensor): The observation tensor from the environment.
        imu_type (str): 'quat', 'euler', or 'projected_gravity'.
        imu_start_idx (int): Index at which the IMU data slice begins.
    Returns:
        torch.Tensor: The relevant slice (shape [num_envs, N]) of the observation.
    """
    if imu_type == "quat":
        # e.g. [w, x, y, z]
        return obs[..., imu_start_idx : imu_start_idx + 4]
    elif imu_type == "projected_gravity":
        # projected gravity is at index 1:4 in the observation
        return obs[..., 1:4]
    else:  # euler
        # both euler and projected_gravity have 3 elements
        return obs[..., imu_start_idx : imu_start_idx + 3]


def round_and_display_imu(imu_values, imu_type, timestep):
    """
    Rounds the IMU values and prints them in a user-friendly way.
    Args:
        imu_values (torch.Tensor): shape [num_envs, N], containing IMU data.
        imu_type (str): 'quat', 'euler', or 'projected_gravity'.
        timestep (int): The current timestep for logging.
    Returns:
        List[float]: The single-environment (index=0) IMU values rounded, for logging.
    """
    print(f"\nTimestep: {timestep}")

    # Convert to Python list
    arr_list = imu_values.tolist()

    if imu_type == "quat":
        # Just round
        imu_rounded = [[round(val, 8) for val in env_vals] for env_vals in arr_list]
        label = "Quaternion (w,x,y,z)"
    elif imu_type == "euler":
        # Convert from rad to deg, then round
        imu_rounded = [[round(math.degrees(val), 3) for val in env_vals] for env_vals in arr_list]
        label = "Euler - degrees (roll,pitch,yaw)"
    else:  # 'projected_gravity'
        # project_gravity is presumably 3 float values
        imu_rounded = [[round(val, 8) for val in env_vals] for env_vals in arr_list]
        label = "Projected Gravity (x,y,z)"

    print(f"IMU {label}: {imu_rounded}")
    return imu_rounded[0]  # Return first env's data


def save_data(timestamps, imu_data, log_dir, config_info, session_timestamp, imu_type="projected_gravity"):
    """Helper function to save IMU data to CSV and create plots.
    
    Args:
        timestamps (list): List of timestep values
        imu_data (list): List of IMU values
        log_dir (str): Base logging directory path
        config_info (dict): Dictionary containing configuration information to save
        session_timestamp (str): Timestamp for the current play session
        imu_type (str): "quat", "euler", or "projected_gravity"
    """
    # Create imu_plots directory in the log directory
    plots_dir = os.path.join(log_dir, "imu_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a directory for this specific run using the session timestamp
    run_dir = os.path.join(plots_dir, session_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Prepare data for CSV and plotting
    if imu_type == "quat":
        columns = ['w', 'x', 'y', 'z']
        plot_labels = ['w (quat)', 'x (quat)', 'y (quat)', 'z (quat)']
        colors = ['red', 'green', 'blue', 'purple']
        y_label = "Quaternion Values"
    else:
        # Both euler and projected_gravity are 3D
        if imu_type == "euler":
            # data are degrees
            columns = ['roll (deg)', 'pitch (deg)', 'yaw (deg)']
            plot_labels = ['Roll', 'Pitch', 'Yaw']
            y_label = "Angle (degrees)"
        else:  # 'projected_gravity'
            columns = ['grav_x', 'grav_y', 'grav_z']
            plot_labels = ['grav_x', 'grav_y', 'grav_z']
            y_label = "Projected Gravity"
        
        colors = ['red', 'green', 'blue']
    
    # Create DataFrame
    df_dict = {'isaaclab_timestep': timestamps}
    for i, col in enumerate(columns):
        df_dict[col] = [frame[i] for frame in imu_data]
    df = pd.DataFrame(df_dict)
    
    # Save to CSV
    csv_filename = f"{session_timestamp}_imu_values.csv"
    csv_path = os.path.join(run_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    
    # Create plot
    plot_filename = f"{session_timestamp}_imu_plot.png"
    plot_path = os.path.join(run_dir, plot_filename)
    
    plt.figure(figsize=(10, 6))
    for i, (label, color) in enumerate(zip(plot_labels, colors)):
        plt.plot(timestamps, [frame[i] for frame in imu_data], label=label, color=color)
    
    plt.xlabel('Timestep')
    plt.ylabel(y_label)
    plt.title(f'IMU {imu_type.capitalize()} Over Time - {session_timestamp}')
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
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

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
            imu_values = extract_imu_values(obs, args_cli.imu_type)

            # Round, convert (if needed), and display
            imu_rounded = round_and_display_imu(imu_values, args_cli.imu_type, timestep)

            # Store the first environment's values for plotting
            timestamps.append(timestep)
            imu_data.append(imu_rounded)

            # Save data every 100 timesteps or at the end of video
            if timestep == args_cli.video_length:
                save_data(timestamps, imu_data, log_dir, config_info, session_timestamp, args_cli.imu_type)

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
