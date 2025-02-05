"""Utility functions for in-memory policy data transformations."""

import math
import torch
import numpy as np
import pandas as pd

def process_imu_data(obs, imu_type, imu_start_idx=3):
    """Extract and process IMU data in memory.
    
    Args:
        obs (torch.Tensor): The observation tensor from the environment.
        imu_type (str): 'quat', 'euler', or 'projected_gravity'.
        imu_start_idx (int): Index at which the IMU data slice begins.
    
    Returns:
        List[float]: The single-environment (index=0) IMU values rounded.
    """
    # Extract IMU values
    imu_dims = 4 if imu_type == "quat" else 3
    imu_values = obs[..., imu_start_idx : imu_start_idx + imu_dims]  # shape [num_envs, imu_dims]
    
    # Convert to Python list and process
    arr_list = imu_values.tolist()

    if imu_type == "quat":
        # Just round quaternion values
        imu_rounded = [[round(val, 8) for val in env_vals] for env_vals in arr_list]
    elif imu_type == "euler":
        # Convert from rad to deg, then round
        imu_rounded = [[round(math.degrees(val), 3) for val in env_vals] for env_vals in arr_list]
    else:  # 'projected_gravity'
        # Round projected gravity values
        imu_rounded = [[round(val, 8) for val in env_vals] for env_vals in arr_list]

    return imu_rounded[0]  # Return first env's data

def build_imu_dataframe(timestamps, imu_data, imu_type="projected_gravity"):
    """Build a DataFrame in memory from timestamps + IMU data.
    
    Args:
        timestamps (list): List of timesteps
        imu_data (list): List of IMU values
        imu_type (str): Type of IMU data ('quat', 'euler', or 'projected_gravity')
    
    Returns:
        tuple: (DataFrame, plot_labels, colors, y_label) needed for plotting or CSV writing
    """
    if imu_type == "quat":
        columns = ['w', 'x', 'y', 'z']
        plot_labels = ['w (quat)', 'x (quat)', 'y (quat)', 'z (quat)']
        colors = ['red', 'green', 'blue', 'purple']
        y_label = "Quaternion Values"
    elif imu_type == "euler":
        columns = ['roll (deg)', 'pitch (deg)', 'yaw (deg)']
        plot_labels = ['Roll', 'Pitch', 'Yaw']
        colors = ['red', 'green', 'blue']
        y_label = "Angle (degrees)"
    else:  # 'projected_gravity'
        columns = ['grav_x', 'grav_y', 'grav_z']
        plot_labels = ['grav_x', 'grav_y', 'grav_z']
        colors = ['red', 'green', 'blue']
        y_label = "Projected Gravity"

    # Build a dictionary for DataFrame
    df_dict = {'isaaclab_timestep': timestamps}
    for i, col in enumerate(columns):
        df_dict[col] = [frame[i] for frame in imu_data]

    df = pd.DataFrame(df_dict)
    return df, plot_labels, colors, y_label

def collect_nn_data(obs, actions, nn_inputs, nn_outputs):
    """Append observations/actions to the nn_inputs/nn_outputs lists in memory.
    
    Args:
        obs (torch.Tensor): Current observation
        actions (torch.Tensor): Current actions
        nn_inputs (list): List to store network inputs
        nn_outputs (list): List to store network outputs
    """
    nn_inputs.append(obs.cpu().numpy())
    nn_outputs.append(actions.cpu().numpy())