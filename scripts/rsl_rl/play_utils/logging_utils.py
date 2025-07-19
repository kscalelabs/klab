"""Utility functions for logging data to disk."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import export_policy_as_onnx

def save_nn_data(nn_inputs, nn_outputs, run_dir, session_timestamp):
    """Save neural network inputs and outputs to disk.
    
    Args:
        nn_inputs (list): List of numpy arrays containing network inputs
        nn_outputs (list): List of numpy arrays containing network outputs
        run_dir (str): Directory to save the data
        session_timestamp (str): Timestamp for the current session
    
    Returns:
        dict: Metadata about saved files and array shapes
    """
    # Convert lists to numpy arrays and stack them
    inputs_array = np.stack(nn_inputs)
    outputs_array = np.stack(nn_outputs)
    
    # Create filenames
    nn_data_filename = f"{session_timestamp}_nn_data.npz"
    metadata_filename = f"{session_timestamp}_nn_metadata.json"
    
    # Create full paths
    nn_data_path = os.path.join(run_dir, nn_data_filename)
    metadata_path = os.path.join(run_dir, metadata_filename)
    
    # Save arrays in compressed format
    np.savez_compressed(
        nn_data_path,
        inputs=inputs_array,
        outputs=outputs_array
    )
    
    # Save metadata about the arrays
    metadata = {
        "inputs_shape": inputs_array.shape,
        "outputs_shape": outputs_array.shape,
        "total_timesteps": len(nn_inputs),
        "files": {
            "data": {
                "filename": nn_data_filename,
                "path": nn_data_path
            },
            "metadata": {
                "filename": metadata_filename,
                "path": metadata_path
            }
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"[INFO] Neural network data saved to {nn_data_path}")
    print(f"[INFO] Metadata saved to {metadata_path}")
    return metadata

def save_imu_csv_and_plot(df, plot_labels, colors, y_label, run_dir, session_timestamp):
    """Save IMU data to CSV and create visualization plot.
    
    Args:
        df (pd.DataFrame): DataFrame containing IMU data
        plot_labels (list): Labels for each IMU component
        colors (list): Colors for plotting each component
        y_label (str): Label for y-axis
        run_dir (str): Directory to save files
        session_timestamp (str): Timestamp for the current session
    """
    # Save to CSV
    csv_filename = f"{session_timestamp}_imu_values.csv"
    csv_path = os.path.join(run_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    
    # Create plot
    plot_filename = f"{session_timestamp}_imu_plot.png"
    plot_path = os.path.join(run_dir, plot_filename)
    
    plt.figure(figsize=(10, 6))
    for i, (label, color) in enumerate(zip(plot_labels, colors)):
        plt.plot(df['isaaclab_timestep'], df.iloc[:, i+1], label=label, color=color)
    
    plt.xlabel('Timestep')
    plt.ylabel(y_label)
    plt.title(f'IMU Data Over Time - {session_timestamp}')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"[INFO] IMU data saved to CSV: {csv_path}")
    print(f"[INFO] IMU plot saved to: {plot_path}")

def save_config_info(config_info, run_dir, session_timestamp):
    """Save configuration information to JSON.
    
    Args:
        config_info (dict): Configuration information to save
        run_dir (str): Directory to save the config
        session_timestamp (str): Timestamp for the current session
    """
    config_filename = f"{session_timestamp}_config.json"
    config_path = os.path.join(run_dir, config_filename)
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=4)
    print(f"[INFO] Config info saved to {config_path}")

def finalize_play_data(timestamps, imu_data, imu_type, nn_inputs, nn_outputs, 
                      config_info, log_dir, session_timestamp):
    """High-level function to save all play data to disk.
    
    Args:
        timestamps (list): List of timesteps
        imu_data (list): List of IMU values
        imu_type (str): Type of IMU data
        nn_inputs (list): List of neural network inputs
        nn_outputs (list): List of neural network outputs
        config_info (dict): Configuration information
        log_dir (str): Base logging directory
        session_timestamp (str): Timestamp for the current session
    
    Returns:
        str: Path to the run directory where data was saved
    """
    # Create run directory
    plots_dir = os.path.join(log_dir, "imu_plots")
    run_dir = os.path.join(plots_dir, session_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save neural network data if provided
    if nn_inputs and nn_outputs:
        nn_metadata = save_nn_data(nn_inputs, nn_outputs, run_dir, session_timestamp)
        config_info["nn_metadata"] = nn_metadata
    
    # Build IMU DataFrame and save CSV/plot
    from play_utils.policy_logging_utils import build_imu_dataframe
    df, plot_labels, colors, y_label = build_imu_dataframe(timestamps, imu_data, imu_type)
    save_imu_csv_and_plot(df, plot_labels, colors, y_label, run_dir, session_timestamp)
    
    # Save configuration info
    save_config_info(config_info, run_dir, session_timestamp)
    
    print(f"[INFO] All data saved in {run_dir}")
    return run_dir

def export_policy_to_onnx(policy_model, paths: dict, run_dir: str) -> dict:
    """Export policy model to ONNX format and copy to run directory.
    
    Args:
        policy_model: The policy model to export
        paths (dict): Dictionary containing path information
        run_dir (str): Directory to copy exported model to
    
    Returns:
        dict: Dictionary containing paths of exported files
    """
    # Export policy to ONNX
    export_model_dir = os.path.join(os.path.dirname(paths["resume_path"]), "exported")
    checkpoint_name = paths["checkpoint_name"]
    onnx_path = os.path.join(export_model_dir, f"policy_{checkpoint_name}.onnx")
    
    print(f"[INFO] Exporting ONNX policy to: {onnx_path}")
    export_policy_as_onnx(policy_model, export_model_dir, filename=f"policy_{checkpoint_name}.onnx")
    
    # Copy ONNX file to run directory with timestamp prefix
    session_onnx_path = os.path.join(run_dir, f"{paths['log_dir_timestamp']}_policy_{checkpoint_name}.onnx")
    shutil.copy2(onnx_path, session_onnx_path)
    print(f"[INFO] Copied ONNX policy to: {session_onnx_path}")
    
    return {"onnx": session_onnx_path}

def copy_config_files(paths: dict, run_dir: str) -> dict:
    """Copy environment and agent config files to run directory.
    
    Args:
        paths (dict): Dictionary containing path information
        run_dir (str): Directory to copy config files to
    
    Returns:
        dict: Dictionary containing paths of copied files
    """
    copied_files = {}
    params_dir = os.path.join(paths["log_dir"], "params")
    
    # Copy env.yaml if it exists
    env_yaml_path = os.path.join(params_dir, "env.yaml")
    if os.path.exists(env_yaml_path):
        session_env_yaml_path = os.path.join(run_dir, f"{paths['log_dir_timestamp']}_env.yaml")
        shutil.copy2(env_yaml_path, session_env_yaml_path)
        print(f"[INFO] Copied env.yaml to: {session_env_yaml_path}")
        copied_files["env_yaml"] = session_env_yaml_path
    
    # Copy agent.yaml if it exists
    agent_yaml_path = os.path.join(params_dir, "agent.yaml")
    if os.path.exists(agent_yaml_path):
        session_agent_yaml_path = os.path.join(run_dir, f"{paths['log_dir_timestamp']}_agent.yaml")
        shutil.copy2(agent_yaml_path, session_agent_yaml_path)
        print(f"[INFO] Copied agent.yaml to: {session_agent_yaml_path}")
        copied_files["agent_yaml"] = session_agent_yaml_path
    
    return copied_files