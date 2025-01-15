import os
import json
import pandas as pd
import matplotlib.pyplot as plt



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
    
    print(f"\nData, plot, video, and config saved in '{run_dir}'")