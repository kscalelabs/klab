import math
import torch

def extract_imu_values(obs, imu_type, imu_start_idx=3):
    """
    Extracts the desired slice of the observation tensor given the imu_type.
    Args:
        obs (torch.Tensor): The observation tensor from the environment.
        imu_type (str): 'quat', 'euler', or 'projected_gravity'.
        imu_start_idx (int): Index at which the IMU data slice begins.
    Returns:
        torch.Tensor: The relevant slice (shape [num_envs, N]) of the observation.
    """
    imu_dims = 4 if imu_type == "quat" else 3
    return obs[..., imu_start_idx : imu_start_idx + imu_dims]

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

    # print(f"IMU {label}: {imu_rounded}")
    return imu_rounded[0]  # Return first env's data
