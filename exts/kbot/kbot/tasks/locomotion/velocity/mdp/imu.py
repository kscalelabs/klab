"""IMU observations for the velocity task."""

import math

import torch
from omni.isaac.lab.utils import math as math_utils
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv


def kscale_imu_euler(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Returns the IMU's orientation in the body frame as Euler angles (roll, pitch, yaw), shape: (num_envs, 3).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    quat_w_b4 = sensor.data.quat_w
    roll_b, pitch_b, yaw_b = math_utils.euler_xyz_from_quat(quat_w_b4)
    # Normalizes to [-pi, pi]
    roll_b = (roll_b + math.pi) % (2 * math.pi) - math.pi
    pitch_b = (pitch_b + math.pi) % (2 * math.pi) - math.pi
    yaw_b = (yaw_b + math.pi) % (2 * math.pi) - math.pi
    return torch.stack([roll_b, pitch_b, yaw_b], dim=-1)


def kscale_imu_quat(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Returns the IMU's orientation as a quaternion (w, x, y, z), shape: (num_envs, 4).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.quat_w