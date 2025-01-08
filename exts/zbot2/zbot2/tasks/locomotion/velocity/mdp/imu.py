from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import RLTaskEnv


def imu_lin_acc(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Returns the IMU's linear acceleration in the body frame, shape: (num_envs, 3).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    # sensor.data.lin_acc_b is a (num_envs, 3) tensor with linear acceleration in body-frame
    return sensor.data.lin_acc_b


def imu_ang_vel(env: RLTaskEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Returns the IMU's angular velocity in the body frame, shape: (num_envs, 3).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.ang_vel_b