import math
import torch
import random

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs.mdp.commands import UniformVelocityCommandCfg
from omni.isaac.lab.managers.command_manager import CommandTerm
from omni.isaac.lab.envs import ManagerBasedEnv  # or ManagerBasedRLEnv, whichever you have

print("Debug - Defining KPeriodicVelocityCommandCfg")

# Define the command class first so we can reference it
class KPeriodicVelocityCommand(CommandTerm):
    """
    A command generator that generates periodic velocity commands.
    """

    def __init__(self, cfg: 'KPeriodicVelocityCommandCfg', env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        # obtain the robot asset
        self.robot = env.scene[cfg.asset_name]
        # We create a buffer for velocity commands:
        self.zero_vel_command = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Initialize metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """
        By convention, the 'command' property returns (num_envs, 3) velocities.
        We'll just return zeros for now, but this will be updated for periodic motion.
        """
        return self.zero_vel_command + 0.5

    def _reset_idx_impl(self, env_ids: torch.Tensor):
        """
        Called when environment indices are reset, but we do nothing special.
        """
        pass

    def _resample_command(self, env_ids):
        """
        Called to resample commands for specific environments.
        For now we do nothing, but this will be updated for periodic motion.
        """
        pass  # Our command buffer is already zeros

    def _update_command(self):
        """
        Called to update commands after resampling.
        For now we do nothing, but this will be updated for periodic motion.
        """
        pass  # Our command buffer is already zeros

    def _update_metrics(self):
        """
        Update metrics for command tracking.
        We track the error between commanded velocity and actual velocity.
        """
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data - track error from commanded velocity
        self.metrics["error_vel_xy"] = torch.norm(
            self.robot.data.root_com_lin_vel_b[:, :2],
            dim=-1,
        ) / max_command_step
        self.metrics["error_vel_yaw"] = torch.abs(
            self.robot.data.root_com_ang_vel_b[:, 2]
        ) / max_command_step

    def step(self, env_ids: torch.Tensor):
        """
        Called each simulation step for the active env_ids.
        This will be updated to generate periodic motion.
        """
        pass

    def _debug_vis_callback(self, event):
        """
        If debug visualization is turned on, you can optionally visualize something.
        We won't bother for now.
        """
        pass

@configclass
class KPeriodicVelocityCommandCfg(UniformVelocityCommandCfg):
    """
    Configuration for periodic velocity commands.
    """

    @configclass
    class Ranges:
        """Ranges for the periodic velocity command."""
        lin_vel_x: tuple[float, float] = (0.0, 0.0)
        lin_vel_y: tuple[float, float] = (0.0, 0.0)
        ang_vel_z: tuple[float, float] = (0.0, 0.0)
        heading: tuple[float, float] | None = (0.0, 0.0)  # Required by parent class

    class_type: type = KPeriodicVelocityCommand  # Set it directly to the command class
    ranges: Ranges = Ranges()  # Default ranges
    heading_command: bool = False  # Override parent class to disable heading commands

print("Debug - KPeriodicVelocityCommandCfg defined")