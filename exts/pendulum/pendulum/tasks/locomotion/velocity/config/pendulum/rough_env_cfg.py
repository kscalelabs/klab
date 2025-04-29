from omni.isaac.lab.utils import configclass

from pendulum.tasks.locomotion.velocity.velocity_env_cfg import \
    LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from pendulum.assets.pendulum import PENDULUM_CFG


@configclass
class PendulumRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.dof_pos_limits.weight = -1.0

        self.observations.policy.height_scan = None
        self.scene.height_scanner = None

@configclass
class PendulumRoughEnvCfg_PLAY(PendulumRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None
