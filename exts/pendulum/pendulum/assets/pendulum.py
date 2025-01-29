import omni.isaac.lab.sim as sim_utils
from pendulum.actuators import IdentifiedActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from pendulum.assets import ISAAC_ASSET_DIR


# NOTE: IsaacSim units info can be found here:
# https://docs.omniverse.nvidia.com/isaacsim/latest/reference_conventions.html

# Super high values
PENDULUM_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit=1.93,     
    velocity_limit=2,
    saturation_effort=1.93,
    stiffness={".*": 18.155406706338038},
    damping={".*": 1.9997577094057506},
    armature={".*": 0.009998463308291579}, 
    friction_static=0.011331680000338948,
    activation_vel=0.1,
    friction_dynamic=0.01593,
)

PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/pendulum.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),  # Example: ~30 cm above ground
        joint_pos={
            "Revolute_1": 0.0,
            "Revolute_2": 0.0,
            "Revolute_3": 0.0,
        },
    ),
    actuators={"pendulum_actuators": PENDULUM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
