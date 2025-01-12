import omni.isaac.lab.sim as sim_utils
from zbot2.actuators import IdentifiedActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from zbot2.assets import ISAAC_ASSET_DIR

ZBOT_BENT_KNEES_POS = {
    "L_Hip_Yaw": 0.0,
    "L_Hip_Roll": 0.0,
    "L_Hip_Pitch": -0.377,
    "L_Knee_Pitch": 0.796,
    "L_Ankle_Pitch": 0.377,
    "R_Hip_Yaw": 0.0,
    "R_Hip_Roll": 0.0,
    "R_Hip_Pitch": 0.377,
    "R_Knee_Pitch": -0.796,
    "R_Ankle_Pitch": -0.377,
}

ZBOT_STRAIGHT_KNEES_POS = {
    "L_Hip_Yaw": 0.0,
    "L_Hip_Roll": 0.0,
    "L_Hip_Pitch": 0.0,
    "L_Knee_Pitch": 0.0,
    "L_Ankle_Pitch": 0.0,
    "R_Hip_Yaw": 0.0,
    "R_Hip_Roll": 0.0,
    "R_Hip_Pitch": 0.0,
    "R_Knee_Pitch": 0.0,
    "R_Ankle_Pitch": 0.0,
}

# NOTE: IsaacSim units info can be found here:
# https://docs.omniverse.nvidia.com/isaacsim/latest/reference_conventions.html

ZBOT2_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit=1.9,            # 1.0 -> 0.98 Nm (max torque)
    velocity_limit=10.0,
    saturation_effort=1.9,       # 2.0 -> 0.98 Nm (max torque)
    stiffness={".*": 21.1},      # 17.68 -> 21.1 N/rad (proportional gain)
    damping={".*": 1.084},       # 0.53 -> 1.084 Nm/(rad/s) (damping)
    armature={".*": 0.045},      # 0.001 -> 0.045 kg*m^2 (armature inertia)
    friction_static=0.03,        # 0.01 -> 0.03 (static friction)
    activation_vel=0.1,
    friction_dynamic=0.01,
)

# TODO: Try these values
# try effort limit 2.5
# ZBOT2_ACTUATOR_CFG = IdentifiedActuatorCfg(
#     joint_names_expr=[".*"],
#     effort_limit=2.0,
#     velocity_limit=60.0,
#     saturation_effort=2.0,
#     stiffness={".*": 17.68},
#     damping={".*": 0.53},
#     armature={".*": 0.0001},
#     friction_static=0.01,
#     activation_vel=0.1,
#     friction_dynamic=0.01,
# )


ZBOT2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/zbot2.usd",
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
        joint_pos=ZBOT_STRAIGHT_KNEES_POS,
    ),
    actuators={"zbot2_actuators": ZBOT2_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
