import omni.isaac.lab.sim as sim_utils
from zbot2.actuators import IdentifiedActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from zbot2.assets import ISAAC_ASSET_DIR

ZBOT_BENT_KNEES_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.377,
    "left_knee_pitch": 0.796,
    "left_ankle_pitch": 0.377,

    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.377,
    "right_knee_pitch": -0.796,
    "right_ankle_pitch": -0.377,
}

ZBOT_STRAIGHT_KNEES_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": 0.0,
    "left_knee_pitch": 0.0,
    "left_ankle_pitch": 0.0,

    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.0,
    "right_knee_pitch": 0.0,
    "right_ankle_pitch": 0.0,
}

ZBOT2_ACTUATOR_CFG = IdentifiedActuatorCfg(
  joint_names_expr=[".*"],
  effort_limit=1.9,           
  velocity_limit=2.0,
  saturation_effort=1.9,
  stiffness={".*": 20.815174050888604},
  damping={".*": 1.9979591919395816},
  armature={".*": 0.009827620008867697},
  friction_static=0.01837093610596295,
  activation_vel=0.1,
  friction_dynamic=0.008773601335918209,
)

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
        pos=(0.0, 0.0, 0.415),  # Example: ~30 cm above ground
        joint_pos=ZBOT_STRAIGHT_KNEES_POS,
    ),
    actuators={"zbot2_actuators": ZBOT2_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
