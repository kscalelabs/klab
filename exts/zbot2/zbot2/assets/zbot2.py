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

ZBOT2_ACTUATOR_CFG = IdentifiedActuatorCfg(
   joint_names_expr=[".*"],
   effort_limit=1.9,            
   velocity_limit=1.0,
   saturation_effort=1.9,
   stiffness={".*": 21.1},
   damping={".*": 1.084},
   armature={".*": 0.045},
   friction_static=0.03,
   activation_vel=0.1,
   friction_dynamic=0.01,
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
