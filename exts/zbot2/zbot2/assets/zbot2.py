import omni.isaac.lab.sim as sim_utils
from zbot2.actuators import IdentifiedActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from zbot2.assets import ISAAC_ASSET_DIR
import math


ZBOT_BENT_KNEES_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": -math.radians(31.6),
    "left_knee": math.radians(65.6),
    "left_ankle": math.radians(31.6),
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": math.radians(31.6),
    "right_knee": -math.radians(65.6),
    "right_ankle": -math.radians(31.6),
    "left_shoulder_yaw": 0.0,
    "left_shoulder_pitch": 0.0,
    "left_elbow": 0.0,
    "left_gripper": 0.0,
    "right_shoulder_yaw": 0.0,
    "right_shoulder_pitch": 0.0,
    "right_elbow": 0.0,
    "right_gripper": 0.0,
}

ZBOT_STRAIGHT_KNEES_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": 0.0,
    "left_knee": 0.0,
    "left_ankle": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.0,
    "right_knee": 0.0,
    "right_ankle": 0.0,
    "left_shoulder_yaw": 0.0,
    "left_shoulder_pitch": 0.0,
    "left_elbow": 0.0,
    "left_gripper": 0.0,
    "right_shoulder_yaw": 0.0,
    "right_shoulder_pitch": 0.0,
    "right_elbow": 0.0,
    "right_gripper": 0.0,
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
        pos=(0.0, 0.0, 0.39),
        joint_pos=ZBOT_BENT_KNEES_POS,
    ),
    actuators={"zbot2_actuators": ZBOT2_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
