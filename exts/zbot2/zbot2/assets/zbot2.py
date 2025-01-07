import omni.isaac.lab.sim as sim_utils
from zbot2.actuators import IdentifiedActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from zbot2.assets import ISAAC_ASSET_DIR

# High torque actuator config for hip pitch and knee joints
ZBOT2_04_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit=120.0,
    velocity_limit=14,
    saturation_effort=560,
    stiffness={".*": 15.0},
    damping={".*": 1.5},
    armature={".*": 1.5e-4 * 81},
    friction_static=0.8,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Medium torque actuator config for hip roll and yaw joints
ZBOT2_03_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit=60.0,
    velocity_limit=14,
    saturation_effort=560,
    stiffness={".*": 15.0},
    damping={".*": 1.5},
    armature={".*": 1.5e-4 * 81},
    friction_static=0.8,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Lower torque actuator config for ankle joints
ZBOT2_01_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit=17.0,
    velocity_limit=14,
    saturation_effort=560,
    stiffness={".*": 15.0},
    damping={".*": 1.5},
    armature={".*": 1.5e-4 * 81},
    friction_static=0.8,
    activation_vel=0.1,
    friction_dynamic=0.02,
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.215),
        joint_pos={
            'left_hip_yaw': 0.0,
            'left_hip_roll': 0.0,
            'left_hip_pitch': 0.0,
            'left_knee_pitch': 0.0,
            'left_ankle_pitch': 0.0,
            'right_hip_yaw': 0.0,
            'right_hip_roll': 0.0,
            'right_hip_pitch': 0.0,
            'right_knee_pitch': 0.0,
            'right_ankle_pitch': 0.0,
        },
    ),
    actuators={
        # High torque actuators (120 Nm) - hip pitch and knee joints
        "left_hip_pitch": ZBOT2_04_ACTUATOR_CFG,
        "right_hip_pitch": ZBOT2_04_ACTUATOR_CFG,
        "left_knee_pitch": ZBOT2_04_ACTUATOR_CFG,
        "right_knee_pitch": ZBOT2_04_ACTUATOR_CFG,
        
        # Medium torque actuators (60 Nm) - hip roll and yaw joints
        "left_hip_roll": ZBOT2_03_ACTUATOR_CFG,
        "right_hip_roll": ZBOT2_03_ACTUATOR_CFG,
        "left_hip_yaw": ZBOT2_03_ACTUATOR_CFG,
        "right_hip_yaw": ZBOT2_03_ACTUATOR_CFG,
        
        # Low torque actuators (17 Nm) - ankle joints
        "left_ankle_pitch": ZBOT2_01_ACTUATOR_CFG,
        "right_ankle_pitch": ZBOT2_01_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)
