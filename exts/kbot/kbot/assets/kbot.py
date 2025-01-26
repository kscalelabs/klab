import omni.isaac.lab.sim as sim_utils
from kbot.actuators import IdentifiedActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from kbot.assets import ISAAC_ASSET_DIR
import math

KBOT_04_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_04"],
    effort_limit=120.0,
    velocity_limit=12.0,
    saturation_effort=120.0,
    stiffness={".*": 300.0},
    damping={".*": 5.0},
    armature={".*": 0.01},
    friction_static=0.0,
    activation_vel=0.1,
    friction_dynamic=0.0,
)

KBOT_03_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_03"],
    effort_limit=60.0,
    velocity_limit=12.0,
    saturation_effort=60.0,
    stiffness={".*": 150.0},
    damping={".*": 5.0},
    armature={".*": 0.01},
    friction_static=0.0,
    activation_vel=0.1,
    friction_dynamic=0.0,
)

KBOT_02_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_02"],
    effort_limit=17.0,
    velocity_limit=12.0,
    saturation_effort=17.0,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
    armature={".*": 0.01},
    friction_static=0.0,
    activation_vel=0.1,
    friction_dynamic=0.0,
)

# KBOT_00_ACTUATOR_CFG = IdentifiedActuatorCfg(
#     joint_names_expr=[".*_00"],
#     effort_limit=14.0,
#     velocity_limit=100.0,
#     saturation_effort=28.0,
#     stiffness={".*": 40.0},
#     damping={".*": 5.0},
#     armature={".*": 0.01},
#     friction_static=0.0,
#     activation_vel=0.1,
#     friction_dynamic=0.0,
# )


KBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/kbot.usd",
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            "left_shoulder_pitch_03": 0.0,
            "left_shoulder_roll_03": 0.0,
            "left_shoulder_yaw_02": -math.radians(80),
            "left_elbow_02": -math.radians(90),
            "left_wrist_02": 0.0,
            "right_shoulder_pitch_03": 0.0,
            "right_shoulder_roll_03": 0.0,
            "right_shoulder_yaw_02": -math.radians(80),
            "right_elbow_02": math.radians(90),
            "right_wrist_02": 0.0,
            # 'left_hip_pitch_04': 0.0,
            "left_hip_pitch_04": math.radians(60),
            "left_hip_roll_03": 0.0,
            "left_hip_yaw_03": 0.0,
            # 'left_knee_04': 0.0,
            "left_knee_04": -math.radians(70),
            # 'left_ankle_02': 0.0,
            "left_ankle_02": math.radians(30),
            # 'right_hip_pitch_04': 0.0,
            "right_hip_pitch_04": -math.radians(60),
            "right_hip_roll_03": 0.0,
            "right_hip_yaw_03": 0.0,
            # 'right_knee_04': 0.0,
            "right_knee_04": math.radians(70),
            # 'right_ankle_02': 0.0,
            "right_ankle_02": math.radians(30),
        },
    ),
    actuators={
        "kbot_04": KBOT_04_ACTUATOR_CFG,
        "kbot_03": KBOT_03_ACTUATOR_CFG,
        "kbot_02": KBOT_02_ACTUATOR_CFG,
        # "kbot_00": KBOT_00_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)
