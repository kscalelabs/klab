import omni.isaac.lab.sim as sim_utils
from kbot.actuators import IdentifiedActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from kbot.assets import ISAAC_ASSET_DIR

KBOT_04_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*hip_pitch_04", ".*knee_04"],
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

KBOT_03_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*hip_roll_03", ".*hip_yaw_03"],
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

KBOT_01_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*ankle_02"],
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.07),
        joint_pos={
            'left_hip_pitch_04': 0.0,
            'left_hip_roll_03': 0.0,
            'left_hip_yaw_03': 0.0,
            'left_knee_04': 0.0,
            'left_ankle_02': 0.0,
            'right_hip_pitch_04': 0.0,
            'right_hip_roll_03': 0.0,
            'right_hip_yaw_03': 0.0,
            'right_knee_04': 0.0,
            'right_ankle_02': 0.0,
        },
    ),
    actuators={
        "kbot_04": KBOT_04_ACTUATOR_CFG,
        "kbot_03": KBOT_03_ACTUATOR_CFG,
        "kbot_01": KBOT_01_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)
