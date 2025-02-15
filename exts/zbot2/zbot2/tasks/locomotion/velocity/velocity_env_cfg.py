from __future__ import annotations

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCasterCfg, ContactSensorCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from omni.isaac.lab.sensors import ImuCfg

import zbot2.tasks.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from zbot2.terrains.terrain_generator_cfg import ROUGH_TERRAINS_CFG


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Z_BOT2_MASTER_BODY_SKELETON",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3,
                                                track_air_time=True, track_pose=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    
    # imu sensor
    kscale_imu_sensor = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/IMU",
        debug_vis=False,
        gravity_bias=(0.0, 0.0, 0.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-1.0, -0.2),
            ang_vel_z=(-0.1, 0.1),
            heading=(-math.pi, math.pi),
        ),
    )

JOINT_NAMES_LIST = [
    "left_shoulder_yaw",    # 00
    "left_shoulder_pitch",  # 01
    "left_elbow",           # 02
    "left_gripper",         # 03
    "right_shoulder_yaw",   # 04
    "right_shoulder_pitch", # 05
    "right_elbow",          # 06
    "right_gripper",        # 07
    "left_hip_yaw",         # 08
    "left_hip_roll",        # 09
    "left_hip_pitch",       # 10
    "left_knee",            # 11
    "left_ankle",           # 12
    "right_hip_yaw",        # 13
    "right_hip_roll",       # 14
    "right_hip_pitch",      # 15
    "right_knee",           # 16
    "right_ankle"           # 17
]

ARM_JOINTS = JOINT_NAMES_LIST[:8]


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=JOINT_NAMES_LIST, scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))

        ######
        # Add IMU values
        ######

        # Projected gravity
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # Unify joints and use positions
        joint_angles = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=JOINT_NAMES_LIST,
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=JOINT_NAMES_LIST,
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (currently identical to policy)."""

        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )

        ######
        # Add IMU values
        ######

        # Critic gets all the IMU values 

        # ## IMU euler angles
        # kscale_imu_euler = ObsTerm(
        #     func=mdp.kscale_imu_euler,
        #     noise=Unoise(n_min=-0.02, n_max=0.02),  # optional noise
        #     params={"sensor_cfg": SceneEntityCfg("kscale_imu_sensor")}
        # )

        # # IMU quaternion
        # kscale_imu_quat = ObsTerm(
        #     func=mdp.kscale_imu_quat,
        #     noise=Unoise(n_min=-0.02, n_max=0.02),  # optional noise
        #     params={"sensor_cfg": SceneEntityCfg("kscale_imu_sensor")}
        # )
        
        # imu_lin_acc = ObsTerm(
        #     func=mdp.imu_lin_acc,
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     params={"asset_cfg": SceneEntityCfg("kscale_imu_sensor")},
        # )

        # imu_ang_vel = ObsTerm(
        #     func=mdp.imu_ang_vel,
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     params={"asset_cfg": SceneEntityCfg("kscale_imu_sensor")},
        # )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # Unify joints and use positions
        joint_angles = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=JOINT_NAMES_LIST,
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=JOINT_NAMES_LIST,
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    # scale_all_link_masses = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.9, 1.1),
    #             "operation": "scale"},
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_distribution_params": (-1.0, 1.0),
    #             "operation": "add"},
    # )

    # scale_all_joint_armature = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "armature_distribution_params": (1.0, 1.05),
    #             "operation": "scale"},
    # )

    # add_all_joint_default_pos = EventTerm(
    #     func=mdp.randomize_joint_default_pos,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "pos_distribution_params": (-0.05, 0.05),
    #             "operation": "add"},
    # )

    # scale_all_joint_friction_model = EventTerm(
    #     func=mdp.randomize_joint_friction_model,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "friction_distribution_params": (0.9, 1.1),
    #             "operation": "scale"},
    # )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Z_BOT2_MASTER_BODY_SKELETON"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.35},
    )
    track_ang_vel_z_world_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.35},
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0) # G1 sets this to 0.0
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_LIST)})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    feet_air_time_positive_biped = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FOOT", "FOOT_2"]),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FOOT", "FOOT_2"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["FOOT", "FOOT_2"]),
        },
    )
    # Penalize ankle joint limits
    # dof_pos_limits = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_ankle", "right_ankle"])},
    # )
    # no joint deviation arms for now
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ARM_JOINTS,
            )
        },
    )
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["a_215_BothFlange_3", "a_215_BothFlange_4"]), "threshold": 1.0},
    # )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "left_hip_roll",
                    "right_hip_roll",
                    "left_hip_yaw",
                    "right_hip_yaw",
                ],
            )
        },
    )
    # joint_deviation_knee = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_knee", "right_knee"])},
    # )
    # joint_encouragement_bend_knee = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_knee_04", "right_knee_04"])},
    # )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    # TODO add
    # joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0)
    # joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # NOTE: these termination joints are chosen because they do not touch each other
    # Choosing joints that touch each other will cause the episode to terminate prematurely
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[

                    # main body
                    # "base",
                    "Z_BOT2_MASTER_BODY_SKELETON",

                    # left arm
                    "Z_BOT2_MASTER_SHOULDER2",
                    "a_215_1Flange",
                    "R_ARM_MIRROR_1",
                    "FINGER_1",

                    # right arm
                    "Z_BOT2_MASTER_SHOULDER2_2",
                    "a_215_1Flange_2",
                    "L_ARM_MIRROR_1",
                    "FINGER_1_2",

                    # left leg
                    "U_HIP_L",
                    "a_215_BothFlange_5",
                    "a_215_BothFlange_9", # thigh
                    # "a_215_BothFlange_13", # shin
                    # "FOOT",

                    # right leg
                    "U_HIP_R",
                    "a_215_BothFlange_6",
                    "a_215_BothFlange_10", # thigh
                    # "a_215_BothFlange_14", # shin
                    # "FOOT_2"
                ]
            ),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # push force follows curriculum
    push_force_levels = CurrTerm(func=mdp.modify_push_force,
                                 params={"term_name": "push_robot", "max_velocity": [3.0, 3.0], "interval": 200 * 24,
                                         "starting_step": 1500 * 24})
    # command vel follows curriculum
    command_vel = CurrTerm(func=mdp.modify_command_velocity,
                           params={"term_name": "track_lin_vel_xy_exp", "max_velocity": [-1.5, 3.0],
                                   "interval": 200 * 24, "starting_step": 5000 * 24})


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
