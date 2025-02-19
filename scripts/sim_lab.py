"""Mujoco validaiton."""
import argparse
import numpy as np
import yaml
from copy import deepcopy
from tqdm import tqdm
from typing import Dict
import os
import time

import mujoco
import mujoco_viewer
import onnx
import onnxruntime as ort
import mediapy as media
import logging

import math


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_gravity_orientation(quaternion):
    """
    Args:
        quaternion: np.ndarray[float, float, float, float]
    
    Returns:
        gravity_orientation: np.ndarray[float, float, float]
    """
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


class Runner:
    def __init__(
        self,
        embodiment: str,
        policy: ort.InferenceSession,
        config: Dict,
        render: bool = True,
        terrain: bool = False,
        in_the_air: bool = False,
    ):
        """
        Initialize the MuJoCo runner.

        Args:
            embodiment: The name of the embodiment
            policy: The policy used for controlling the simulation
            config: The configuration object containing simulation settings
            render: Whether to render the simulation
            terrain: Whether to render the terrain
        """
        self.policy = policy
        self.render = render
        self.frames = []
        self.framerate = 30
        self.in_the_air = in_the_air

        # Initialize model
        if terrain:
            mujoco_model_path = f"resources/{embodiment}/robot_terrain.xml"
        elif in_the_air:
            mujoco_model_path = f"resources/{embodiment}/robot_air.xml"
        else:
            mujoco_model_path = f"resources/{embodiment}/robot.xml"
        
        logger.info(f"Using robot file: {os.path.abspath(mujoco_model_path)}")
        
        num_actions = 20  # Fixed number of actions for kbot
        
        # Set up joint mappings first so we have access to joint names
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.data = mujoco.MjData(self.model)
        self._setup_joint_mappings(config)
        
        # Get actuator configs
        kbot_actuators = config["scene"]["robot"]["actuators"]
        kbot_04_cfg = kbot_actuators["kbot_04"]
        kbot_03_cfg = kbot_actuators["kbot_03"]
        kbot_02_cfg = kbot_actuators["kbot_02"]

        # Initialize arrays for actuator parameters
        robot_effort = np.zeros(num_actions)
        robot_stiffness = np.zeros(num_actions)
        robot_damping = np.zeros(num_actions)
        friction_static = np.zeros(num_actions)
        friction_dynamic = np.zeros(num_actions)
        activation_vel = np.zeros(num_actions)

        def get_actuator_cfg_for_joint(name: str):
            """Return actuator config block based on joint name suffix."""
            if name.endswith("_04"):
                return kbot_04_cfg
            elif name.endswith("_03"):
                return kbot_03_cfg
            elif name.endswith("_02"):
                return kbot_02_cfg
            else:
                raise ValueError(f"Joint name {name} does not match _02, _03, or _04 suffix")

        # Get the joint names in MuJoCo order
        mujoco_joint_names = []
        if self.in_the_air:
            for ii in range(0, len(self.data.ctrl)):
                mujoco_joint_names.append(self.data.joint(ii).name)
        else:
            for ii in range(1, len(self.data.ctrl) + 1):
                mujoco_joint_names.append(self.data.joint(ii).name)

        # Fill arrays with correct parameters for each joint
        for i, joint_name in enumerate(mujoco_joint_names):
            cfg = get_actuator_cfg_for_joint(joint_name)
            # Add small epsilon to friction values to avoid numerical issues
            FRICTION_EPS = 1e-6
            
            robot_effort[i] = cfg["effort_limit"]
            robot_stiffness[i] = cfg["stiffness"][".*"]
            robot_damping[i] = cfg["damping"][".*"]
            friction_static[i] = cfg["friction_static"] + FRICTION_EPS
            friction_dynamic[i] = cfg["friction_dynamic"] + FRICTION_EPS
            activation_vel[i] = cfg["activation_vel"]

            # Print parameters for this joint
            logger.debug(f"\nJoint {joint_name} parameters:")
            logger.debug(f"  Effort limit: {robot_effort[i]}")
            logger.debug(f"  Stiffness: {robot_stiffness[i]}")
            logger.debug(f"  Damping: {robot_damping[i]}")
            logger.debug(f"  Static friction: {friction_static[i]}")
            logger.debug(f"  Dynamic friction: {friction_dynamic[i]}")
            logger.debug(f"  Activation velocity: {activation_vel[i]}")

        self.model_info = {
            "sim_dt": config["sim"]["dt"],
            "sim_decimation": config["decimation"],
            "tau_factor": [1] * num_actions,
            "num_actions": num_actions,
            "robot_effort": robot_effort,
            "robot_stiffness": robot_stiffness,
            "robot_damping": robot_damping,
            "action_scale": config["actions"]["joint_pos"]["scale"],
            "friction_static": friction_static,
            "friction_dynamic": friction_dynamic,
            "activation_vel": activation_vel,
        }
        logger.info(f"Action scale: {self.model_info['action_scale']}")
        self.model.opt.timestep = self.model_info["sim_dt"]
        
        # Set up control parameters
        self.tau_limit = np.array(self.model_info["robot_effort"]) * self.model_info["tau_factor"]
        self.kps = np.array(self.model_info["robot_stiffness"])
        self.kds = np.array(self.model_info["robot_damping"])
        
        # Initialize default position
        try:
            self.data.qpos = self.model.keyframe("default").qpos
            self.default = deepcopy(self.model.keyframe("default").qpos)[-self.model_info["num_actions"]:]
            logging.info(f"Default position: {self.default}")
        except:
            logger.warning("No default position found, using zero initialization")
            self.default = np.zeros(self.model_info["num_actions"])
        
        # Initialize simulation state
        self.data.qvel = np.zeros_like(self.data.qvel)
        self.data.qacc = np.zeros_like(self.data.qacc)
        
        # Initialize viewer
        if self.render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, "offscreen")
        self.viewer.cam.distance += 2.0  # Zoom out the render camera

        # Initialize control variables
        self.target_q = np.zeros((self.model_info["num_actions"]), dtype=np.double)
        self.last_action = np.zeros((self.model_info["num_actions"]), dtype=np.double)
        self.count_lowlevel = 0
        logger.debug(f"Model info: {self.model_info}")
        
    def _setup_joint_mappings(self, config):
        """Set up mappings between MuJoCo and Isaac joint names."""
        mujoco_joint_names = []
        mujoco.mj_step(self.model, self.data)

        if self.in_the_air:
            for ii in range(0, len(self.data.ctrl)):
                mujoco_joint_names.append(self.data.joint(ii).name)
        else:
            for ii in range(1, len(self.data.ctrl) + 1):
                mujoco_joint_names.append(self.data.joint(ii).name)

        # TODO - load this at runtime
        isaac_joint_names = [
            "left_hip_pitch_04", 
            "left_shoulder_pitch_03", 
            "right_hip_pitch_04", 
            "right_shoulder_pitch_03", 
            "left_hip_roll_03", 
            "left_shoulder_roll_03", 
            "right_hip_roll_03", 
            "right_shoulder_roll_03", 
            "left_hip_yaw_03", 
            "left_shoulder_yaw_02", 
            "right_hip_yaw_03", 
            "right_shoulder_yaw_02", 
            "left_knee_04",
            "left_elbow_02", 
            "right_knee_04", 
            "right_elbow_02", 
            "left_ankle_02", 
            "left_wrist_02", 
            "right_ankle_02", 
            "right_wrist_02"
        ]

        print(mujoco_joint_names)

        for ii in range(len(mujoco_joint_names)):
            logging.info(f"{mujoco_joint_names[ii]} -> {isaac_joint_names[ii]}")

        # Create mappings
        self.mujoco_to_isaac_mapping = {
            mujoco_joint_names[i]: isaac_joint_names[i]
            for i in range(len(mujoco_joint_names))
        }
        self.isaac_to_mujoco_mapping = {
            isaac_joint_names[i]: mujoco_joint_names[i]
            for i in range(len(isaac_joint_names))
        }
        # TODO - update the values
        # self.test_mappings()

    def map_isaac_to_mujoco(self, isaac_position):
        """
        Maps joint positions from Isaac format to MuJoCo format.
        
        Args:
            isaac_position (np.array): Joint positions in Isaac order.
        
        Returns:
            np.array: Joint positions in MuJoCo order.
        """
        mujoco_position = np.zeros(len(self.isaac_to_mujoco_mapping))
        for isaac_index, isaac_name in enumerate(self.isaac_to_mujoco_mapping.keys()):
            mujoco_index = list(self.isaac_to_mujoco_mapping.values()).index(isaac_name)
            mujoco_position[mujoco_index] = isaac_position[isaac_index]
        return mujoco_position

    def map_mujoco_to_isaac(self, mujoco_position):
        """
        Maps joint positions from MuJoCo format to Isaac format.
        
        Args:
            mujoco_position (np.array): Joint positions in MuJoCo order.
        
        Returns:
            np.array: Joint positions in Isaac order.
        """
        # Create an array for Isaac positions based on the mapping
        isaac_position = np.zeros(len(self.mujoco_to_isaac_mapping))
        for mujoco_index, mujoco_name in enumerate(self.mujoco_to_isaac_mapping.keys()):
            isaac_index = list(self.mujoco_to_isaac_mapping.values()).index(mujoco_name)
            isaac_position[isaac_index] = mujoco_position[mujoco_index]
        return isaac_position

    def test_mappings(self):
        mujoco_position = np.array([
            -0.9, -0.9,
            0.2, 0.2,
            -0.377, 0.377,
            0.796, -0.796,
            0.377, -0.377,
            0.1, -0.1,
            0.2, -0.2,
            0.3, -0.3,
            0.4, -0.4,
            0.5, -0.5
        ])
        isaac_position = np.array([
            -0.9, -0.9,  # hip pitch
            0.2, 0.2,    # hip roll
            -0.377, 0.377,  # hip yaw
            0.796, -0.796,  # knee
            0.377, -0.377,  # ankle
            0.1, -0.1,  # shoulder pitch
            0.2, -0.2,  # shoulder roll
            0.3, -0.3,  # shoulder yaw
            0.4, -0.4,  # elbow
            0.5, -0.5   # wrist
        ])

        # Map positions
        isaac_to_mujoco = self.map_isaac_to_mujoco(isaac_position)
        mujoco_to_isaac = self.map_mujoco_to_isaac(mujoco_position)

        # Print mappings in a side-by-side format for easy comparison
        print("\nMapping Comparison:")
        print("-" * 80)
        print(f"{'Index':<6}{'MuJoCo':<25}{'Isaac':<25}{'MuJoCo Value':<15}{'Isaac Value':<15}")
        print("-" * 80)
        for i in range(len(mujoco_position)):
            mujoco_name = list(self.mujoco_to_isaac_mapping.keys())[i]
            isaac_name = self.mujoco_to_isaac_mapping[mujoco_name]
            print(f"{i:<6}{mujoco_name:<25}{isaac_name:<25}{mujoco_position[i]:>12.3f}{isaac_to_mujoco[i]:>15.3f}")
        print("-" * 80)

        assert np.allclose(mujoco_position, isaac_to_mujoco)
        assert np.allclose(isaac_position, mujoco_to_isaac)

    def step(self, x_vel_cmd: float, y_vel_cmd: float, yaw_vel_cmd: float):
        """
        Execute one step of the simulation.

        Args:
            x_vel_cmd: X velocity command
            y_vel_cmd: Y velocity command
            yaw_vel_cmd: Yaw velocity command
        """
        # Last 20 dof are our "active" joints
        q = self.data.qpos[-self.model_info["num_actions"]:]
        dq = self.data.qvel[-self.model_info["num_actions"]:]

        # Read orientation sensor -> pass to get_gravity_orientation
        # Make sure the sensor name "orientation" is correct in your XML
        # Try both possible sensor names for orientation
        try:
            orientation_quat = self.data.sensor("orientation").data  # shape (4,)
        except:
            try:
                orientation_quat = self.data.sensor("base_link_quat").data  # shape (4,)
            except:
                raise Exception("Could not find orientation sensor - tried names 'orientation' and 'base_link_quat'")
        projected_gravity = get_gravity_orientation(orientation_quat)

        # Get IMU readings
        imu_ang_vel = self.data.sensor("angular-velocity").data
        imu_lin_acc = self.data.sensor("linear-acceleration").data

        # Debug IMU values periodically
        if self.count_lowlevel % 100 == 0:  # Print every 100 steps to avoid spam
            logger.debug(f"\nIMU Debug:")
            logger.debug(f"Angular velocity (rad/s): {imu_ang_vel}")
            logger.debug(f"Linear acceleration (m/s²): {imu_lin_acc}")
            logger.debug(f"Gravity projection: {projected_gravity}")

        # Build the observation only if it's time to do policy inference
        if self.count_lowlevel % self.model_info["sim_decimation"] == 0:
            # Position offset from default
            cur_pos_obs = q - self.default
            # Joint velocities
            cur_vel_obs = dq

            # 3D velocity commands
            vel_cmd = np.array([x_vel_cmd, y_vel_cmd, yaw_vel_cmd], dtype=np.float32)

            # Map from MuJoCo -> Isaac indexing
            cur_pos_isaac = self.map_mujoco_to_isaac(cur_pos_obs).astype(np.float32)
            cur_vel_isaac = self.map_mujoco_to_isaac(cur_vel_obs).astype(np.float32)
            last_act_isaac = self.last_action.astype(np.float32)

            # Now build the entire 72-D observation:
            #   3 (vel_cmd) + 3 (imu_ang_vel) + 3 (imu_lin_acc) + 3 (proj_grav)
            #   + 20 (joint pos) + 20 (joint vel) + 20 (last_action) = 72
            obs = np.concatenate([
                vel_cmd,                           # 3
                # imu_lin_acc.astype(np.float32),   # 3
                # imu_ang_vel.astype(np.float32),   # 3
                projected_gravity.astype(np.float32),  # 3
                cur_pos_isaac,                     # 20
                cur_vel_isaac,                     # 20
                last_act_isaac                     # 20
            ])


            # Run the ONNX policy
            input_name = self.policy.get_inputs()[0].name
            curr_actions = self.policy.run(
                None, {input_name: obs.reshape(1, -1).astype(np.float32)}
            )[0][0]  # shape (20,)


            # Update last_action
            self.last_action = curr_actions.copy()

            # Scale actions, then map Isaac->MuJoCo indexing
            curr_actions_scaled = curr_actions * self.model_info["action_scale"]
            self.target_q = self.map_isaac_to_mujoco(curr_actions_scaled)

            # Render if needed
            if self.render:
                self.viewer.render()
            else:
                # Offscreen mode -> record frames for a video
                self.frames.append(self.viewer.read_pixels(camid=1))

        # PD control for the step
        tau = (
            self.kps * (self.default + self.target_q - q)
            - self.kds * dq
        )
        # add friction logic from Isaac Lab
        tau -= (
            self.model_info["friction_static"] * np.tanh(dq / self.model_info["activation_vel"])
            + self.model_info["friction_dynamic"] * dq
        )

        # Apply torques & step
        self.data.ctrl = tau
        mujoco.mj_step(self.model, self.data)

        self.count_lowlevel += 1


    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
        
    def save_video(self, filename: str = "episode.mp4"):
        """Save recorded frames as a video file."""
        if not self.render and self.frames:
            media.write_video(filename, self.frames, fps=self.framerate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--embodiment", type=str, default="kbot", help="Embodiment name.")
    parser.add_argument("--sim_duration", type=float, default=5, help="Simulation duration in seconds.")
    parser.add_argument("--model_path", type=str, default="example_model", help="Model path.")
    parser.add_argument("--terrain", action="store_true", help="Render the terrain.")
    parser.add_argument("--air", action="store_true", help="Run in the air.")
    parser.add_argument("--render", action="store_true", help="Render the terrain.")
    args = parser.parse_args()

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = -0.0, 0.0, 0.0

    # Get the most recent yaml and onnx files from the checkpoint directory
    yaml_files = [f for f in os.listdir(args.model_path) if f.endswith('env.yaml')]
    policy_files = [f for f in os.listdir(args.model_path) if f.endswith('.onnx')]
    
    if not yaml_files or not policy_files:
        raise FileNotFoundError(f"Could not find env.yaml and .onnx files in {args.model_path}")
        
    yaml_file = yaml_files[0]  # Use first found yaml
    policy_file = policy_files[0]  # Use first found onnx
    
    policy_path = os.path.join(args.model_path, policy_file)
    yaml_path = os.path.join(args.model_path, yaml_file)
    
    logger.info(f"Loading policy from: {os.path.abspath(policy_path)}")
    logger.info(f"Loading config from: {os.path.abspath(yaml_path)}")
    
    policy = onnx.load(policy_path)
    session = ort.InferenceSession(policy.SerializeToString())

    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    runner = Runner(
        embodiment=args.embodiment,
        policy=session,
        config=config,
        render=args.render,
        terrain=args.terrain,
        in_the_air=args.air
    )
    
    for _ in tqdm(range(int(args.sim_duration / config["sim"]["dt"])), desc="Simulating..."):
        runner.step(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)

    # Create mujoco_videos directory in model path if it doesn't exist
    logger.info(f"Saving video...")
    video_dir = os.path.join(args.model_path, "mujoco_videos")
    os.makedirs(video_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_path = os.path.join(video_dir, f"sim_video_{timestamp}.mp4")
    runner.save_video(video_path)
    logger.info(f"Saved video to: {os.path.abspath(video_path)}")

    runner.close()