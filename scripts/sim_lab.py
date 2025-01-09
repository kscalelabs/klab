"""Mujoco validaiton.

Run:
    python sim/sim_lab.py --load_model lab_model/policy.onnx
"""
import argparse
import numpy as np
import yaml
from copy import deepcopy
from tqdm import tqdm
from typing import Dict

import mujoco
import mujoco_viewer
import onnx
import onnxruntime as ort
import mediapy as media


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
    def __init__(self, embodiment: str, policy: ort.InferenceSession, config: Dict, render: bool = True):
        """
        Initialize the MuJoCo runner.

        Args:
            embodiment: The name of the embodiment
            policy: The policy used for controlling the simulation
            config: The configuration object containing simulation settings
            render: Whether to render the simulation
        """
        self.policy = policy
        self.render = render
        self.frames = []
        self.framerate = 30
        
        # Initialize model
        mujoco_model_path = f"resources/{embodiment}/robot_fixed.xml"
        num_actions = len(config["observations"]["policy"]["joint_angles"]["params"]["asset_cfg"]["joint_names"])
        
        self.model_info = {
            "sim_dt": config["sim"]["dt"],
            "sim_decimation": config["decimation"],
            "tau_factor": [1] * num_actions,
            "num_actions": num_actions,
            "robot_effort": [config["scene"]["robot"]["actuators"]["zbot2_actuators"]["effort_limit"]] * num_actions,
            "robot_stiffness": [config["scene"]["robot"]["actuators"]["zbot2_actuators"]["stiffness"][".*"]] * num_actions,
            "robot_damping": [config["scene"]["robot"]["actuators"]["zbot2_actuators"]["damping"][".*"]] * num_actions,
        }
        
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.model.opt.timestep = self.model_info["sim_dt"]
        self.data = mujoco.MjData(self.model)
        
        # Set up control parameters
        self.tau_limit = np.array(self.model_info["robot_effort"]) * self.model_info["tau_factor"]
        self.kps = np.array(self.model_info["robot_stiffness"])
        self.kds = np.array(self.model_info["robot_damping"])
        
        # Initialize default position
        try:
            self.data.qpos = self.model.keyframe("default").qpos
            self.default = deepcopy(self.model.keyframe("default").qpos)[-self.model_info["num_actions"]:]
            print("Default position:", self.default)
        except:
            print("No default position found, using zero initialization")
            self.default = np.zeros(self.model_info["num_actions"])
        
        # Set up joint mappings
        self._setup_joint_mappings(config)
        
        # Initialize simulation state
        self.data.qvel = np.zeros_like(self.data.qvel)
        self.data.qacc = np.zeros_like(self.data.qacc)
        
        # Initialize viewer
        if self.render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, "offscreen")
        
        # Initialize control variables
        self.target_q = np.zeros((self.model_info["num_actions"]), dtype=np.double)
        self.last_action = np.zeros((self.model_info["num_actions"]), dtype=np.double)
        self.count_lowlevel = 0
        
    def _setup_joint_mappings(self, config):
        """Set up mappings between MuJoCo and Isaac joint names."""
        mujoco_joints_names = []
        mujoco.mj_step(self.model, self.data)
        for ii in range(1, len(self.data.ctrl) + 1):
            print(self.data.joint(ii).id, self.data.joint(ii).name)
            mujoco_joints_names.append(self.data.joint(ii).name)

        isaac_joints_names = config["observations"]["policy"]["joint_angles"]["params"]["asset_cfg"]["joint_names"]

        self.mujoco_to_isaac_idx = {
            mujoco_joints_names[ii]: isaac_joints_names.index(mujoco_joints_names[ii]) 
            for ii in range(len(mujoco_joints_names))
        }
        self.isaac_to_mujoco_idx = {
            isaac_joints_names[ii]: mujoco_joints_names.index(isaac_joints_names[ii]) 
            for ii in range(len(isaac_joints_names))
        }
        
    def step(self, x_vel_cmd: float, y_vel_cmd: float, yaw_vel_cmd: float):
        """
        Execute one step of the simulation.
        
        Args:
            x_vel_cmd: X velocity command
            y_vel_cmd: Y velocity command
            yaw_vel_cmd: Yaw velocity command
        """
        q = self.data.qpos[-self.model_info["num_actions"]:][list(self.mujoco_to_isaac_idx.values())]
        dq = self.data.qvel[-self.model_info["num_actions"]:][list(self.mujoco_to_isaac_idx.values())]
        projected_gravity = get_gravity_orientation(self.data.sensor("orientation").data)
        
        if self.count_lowlevel % self.model_info["sim_decimation"] == 0:
            obs = np.concatenate([
                np.array([x_vel_cmd, y_vel_cmd, yaw_vel_cmd], dtype=np.float32).reshape(-1),
                np.array([projected_gravity], dtype=np.float32).reshape(-1),
                q.astype(np.float32),
                dq.astype(np.float32),
                self.last_action.astype(np.float32)
            ])
            
            input_name = self.policy.get_inputs()[0].name
            curr_actions = self.policy.run(None, {input_name: obs.reshape(1, -1).astype(np.float32)})[0][0]
            
            self.target_q = curr_actions[list(self.isaac_to_mujoco_idx.values())]
            self.last_action = curr_actions.copy()
            
            if self.render:
                self.viewer.render()
            else:
                self.frames.append(self.viewer.read_pixels())
        
        # Generate PD control
        tau = self.kps * (self.target_q + self.default - q) - self.kds * dq
        # Clamp torques
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)
        
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
    parser.add_argument("--embodiment", type=str, default="zbot2", help="Embodiment name.")
    parser.add_argument("--sim_duration", type=float, default=10.0, help="Simulation duration in seconds.")
    parser.set_defaults(render=True)
    args = parser.parse_args()

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.2, 0.0, 0.0

    policy = onnx.load("example_model/exported/policy.onnx")

    # In the run_mujoco function, replace the policy.run line with:
    session = ort.InferenceSession(policy.SerializeToString())

    with open("example_model/params/env.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    runner = Runner(
        embodiment=args.embodiment,
        policy=session,
        config=config,
        render=args.render
    )

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.2, 0.0, 0.0
    
    for _ in tqdm(range(int(args.sim_duration / config["sim"]["dt"])), desc="Simulating..."):
        runner.step(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)

    runner.save_video()
    runner.close()
