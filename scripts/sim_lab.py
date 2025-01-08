"""Mujoco validaiton.

Run:
    python sim/sim_lab.py --load_model lab_model/policy.onnx --embodiment zbot2
"""
import argparse
import numpy as np
import os
import yaml
from copy import deepcopy
from tqdm import tqdm
from typing import Dict

import mujoco
import mujoco_viewer
import onnx
import onnxruntime as ort
import mediapy as media


def run(
    embodiment: str,
    policy: ort.InferenceSession,
    config: Dict,
    render: bool = True,
    sim_duration: float = 10.0,
) -> None:
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        embodiment: The name of the embodiment.
        policy: The policy used for controlling the simulation.
        config: The configuration object containing simulation settings.
        render: Whether to render the simulation.
        sim_duration: The duration of the simulation in seconds.
    """
    model_dir = os.environ.get("MODEL_DIR", "sim/resources")
    mujoco_model_path = f"{model_dir}/{embodiment}/robot_fixed.xml"

    model = mujoco.MjModel.from_xml_path(mujoco_model_path)

    model_info = {
        "sim_dt": config["sim"]["dt"],
        "sim_decimation": config["decimation"],
        "tau_factor": [3]*10,
        "num_actions": 10,
        "num_observations": 39,
        "robot_effort": [1]*10,
        "robot_stiffness": [17.68]*10,
        "robot_damping": [0.53]*10,
    }
    model.opt.timestep = model_info["sim_dt"]
    data = mujoco.MjData(model)

    tau_limit = np.array(model_info["robot_effort"]) * model_info["tau_factor"]
    kps = np.array(model_info["robot_stiffness"])
    kds = np.array(model_info["robot_damping"])

    try:
        data.qpos = model.keyframe("default").qpos
        default = deepcopy(model.keyframe("default").qpos)[-model_info["num_actions"] :]
        print("Default position:", default)
    except:
        print("No default position found, using zero initialization")
        default = np.zeros(model_info["num_actions"])  # 3 for pos, 4 for quat, cfg.num_actions for joints

    mujoco.mj_step(model, data)
    for ii in range(len(data.ctrl) + 1):
        print(data.joint(ii).id, data.joint(ii).name)

    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)

    frames = []
    framerate = 30
    if render:
        viewer = mujoco_viewer.MujocoViewer(model, data)
    else:
        viewer = mujoco_viewer.MujocoViewer(model, data, "offscreen")

    target_q = np.zeros((model_info["num_actions"]), dtype=np.double)
    last_action = np.zeros((model_info["num_actions"]), dtype=np.double)
    count_lowlevel = 0

    for _ in tqdm(range(int(sim_duration / model_info["sim_dt"])), desc="Simulating..."):
        # Obtain an observation
        mujoco_to_isaac = [1,0,2,3,4,6,5,7,8,9]
        q = data.qpos[-model_info["num_actions"] :][mujoco_to_isaac]
        dq = data.qvel[-model_info["num_actions"] :][mujoco_to_isaac]
        accelerometer = data.sensor("linear-acceleration").data
        gyroscope = data.sensor("angular-velocity").data

        # 1000hz -> 250hz
        if count_lowlevel % model_info["sim_decimation"] == 0:
            obs = np.concatenate(
                [
                    np.array([x_vel_cmd, y_vel_cmd, yaw_vel_cmd], dtype=np.float32).reshape(-1), 
                    np.array([accelerometer], dtype=np.float32).reshape(-1), 
                    np.array([gyroscope], dtype=np.float32).reshape(-1), 
                    q.astype(np.float32), 
                    dq.astype(np.float32), 
                    last_action.astype(np.float32)
                ]
            )

            input_name = policy.get_inputs()[0].name
            curr_actions = policy.run(None, {input_name: obs.reshape(1, -1).astype(np.float32)})[0][0]

            target_q = curr_actions
            last_action = curr_actions.copy()

            if render:
                viewer.render()
            else:
                frames.append(viewer.read_pixels())
        # Generate PD control
        tau = kps * (target_q + default - q) - kds * dq
        # Clamp torques
        tau = np.clip(tau, -tau_limit, tau_limit)  

        data.ctrl = tau
        mujoco.mj_step(model, data)


        count_lowlevel += 1

    if render:
        viewer.close()

    media.write_video("episode.mp4", frames, fps=framerate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--embodiment", type=str, required=True, help="Embodiment name.")
    parser.set_defaults(render=True)
    args = parser.parse_args()

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.1, 0.0, 0.0

    policy = onnx.load("example_model/exported/policy.onnx")

    # In the run_mujoco function, replace the policy.run line with:
    session = ort.InferenceSession(policy.SerializeToString())

    with open("example_model/params/env.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    run(
        embodiment=args.embodiment,
        policy=session,
        config=config,
        render=False,
    )
