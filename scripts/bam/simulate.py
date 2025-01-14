"""Simulation adaptation.

# friction, stiffness, damping can be set each time articulation is called

# armature is set at the start of the simulation
#IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/assets/articulation/articulation.py#L486
# update armature value
# if write_joint_armature_to_sim is not specified updates all envs and joints
env.env.env.scene._articulations["robot"].write_joint_armature_to_sim(model.armature.value)
"""
import math
import torch
import numpy as np
from rsl_rl.runners import OnPolicyRunner
from dataclasses import dataclass
from typing import List


@dataclass
class Rollout:
    phase_1: float
    phase_2: float
    phase_3: float
    amplitude: float
    frequency: float
    joint_position_1: List[np.ndarray]
    joint_position_2: List[np.ndarray]
    joint_position_3: List[np.ndarray]
    actions: List[np.ndarray]


def sin_func(timestep, rollout_data, env, decimation_factor=4) -> torch.Tensor:
    """Generate actions using superimposed sine waves.
    
    Args:
        obs: Current observation
        timestep: Current timestep
        rollout_data: Rollout dataclass containing sine wave parameters
        env: Environment instance to get simulation dt
    
    Returns:
        torch.Tensor: Actions generated from sine waves
    """
    # Get simulation dt from physics
    dt = env.unwrapped.physics_dt * decimation_factor

    # Time variable for sine waves using actual dt
    t = timestep * dt

    signs = [1, 1, 1]

    # Generate sine waves for each component
    wave1 = signs[0] * np.deg2rad(rollout_data.amplitude * math.sin(2 * np.pi * rollout_data.frequency * t + np.deg2rad(rollout_data.phase_1)))
    wave2 = signs[1] * np.deg2rad(rollout_data.amplitude * math.sin(2 * np.pi * rollout_data.frequency * t + np.deg2rad(rollout_data.phase_2)))
    wave3 = signs[2] * np.deg2rad(rollout_data.amplitude * math.sin(2 * np.pi * rollout_data.frequency * t + np.deg2rad(rollout_data.phase_3)))

    actions = torch.tensor([wave1, wave2, wave3]).unsqueeze(0)

    return actions


def set_model_parameters(env, model):
    """Set the model parameters for the environment.

    Args:
        env: The environment.
        model: The model.
    
    Returns:
        The environment with the model parameters set.
    """
    print("Current Parameters", model.damping.value, model.stiffness.value, model.friction_static.value, model.friction_dynamic.value, model.armature.value)
    env_actuators = env.env.unwrapped.scene._articulations["robot"].actuators["pendulum_actuators"]

    # set damping 
    new_damping_tensor = torch.full_like(env_actuators.damping, model.damping.value)
    env_actuators.damping = new_damping_tensor

    # set stiffness 
    new_stiffness_tensor = torch.full_like(env_actuators.stiffness, model.stiffness.value)
    env_actuators.stiffness = new_stiffness_tensor

    # set friction
    new_friction_static_tensor = torch.full_like(env_actuators.friction_static, model.friction_static.value)
    env_actuators.friction_static = new_friction_static_tensor
    new_friction_dynamic_tensor = torch.full_like(env_actuators.friction_dynamic, model.friction_dynamic.value)
    env_actuators.friction_dynamic = new_friction_dynamic_tensor

    # set armature - armature
    env_armature = env.env.unwrapped.scene._articulations["robot"].actuators["pendulum_actuators"].armature
    new_armature_tensor = torch.full_like(env_armature, model.armature.value)
    env.env.unwrapped.scene._articulations["robot"].write_joint_armature_to_sim(new_armature_tensor)

    return env


def rollout(simulation_app, agent_cfg, env, model, resume_path, 
            rollout_length=100, observed_data=None) -> Rollout:
    """Rollout the environment with the model.

    Args:
        simulation_app: The simulation app.
        agent_cfg: The agent configuration.
        env: The environment.
        model: The model.
        resume_path: The path to the resume model.
        rollout_length: The length of the rollout.
        observed_data: The observed data.
    
    Returns:
        The rollout data.
    """
    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env, 
        agent_cfg.to_dict(), 
        log_dir=None, 
        device=agent_cfg.device
    )
    ppo_runner.load(resume_path)

    # reset environment
    with torch.inference_mode():
        env.reset()
    env = set_model_parameters(env, model)

    timestep = 0

    # rollout_data = Rollout(
    #     joint_position_1=[], joint_position_2=[], joint_position_3=[], 
    #     actions=[],
    #     phase_1=observed_data["phase_offset"] * 0, 
    #     phase_2=observed_data["phase_offset"] * 1, 
    #     phase_3=observed_data["phase_offset"] * 2, 
    #     amplitude=observed_data["amplitude"], 
    #     frequency=observed_data["frequency"]
    # )

    rollout_data = Rollout(
        joint_position_1=[], joint_position_2=[], joint_position_3=[], 
        actions=[],
        phase_1=observed_data.phase_offset * 0, 
        phase_2=observed_data.phase_offset * 1, 
        phase_3=observed_data.phase_offset * 2, 
        amplitude=observed_data.amplitude, 
        frequency=observed_data.frequency
    )


    while simulation_app.is_running():
        if timestep == rollout_length:
            break

        with torch.inference_mode():
            joint_pos = env.env.unwrapped.scene._articulations["robot"].data.joint_pos.detach().cpu().numpy()
            rollout_data.joint_position_1.append(np.rad2deg(joint_pos[0][0]))
            rollout_data.joint_position_2.append(np.rad2deg(joint_pos[0][1]))
            rollout_data.joint_position_3.append(np.rad2deg(joint_pos[0][2]))
            if observed_data is not None:
                actions = torch.tensor(
                    [
                        np.deg2rad(observed_data.data[timestep]["actuators"]["1"]["commanded_state"]), 
                        np.deg2rad(observed_data.data[timestep]["actuators"]["2"]["commanded_state"]), 
                        np.deg2rad(observed_data.data[timestep]["actuators"]["3"]["commanded_state"])
                    ]
                ).reshape(1, 3)
            else:
                actions = sin_func(timestep, rollout_data, env)

            _, _, _, _ = env.step(actions)
            rollout_data.actions.append(actions.detach().cpu().numpy())

        timestep += 1

    return rollout_data
