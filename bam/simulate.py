"""This file contains the code for simulating the environment.

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


def set_model_parameters(env, model):
    breakpoint()
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


def rollout(simulation_app, agent_cfg, rollout_data, env, model, resume_path):
    """Rollout the environment with the model.

    Args:
        simulation_app: The simulation app.
        agent_cfg: The agent configuration.
        rollout_data: The rollout data.
        env: The environment.
        model: The model.
        resume_path: The path to the resume model.
    
    Returns:
        The rollout data.
    """
    env = set_model_parameters(env, model)

    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env, 
        agent_cfg.to_dict(), 
        log_dir=None, 
        device=agent_cfg.device
    )
    ppo_runner.load(resume_path)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    while simulation_app.is_running():
        if timestep == 100:
            with torch.inference_mode():
                env.reset()
                break
        with torch.inference_mode():
            actions = sin_func(obs, timestep, rollout_data, env)
            
            obs, _, _, _ = env.step(actions)
            rollout_data.obs.append(obs.detach().cpu().numpy())
            rollout_data.actions.append(actions.detach().cpu().numpy())
            rollout_data.rewards.append(np.array([0.0]))
            rollout_data.dones.append(np.array([False]))
            rollout_data.infos.append(np.array([{}]))

        timestep += 1

    # TODO test that
    # env.close()

    return rollout_data


def sin_func(obs, timestep, rollout_data, env):
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
    dt = env.unwrapped.physics_dt

    # Time variable for sine waves using actual dt
    t = timestep * dt
    
    # Generate sine waves for each component
    wave1 = rollout_data.amplitude_1 * torch.sin(2 * np.pi * rollout_data.frequency_1 * t + rollout_data.phase_1)
    wave2 = rollout_data.amplitude_2 * torch.sin(2 * np.pi * rollout_data.frequency_2 * t + rollout_data.phase_2)
    wave3 = rollout_data.amplitude_3 * torch.sin(2 * np.pi * rollout_data.frequency_3 * t + rollout_data.phase_3)
    
    actions = torch.cat([wave1, wave2, wave3], dim=0)
    
    return actions


def compute_sine_wave(amplitude_deg, frequency, t):
    sine_pos_radians = torch.zeros(3)
    num_actuators = 3
    for idx in range(num_actuators):
        # Convert phase offset from degrees to radians and apply
        phase = math.radians(idx)
        # Add sine wave to initial position
        sine_pos_degree = amplitude_deg * math.sin(2 * math.pi * frequency * t + phase)
        sine_pos_radians[idx] = math.radians(sine_pos_degree)
    return sine_pos_radians