"""This file contains the code for simulating the environment.

# friction, stiffness, damping can be set each time articulation is called

# armature is set at the start of the simulation
# IsaacLab/blob/main/source/extensions/omni.isaac.lab/omni/isaac/lab/assets/articulation/articulation.py#L486
# update armature value
# if write_joint_armature_to_sim is not specified updates all envs and joints
env.env.env.scene._articulations["robot"].write_joint_armature_to_sim(model.armature.value)
"""
import torch
import numpy as np
from rsl_rl.runners import OnPolicyRunner

URDF_NAME = "pendulum.urdf"

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
    env.env.env.scene._articulations[URDF_NAME].write_joint_armature_to_sim(model.armature.value)

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
        if timestep == 10:
            with torch.inference_mode():
                env.reset()
                break
        with torch.inference_mode():
            # TODO wesley ali and passing frictio and armature
            # TODO sin wave logic
            # actions = sin_func(obs)
            actions = torch.rand(2, 10)

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