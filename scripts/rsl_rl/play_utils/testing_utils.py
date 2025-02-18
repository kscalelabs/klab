""" Utility functions for testing """

import torch
import json


def test_joints(actions, timestep, env, env_cfg, obs, joint_test_duration=50):
    """Test the joints of the robot by moving them in positive and negative directions."""
    # Get joint names and indices

    asset = env.unwrapped.scene["robot"]
    root_states = asset.data.default_root_state[0].clone()
    asset.write_root_link_pose_to_sim(root_states[:7])
    asset.write_root_com_velocity_to_sim(root_states[7:13])


    joint_name_to_starting_pos = env_cfg.scene.robot.init_state.joint_pos

    joint_names = env_cfg.observations.policy.joint_pos.params['asset_cfg'].joint_names

    current_joint_pos = obs[0, 6:6+20]

    # Initialize a counter to track joint index
    joint_index = timestep // joint_test_duration % len(joint_names)

    # Determine the direction for the joint action (positive or negative)
    direction = 1 if (timestep // (joint_test_duration // 2)) % 2 == 0 else -1

    # Get the current joint name
    joint_name = joint_names[joint_index]

    # Set the action for the current joint
    actions = torch.zeros_like(actions)

    actions[:, joint_index] = direction * 9.5

    print(f"index: {joint_index}, name: {joint_name}, curr_pos: {current_joint_pos[joint_index]:.2f}, action: {actions[:, joint_index].item():.2f} direction: {'Positive' if direction == 1 else 'Negative'}")


    # breakpoint()

    return actions

