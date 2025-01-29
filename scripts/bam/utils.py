"""Utility functions for plotting."""

import matplotlib.pyplot as plt


def plot_positions(result, real_positions, timesteps, commanded_positions=None, steps=0) -> None:
    """Plot the positions of the simulated and real joints.

    Args:
        result (Rollout): The result of the simulation.
        real_positions (dict): The real positions of the joints.
        timesteps (int): The number of timesteps to plot.
        commanded_positions (dict): The commanded positions of the joints.
        steps (int): The number of steps to plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Left subplot - Simulated vs Real positions
    ax1.plot(result.joint_position_1, label='Sim joint_1', color='red')
    ax1.plot(result.joint_position_2, label='Sim joint_2', color='green')
    ax1.plot(result.joint_position_3, label='Sim joint_3', color='blue')
    ax1.plot(real_positions["1"][steps:steps + timesteps], label='Real joint_1', color='orange', linestyle='--')
    ax1.plot(real_positions["2"][steps:steps + timesteps], label='Real joint 2', color='black', linestyle='--')
    ax1.plot(real_positions["3"][steps:steps + timesteps], label='Real joint 3', color='purple', linestyle='--')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Position')
    ax1.set_title('Comparison of simulated vs logged positions')
    ax1.legend()
    ax1.grid(True)

    # Right subplot - Simulated vs Commanded positions
    ax2.plot(result.joint_position_1, label='Sim joint_1', color='red')
    ax2.plot(result.joint_position_2, label='Sim joint_2', color='green')
    ax2.plot(result.joint_position_3, label='Sim joint_3', color='blue')
    ax2.plot(commanded_positions["1"][steps:steps + timesteps], label='Commanded joint 1', color='red', linestyle='-.')
    ax2.plot(commanded_positions["2"][steps:steps + timesteps], label='Commanded joint 2', color='green', linestyle='-.')
    ax2.plot(commanded_positions["3"][steps:steps + timesteps], label='Commanded joint 3', color='blue', linestyle='-.')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position')
    ax2.set_title('Comparison of simulated vs commanded positions')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save and close
    plt.savefig('simulation_comparison.png')
    plt.close()

