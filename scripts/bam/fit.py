"""Fit the model to the data.

It first loads the simulation launcher that works in the background.
Run:
    export PYTHONPATH=/home/dpsh/isaac_gpr:$PYTHONPATH
    /home/dpsh/IsaacLab/isaaclab.sh -p scripts/rsl_rl/fit.py --task Velocity-Rough-Gpr-Play-v0 --headless
"""

import argparse
import json
from omni.isaac.lab.app import AppLauncher

# local imports
from scripts.rsl_rl import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_dir", type=str, default="scripts/bam/trajectories", help="Directory to save the logs.")
parser.add_argument("--n_trials", type=int, default=500, help="Number of trials to run.")
parser.add_argument("--n_workers", type=int, default=1, help="Number of workers to run.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os

# Import extensions to set up environment tasks
import pendulum.tasks  # noqa: F401

from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# parse configuration
env_cfg = parse_env_cfg(
    args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)
agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

# specify directory for logging experiments
log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
log_root_path = os.path.abspath(log_root_path)
print(f"[INFO] Loading experiment from directory: {log_root_path}")
resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
log_dir = os.path.dirname(resume_path)

# create isaac environment
env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env)

# Import rest of the code
from copy import deepcopy
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import optuna
import sys
import wandb

from scripts.bam.model import Model
from scripts.bam import simulate
from scripts.bam.data_logs import Logs
from scripts.bam import message


# Json params file
params_json_filename = "sts_3215"
if not params_json_filename.endswith(".json"):
    params_json_filename = f"output/params_{params_json_filename}.json"
json.dump({}, open(params_json_filename, "w"))


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


def compute_score(model: Model, observed_data: dict, rollout_length: int = 330) -> float:
    """Compute the score for the model.

    Args:
        model: The model with the parameters to be optimized.
        observed_data: The observed data.
    
    Returns:
        The score.
    """
    result = simulate.rollout(
        simulation_app, agent_cfg, env, model, resume_path, 
        rollout_length=rollout_length, 
        observed_data=observed_data
    )

    real_positions = observed_data.get_actuator_positions("relative_position")
    commanded_positions = observed_data.get_actuator_positions(key="commanded_state")
    timesteps = len(result.joint_position_1)

    plot_positions(result, real_positions, timesteps, commanded_positions)

    overall_score = np.mean(np.abs(np.array(result.joint_position_1) - np.array(real_positions["1"][:timesteps])))
    overall_score += np.mean(np.abs(np.array(result.joint_position_2) - np.array(real_positions["2"][:timesteps])))
    overall_score += np.mean(np.abs(np.array(result.joint_position_3) - np.array(real_positions["3"][:timesteps])))
    
    return overall_score / 3


def compute_scores(model: Model, compute_logs=None) -> float:
    """Compute the scores for the model.

    Args:
        model: The model.
        compute_logs: The logs to compute the scores for.
    
    Returns:
        The scores.
    """
    scores = 0
    for log in compute_logs.logs:
        scores += compute_score(model, log)

    return scores / len(compute_logs.logs)


def objective(trial) -> float:
    """Objective function for the optimization.

    Args:
        trial: The trial.
    
    Returns:
        The score.
    """
    model = Model()

    parameters = model.get_parameters()
    for name in parameters:
        parameter = parameters[name]
        if parameter.optimize:
            parameter.value = trial.suggest_float(name, parameter.min, parameter.max)

    return compute_scores(model, logs)


def monitor(study, trial):
    global last_log, wandb_run
    elapsed = time.time() - last_log

    if elapsed > 0.2:
        last_log = time.time()
        data = deepcopy(study.best_params)
        trial_number = trial.number
        best_value = study.best_value
        wandb_log = {
            "optim/best_value": best_value,
            "optim/trial_number": trial_number,
        }

        model = Model()
        model_parameters = model.get_parameters()
        for key in model_parameters:
            if key not in data:
                data[key] = model_parameters[key].value
        data["model"] = "isaac_actuator"
        data["actuator"] = "sts_3215"

        json.dump(data, open(params_json_filename, "w"))

        print()
        message.bright(f"[Trial {trial_number}, Best score: {best_value}]")
        print(
            message.emphasis(f"Best params found (saved to {params_json_filename}): ")
        )
        for key in data:
            infos, warning = None, None

            if key in model_parameters:
                if model_parameters[key].optimize:
                    infos = f"min: {model_parameters[key].min}, max: {model_parameters[key].max}"
                else:
                    warning = "not optimized"

            message.print_parameter(key, data[key], infos, warning)

            if type(data[key]) == float:
                wandb_log[f"params/{key}"] = data[key]

        if wandb_run is not None:
            wandb.log(wandb_log)
    sys.stdout.flush()


if __name__ == "__main__":
    logs = Logs(args_cli.log_dir)
    last_log = time.time()
    wandb_run = None

    study_name = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study_url = f"sqlite:///study.db"
    sampler = optuna.samplers.CmaEsSampler(
        restart_strategy="bipop"
    )

    def optuna_run(enable_monitoring=True):
        if args_cli.n_workers > 1:
            study = optuna.load_study(study_name=study_name, storage=study_url)
        else:
            study = optuna.create_study(sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        callbacks = []
        if enable_monitoring:
            callbacks = [monitor]
        study.optimize(objective, n_trials=args_cli.n_trials, n_jobs=args_cli.n_workers, callbacks=callbacks)

    optuna_run(True)
    simulation_app.close()
