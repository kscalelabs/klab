"""Fit the model to the data.

Run:
    export PYTHONPATH=/home/dpsh/isaac_gpr:$PYTHONPATH
    /home/dpsh/IsaacLab/isaaclab.sh -p scripts/rsl_rl/fit.py --task Velocity-Rough-Gpr-Play-v0 --headless
"""

import argparse
import json
import pandas as pd
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
from datetime import datetime
import sys
import numpy as np
import json
from copy import deepcopy
import json
import time
import optuna
import wandb
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
from scripts.bam.model import Model
from scripts.bam import simulate
from scripts.bam.logs import Logs
from scripts.bam import message


global COUNTER
COUNTER = 0


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


# Json params file
params_json_filename = "sts_3215"
if not params_json_filename.endswith(".json"):
    params_json_filename = f"output/params_{params_json_filename}.json"
json.dump({}, open(params_json_filename, "w"))


def get_actuator_positions(log: dict) -> list[float]:
    """
    Returns a list of positions for the specified actuator from a single log.
    """
    positions = {"1": [], "2": [], "3": []}

    for actuator_idx in range(1, 4):
        for timestep in log["data"]:
            position = timestep["actuators"][str(actuator_idx)]["relative_position"]
            positions[str(actuator_idx)].append(position)

    return positions


def plot_positions(result, real_positions, timesteps, COUNTER) -> None:
    """Plot the positions of the simulated and real joints.

    Args:
        result (Rollout): The result of the simulation.
        real_positions (dict): The real positions of the joints.
        COUNTER (int): The counter for the plot.
    
    Returns:
        None
    """
    steps = 43 # int(1 / result.frequency / 0.02)
    plt.figure(figsize=(10, 6))
    plt.plot(result.joint_position_1, label='Sim joint_1', color='red')
    plt.plot(result.joint_position_2, label='Sim joint_2', color='red')
    plt.plot(result.joint_position_3, label='Sim joint_3', color='red')
    plt.plot(real_positions["1"][steps:steps + timesteps], label='Real joint_1', color='blue')
    plt.plot(real_positions["2"][steps:steps + timesteps], label='Real joint 2', color='blue')
    plt.plot(real_positions["3"][steps:steps + timesteps], label='Real joint 3', color='blue')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.title('Comparison of Simulated vs Logged Positions')
    plt.legend()
    plt.grid(True)

    # Save plot
    plt.savefig(f'position_comparison_{COUNTER}.png')
    plt.close()


def compute_score(model: Model, log: dict) -> float:
    rollout_data = Rollout(
        joint_position_1=[], joint_position_2=[], joint_position_3=[], 
        actions=[],
        phase_1=log["phase_offset"] * 1, phase_2=log["phase_offset"] * 2, phase_3=log["phase_offset"] * 3, 
        amplitude=log["amplitude"], 
        frequency=log["frequency"]
    )

    result = simulate.rollout(simulation_app, agent_cfg, rollout_data, env, model, resume_path)
    
    real_positions = get_actuator_positions(log)
    timesteps = len(result.joint_position_1)

    # Create figure and axis
    if True:
        plot_positions(result, real_positions, timesteps, COUNTER)

    steps = 43
    return np.mean(np.abs(np.array(result.joint_position_1) - np.array(real_positions["1"][steps:steps + timesteps])))


def compute_scores(model: Model, compute_logs=None):
    scores = 0
    for log in compute_logs.logs:
        scores += compute_score(model, log)

    return scores / len(compute_logs.logs)


def make_model() -> Model:
    model = Model()

    return model


def objective(trial):
    model = make_model()

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

        model = make_model()
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
    logdir = "scripts/bam/data_wesley"
    validation_kp = 0
    workers = 1
    trials = 500
    method = "cmaes"
    eval = False
    logs = Logs(logdir)
    last_log = time.time()
    wandb_run = None

    study_name = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Study URL (when multiple workers are used)
    study_url = f"sqlite:///study.db"

    if method == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(
            restart_strategy="bipop"
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    def optuna_run(enable_monitoring=True):
        if workers > 1:
            study = optuna.load_study(study_name=study_name, storage=study_url)
        else:
            study = optuna.create_study(sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        callbacks = []
        if enable_monitoring:
            callbacks = [monitor]
        study.optimize(objective, n_trials=trials, n_jobs=1, callbacks=callbacks)

    optuna_run(True)
    simulation_app.close()
