"""Fit the model to the data.

Run:
    export PYTHONPATH=/home/dpsh/isaac_gpr:$PYTHONPATH
    /home/dpsh/IsaacLab/isaaclab.sh -p scripts/rsl_rl/fit.py --task Velocity-Rough-Gpr-Play-v0 --headless
"""

import argparse
from omni.isaac.lab.app import AppLauncher

# local imports
from scripts.rsl_rl import cli_args  # isort: skip
import copy

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
import torch

# Import extensions to set up environment tasks
import gpr.tasks  # noqa: F401

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner


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

# import socket
from datetime import datetime
import sys
import numpy as np
import json
from copy import deepcopy
import json
import time
import optuna
import wandb

from scripts.bam.model import Model
from scripts.bam import simulate
from scripts.bam.logs import Logs
from scripts.bam import message


from dataclasses import dataclass
from typing import List


@dataclass
class Rollout:
    phase_1: float
    phase_2: float
    phase_3: float
    amplitude_1: float
    amplitude_2: float
    amplitude_3: float
    frequency_1: float
    frequency_2: float
    frequency_3: float
    obs: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[np.ndarray]
    dones: List[np.ndarray]
    infos: List[np.ndarray]


# Json params file
params_json_filename =  "sts_3215"
if not params_json_filename.endswith(".json"):
    params_json_filename = f"output/params_{params_json_filename}.json"
json.dump({}, open(params_json_filename, "w"))


def compute_score(model: Model, log: dict) -> float:
    rollout_data = Rollout(
        obs=[], actions=[], rewards=[], dones=[], infos=[],
        phase_1=0.0, phase_2=0.0, phase_3=0.0,
        amplitude_1=0.0, amplitude_2=0.0, amplitude_3=0.0,
        frequency_1=0.0, frequency_2=0.0, frequency_3=0.0
    )

    result = simulate.rollout(simulation_app, agent_cfg, rollout_data, env, model, resume_path)
    log_positions = np.array([entry["position"] for entry in log["entries"]])
    # TODO TEMP pfb30
    positions = np.random.rand(log_positions.shape[0])
    return np.mean(np.abs(positions - log_positions))


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
    logdir = "data_feetech"
    validation_kp = 0
    workers = 1
    trials = 500
    method = "cmaes"
    eval = False
    logs = Logs(logdir)
    if not eval and validation_kp > 0:
        validation_logs = logs.split(validation_kp)
        print(f"{len(validation_logs.logs)} logs splitted for validation")
        if len(validation_logs.logs) == 0:
            raise ValueError("No logs for validation")

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
