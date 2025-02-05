"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import json
import wandb
import torch
import threading
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import kbot.tasks  # noqa: F401
import zbot2.tasks  # noqa: F401

import shutil
import traceback
import re

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

def log_copy_of_src_code_local(log_dir: str) -> None:
    """
    Copies all Python files in {klab_path}/exts into a local
    'src_code_copy' folder within log_dir, maintaining the original folder structure.

    If not found, does nothing.
    """
    try:
        # Match everything up to and including "/klab"
        match = re.search(r"^(.*?/klab)", log_dir)
        if not match:
            return

        print(f"[INFO] Saving source code copy to: {os.path.join(log_dir, 'src_code_copy')}")

        klab_path = match.group(1)
        zbot2_path = os.path.join(klab_path, "exts")
        src_code_copy_dir = os.path.join(log_dir, "src_code_copy")

        os.makedirs(src_code_copy_dir, exist_ok=True)

        for root, dirs, files in os.walk(zbot2_path):
            # Filter out .py files
            py_files = [f for f in files if f.endswith(".py")]
            if not py_files:
                continue

            # Relative subdirectory under 'exts' mirrored in 'src_code_copy'
            rel_path = os.path.relpath(root, zbot2_path)
            dst_dir = os.path.join(src_code_copy_dir, rel_path)
            os.makedirs(dst_dir, exist_ok=True)

            # Copy each .py file into the mirrored structure
            for fname in py_files:
                src = os.path.join(root, fname)
                dst = os.path.join(dst_dir, fname)
                shutil.copy2(src, dst)

    except Exception:
        print("Warning: log_copy_of_src_code_local() encountered an error:")
        traceback.print_exc()

def save_wandb_run_info(log_dir: str, wandb_run) -> None:
    """Save wandb run info to a JSON file."""
    wandb_info = {
        "run_id": wandb_run.id,
        "run_name": wandb_run.name,
        "project": wandb_run.project,
    }
    wandb_file = os.path.join(log_dir, "wandb_run_info.json")
    with open(wandb_file, "w") as f:
        json.dump(wandb_info, f)
    print(f"[INFO] Saved wandb run info to: {wandb_file}")
    
def wait_for_wandb_run(log_dir: str, timeout: float = 60.0):
    """Wait for wandb run to be initialized and save its info."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if wandb.run is not None:
            save_wandb_run_info(log_dir, wandb.run)
            return
        time.sleep(1.0)
    print("[WARNING] Timeout waiting for wandb run to initialize")


def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # copy zbot2 py files to log_dir
    log_copy_of_src_code_local(log_dir)

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    wandb_timeout = 60.0
    wandb_thread = threading.Thread(target=wait_for_wandb_run, args=(log_dir, wandb_timeout))
    wandb_thread.start()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    wandb_thread.join(timeout=wandb_timeout)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close the sim app
    simulation_app.close()
