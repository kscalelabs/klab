"""Utility functions for logging with wandb."""

import os
import json
import wandb
from typing import Any


def load_wandb_run_info(log_dir: str) -> dict:
    """Load wandb run info from a previous training session.
    
    Args:
        log_dir (str): Directory containing wandb run info
    
    Returns:
        dict: Wandb run information or None if not found
    """
    wandb_file = os.path.join(log_dir, "wandb_run_info.json")
    if os.path.exists(wandb_file):
        with open(wandb_file, "r") as f:
            return json.load(f)
    return None

def init_wandb(log_dir: str, cli_args: dict) -> Any:
    """Initialize wandb run, either resuming an existing one or creating a new one.
    
    Args:
        log_dir (str): Directory containing wandb run info
        cli_args (dict): Command line arguments for config
    
    Returns:
        wandb.Run: Initialized wandb run
    """
    # Try to load existing run info
    wandb_run_info = load_wandb_run_info(log_dir)
    
    if wandb_run_info:
        print(f"[INFO] Using existing wandb run: {wandb_run_info['run_id']}")
        wandb_run = wandb.init(
            project=wandb_run_info["project"],
            id=wandb_run_info["run_id"],
            resume="must",
            mode="online"
        )
    else:
        print("[INFO] No existing wandb run found. Creating new run.")
        auto_parsed_run_name = os.path.basename(log_dir)
        wandb_run = wandb.init(
            project="isaac-eval",
            name=auto_parsed_run_name,
            mode="online"
        )
    
    return wandb_run

def upload_videos_to_wandb(wandb_run: Any, run_dir: str):
    """Upload videos from run directory to wandb.
    
    Args:
        wandb_run (wandb.Run): Active wandb run
        run_dir (str): Directory containing video files
    """
    if wandb_run is None:
        return

    video_files = [f for f in os.listdir(run_dir) if f.endswith(".mp4")]
    print(f"[INFO] Found {len(video_files)} video files")
    
    for video_file in video_files:
        video_path_full = os.path.join(run_dir, video_file)
        try:
            print(f"[INFO] Uploading video: {video_file}")
            wandb_run.log({
                "play/video": wandb.Video(
                    video_path_full,
                    fps=30,
                    format="mp4",
                    caption=f"Play episode: {video_file}"
                )
            })
        except Exception as e:
            print(f"[ERROR] Failed to upload video {video_file}: {str(e)}")

def finish_wandb_run(wandb_run: Any):
    """Safely finish a wandb run.
    
    Args:
        wandb_run (wandb.Run): Active wandb run to finish
    """
    if wandb_run is not None:
        wandb_run.finish()