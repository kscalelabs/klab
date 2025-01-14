# Installation

## Prerequisites

Only ubuntu 22.04 works so check your version
```bash
lsb_release -a
#No LSB modules are available.
#Distributor ID:	Ubuntu
#Description:	Ubuntu 22.04.5 LTS
#Release:	22.04
#Codename:	jammy
```

## Clone repo and lfs pull

```bash
# cd to where you want to install the repo
# git clone the repo
cd klab
# Install Git LFS
sudo apt-get update
sudo apt-get install git-lfs
git lfs install

# Pull large files (like USD models)
git lfs pull
```

## Make conda env

```bash
conda create -n env_isaacsim python=3.10
conda activate env_isaacsim
pip install isaacsim==4.2.0.2 --extra-index-url https://pypi.nvidia.com
pip install isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com
```

## Test IsaacSim can launch

logs  are at /home/dpsh/.nvidia-omniverse/
```bash
isaacsim
```

## Isaac Lab
```bash
# cd to repo root
# Get Isaaclab as a submodule
git submodule update --init --recursive 
```

### Install additional packages
```bash
sudo apt install cmake build-essential
cd IsaacLab
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

# Usage

## Test empty sim

### Option 1: Using the isaaclab.sh executable
Note: this works for both the bundled python and the virtual environment
```bash
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

### Option 2: Using python in your virtual environment
```bash
python source/standalone/tutorials/00_sim/create_empty.py
```

## Run with Robot

### Install GPR extension
```bash
cd exts/gpr
python -m pip install -e .
```

### Run training / playing

#### For GPR / Kbot

Training an agent with RSL-RL on Velocity-Rough-Kbot-v0:

```bash
# run script for training
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/train.py --task Velocity-Rough-Kbot-v0
# run script for playing
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/play.py --task Velocity-Rough-Kbot-Play-v0
```

#### For zbot2

```bash
# run training
# cd to repo root (klab folder)
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/train.py --task Velocity-Rough-Zbot2-v0

# run play
# cd to repo root (klab folder)
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/play.py --task Velocity-Rough-Zbot2-Play-v0
```

#### Play from checkpoint

If you want to play from a checkpoint, here is an example command:
```bash
# to load checkpoint from
# klab/logs/rsl_rl/zbot2_rough/2025-01-08_19-33-44/model_200.pt
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/play.py   --task Velocity-Rough-Zbot2-Play-v0 \
  --num_envs 1 \
  --resume true \
  --load_run 2025-01-08_19-33-44 \
  --checkpoint model_200.pt
```

#### Saving imu plots 

Use `play_imu.py` to save imu plots

```bash
# NOTE: turn off termination so that it doesn't stop the moment it falls
# NOTE: The loaded checkpoint has to match the current obs config
# The imu plot and data will be the same length as the video
# imu_type is projected_gravity by default
# example command:
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/play_imu.py  \
  --task Velocity-Rough-Zbot2-Play-v0 \
  --num_envs 1 \
  --video \
  --video_length 100 \
  --imu_type projected_gravity \
  --load_run 2025-01-09_04-50-36 \
  --checkpoint model_0.pt 
```


# Adding a new robot from URDF

Instructions in [AddNewRobot.md](AddNewRobot.md)

# Troubleshooting

## Wandb logging

Wandb logging relies on rsl_rl library's wandb logging, so there are a few things to keep in mind.

You need to set the WANDB_USERNAME to the project's entity name.

```bash
export WANDB_USERNAME=project_entity_name
```

Also, to set the wandb experiment name to the log_dir folder name, you need to change a file in the rsl_rl library.

In the `rsl_rl/utils/wandb_utils.py` file, change the wandb.run.name to the last folder in log_dir path as follows:

```python
# Change generated name to project-number format            
wandb.run.name = project + wandb.run.name.split("-")[-1] # <--- After this line

# ALI CHANGES
# Change wandb run name to the last folder in log_dir path
wandb.run.name = os.path.basename(log_dir)                     # <--- Add this line
```

## Inotify limit 

If you see this in the logs 
```
Failed to create an inotify instance. Your system may be at the limit of inotify instances. The limit is listed in `/proc/sys/fs/inotify/max_user_watches` but can be modified. A reboot or logging out and back in may also resolve the issue.
```
Just increase the limits via the following commands:
```bash
sudo sysctl fs.inotify.max_user_instances=8192
sudo sysctl fs.inotify.max_user_watches=524288
sudo sysctl -p
```

## VSCode Environment

For proper indexing of the isaaclab package, you need to set the PYTHONPATH environment variable for Vscode. 

1. Create a .env file in the root of the Vscode workspace with the following content:
```bash
PYTHONPATH=/path/to/klab/IsaacLab/source/extensions/omni.isaac.lab:/path/to/klab/IsaacLab/source/extensions/omni.isaac.lab_assets:/path/to/klab/IsaacLab/source/extensions/omni.isaac.lab_tasks
```

2. Create a .vscode/settings.json file with the following content:
```json
{
    "python.envFile": "${workspaceFolder}/.env"
}
```

