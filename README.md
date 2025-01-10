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
cd IsaacLab
```

### Install additional packages
```bash
sudo apt install cmake build-essential
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

Training an agent with RSL-RL on Velocity-Rough-Gpr-v0:

```bash
# run script for training
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/train.py --task Velocity-Rough-Gpr-v0
# run script for playing
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/play.py --task Velocity-Rough-Gpr-Play-v0
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