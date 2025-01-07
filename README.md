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
git clone https://github.com/isaac-sim/IsaacLab.git
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

Training an agent with RSL-RL on Velocity-Rough-Gpr-v0:

```bash
# run script for training
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/train.py --task Velocity-Rough-Gpr-v0
# run script for playing
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/play.py --task Velocity-Rough-Gpr-Play-v0
```

# Adding a new robot from URDF

docs [here](https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_import_urdf.html).


## launch the omniverse launcher
just click on the icon in the dock

click launch

## start empty isaacsim
again just click on the icon in the gui 

## urdf importer

In the empty isaacsim window

IsaacUtils -> Workflows -> URDF Importer

## select input and output 

Input file should be the urdf file 

output folder should be the folder where you want to save the robot usd files 

The settings used for the import are in the file `klab/exts/zbot2/zbot2/assets/Robots/zbot2_urdf_to_usd_import_settings.yaml`

**NOTE:** the fix_base_link should be set to false, otherwise the robot base will not move freely

example input and output folders:

```bash
# input file
zbot2/
├── joints.py
├── meshes
│   ├── FINGER_1_1.stl
│   ...
└── robot_fixed.urdf # < ---- select this

# output folder 
klab/exts/zbot2/ # <--- select this

# output folder result
zbot2/
├── import_settings.txt
└── robot
    ├── instanceable_meshes.usd
    └── robot_fixed.usd
```


