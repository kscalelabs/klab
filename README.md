# Installation

## Isaac Lab
### Only ubunut 22.04 works
```bash
lsb_release -a
#No LSB modules are available.
#Distributor ID:	Ubuntu
#Description:	Ubuntu 22.04.5 LTS
#Release:	22.04
#Codename:	jammy
```

```bash
conda create -n env_isaacsim python=3.10
conda activate env_isaacsim
pip install isaacsim==4.2.0.2 --extra-index-url https://pypi.nvidia.com
pip install isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com
```

# First running
# logs at /home/dpsh/.nvidia-omniverse/
```bash
isaacsim
```

# Isaac Lab
```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

# Install additional packages
```bash
sudo apt install cmake build-essential
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

# Option 1: Using the isaaclab.sh executable
# note: this works for both the bundled python and the virtual environment
```bash
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

# Option 2: Using python in your virtual environment
```bash
python source/standalone/tutorials/00_sim/create_empty.py
```


## Install GPR extension
```bash
cd exts/gpr
python -m pip install -e .
```

## Run

Training an agent with RSL-RL on Velocity-Rough-Gpr-v0:

```bash
# run script for training
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/train.py --task Velocity-Rough-Gpr-v0
# run script for playing
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/rsl_rl/play.py --task Velocity-Rough-Gpr-Play-v0
```
