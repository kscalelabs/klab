# Add a new robot

Note these are not comprehensive instructions, just a guide to get you started. Look at the docs [here](https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/ext_omni_isaac_urdf.html) for more details.



## Create a new extension folder
You first need to make a new extension folder for your robot in `klab/exts/robot_name`. So make a copy of the `gpr` or `zbot2` folder to start with, and then rename it to your robot name.

**NOTE:** You will need to rename both filenames and mentions of the name in the code. (Preserve case i.e. gpr -> robot_name and GPR -> ROBOT_NAME)

## Edit the values 

Then update the mapping to your joints in `klab/exts/robot_name/robot_name/assets/robot_name.py` and `klab/exts/robot_name/robot_name/tasks/locomotion/velocity/velocity_env_cfg.py`

## Import your robot

in `klab/scripts/rsl_rl/play.py` and `klab/scripts/rsl_rl/train.py` update the lines 

```python
# Import extensions to set up environment tasks
import kbot.tasks  # noqa: F401
import zbot2.tasks  # noqa: F401
```

to 

```python
# Import extensions to set up environment tasks
import kbot.tasks  # noqa: F401
import zbot2.tasks  # noqa: F401
import robot_name.tasks  # noqa: F401
```


# Importing a new robot from URDF

Skim the [Isaac Sim URDF Import Tutorial](https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_import_urdf.html).


## launch the omniverse launcher
just click on the icon in the dock

or use this command 

```bash
# cd ~/Desktop
./omniverse-launcher-linux.AppImage
```

click launch

## start empty isaacsim
again just click on the icon in the gui 

## urdf importer

In the empty isaacsim window

IsaacUtils -> Workflows -> URDF Importer

## select input and output 

Input file should be the urdf file 

output folder should be the folder where you want to save the robot usd files 

### Check the settings

**NOTE: the fix_base_link should be set to false, otherwise the robot base will not move freely**

Use these settings:
```yaml
label: "Basic import settings for robot_name"

import_options:
  merge_fixed_joints: true
  replace_cylinders_with_capsules: false
  fix_base_link: false # important to set to false, otherwise the robot base will not move freely
  import_inertia_tensor: false
  stage_units_per_meter: 1.00
  link_density: 0.00
  joint_drive_type: "Position"
  override_joint_dynamics: false
  joint_drive_strength: 10000.00
  joint_position_damping: 1000.00
  clear_stage: false
  normals_subdivision: "bilinear"
  convex_decomposition: false
  self_collision: false
  collision_from_visuals: false
  create_physics_scene: true
  create_instanceable_asset: true
  instanceable_usd_path: "./instanceable_meshes.usd"
  parse_mimic_joint_tag: true

input:
  input_file: "/path/to/robot_fixed.urdf"

import:
  output_directory: "/path/to/output/folder"
```

example input and output folders:

```bash
# input file
robot_name/
├── joints.py
├── meshes
│   ├── FINGER_1_1.stl
│   ...
└── robot_fixed.urdf # < ---- select this

# output folder 
klab/exts/robot_name/ # <--- select this

# output folder result
robot_name/
├── import_settings.txt
└── robot
    ├── instanceable_meshes.usd
    └── robot_fixed.usd
```

## Move output files

You should have created an extension folder for your robot in `klab/exts/<robot_name>`.

Move the `robot_fixed.usd` file to the `Robots` folder in your extension folder.

```bash
# move instanceable_meshes.usd to here
klab/exts/robot_name/robot_name/assets/Robots/Props_test/instanceable_meshes.usd

# move robot_fixed.usd to here and rename it to robot_name.usd
klab/exts/robot_name/robot_name/assets/Robots/robot_name.usd
```

