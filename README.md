# RBY1 Robot Integration for LIBERO Benchmark

> ⚠️ **Work In Progress (개발 중)**  
> This project is currently under active development. Features may be incomplete or change without notice.  
> 이 프로젝트는 현재 개발 중입니다. 기능이 불완전하거나 예고 없이 변경될 수 있습니다.

This repository contains the integration of the **RBY1 humanoid robot** into the **LIBERO benchmark** environment for evaluating Vision-Language-Action (VLA) models, specifically **MolmoAct**.

## Overview

This project adapts the [MolmoAct](https://github.com/allenai/MolmoAct) VLA model (originally trained on Panda robot) to work with the RBY1 humanoid robot in the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulation environment built on [robosuite](https://github.com/ARISE-Initiative/robosuite).

### Key Features
- RBY1 robot model integration with robosuite
- LIBERO benchmark environment adaptation for RBY1
- Evaluation scripts for MolmoAct VLA model on RBY1
- Trajectory visualization for debugging and analysis

## Quick Start: Loading RBY1 in robosuite

### Basic Example
```python
import robosuite as suite

# Create environment with RBY1 robot
env = suite.make(
    env_name="Lift",                    # or any robosuite task
    robots="RBY1Single",                # Use RBY1 single-arm robot
    has_renderer=True,                  # Enable visualization
    has_offscreen_renderer=False,
    use_camera_obs=False,
    controller_configs=suite.load_controller_config(
        default_controller="OSC_POSE"
    ),
)

# Reset and run
obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
```

### Using Custom Controller Config
```python
import robosuite as suite
import json

# Load RBY1-specific controller config
controller_config = suite.load_controller_config(
    custom_fpath="robosuite/robosuite/controllers/config/robots/default_rby1_single.json"
)

env = suite.make(
    env_name="Lift",
    robots="RBY1Single",
    controller_configs=controller_config,
    has_renderer=True,
)
```

### Accessing Robot State
```python
obs = env.reset()

# Get end-effector position and orientation
eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id]
eef_quat = env.robots[0].controller.ee_ori_mat

# Get joint positions
joint_pos = env.robots[0]._joint_positions

print(f"EEF Position: {eef_pos}")
print(f"Joint Positions: {joint_pos}")
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- MuJoCo physics engine

### Setup

1. **Clone this repository:**
```bash
git clone https://github.com/RB3159/rby1_LIBERO.git
cd rby1_LIBERO
```

2. **Install robosuite with RBY1 support:**
```bash
cd robosuite
pip install -e .
```

3. **Install LIBERO:**
```bash
cd experiments/LIBERO
pip install -e .
```

4. **Install MolmoAct dependencies:**
```bash
pip install transformers torch torchvision
pip install scipy opencv-python pillow
```

## Project Structure

```
rby1_LIBERO/
├── robosuite/                          # Modified robosuite with RBY1 support
│   └── robosuite/
│       ├── models/robots/manipulators/
│       │   └── rby1_single_robot.py    # RBY1 robot model definition
│       └── controllers/config/robots/
│           └── default_rby1_single.json # RBY1 OSC controller config
├── experiments/
│   ├── libero/
│   │   ├── rby1_libero_eval.py         # RBY1 evaluation script
│   │   └── run_libero_eval.py          # Original Panda evaluation
│   └── LIBERO/                         # LIBERO benchmark (submodule)
│       └── libero/envs/
│           └── bddl_base_domain.py     # Modified for RBY1 positioning
└── rby1a/
    └── mujoco/                         # RBY1 MuJoCo model files
```

## RBY1 Robot Configuration

### Initial Joint Configuration (init_qpos)
The RBY1 robot's initial arm configuration is optimized to match Panda's EEF orientation:

```python
# Target: EEF Euler ≈ [0, 7, 0] degrees (gripper pointing down)
r_arm_qpos = np.array([-0.35, -0.60, -1.10, -1.00, -3.00, -1.74, -2.00])
```

### Joint Definitions
| Joint | Name | Function | Limits (rad) |
|-------|------|----------|--------------|
| arm_0 | Shoulder Yaw | Y-axis positioning | [-2.356, 2.356] |
| arm_1 | Shoulder Pitch | Up/Down movement | [-3.142, 0.050] |
| arm_2 | Shoulder Roll | Arm rotation | [-2.094, 2.094] |
| arm_3 | Elbow | Arm flexion | [-2.618, 0.010] |
| arm_4 | Wrist Roll | Gripper roll | [-6.283, 6.283] |
| arm_5 | Wrist Pitch | Gripper direction | [-1.745, 2.007] |
| arm_6 | Wrist Yaw | Gripper yaw | [-2.967, 2.967] |

### OSC Controller Configuration
```json
{
    "type": "OSC_POSE",
    "input_type": "delta",
    "input_ref_frame": "base",
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "kp": [500, 500, 500, 500, 500, 500]
}
```

## Usage

### Running Evaluation

**Evaluate RBY1 on LIBERO Spatial:**
```bash
cd experiments/libero
export PYTHONPATH=$PWD/../LIBERO:$PWD/../../robosuite:$PYTHONPATH
export HF_HOME=~/molmoact_data/huggingface

python rby1_libero_eval.py \
    --task spatial \
    --task_id 0 \
    --checkpoint allenai/MolmoAct-7B-D-LIBERO-Spatial-0812
```

**Available task suites:**
- `spatial` - Spatial reasoning tasks
- `object` - Object manipulation tasks  
- `goal` - Goal-oriented tasks
- `10` - 10-task benchmark

## Key Modifications from Original

### 1. Robot Base Positioning
RBY1 requires a y-axis offset due to its different kinematic structure:
```python
# In bddl_base_domain.py
if "RBY1Single" in robot_name:
    y_offset = 0.45  # RBY1 needs Y offset
else:
    y_offset = 0.0   # Panda at origin
```

### 2. EEF Orientation Matching
init_qpos optimized so RBY1's EEF orientation matches Panda:
- **Panda EEF Euler:** [0, 7, 0] degrees
- **RBY1 EEF Euler:** [~0, ~6, ~3] degrees (after optimization)

### 3. Action Processing
MolmoAct outputs unnormalized delta actions. OSC controller scales them:
```
actual_delta = action × output_max
e.g., action=0.8 → delta=0.8×0.05=0.04m
```

## Current Status & Known Issues

### ✅ Completed
- RBY1 robot model integration with robosuite
- OSC_POSE controller configuration
- Initial joint position optimization
- LIBERO environment adaptation
- Basic trajectory visualization

### 🔄 In Progress
- VLA model fine-tuning for RBY1
- Task success rate evaluation
- Gripper action calibration

### ❌ Known Issues
- First few steps may show delta=0 (OSC stabilization)
- Some tasks may require additional init_qpos tuning
- Gripper timing differences from Panda

## Troubleshooting

### Robot not moving (delta = 0)
- Normal for first few steps (OSC controller stabilization)
- Ensure init_qpos produces valid EEF orientation

### Wrong movement direction  
- Check EEF coordinate frame alignment
- Verify `input_ref_frame: "base"` in controller config

### Gripper issues
- RBY1 convention: -1 = open, +1 = close
- Use `invert_gripper_action()` for LIBERO compatibility

## References

- **MolmoAct:** https://github.com/allenai/MolmoAct
- **LIBERO:** https://github.com/Lifelong-Robot-Learning/LIBERO  
- **robosuite:** https://github.com/ARISE-Initiative/robosuite
- **RBY1 SDK:** https://github.com/RainbowRobotics/rby1-sdk

## License

Based on:
- MolmoAct: Apache 2.0 License
- LIBERO: MIT License
- robosuite: MIT License
