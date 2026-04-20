# OneHBC: A Unified Framework for Humanoid Body Control

# OneHBC

> A Unified Framework for Humanoid Body Control Training
> Based on IsaacLab Newton Physics Engine
> 
> 

OneHBC is a dedicated research framework for training humanoid robot locomotion and whole\-body control policies using NVIDIA IsaacLab with the Newton physics engine\. It supports high\-performance end\-to\-end reinforcement learning and motion imitation for humanoid robots, with a focus on speed tracking, AMP\-based motion imitation, and whole\-body trajectory tracking\. Future extensions will support general whole\-body VLA \(Vision\-Language\-Action\) control\.

---

## Features

- **Velocity Tracking Control**: Omnidirectional speed command tracking for robust locomotion

- **AMP \(Adversarial Motion Priors\)**: High\-quality natural motion imitation learning

- **Whole\-Body Trajectory Tracking**: Accurate task\-space and joint\-space trajectory following

- **Under Development**: General whole\-body VLA \(Vision\-Language\-Action\) control pipeline

---

## Environment Requirements

- OS: Ubuntu 22\.04 / 24\.04

- IsaacLab: `>=3.0`

- Python: 3\.12

- CUDA: 12\.8 or higher

---

## Installation

### 1\. Set Up Conda Environment and Dependencies

```bash
conda create -n onehbc python=3.12
conda activate onehbc
```


### 2\. install IsaacLab Newton Physics Engine by following the [official installation guide](https://isaac-sim.github.io/IsaacLab/develop/source/experimental-features/newton-physics-integration/installation.html)

### 3\. Clone OneHBC

```bash
cd IsaacLab
git clone https://github.com/HongtuZ/OneHBC.git
cd OneHBC
```

### 4\. Install Rsl-rl Library

```bash
cd rsl-rl
pip install -e .
```
---

## Training Commands

### Velocity Tracking Control

```bash
# Train
python scripts/rsl_rl/train.py --task OneHBC-RL-Flat-THS23Dof-v0 --viz none --num_envs 4096

# Train with visualization
python scripts/rsl_rl/train.py --task OneHBC-RL-Flat-THS23Dof-v0 --viz newton --num_envs 4096

### AMP Imitation Learning
TODO: Add AMP training and evaluation commands

### Whole\-Body Trajectory Tracking
TODO: Add whole-body trajectory tracking training and evaluation commands
```