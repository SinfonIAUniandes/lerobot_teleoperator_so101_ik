# SO101 Inverse Kinematics Web Teleoperator
<img width="709" height="325" alt="image" src="https://github.com/user-attachments/assets/83e4ac9d-454e-4fda-a746-84832d745ca1" />

A custom teleoperation plugin for the [LeRobot](https://github.com/huggingface/lerobot) framework. This package spins up a local Viser web server with a 3D interface, allowing you to control a robotic arm using a drag-and-drop target gizmo.

It uses [Pyroki](https://github.com/chungmin99/pyroki) to solve inverse kinematics in real time and outputs the required joint angles. The package is designed to work with both simulated robots (such as MuJoCo) and real SO-101 hardware.

- `so101physicalwrapper.py` adds a robot config named `so101_physical_wrapped`
- it wraps the standard `SOFollower` robot and exposes end-effector pose and depth features
- it is intended for real hardware recordings and for high-level LeRobot pipelines that need an easy robot type for physical experiments

## Installation

This plugin requires a working installation of LeRobot and the Pyroki IK solver.

### 1. Clone the repository

```bash
git clone https://github.com/SinfonIAUniandes/lerobot_teleoperator_so101_ik
cd lerobot_teleoperator_so101_ik
```

### 2. Install the UI and IK dependencies

This lightweight requirements file installs the web UI and URDF loaders without conflicting with your core LeRobot environment:

```bash
pip install -r requirements.txt
```

### 3. Install the plugin

Install the package in editable mode so it registers with the LeRobot CLI:

```bash
pip install -e .

```

## Usage

Once installed, the teleoperator is automatically discovered by the LeRobot CLI and can be referenced using `--teleop.type=so101_ik`.

**Teleoperate a robot (Example using the SO101 MuJoCo sim):**

```bash
lerobot-teleoperate \
  --robot.type=so101_mujoco \
  --teleop.type=so101_ik

```

**Teleoperate the physical SO101 using the browser IK target:**

```bash
lerobot-teleoperate \
  --robot.type=so101_physical_wrapped \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm_test4 \
  --teleop.type=so101_ik

```

**Teleoperate the physical SO101 using a leader robot:**

```bash
lerobot-teleoperate \
  --robot.type=so101_physical_wrapped \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm_test4 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader_arm_test2

```

**Recording Data:**

1. Run the `lerobot-record` command with this teleoperator.
2. Open your web browser and navigate to `http://localhost:8080`.
3. Use the 3D target to guide the arm while LeRobot records the joint states and camera frames.

```bash
lerobot-record \
  --robot.type=so101_mujoco \
  --teleop.type=so101_ik \
  --dataset.repo_id=local/ik_test \
  --dataset.single_task="follow the target"
```

**Recording data from the physical SO101 using the browser IK target:**

```bash
lerobot-record \
  --robot.type=so101_physical_wrapped \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm_test4 \
  --teleop.type=so101_ik \
  --dataset.repo_id=local/so101_real_ik_demo \
  --dataset.single_task="reach target"
```

**Recording data from the physical SO101 using a leader robot:**

```bash
lerobot-record \
  --robot.type=so101_physical_wrapped \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm_test4 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader_arm_test2 \
  --dataset.repo_id=local/my_physical_dataset \
  --dataset.single_task="grab the tape"
```

## Notes

- `so101_ik` is the web teleoperator that drives the target gizmo.
- `so101_physical_wrapped` is the robot wrapper used for real hardware data collection.
- The real-robot path is typically run through the script in `FLAG-Embodied-data`, not directly through `lerobot-record`, because the physical recording flow includes the additional robot wrapper and runtime logic for hardware acquisition.
- In Linux the robot serial port is usually something like `/dev/ttyACM0`; in Windows it will be a COM port such as `COM9`.

## Supported workflows

- MuJoCo + web IK target
- real SO-101 + hardware leader
- real SO-101 + browser IK target
- dataset capture for LeRobot training pipelines

