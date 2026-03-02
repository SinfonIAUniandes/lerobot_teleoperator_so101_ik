# SO101 Inverse Kinematics Web Teleoperator

A custom teleoperation plugin for the [LeRobot](https://github.com/huggingface/lerobot) framework. This package spins up a local Viser web server with a 3D interface, allowing you to control a robotic arm using a drag-and-drop target gizmo. 

It uses [Pyroki](https://github.com/chungmin99/pyroki) to solve the Inverse Kinematics (IK) in real-time and outputs the required joint angles. **It is hardware-agnostic** and can be used to control both simulated robots (like MuJoCo) and physical hardware.

## Installation

This plugin requires a working installation of LeRobot and the Pyroki IK solver.

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd lerobot_teleoperator_so101_ik

```

**2. Install the UI and IK dependencies**
This lightweight requirements file installs the web UI and URDF loaders without conflicting with your core LeRobot environment:

```bash
pip install -r requirements.txt

```

**3. Install the plugin**
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
