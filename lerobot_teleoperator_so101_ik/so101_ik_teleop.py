import numpy as np
import threading
import time
from typing import Any
from lerobot.teleoperators.teleoperator import Teleoperator
from .config_so101_ik_teleop import So101IkTeleopConfig

import viser
from viser.extras import ViserUrdf
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description
from .pyroki_snippets import solve_ik

class So101IkTeleop(Teleoperator):
    config_class = So101IkTeleopConfig
    name = "so101_ik"

    def __init__(self, config: So101IkTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self.viser_server = None
        
        # Threading mechanisms
        self._ik_thread = None
        self._lock = threading.Lock()
        
        # State variables protected by the lock
        self._latest_q_sol = None
        self._latest_gripper = 0.0

        self.scale = 50.0  # Scale factor for the robot arm joint angles
        
        self.ik_joint_mapping = {
            "1": "shoulder_pan", "2": "shoulder_lift", "3": "elbow_flex",
            "4": "wrist_flex", "5": "wrist_roll"
        }

    def _ik_worker(self):
        """Background thread that continuously solves IK based on the UI target."""
        while self._is_connected:
            # Read UI state safely
            try:
                target_pos = np.array(self.ik_web_target.position)
                target_quat = np.array(self.ik_web_target.wxyz)
                gripper_val = self.gripper_slider.value
            except Exception:
                # Catch errors if UI is closing
                break

            # Solve IK (this blocks the worker thread, but not the main loop)
            q_sol = solve_ik(
                robot=self.robot,
                target_link_name=self.config.target_link,
                target_position=target_pos,
                target_wxyz=target_quat,
            )

            if q_sol is not None:
                # JAX arrays are immutable, so we must make a standard numpy copy first
                q_vis = np.array(q_sol)
                
                # --- THE FIX: Inject the gripper value for the visualizer ---
                # Based on your URDF, the gripper joint is named "6"
                if "6" in self.urdf_joints:
                    gripper_idx = self.urdf_joints.index("6")
                    q_vis[gripper_idx] = gripper_val
                # Update UI Ghost
                self.urdf_vis.update_cfg(q_vis)
                
                # Safely update the latest solution for get_action() to read
                with self._lock:
                    self._latest_q_sol = q_sol
                    self._latest_gripper = gripper_val

            # Prevent CPU pegging
            time.sleep(0.01)

    def connect(self) -> None:
        self.urdf = load_robot_description(self.config.urdf_name)
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.urdf_joints = [j.name for j in self.urdf.actuated_joints]
        
        self.viser_server = viser.ViserServer(port=self.config.viser_port)
        self.viser_server.scene.add_grid("/ground", width=2, height=2)
        self.urdf_vis = ViserUrdf(self.viser_server, self.urdf, root_node_name="/ghost_robot")
        
        self.ik_web_target = self.viser_server.scene.add_transform_controls(
            "/ik_target", 
            scale=0.1, 
            # Updated to your safe starting position
            position=(0.00931305, -0.27034248, 0.26730747), 
            wxyz=(0.707, -0.707, 0.0, 0.0)
        )
        self.gripper_slider = self.viser_server.gui.add_slider(
            "Gripper", min=0.0, max=1.0, step=0.01, initial_value=0.0
        )
        
        # --- THE FIX: JAX Warm-up ---
        print("\n--- Compiling JAX IK Solver ---")
        print("This will take ~10-20 seconds. Please wait...")
        dummy_pos = np.array([0.3, 0.0, 0.2])
        dummy_quat = np.array([0.0, 0.0, 0.0, 0.0])
        solve_ik(
            robot=self.robot,
            target_link_name=self.config.target_link,
            target_position=dummy_pos,
            target_wxyz=dummy_quat,
        )
        print("--- JAX Compilation Complete! ---\n")

        self._is_connected = True

        # Start the background solver thread
        self._ik_thread = threading.Thread(target=self._ik_worker, daemon=True)
        self._ik_thread.start()

    def disconnect(self) -> None:
        self._is_connected = False
        if self.viser_server:
            self.viser_server.stop()
        if self._ik_thread:
            self._ik_thread.join(timeout=1.0)

    def reset_target(self) -> None:
        """Snaps the Viser IK target and gripper back to their starting safe coordinates."""
        if hasattr(self, 'ik_web_target'):
            # These are the safe starting coordinates defined in your connect() method
            self.ik_web_target.position = (0.00931305, -0.27034248, 0.26730747)
            self.ik_web_target.wxyz = (0.707, -0.707, 0.0, 0.0)
            
        if hasattr(self, 'gripper_slider'):
            self.gripper_slider.value = 0.0

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def get_action(self) -> dict:
        """Returns joint commands in Radians, which the robot driver handles."""
        with self._lock:
            q_sol = self._latest_q_sol
            gripper_val = self._latest_gripper

        # We use 0.0 as the base (the robot's calibrated center)
        action_dict = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": float(gripper_val) * self.scale,  # Scale gripper value to match expected range,
        }

        if q_sol is not None:
            # Map the IK solution URDF indices to the .pos string keys
            # These values are in Radians.
            if "1" in self.urdf_joints: action_dict["shoulder_pan.pos"] = float(q_sol[self.urdf_joints.index("1")]) * self.scale
            if "2" in self.urdf_joints: action_dict["shoulder_lift.pos"] = float(q_sol[self.urdf_joints.index("2")]) * self.scale
            if "3" in self.urdf_joints: action_dict["elbow_flex.pos"] = float(q_sol[self.urdf_joints.index("3")])  * self.scale
            if "4" in self.urdf_joints: action_dict["wrist_flex.pos"] = float(q_sol[self.urdf_joints.index("4")]) * self.scale
            if "5" in self.urdf_joints: action_dict["wrist_roll.pos"] = float(q_sol[self.urdf_joints.index("5")]) * self.scale
            
        return action_dict
    
    @property
    def action_features(self) -> dict:
        # Register the features with the exact .pos suffix the pipeline expects
        return {
            "shoulder_pan.pos": float, "shoulder_lift.pos": float, "elbow_flex.pos": float,
            "wrist_flex.pos": float, "wrist_roll.pos": float, "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass