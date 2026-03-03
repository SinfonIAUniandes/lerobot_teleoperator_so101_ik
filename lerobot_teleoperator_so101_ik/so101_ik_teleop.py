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
            "/ik_target", scale=0.1, position=(0.3, 0.0, 0.2), wxyz=(1.0, 0.0, 0.0, 0.0)
        )
        self.gripper_slider = self.viser_server.gui.add_slider(
            "Gripper", min=0.0, max=1.0, step=0.01, initial_value=0.0
        )
        
        # --- THE FIX: JAX Warm-up ---
        print("\n--- Compiling JAX IK Solver ---")
        print("This will take ~10-20 seconds. Please wait...")
        dummy_pos = np.array([0.3, 0.0, 0.2])
        dummy_quat = np.array([1.0, 0.0, 0.0, 0.0])
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

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def get_action(self) -> dict[str, Any]:
        """Instantly returns the latest solved joint commands."""
        action = {
            "shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
            "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 0.0
        }
        
        # Safely grab the latest solution without blocking
        with self._lock:
            q_sol = self._latest_q_sol
            gripper_val = self._latest_gripper

        if q_sol is not None:
            for u_idx, u_name in enumerate(self.urdf_joints):
                if u_name in self.ik_joint_mapping:
                    action_key = self.ik_joint_mapping[u_name]
                    action[action_key] = float(q_sol[u_idx])
            
        action["gripper"] = gripper_val
                    
        return action
    
    @property
    def action_features(self) -> dict:
        return {
            "shoulder_pan": float, "shoulder_lift": float, "elbow_flex": float,
            "wrist_flex": float, "wrist_roll": float, "gripper": float,
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