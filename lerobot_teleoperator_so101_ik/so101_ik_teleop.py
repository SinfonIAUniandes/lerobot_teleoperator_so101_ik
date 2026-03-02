import numpy as np
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
        
        # Map URDF joint names to LeRobot action keys
        self.ik_joint_mapping = {
            "1": "shoulder_pan", "2": "shoulder_lift", "3": "elbow_flex",
            "4": "wrist_flex", "5": "wrist_roll"
        }

    def connect(self) -> None:
        # 1. Initialize Pyroki Robot for IK
        self.urdf = load_robot_description(self.config.urdf_name)
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.urdf_joints = [j.name for j in self.urdf.actuated_joints]
        
        # 2. Start Viser Web Server
        self.viser_server = viser.ViserServer(port=self.config.viser_port)
        self.viser_server.scene.add_grid("/ground", width=2, height=2)
        self.urdf_vis = ViserUrdf(self.viser_server, self.urdf, root_node_name="/ghost_robot")
        
        # 3. Add UI Controls
        self.ik_web_target = self.viser_server.scene.add_transform_controls(
            "/ik_target", scale=0.1, position=(0.3, 0.0, 0.2), wxyz=(1.0, 0.0, 0.0, 0.0)
        )
        self.gripper_slider = self.viser_server.gui.add_slider(
            "Gripper", min=0.0, max=1.0, step=0.01, initial_value=0.0
        )
        
        self._is_connected = True

    def disconnect(self) -> None:
        if self.viser_server:
            self.viser_server.stop()
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def get_action(self) -> dict[str, Any]:
        """Reads UI, solves IK, and returns joint commands."""
        target_pos = np.array(self.ik_web_target.position)
        target_quat = np.array(self.ik_web_target.wxyz)
        
        # Solve IK
        q_sol = solve_ik(
            robot=self.robot,
            target_link_name=self.config.target_link,
            target_position=target_pos,
            target_wxyz=target_quat,
        )
        
        # Initialize default action dictionary
        action = {
            "shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0,
            "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": self.gripper_slider.value
        }
        
        if q_sol is not None:
            # Update ghost robot in Viser UI
            self.urdf_vis.update_cfg(q_sol)
            
            # Map solver output to action dictionary
            for u_idx, u_name in enumerate(self.urdf_joints):
                if u_name in self.ik_joint_mapping:
                    action_key = self.ik_joint_mapping[u_name]
                    action[action_key] = float(q_sol[u_idx])
                    
        return action
    
    @property
    def action_features(self) -> dict:
        # Define the structure of the commands your teleoperator sends
        # This matches the names of the actuators in your MuJoCo XML
        return {
            "shoulder_pan": float,
            "shoulder_lift": float,
            "elbow_flex": float,
            "wrist_flex": float,
            "wrist_roll": float,
            "gripper": float,
        }

    @property
    def feedback_features(self) -> dict:
        # The web UI doesn't take force-feedback data back from the robot
        return {}

    @property
    def is_calibrated(self) -> bool:
        # Software UIs don't need physical calibration
        return True

    def calibrate(self) -> None:
        # No-op for software UI
        pass

    def configure(self) -> None:
        # No-op for software UI
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # A web UI doesn't have force feedback, so this is a no-op
        pass