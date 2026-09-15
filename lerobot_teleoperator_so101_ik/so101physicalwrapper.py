import numpy as np
import cv2
from dataclasses import dataclass
from typing import Any
from scipy.spatial.transform import Rotation

from robot_descriptions.loaders.yourdfpy import load_robot_description

from lerobot.robots.robot import RobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

@RobotConfig.register_subclass("so101_physical_wrapped")
@dataclass
class So101PhysicalWrapperConfig(SOFollowerRobotConfig):
    urdf_name: str = "so_arm101_description"
    target_link: str = "gripper"
    scale: float = 50.0

class So101PhysicalWrapper(SOFollower):
    config_class = So101PhysicalWrapperConfig
    name = "so101_physical_wrapped"

    def __init__(self, config: So101PhysicalWrapperConfig):
        super().__init__(config)
        
        self.urdf = load_robot_description(self.config.urdf_name)
        self.urdf_joints = [j.name for j in self.urdf.actuated_joints]
        
        self.joint_mapping = {
            "shoulder_pan": "1", "shoulder_lift": "2", "elbow_flex": "3",
            "wrist_flex": "4", "wrist_roll": "5", "gripper": "6"
        }

    @property
    def observation_features(self) -> dict:
        obs = super().observation_features.copy()
        
        obs.update({
            "ee_pos_x": float, "ee_pos_y": float, "ee_pos_z": float,
            "ee_quat_x": float, "ee_quat_y": float, "ee_quat_z": float, "ee_quat_w": float,
            "cam_high_depth": (480, 640, 3),
            "cam_high_depth_vis": (480, 640, 3) 
        })
        return obs

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        
        # 1. Kinematics
        cfg = {}
        for lerobot_name, urdf_num in self.joint_mapping.items():
            if urdf_num in self.urdf_joints:
                val = obs.get(f"{lerobot_name}.pos", 0.0)
                cfg[urdf_num] = val / self.config.scale
                
        self.urdf.update_cfg(cfg)
        ee_transform = self.urdf.scene.graph.get(self.config.target_link)[0]
        
        ee_pos = ee_transform[:3, 3]
        ee_quat = Rotation.from_matrix(ee_transform[:3, :3]).as_quat()
        
        obs.update({
            "ee_pos_x": float(ee_pos[0]), "ee_pos_y": float(ee_pos[1]), "ee_pos_z": float(ee_pos[2]),
            "ee_quat_x": float(ee_quat[0]), "ee_quat_y": float(ee_quat[1]), 
            "ee_quat_z": float(ee_quat[2]), "ee_quat_w": float(ee_quat[3])
        })

        # 2. Depth Encoding and Visualization
        cam_high = self.cameras.get("cam_high")
        
        if cam_high and hasattr(cam_high, "latest_depth_frame") and cam_high.latest_depth_frame is not None:
            # Safely extract the 2D array whether LeRobot stored it as (H, W) or (H, W, 1)
            raw_depth_mm = np.squeeze(cam_high.latest_depth_frame)
            
            # --- Math Data ---
            depth_mm = np.clip(raw_depth_mm, 0, 65535).astype(np.uint16)
            encoded_depth = np.zeros((raw_depth_mm.shape[0], raw_depth_mm.shape[1], 3), dtype=np.uint8)
            encoded_depth[..., 0] = (depth_mm >> 8) & 0xFF  
            encoded_depth[..., 1] = depth_mm & 0xFF         
            obs["cam_high_depth"] = encoded_depth
            
            # --- Visual Data ---
            max_depth_mm = 3000.0  # 3 meters
            depth_visual = raw_depth_mm.copy().astype(np.float32)
            bg_mask = (depth_visual == 0.0) | np.isinf(depth_visual) | np.isnan(depth_visual)
            depth_visual[bg_mask] = max_depth_mm
            
            depth_normalized = np.clip(depth_visual, 0, max_depth_mm) / max_depth_mm
            depth_8bit = (depth_normalized * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(255 - depth_8bit, cv2.COLORMAP_JET)
            
            obs["cam_high_depth_vis"] = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
            
        else:
            obs["cam_high_depth"] = np.zeros((480, 640, 3), dtype=np.uint8)
            obs["cam_high_depth_vis"] = np.zeros((480, 640, 3), dtype=np.uint8)

        return obs