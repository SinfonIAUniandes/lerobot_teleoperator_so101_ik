from dataclasses import dataclass
from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("so101_ik")
@dataclass
class So101IkTeleopConfig(TeleoperatorConfig):
    urdf_name: str = "so_arm101_description"
    target_link: str = "gripper"
    viser_port: int = 8080