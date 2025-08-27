"""Controller and joystick data types"""
from dataclasses import dataclass, field
from typing import Dict
import numpy as np

@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Position':
        return cls(x=data.get("x", 0.0), y=data.get("y", 0.0), z=data.get("z", 0.0))
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'Position':
        if array.shape != (3,):
            raise ValueError("Array must have shape (3,)")
        return cls(x=array[0], y=array[1], z=array[2])
    
@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w])
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Quaternion':
        return cls(
            x=data.get("x", 0.0), 
            y=data.get("y", 0.0), 
            z=data.get("z", 0.0), 
            w=data.get("w", 1.0)
        )
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'Quaternion':
        if array.shape != (4,):
            raise ValueError("Array must have shape (4,)")
        return cls(x=array[0], y=array[1], z=array[2], w=array[3])

@dataclass
class Thumbstick:
    x: float = 0.0
    y: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Thumbstick':
        return cls(x=data.get("x", 0.0), y=data.get("y", 0.0))

@dataclass
class ControllerData:
    position: Position = field(default_factory=Position)
    rotation: Quaternion = field(default_factory=Quaternion)
    thumbstick: Thumbstick = field(default_factory=Thumbstick)
    index_trigger: float = 0.0
    hand_trigger: float = 0.0
    buttons: Dict[str, bool] = field(default_factory=lambda: {
        "A": False, "B": False, "X": False, "Y": False,
    }) 
    hand_side: str = "unknown"  # "left" or "right"

    def get_controller_position(self) -> np.ndarray:
        return self.position.to_array()
    
    def get_controller_rotation(self) -> np.ndarray:
        return self.rotation.to_array()
    
    @classmethod
    def from_dict(cls, data: Dict, hand_side: str = "unknown") -> 'ControllerData':
        return cls(
            position=Position.from_dict(data.get("position", {})),
            rotation=Quaternion.from_dict(data.get("rotation", {})),
            thumbstick=Thumbstick.from_dict(data.get("thumbstick", {})),
            index_trigger=data.get("index_trigger", 0.0),
            hand_trigger=data.get("hand_trigger", 0.0),
            buttons={btn["buttonName"]: btn["pressed"] for btn in data.get("buttons", [])},
            hand_side=hand_side
        )

@dataclass
class HeadsetPose:
    position: Position = field(default_factory=Position)
    rotation: Quaternion = field(default_factory=Quaternion)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HeadsetPose':
        return cls(
            position=Position.from_dict(data.get("position", {})),
            rotation=Quaternion.from_dict(data.get("rotation", {}))
        )