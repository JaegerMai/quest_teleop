"""Hand tracking data types"""
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum
import numpy as np
from teleop.joystick_data_types import Position, Quaternion

class HandJoint(Enum):
    """Hand joint enumeration"""
    PALM = "XRHand_Palm"
    WRIST = "XRHand_Wrist"
    
    # Thumb
    THUMB_METACARPAL = "XRHand_ThumbMetacarpal"
    THUMB_PROXIMAL = "XRHand_ThumbProximal"
    THUMB_DISTAL = "XRHand_ThumbDistal"
    THUMB_TIP = "XRHand_ThumbTip"
    
    # Index finger
    INDEX_METACARPAL = "XRHand_IndexMetacarpal"
    INDEX_PROXIMAL = "XRHand_IndexProximal"
    INDEX_INTERMEDIATE = "XRHand_IndexIntermediate"
    INDEX_DISTAL = "XRHand_IndexDistal"
    INDEX_TIP = "XRHand_IndexTip"
    
    # Middle finger
    MIDDLE_METACARPAL = "XRHand_MiddleMetacarpal"
    MIDDLE_PROXIMAL = "XRHand_MiddleProximal"
    MIDDLE_INTERMEDIATE = "XRHand_MiddleIntermediate"
    MIDDLE_DISTAL = "XRHand_MiddleDistal"
    MIDDLE_TIP = "XRHand_MiddleTip"
    
    # Ring finger
    RING_METACARPAL = "XRHand_RingMetacarpal"
    RING_PROXIMAL = "XRHand_RingProximal"
    RING_INTERMEDIATE = "XRHand_RingIntermediate"
    RING_DISTAL = "XRHand_RingDistal"
    RING_TIP = "XRHand_RingTip"
    
    # Little finger
    LITTLE_METACARPAL = "XRHand_LittleMetacarpal"
    LITTLE_PROXIMAL = "XRHand_LittleProximal"
    LITTLE_INTERMEDIATE = "XRHand_LittleIntermediate"
    LITTLE_DISTAL = "XRHand_LittleDistal"
    LITTLE_TIP = "XRHand_LittleTip"
    
    @classmethod
    def get_joint_names(cls) -> List[str]:
        """Get a list of all joint names"""
        return [joint.value for joint in cls]
    
    @classmethod
    def get_joint_count(cls) -> int:
        """Get the number of joints"""
        return len(cls)
    
    @classmethod
    def get_joint_by_index(cls, index: int) -> 'HandJoint':
        """Get joint by index"""
        joints = list(cls)
        if 0 <= index < len(joints):
            return joints[index]
        raise IndexError(f"Joint index {index} out of range")

@dataclass
class Joint:
    position: Position = field(default_factory=Position)
    rotation: Quaternion = field(default_factory=Quaternion)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Joint':
        return cls(
            position=Position.from_dict(data.get("position", {})),
            rotation=Quaternion.from_dict(data.get("rotation", {}))
        )

@dataclass
class HandData:
    is_tracked: bool = False
    confidence: float = 0.0
    hand_side: str = "unknown"  # "left" or "right"
    joints: List[Joint] = field(default_factory=list)
    
    def get_joint_position(self, joint: HandJoint) -> np.ndarray:
        """Get the position of the specified joint"""
        joint_index = list(HandJoint).index(joint)
        if joint_index < len(self.joints):
            return self.joints[joint_index].position.to_array()
        return np.zeros(3)
    
    def get_joint_rotation(self, joint: HandJoint) -> np.ndarray:
        """Get the rotation of the specified joint"""
        joint_index = list(HandJoint).index(joint)
        if joint_index < len(self.joints):
            return self.joints[joint_index].rotation.to_array()
        return np.array([0, 0, 0, 1])
    
    def get_palm_position(self) -> np.ndarray:
        """Get palm position"""
        return self.get_joint_position(HandJoint.PALM)
    
    def get_palm_rotation(self) -> np.ndarray:
        """Get palm rotation"""
        return self.get_joint_rotation(HandJoint.PALM)
    
    @classmethod
    def from_dict(cls, data: Dict, hand_side: str = "unknown") -> 'HandData':
        joints = []
        for joint_data in data.get("joints", []):
            joints.append(Joint.from_dict(joint_data))
        
        return cls(
            is_tracked=data.get("isTracked", False),
            confidence=data.get("confidence", 0.0),
            hand_side=hand_side,
            joints=joints
        )