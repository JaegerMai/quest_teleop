'''Receives and processes hand tracking data from Meta Quest '''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import threading
import numpy as np
import socket
import json
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
from teleop.joystick_data_types import Position, Quaternion
from teleop.hand_types import HandJoint, HandData, Joint
from teleop.meta_quest import MetaQuest

class MetaQuestHand(MetaQuest):
    """Meta Quest hand tracking class that extends the base MetaQuest functionality"""
    def __init__(self, use_headset_as_base_frame = True, is_palm_fixed: bool = False):
        
        if is_palm_fixed:
            use_headset_as_base_frame = False  # If palm is fixed, we don't use headset as base frame
        super().__init__(use_headset_as_base_frame)

        self.is_palm_fixed = is_palm_fixed

        self.hand_data: Dict[str, HandData] = {
            "left": HandData(hand_side="left"),
            "right": HandData(hand_side="right")
        }
    
    def _update_hand_data(self, hand_data: HandData, data: Dict):
        """Update hand tracking data from received UDP data
        """
        hand_data.is_tracked = data.get("isTracked", False)
        hand_data.confidence = data.get("confidence", 0.0)
        
        # Update joint data
        joints_data = data.get("joints", [])
        hand_data.joints.clear()
        
        for joint_data in joints_data:
            joint_raw = Joint.from_dict(joint_data)
            # Remap position and orientation to robot frame
            joint_position = self._remapping_position(joint_raw.position.to_array())
            joint_rotation = self._remapping_orientation(joint_raw.rotation.to_array())
            joint = Joint(
                position=Position.from_array(joint_position),
                rotation=Quaternion.from_array(joint_rotation)
            )
            hand_data.joints.append(joint)
    
    def process_data(self):
        """Override parent method to include hand tracking data processing"""
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
                data_dict = json.loads(data.decode('utf-8'))
                
                # Process headset and controller data (call parent method)
                if 'headset' in data_dict:
                    self._update_headset_pose(data_dict['headset'])
                if 'leftController' in data_dict:
                    self._update_controller_data(self.controller_data["left"], data_dict['leftController'])
                if 'rightController' in data_dict:
                    self._update_controller_data(self.controller_data["right"], data_dict['rightController'])
                
                # Process hand tracking data
                if 'leftHand' in data_dict:
                    self._update_hand_data(self.hand_data["left"], data_dict['leftHand'])
                if 'rightHand' in data_dict:
                    self._update_hand_data(self.hand_data["right"], data_dict['rightHand'])
                    
            except json.JSONDecodeError as e:
                print(f"[Warning] JSON decode error: {e}")
                continue

    def get_hand_joint_position(self, hand_side: str, joint: HandJoint) -> np.ndarray:
        """Get specific hand joint position
        Args:
            hand_side: "left" or "right"
            joint: Hand joint enum
        Returns:
            Joint position as numpy array [x, y, z]
        """
        if hand_side not in self.hand_data:
            raise ValueError(f"Invalid hand side: {hand_side}")
        
        hand = self.hand_data[hand_side]
        if not hand.is_tracked:
            return np.zeros(3)
        
        joint_position = hand.get_joint_position(joint)
        
        if self.is_palm_fixed:
            joint_position = joint_position - hand.get_palm_position()
            palm_orientation = hand.get_palm_rotation()
            r_palm = R.from_quat(palm_orientation)
            joint_position = r_palm.inv().apply(joint_position)

        # Apply transformation (same logic as palm position)
        pos_scaled = joint_position * self.pos_scale
        
        if self.use_headset_as_base_frame:
            pos_scaled = pos_scaled - self.get_headset_position()
            pos_scaled = R.from_quat(self.get_headset_orientation()).inv().apply(pos_scaled)
        
        pos_scaled += self.pos_offset[hand_side].to_array()
        
        return pos_scaled
    
    def get_hand_joint_orientation(self, hand_side: str, joint: HandJoint) -> np.ndarray:
        """Get specific hand joint orientation
        
        Args:
            hand_side: "left" or "right"
            joint: Hand joint enum
            
        Returns:
            Joint orientation as quaternion [x, y, z, w]
        """
        if hand_side not in self.hand_data:
            raise ValueError(f"Invalid hand side: {hand_side}")
        
        hand = self.hand_data[hand_side]
        if not hand.is_tracked:
            return np.array([0, 0, 0, 1])
        
        joint_orientation = hand.get_joint_rotation(joint)

        if self.is_palm_fixed:
            # If palm is fixed, use palm orientation to adjust joint orientation
            palm_orientation = hand.get_palm_rotation()
            r_palm = R.from_quat(palm_orientation)
            r_joint = R.from_quat(joint_orientation)
            joint_orientation = (r_palm.inv() * r_joint).as_quat()

        # Apply coordinate transformation
        if self.use_headset_as_base_frame:
            r_headset = R.from_quat(self.get_headset_orientation())
            r_joint = R.from_quat(joint_orientation)
            joint_orientation = (r_headset.inv() * r_joint).as_quat()
        
        # Apply orientation offset
        ori_offset = self.ori_offset[hand_side].to_array()
        ori_adjusted = R.from_quat(joint_orientation) * R.from_quat(ori_offset).inv()
        
        return ori_adjusted.as_quat()
    
    def get_all_hand_joint_positions(self, hand_side: str) -> np.ndarray:
        """Get all hand joint positions as a matrix
        
        Args:
            hand_side: "left" or "right"
            
        Returns:
            numpy array of shape (N, 3) where N is number of joints
            Each row represents [x, y, z] position of a joint
        """
        if hand_side not in self.hand_data:
            raise ValueError(f"Invalid hand side: {hand_side}")
        
        hand = self.hand_data[hand_side]
        if not hand.is_tracked or not hand.joints:
            return np.zeros((HandJoint.get_joint_count(), 3))
        
        joint_positions = []
        
        for joint_enum in HandJoint:
            try:
                position = self.get_hand_joint_position(hand_side, joint_enum)
                joint_positions.append(position)
            except IndexError:
                # If joint data is insufficient, fill with zeros
                joint_positions.append(np.zeros(3))
        
        return np.array(joint_positions)

    def get_all_hand_joint_orientations(self, hand_side: str) -> np.ndarray:
        """Get all hand joint orientations as a matrix
        
        Args:
            hand_side: "left" or "right"
            
        Returns:
            numpy array of shape (N, 4) where N is number of joints
            Each row represents [x, y, z, w] quaternion of a joint
        """
        if hand_side not in self.hand_data:
            raise ValueError(f"Invalid hand side: {hand_side}")
        
        hand = self.hand_data[hand_side]
        if not hand.is_tracked or not hand.joints:
            return np.tile([0, 0, 0, 1], (HandJoint.get_joint_count(), 1))
        
        joint_orientations = []
        
        for joint_enum in HandJoint:
            try:
                orientation = self.get_hand_joint_orientation(hand_side, joint_enum)
                joint_orientations.append(orientation)
            except IndexError:
                # If joint data is insufficient, fill with identity quaternion
                joint_orientations.append(np.array([0, 0, 0, 1]))
        
        return np.array(joint_orientations)
    

