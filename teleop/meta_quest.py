'''Receives and processes joystick data from Meta Quest '''
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
from teleop.joystick_data_types import Position, Quaternion, Thumbstick, ControllerData

class MetaQuest:
    def __init__(self, use_headset_as_base_frame: bool = True):
        self.use_headset_as_base_frame = use_headset_as_base_frame
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', 8888))
        
        # Joystick data
        self.controller_data: Dict[str, ControllerData] = {
            "left": ControllerData(hand_side="left"),
            "right": ControllerData(hand_side="right")
        }
        self.headset_pose = {
            "position": Position(),
            "orientation": Quaternion()
        }
        
        self.pos_scale = 1.0  # Scale factor for position data
        self.pos_offset: Dict[str, Position] = {
            "left": Position(),
            "right": Position()
        }
        self.ori_offset: Dict[str, Quaternion] = {
            "left": Quaternion(),
            "right": Quaternion()
        }
        
        # Thread for receiving data
        self.thread = threading.Thread(target=self.process_data)
        self.thread.daemon = True  
        self.thread.start()
        
    def process_data(self):
        while True:
            data, _ = self.sock.recvfrom(65536)
            try:
                data_dict = json.loads(data.decode('utf-8'))
                self._update_headset_pose(data_dict['headset'])
                self._update_controller_data(self.controller_data["left"], data_dict['leftController'])
                self._update_controller_data(self.controller_data["right"], data_dict['rightController'])
            except json.JSONDecodeError as e:
                    print(f"[Warning] JSON decode error: {e}")
                    continue

    def _update_headset_pose(self, data: Dict):
        position_raw = Position.from_dict(data.get("position", {}))
        position = self._remapping_position(position_raw.to_array()) # Remapping position to robot frame
        self.headset_pose["position"] = Position.from_array(position)

        orientation_raw = Quaternion.from_dict(data.get("rotation", {}))
        orientation = self._remapping_orientation(orientation_raw.to_array()) # Remapping orientation to robot frame
        self.headset_pose["orientation"] = Quaternion.from_array(orientation)

    def _update_controller_data(self, controller: ControllerData, data: Dict):
        position_raw = Position.from_dict(data.get("position", {}))
        position = self._remapping_position(position_raw.to_array()) # Remapping position to robot frame
        controller.position = Position.from_array(position)
        
        orientation_raw = Quaternion.from_dict(data.get("rotation", {}))
        orientation = self._remapping_orientation(orientation_raw.to_array()) # Remapping orientation to robot frame
        controller.rotation = Quaternion.from_array(orientation)
        
        controller.thumbstick = Thumbstick.from_dict(data.get("thumbstick", {}))
        controller.index_trigger = data.get("index_trigger", 0.0)
        controller.hand_trigger = data.get("hand_trigger", 0.0)

        # Map button names to controller buttons
        buttons_list = data.get("buttons", [])
        for btn in buttons_list:
            name = btn.get("buttonName", "")
            pressed = btn.get("pressed", False)
            if name == "One":
                controller.buttons["A"] = pressed
            elif name == "Two":
                controller.buttons["B"] = pressed
            elif name == "Three":
                controller.buttons["X"] = pressed
            elif name == "Four":
                controller.buttons["Y"] = pressed     
    
    def get_headset_position(self) -> np.ndarray:
        pos_raw = self.headset_pose["position"].to_array()
        return pos_raw * self.pos_scale
    
    def get_headset_orientation(self) -> np.ndarray:
        return self.headset_pose["orientation"].to_array()
    
    def get_joystick_position(self, hand_side: str) -> np.ndarray:
        if hand_side not in self.controller_data:
            raise ValueError(f"Invalid hand side: {hand_side}. Use 'left' or 'right'.")
        pos_raw = self.controller_data[hand_side].get_controller_position()
        pos_scaled = pos_raw * self.pos_scale
        
        # if use_headset_as_base_frame is True, adjust position based on headset position
        if self.use_headset_as_base_frame:
            pos_scaled = pos_scaled - self.get_headset_position()
            pos_scaled = R.from_quat(self.get_headset_orientation()).inv().apply(pos_scaled)

        # apply position offset
        pos_scaled += self.pos_offset[hand_side].to_array()

        return pos_scaled
    
    def get_joystick_orientation(self, hand_side: str) -> np.ndarray:
        if hand_side not in self.controller_data:
            raise ValueError(f"Invalid hand side: {hand_side}. Use 'left' or 'right'.")
        ori_raw = self.controller_data[hand_side].get_controller_rotation()
        
        # if use_headset_as_base_frame is True, adjust orientation based on headset orientation
        if self.use_headset_as_base_frame:
            r_headset = R.from_quat(self.get_headset_orientation())
            r_ori = R.from_quat(ori_raw)

            ori_raw = (r_headset.inv() * r_ori).as_quat()

        # apply orientation offset
        ori_offset = self.ori_offset[hand_side].to_array()
        ori_adjusted = R.from_quat(ori_raw) * R.from_quat(ori_offset).inv()

        return ori_adjusted.as_quat()

    def set_pos_scale(self, scale: float):
        self.pos_scale = scale
        
    def set_offset(self, hand_side: str, offset_type: str, offset_value: np.ndarray):
        """ Set position or orientation offset for a specific hand side.
        Args:
            hand_side (str): "left" or "right"
            offset_type (str): "pos" for position, "ori" for orientation
            offset_value (np.ndarray): The offset value as a numpy array, the shape of pos is (3,1) and ori is (4,)
        """
        if hand_side not in self.pos_offset or hand_side not in self.ori_offset:
            raise ValueError(f"Invalid hand side: {hand_side}. Use 'left' or 'right'.")
        
        if offset_type == "pos":
            self.pos_offset[hand_side] = Position.from_array(offset_value)
        elif offset_type == "ori":
            self.ori_offset[hand_side] = Quaternion.from_array(offset_value)
        else:
            raise ValueError(f"Invalid offset type: {offset_type}. Use 'pos' or 'ori'.")
        
    def _remapping_position(self, position: np.ndarray) -> np.ndarray:
        '''Remap position frame from meta quest to robot frame.
            meta quest frame is left-handed(z-axis to the front), 
            robot frame is right-handed(x-axis to the front).'''
        mat = np.array([
                [0, 0, 1],
                [-1, 0, 0],
                [0, 1, 0],
            ])
        position = mat @ position.reshape(3, 1)
        return position.flatten()

    def _remapping_orientation(self, orientation: np.ndarray) -> np.ndarray:
        """
        Remap orientation from Meta Quest frame to robot frame.
        
        Transforms orientation from Meta Quest's left-handed coordinate system (Z-axis forward)
        to robot's right-handed coordinate system (X-axis forward).
        
        The transformation achieves the mapping: [roll, pitch, yaw] → [-yaw, roll, -pitch]
        by performing axis swapping and sign flipping operations using quaternion operations
        to avoid gimbal lock issues that would occur with direct Euler angle manipulation.
        
        Args:
            orientation (np.ndarray): Input quaternion as [x, y, z, w] from Meta Quest
            
        Returns:
            np.ndarray: Transformed quaternion as [x, y, z, w] in robot frame
        """
        
        # Convert quaternion to rotation matrix (body-to-world)
        rotation_matrix = R.from_quat(orientation).as_matrix()

        # Transpose to get world-to-body transformation
        world_to_body = rotation_matrix.T

        # Define axis remapping matrix Q for coordinate transformation
        # Equivalent to: [roll, pitch, yaw] → [-yaw, roll, -pitch]
        axis_remap_matrix = np.array([[ 0,  0, -1],
                                     [ 1,  0,  0],
                                     [ 0, -1,  0]])

        # Apply conjugation transformation: Q @ M_inv @ Q.T
        transformed_matrix = axis_remap_matrix @ world_to_body @ axis_remap_matrix.T
        
        # Convert back to body-to-world representation
        body_to_world = transformed_matrix.T

        # Convert rotation matrix back to quaternion
        transformed_quaternion = R.from_matrix(body_to_world).as_quat()
        
        return transformed_quaternion

        