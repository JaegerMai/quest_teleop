import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solar.third_party.quest_teleop.teleop.meta_quest import MetaQuest
from typing import Dict
from scipy.spatial.transform import Rotation as R

# constants
RIGHT = 'right'
LEFT  = 'left'

CONTROLLER_CONFIG = {
    "left": {
        "pos": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "ori": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    },
    "right": {
        "pos": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "ori": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    }
}

class MetaQuestController:
    def __init__(self, is_mirror_control=True):
        self.is_mirror_control = is_mirror_control
        self.meta_quest = MetaQuest(use_headset_as_base_frame=True)
        self.meta_quest.set_pos_scale(1.0)
        self.meta_quest.set_offset(LEFT, 'ori', CONTROLLER_CONFIG[LEFT]['ori'])
        self.meta_quest.set_offset(RIGHT, 'ori', CONTROLLER_CONFIG[RIGHT]['ori'])
        self.meta_quest.set_offset(LEFT, 'pos', CONTROLLER_CONFIG[LEFT]['pos'])
        self.meta_quest.set_offset(RIGHT, 'pos', CONTROLLER_CONFIG[RIGHT]['pos'])

        # Current state arrays
        self.joystick_ori: Dict[str, np.ndarray] = {
            "left": CONTROLLER_CONFIG[LEFT]['ori'],
            "right": CONTROLLER_CONFIG[RIGHT]['ori']
        }
        self.joystick_pos: Dict[str, np.ndarray] = {
            "left": CONTROLLER_CONFIG[LEFT]['pos'],
            "right": CONTROLLER_CONFIG[RIGHT]['pos']
        }
        
        # Control state tracking 
        self.unlock_pos: Dict[str, np.ndarray] = {
            "left": CONTROLLER_CONFIG[LEFT]['pos'].copy(),
            "right": CONTROLLER_CONFIG[RIGHT]['pos'].copy()
        }
        self.last_arm_pos: Dict[str, np.ndarray] = {
            "left": CONTROLLER_CONFIG[LEFT]['pos'].copy(),
            "right": CONTROLLER_CONFIG[RIGHT]['pos'].copy()
        }
        self.current_arm_pos: Dict[str, np.ndarray] = {
            "left": CONTROLLER_CONFIG[LEFT]['pos'].copy(),
            "right": CONTROLLER_CONFIG[RIGHT]['pos'].copy()
        }
        self.current_arm_ori: Dict[str, np.ndarray] = {
            "left": CONTROLLER_CONFIG[LEFT]['ori'].copy(),
            "right": CONTROLLER_CONFIG[RIGHT]['ori'].copy()
        }
        self.target_arm_pos: Dict[str, np.ndarray] = {
            "left": CONTROLLER_CONFIG[LEFT]['pos'].copy(),
            "right": CONTROLLER_CONFIG[RIGHT]['pos'].copy()
        }
        self.target_arm_ori: Dict[str, np.ndarray] = {
            "left": CONTROLLER_CONFIG[LEFT]['ori'].copy(),
            "right": CONTROLLER_CONFIG[RIGHT]['ori'].copy()
        }

        # Configuration
        self.trigger_threshold = 0.3
        
    def is_joystick_unlocked(self, hand_side: str) -> bool:
        """Check if joystick is unlocked for control
        Args:
            hand_side (str): "left" or "right"
        Returns:
            bool: True if hand trigger is pressed above threshold
        """
        return self.meta_quest.controller_data[hand_side].hand_trigger > self.trigger_threshold

    def update_joystick_state(self, hand_side: str, current_arm_pos: np.ndarray, current_arm_ori: np.ndarray):
        if hand_side not in self.meta_quest.controller_data:
            raise ValueError(f"Invalid hand side: {hand_side}")
        self.current_arm_pos[hand_side] = current_arm_pos
        self.current_arm_ori[hand_side] = current_arm_ori
        self._update_joystick_control(hand_side)

    def _update_joystick_control(self, hand_side: str):
        """Update control state for one hand
        
        Args:
            hand_side (str): "left" or "right"
        """
        if self.is_joystick_unlocked(hand_side):
            # Update orientation and position from VR controller
            self.joystick_ori[hand_side] = self.meta_quest.get_joystick_orientation(hand_side)
            self.joystick_pos[hand_side] = self.meta_quest.get_joystick_position(hand_side)
            
            # Calculate relative position movement
            delta_pos = self.joystick_pos[hand_side] - self.unlock_pos[hand_side]
            if self.is_mirror_control:
                mirror_pos, mirror_ori = self.mirror_control(delta_pos, self.joystick_ori[hand_side])
                self.target_arm_pos[hand_side] = self.last_arm_pos[hand_side] + mirror_pos
                self.target_arm_ori[hand_side] = mirror_ori
            else:
                self.target_arm_pos[hand_side] = self.last_arm_pos[hand_side] + delta_pos
                self.target_arm_ori[hand_side] = self.joystick_ori[hand_side]

        else:
            # Controller not unlocked - save current state
            self.last_arm_pos[hand_side] = self.current_arm_pos[hand_side].copy()
            self.unlock_pos[hand_side] = self.meta_quest.get_joystick_position(hand_side).copy()
    
    def mirror_control(self, pos: np.ndarray, ori: np.ndarray):
        # Mirror the position and orientation, the same as meta_quest remapping function
        pos_mirror = pos.copy()
        mat = np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ])
        pos_mirror = mat @ pos_mirror.reshape(3, 1)
        pos_mirror = pos_mirror.flatten()

        ori_mirror = ori.copy()
        rotation_matrix = R.from_quat(ori_mirror).as_matrix()
        world_to_body = rotation_matrix.T
        axis_remap_matrix = np.array([[-1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, 1]])
        transformed_matrix = axis_remap_matrix @ world_to_body @ axis_remap_matrix.T
        body_to_world = transformed_matrix.T
        ori_mirror = R.from_matrix(body_to_world).as_quat()
        
        return pos_mirror, ori_mirror
        
    def input2action(self, controller_type="OSC_POSE"):
        if not self.is_joystick_unlocked(LEFT):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
            # gripper can still be controlled when joystick is locked
            gripper_control = np.array(self.meta_quest.controller_data[LEFT].thumbstick.to_array())
            return action, gripper_control
        
        dpos = (self.target_arm_pos[LEFT] - self.current_arm_pos[LEFT]).flatten() * 10

        # Calculate delta rotation
        drot = np.zeros(3, dtype=np.float64)
        d_rotation_mat = R.from_quat(self.target_arm_ori[LEFT]) * R.from_quat(self.current_arm_ori[LEFT]).inv()

        if controller_type == "OSC_POSE":
            drot = d_rotation_mat.as_rotvec()
        if controller_type == "OSC_YAW":
            drot = np.array([0, 0, d_rotation_mat.as_rotvec()[2]])
        if controller_type == "OSC_POSITION":
            drot = np.zeros(3, dtype=np.float64)
        
        grasp = 1.0 if self.meta_quest.controller_data[LEFT].index_trigger > 0.5 else -1.0
        grasp = np.array([grasp], dtype=np.float64)
        dpos = np.clip(dpos, -1.0, 1.0)
        drot = np.clip(drot, -0.3, 0.3)
        action = np.concatenate([dpos, drot, grasp])
        gripper_control = np.array(self.meta_quest.controller_data[LEFT].thumbstick.to_array())
        return action, gripper_control
