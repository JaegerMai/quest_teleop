"""PyBullet simulation environments for VR visualization"""
import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict
from .hand_types import HandJoint


class PyBulletSimulator:
    """PyBullet simulation environment wrapper for basic VR control"""
    
    def __init__(self, gui: bool = True, gravity: float = -9.81):
        """Initialize PyBullet simulation environment
        """
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, gravity)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load ground plane
        self.ground_id = p.loadURDF("plane.urdf")
        
        # Store body IDs and debug line IDs
        self.bodies = {}
        self.debug_lines = {}
        self._frame_counter = 0
        self._draw_frequency = 10  # Draw frames every 30 simulation steps
        
    def create_box(self, name: str, position: np.ndarray, size: np.ndarray = None, 
                   mass: float = 1.0, color: np.ndarray = None) -> int:
        """Create a box object in the simulation
        """
        if size is None:
            size = np.array([0.2, 0.2, 0.2])
        if color is None:
            color = np.array([1, 0, 0, 0.5])
            
        half_extents = size / 2
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        self.bodies[name] = body_id
        return body_id
    
    def set_pose(self, name: str, position: np.ndarray, quaternion: np.ndarray) -> bool:
        """Set both position and orientation of an object
        """
        if name not in self.bodies:
            return False
            
        # Validate inputs
        if len(position) != 3 or len(quaternion) != 4:
            print(f"Warning: Invalid pose data for {name}")
            return False
            
        if np.any(np.isnan(position)) or np.any(np.isnan(quaternion)):
            print(f"Warning: NaN values in pose data for {name}")
            return False
            
        p.resetBasePositionAndOrientation(self.bodies[name], position, quaternion)
        return True
    
    def change_color(self, name: str, color: np.ndarray) -> bool:
        """Change the color of an object
        """
        if name not in self.bodies:
            return False
        p.changeVisualShape(self.bodies[name], -1, rgbaColor=color)
        return True
    
    def draw_body_frame(self, name: str, axis_len: float = 0.3) -> bool:
        """Draw coordinate frame for an object with optimized frequency
        """
        if name not in self.bodies:
            return False
            
        # Only draw every N frames to reduce computational load
        if self._frame_counter % self._draw_frequency != 0:
            return False
            
        pos, orn = p.getBasePositionAndOrientation(self.bodies[name])
        
        # Validate pose data
        if np.any(np.isnan(pos)) or np.any(np.isnan(orn)):
            return False
            
        # Clear old debug lines for this object
        if name in self.debug_lines:
            for line_id in self.debug_lines[name]:
                try:
                    p.removeUserDebugItem(line_id)
                except:
                    pass
        
        # Create rotation matrix and axis vectors
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        x_axis = rot_matrix[:, 0] * axis_len
        y_axis = rot_matrix[:, 1] * axis_len  
        z_axis = rot_matrix[:, 2] * axis_len
        
        # Draw coordinate axes with persistent lines
        line_ids = []
        line_ids.append(p.addUserDebugLine(pos, pos + x_axis, [1, 0, 0], lineWidth=5, lifeTime=0.3))
        line_ids.append(p.addUserDebugLine(pos, pos + y_axis, [0, 1, 0], lineWidth=5, lifeTime=0.3))
        line_ids.append(p.addUserDebugLine(pos, pos + z_axis, [0, 0, 1], lineWidth=5, lifeTime=0.3))
        
        self.debug_lines[name] = line_ids
        return True
    
    def step_simulation(self):
        """Execute one simulation step and increment frame counter"""
        p.stepSimulation()
        self._frame_counter += 1
    
    def clear_debug_lines(self):
        """Clear all debug lines"""
        for name, line_ids in self.debug_lines.items():
            for line_id in line_ids:
                try:
                    p.removeUserDebugItem(line_id)
                except:
                    pass
        self.debug_lines.clear()
    
    def disconnect(self):
        """Disconnect from PyBullet and clean up resources"""
        self.clear_debug_lines()
        p.disconnect(self.physics_client)
    
    def get_body_id(self, name: str) -> int:
        """Get PyBullet body ID for an object
        """
        return self.bodies.get(name)


class PyBulletHandSimulator:
    """PyBullet simulation environment for hand tracking visualization"""
    
    def __init__(self, gui: bool = True, gravity: float = -9.81, is_palm_fixed: bool = False):
        """Initialize PyBullet simulation environment
        
        Args:
            gui: Whether to use GUI mode
            gravity: Gravity value
            is_palm_fixed: Whether palm is fixed (affects visualization mode)
        """
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, gravity)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load ground plane
        self.ground_id = p.loadURDF("plane.urdf")
        
        # Store body IDs and debug line IDs
        self.bodies = {}
        self.debug_lines = {}
        self._frame_counter = 0
        self._draw_frequency = 10  # Draw frames every 10 simulation steps
        
        self.is_palm_fixed = is_palm_fixed
        
        # Initialize joint visualization
        self._create_hand_joint_visualization()
        
        if not is_palm_fixed:
            self._create_headset_visualization()
        else:
            # Set camera view from top when palm is fixed
            self._set_top_down_camera_view()
        
    def _create_hand_joint_visualization(self):
        """Create balls for hand joint visualization similar to leap_pybullet"""
        # Ball settings
        ball_radius = 0.02
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        small_ball_radius = 0.015
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius)
        baseMass = 0.001
        basePosition = [0, 0, 0]
        
        # Define colors for different fingers
        colors = {
            'THUMB': [1, 0, 0, 1],     # Red
            'INDEX': [0, 1, 0, 1],     # Green
            'MIDDLE': [0, 0, 1, 1],    # Blue
            'RING': [1, 1, 0, 1],      # Yellow
            'LITTLE': [1, 0, 1, 1],    # Magenta
            'WRIST': [0, 0, 0, 1],     # Black
            'PALM': [0.5, 0.5, 0.5, 1] # Gray
        }
        
        # Create balls for all hand joints for both hands
        for hand_side in ['left', 'right']:
            for joint_enum in HandJoint:
                joint_name = joint_enum.name
                ball_key = f'{hand_side}_{joint_name}'
                
                # Use larger balls for fingertips, smaller for other joints
                if 'TIP' in joint_name:
                    _ball_shape = ball_shape
                else:
                    _ball_shape = small_ball_shape
                
                ball = p.createMultiBody(
                    baseMass=baseMass, 
                    baseCollisionShapeIndex=_ball_shape, 
                    basePosition=basePosition
                )
                self.bodies[ball_key] = ball
                
                # Determine color based on finger
                finger_name = joint_name.split('_')[0]  # Extract finger name
                color = colors.get(finger_name, [1, 1, 1, 1])  # Default white if not found
                
                p.changeVisualShape(ball, -1, rgbaColor=color)
                p.setCollisionFilterGroupMask(ball, -1, 0, 0)
        
        # Add palm offset positions for when palm is not fixed
        if not self.is_palm_fixed:
            self.palm_offset = {
                'left': np.array([0.0, 0.0, 0.5]),
                'right': np.array([0.0, 0.0, 0.5])
            }
        else:
            # When palm is fixed, set palm positions for visualization
            self.palm_offset = {
                'left': np.array([0.0, 0.0, 0.0]),   # Better symmetry around origin
                'right': np.array([0.0, 0.0, 0.0])
            }
    
    def _set_top_down_camera_view(self):
        """Set camera to look down from z-axis when palm is fixed"""
        camera_distance = 1.0   # Reduced distance for better view
        camera_yaw = 0  # Looking along x-axis
        camera_pitch = -89  # Almost straight down
        camera_target_position = [0.0, 0.0, 0.0]  # Looking at origin
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target_position
        )
    
    def _create_headset_visualization(self):
        """Create headset visualization as a rectangular box"""
        if not self.is_palm_fixed:
            headset_size = np.array([0.25, 0.15, 0.1])
            headset_color = np.array([0, 0.8, 0, 0.7])  # Green
            
            self.create_box(
                "headset",
                position=np.array([0, 0, 0.3]),  # Head height
                size=headset_size,
                color=headset_color
            )
    
    def create_box(self, name: str, position: np.ndarray, size: np.ndarray = None, 
                   mass: float = 1.0, color: np.ndarray = None) -> int:
        """Create a box object in the simulation"""
        if size is None:
            size = np.array([0.2, 0.2, 0.2])
        if color is None:
            color = np.array([1, 0, 0, 0.5])
            
        half_extents = size / 2
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        self.bodies[name] = body_id
        return body_id
    
    def set_pose(self, name: str, position: np.ndarray, quaternion: np.ndarray = None) -> bool:
        """Set position and optionally orientation of an object"""
        if name not in self.bodies:
            return False
            
        # Validate inputs
        if len(position) != 3:
            print(f"Warning: Invalid position data for {name}")
            return False
            
        if np.any(np.isnan(position)):
            print(f"Warning: NaN values in position data for {name}")
            return False
        
        if quaternion is not None:
            if len(quaternion) != 4 or np.any(np.isnan(quaternion)):
                print(f"Warning: Invalid quaternion data for {name}")
                return False
            p.resetBasePositionAndOrientation(self.bodies[name], position, quaternion)
        else:
            # Only set position
            current_pos, current_orn = p.getBasePositionAndOrientation(self.bodies[name])
            p.resetBasePositionAndOrientation(self.bodies[name], position, current_orn)
        
        return True
    
    def update_hand_joints(self, hand_side: str, joint_positions: np.ndarray):
        """Update hand joint positions in simulation
        
        Args:
            hand_side: "left" or "right"
            joint_positions: Array of shape (N, 3) with joint positions
        """
        if joint_positions.shape[0] == 0:
            return
        
        # Apply palm-fixed rotation if needed
        if self.is_palm_fixed and joint_positions.shape[0] > 0:
            joint_positions = self._apply_palm_fixed_rotation(hand_side, joint_positions)
            
        for i, joint_enum in enumerate(HandJoint):
            if i >= joint_positions.shape[0]:
                break
                
            joint_name = joint_enum.name
            ball_key = f'{hand_side}_{joint_name}'
            
            if ball_key in self.bodies:
                position = joint_positions[i]
                
                # Add palm offset
                position = position + self.palm_offset[hand_side]
                
                self.set_pose(ball_key, position)
    
    def _apply_palm_fixed_rotation(self, hand_side: str, joint_positions: np.ndarray) -> np.ndarray:
        """Apply 90-degree clockwise rotation around palm for palm-fixed mode
        
        Args:
            hand_side: "left" or "right"
            joint_positions: Original joint positions
            
        Returns:
            Rotated joint positions
        """
        if joint_positions.shape[0] == 0:
            return joint_positions
            
        # Get palm position (assuming it's the first joint - PALM)
        palm_pos = joint_positions[0] if joint_positions.shape[0] > 0 else np.zeros(3)
        
        # Create 90-degree clockwise rotation around z-axis
        # Clockwise rotation matrix around z-axis
        rotation_angle = np.pi/2  
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle), np.cos(rotation_angle), 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to all joints relative to palm
        rotated_positions = np.zeros_like(joint_positions)
        for i in range(joint_positions.shape[0]):
            # Translate to palm center
            relative_pos = joint_positions[i] - palm_pos
            # Apply rotation
            rotated_relative = rotation_matrix @ relative_pos
            # Translate back
            rotated_positions[i] = rotated_relative + palm_pos
            
        return rotated_positions
    
    def draw_body_frame(self, name: str, axis_len: float = 0.1) -> bool:
        """Draw coordinate frame for an object with optimized frequency"""
        if name not in self.bodies:
            return False
            
        # Only draw every N frames to reduce computational load
        if self._frame_counter % self._draw_frequency != 0:
            return False
            
        pos, orn = p.getBasePositionAndOrientation(self.bodies[name])
        
        # Validate pose data
        if np.any(np.isnan(pos)) or np.any(np.isnan(orn)):
            return False
            
        # Clear old debug lines for this object
        if name in self.debug_lines:
            for line_id in self.debug_lines[name]:
                try:
                    p.removeUserDebugItem(line_id)
                except:
                    pass
        
        # Create rotation matrix and axis vectors
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        x_axis = rot_matrix[:, 0] * axis_len
        y_axis = rot_matrix[:, 1] * axis_len  
        z_axis = rot_matrix[:, 2] * axis_len
        
        # Draw coordinate axes with persistent lines
        line_ids = []
        line_ids.append(p.addUserDebugLine(pos, pos + x_axis, [1, 0, 0], lineWidth=3, lifeTime=0.3))
        line_ids.append(p.addUserDebugLine(pos, pos + y_axis, [0, 1, 0], lineWidth=3, lifeTime=0.3))
        line_ids.append(p.addUserDebugLine(pos, pos + z_axis, [0, 0, 1], lineWidth=3, lifeTime=0.3))
        
        self.debug_lines[name] = line_ids
        return True
    
    def step_simulation(self):
        """Execute one simulation step and increment frame counter"""
        p.stepSimulation()
        self._frame_counter += 1
    
    def clear_debug_lines(self):
        """Clear all debug lines"""
        for name, line_ids in self.debug_lines.items():
            for line_id in line_ids:
                try:
                    p.removeUserDebugItem(line_id)
                except:
                    pass
        self.debug_lines.clear()
    
    def disconnect(self):
        """Disconnect from PyBullet and clean up resources"""
        self.clear_debug_lines()
        p.disconnect(self.physics_client)
