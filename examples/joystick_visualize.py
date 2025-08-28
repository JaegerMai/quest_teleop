import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import argparse
import pybullet as p
from quest_teleop.meta_quest import MetaQuest
from quest_teleop.pybullet_simulators import PyBulletSimulator
from typing import Dict

# Controller configuration constants
CONTROLLER_CONFIG = {
    "left": {
        "pos": np.array([0.0, 0.5, 0.0], dtype=np.float64),
        "ori": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    },
    "right": {
        "pos": np.array([0.0, -0.5, 0.0], dtype=np.float64),
        "ori": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    }
}

class MetaQuestController:
    """Controller for Meta Quest VR input and PyBullet simulation integration"""
    
    def __init__(self, simulator: PyBulletSimulator, meta_quest: MetaQuest, debug: bool = False):
        """Initialize the controller
        
        Args:
            simulator (PyBulletSimulator): PyBullet simulation instance
            meta_quest (MetaQuest): Meta Quest VR interface instance
            debug (bool): Whether to enable debug visualization (draw body frames)
        """
        self.simulator = simulator
        self.meta_quest = meta_quest
        self.debug = debug

        # Current state arrays
        self.joystick_ori: Dict[str, np.ndarray] = {
            "left": np.array([0, 0, 0, 1], dtype=np.float64),
            "right": np.array([0, 0, 0, 1], dtype=np.float64)
        }
        self.joystick_pos: Dict[str, np.ndarray] = {
            "left": np.zeros(3, dtype=np.float64),
            "right": np.zeros(3, dtype=np.float64)
        }
        self.headset_ori = np.array([0, 0, 0, 1], dtype=np.float64)
        self.headset_pos = np.zeros(3, dtype=np.float64)
        self.control_color: Dict[str, np.ndarray] = {
            "left": np.array([0, 0, 1, 1], dtype=np.float64),  # Default blue
            "right": np.array([0, 0, 1, 1], dtype=np.float64)
        }
        
        # Control state tracking - Use unified configuration
        self.unlock_pos: Dict[str, np.ndarray] = {
            "left": np.zeros(3, dtype=np.float64),
            "right": np.zeros(3, dtype=np.float64)
        }
        self.last_cube_pos: Dict[str, np.ndarray] = {
            "left": CONTROLLER_CONFIG["left"]["pos"].copy(),
            "right": CONTROLLER_CONFIG["right"]["pos"].copy()
        }
        self.current_cube_pos: Dict[str, np.ndarray] = {
            "left": CONTROLLER_CONFIG["left"]["pos"].copy(),
            "right": CONTROLLER_CONFIG["right"]["pos"].copy()
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
    
    def _get_control_color(self, hand_side: str) -> np.ndarray:
        """Determine control color based on button presses and trigger state
        """
        controller = self.meta_quest.controller_data[hand_side]
        color_intensity = 1.0 - controller.index_trigger
        
        # Color mapping based on button presses
        if hand_side == "left":
            if controller.buttons.get("X", False):
                return np.array([1, 0, 0, color_intensity])  # Red
            elif controller.buttons.get("Y", False):
                return np.array([0, 1, 0, color_intensity])  # Green
        else:  # right
            if controller.buttons.get("A", False):
                return np.array([1, 0, 0, color_intensity])  # Red
            elif controller.buttons.get("B", False):
                return np.array([0, 1, 0, color_intensity])  # Green
        
        return np.array([0, 0, 1, color_intensity])  # Default blue
    
    def _update_hand_control(self, hand_side: str):
        """Update control state for one hand
        
        Args:
            hand_side (str): "left" or "right"
        """
        if self.is_joystick_unlocked(hand_side):
            # Update orientation and position from VR controller
            self.joystick_ori[hand_side] = self.meta_quest.get_joystick_orientation(hand_side)
            self.joystick_pos[hand_side] = self.meta_quest.get_joystick_position(hand_side)
            
            # Update color based on button state
            self.control_color[hand_side] = self._get_control_color(hand_side)
            
            # Calculate relative position movement
            delta_pos = self.joystick_pos[hand_side] - self.unlock_pos[hand_side]
            self.current_cube_pos[hand_side] = self.last_cube_pos[hand_side] + delta_pos
        else:
            # Controller not unlocked - save current state
            self.last_cube_pos[hand_side] = self.current_cube_pos[hand_side].copy()
            self.unlock_pos[hand_side] = self.meta_quest.get_joystick_position(hand_side).copy()

    def update_joystick_data(self):
        """Update all joystick and headset data from Meta Quest"""
        # Update both hands
        self._update_hand_control("left")
        self._update_hand_control("right")

        # Update headset data
        self.headset_ori = self.meta_quest.get_headset_orientation()
        self.headset_pos = self.meta_quest.get_headset_position()
        
        # Reset headset position if using headset as base frame
        if self.meta_quest.use_headset_as_base_frame:
            self.headset_pos = np.zeros(3, dtype=np.float64)
    
    def update_simulation(self):
        """Update the simulation with the latest VR input data"""
        # Update all VR input data
        self.update_joystick_data()
        
        # Update simulation objects
        self._update_cube("left_cube", "left")
        self._update_cube("right_cube", "right")
        self._update_headset()
            
        # Step the simulation
        self.simulator.step_simulation()
    
    def _update_cube(self, cube_name: str, hand_side: str):
        """Update a cube object in the simulation
        
        Args:
            cube_name (str): Name of the cube in simulation
            hand_side (str): Associated hand side ("left" or "right")
        """
        if cube_name in self.simulator.bodies:
            success = self.simulator.set_pose(
                cube_name, 
                self.current_cube_pos[hand_side], 
                self.joystick_ori[hand_side]
            )
            # print(f"Orientation for {hand_side} cube: {self.joystick_ori[hand_side]}") # Debugging line
            if success:
                self.simulator.change_color(cube_name, self.control_color[hand_side])
                if self.debug:
                    self.simulator.draw_body_frame(cube_name)
    
    def _update_headset(self):
        """Update headset representation in simulation"""
        if "headset" in self.simulator.bodies:
            success = self.simulator.set_pose("headset", self.headset_pos, self.headset_ori)
            if success and self.debug:
                self.simulator.draw_body_frame("headset")

def setup_meta_quest() -> MetaQuest:
    """Configure and initialize Meta Quest interface
    
    Returns:
        MetaQuest: Configured Meta Quest instance
    """
    meta_quest = MetaQuest(use_headset_as_base_frame=True)
    
    # Configure scaling and offsets
    meta_quest.set_pos_scale(2.0)
    meta_quest.set_offset("left", "pos", CONTROLLER_CONFIG["left"]["pos"])
    meta_quest.set_offset("right", "pos", CONTROLLER_CONFIG["right"]["pos"])
    meta_quest.set_offset("left", "ori", CONTROLLER_CONFIG["left"]["ori"])
    meta_quest.set_offset("right", "ori", CONTROLLER_CONFIG["right"]["ori"])

    return meta_quest

def create_simulation_objects(simulator: PyBulletSimulator):
    """Create all objects in the simulation
    
    Args:
        simulator (PyBulletSimulator): Simulator instance to add objects to
    """
    # Create cubes for left and right controllers (mass=0 makes them kinematic)
    simulator.create_box(
        "left_cube", 
        position=CONTROLLER_CONFIG["left"]["pos"], 
        size=np.array([0.2, 0.2, 0.5]),
        mass=0.0  # Kinematic body - no physics simulation
    )
    simulator.create_box(
        "right_cube", 
        position=CONTROLLER_CONFIG["right"]["pos"], 
        size=np.array([0.2, 0.2, 0.5]),
        mass=0.0  # Kinematic body - no physics simulation
    )
    
    # Create headset representation (also kinematic)
    simulator.create_box(
        "headset", 
        position=np.array([0, 0, 0.5]), 
        size=np.array([0.2, 0.2, 0.4]), 
        color=np.array([0, 1, 0, 0.5]),
        mass=0.0  # Kinematic body - no physics simulation
    )

def main():
    """Main function to run the VR teleoperation demo"""
    parser = argparse.ArgumentParser(description='Meta Quest VR Teleoperation Demo')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (show body frames)')
    args = parser.parse_args()
    
    print(f"Initializing VR Teleoperation Demo (Debug: {args.debug})...")
    
    try:
        # Initialize PyBullet simulator
        simulator = PyBulletSimulator(gui=True)
        print("PyBullet simulator initialized")
        
        # Set window title
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
        # Note: PyBullet doesn't have direct window title setting, but we can add text overlay
        
        # Create simulation objects
        create_simulation_objects(simulator)
        print("Simulation objects created")

        # Initialize Meta Quest interface
        meta_quest = setup_meta_quest()
        print("Meta Quest interface initialized")
        
        # Create controller
        controller = MetaQuestController(simulator, meta_quest, debug=args.debug)
        print("Controller initialized")
        
        # Add instruction text overlay
        instruction_text = p.addUserDebugText(
            text="Press hand trigger to activate control",
            textPosition=[0, 0, 0.5],
            textColorRGB=[1, 1, 0],  # Yellow text
            textSize=1.5,
            lifeTime=0  # Permanent text
        )
        
        print("Starting main loop (Press Ctrl+C to stop)...")
        
        # Main simulation loop
        frame_count = 0
        start_time = time.time()
        
        while True:
            controller.update_simulation()
            time.sleep(1/60)  # Run at 60Hz instead of 240Hz to reduce jitter
            
            frame_count += 1
            if frame_count % 600 == 0:  # Print FPS every 10 seconds at 60Hz
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Running at {fps:.1f} FPS")
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        simulator.disconnect()
        print("Disconnected from PyBullet.")
        
if __name__ == "__main__":
    main()