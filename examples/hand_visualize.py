import pybullet as p
import pybullet_data
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import argparse
import traceback
from teleop.meta_quest_hand import MetaQuestHand
from teleop.hand_types import HandJoint
from teleop.pybullet_simulators import PyBulletHandSimulator
from scipy.spatial.transform import Rotation as R
from typing import Dict

class MetaQuestHandController:
    """Controller for Meta Quest hand tracking and PyBullet simulation integration"""
    
    def __init__(self, simulator: PyBulletHandSimulator, meta_quest_hand: MetaQuestHand, debug: bool = False):
        """Initialize the controller
        
        Args:
            simulator: PyBullet simulation instance
            meta_quest_hand: Meta Quest hand tracking interface instance
            debug: Whether to enable debug visualization (draw body frames)
        """
        self.simulator = simulator
        self.meta_quest_hand = meta_quest_hand
        self.debug = debug
        
        # Current state tracking
        self.headset_pos = np.array([0, 0, 0.5], dtype=np.float64)
        self.headset_ori = np.array([0, 0, 0, 1], dtype=np.float64)
        
    def update_hand_data(self):
        """Update all hand tracking data from Meta Quest"""
        # Update both hands joint positions
        for hand_side in ['left', 'right']:
            # Get all joint positions for this hand
            joint_positions = self.meta_quest_hand.get_all_hand_joint_positions(hand_side)
            
            # Update simulation with joint positions
            self.simulator.update_hand_joints(hand_side, joint_positions)
        
        # Update headset data if not in palm fixed mode
        if not self.simulator.is_palm_fixed:
            # self.headset_pos = self.meta_quest_hand.get_headset_position()
            self.headset_ori = self.meta_quest_hand.get_headset_orientation()
    
    def update_simulation(self):
        """Update the simulation with the latest hand tracking data"""
        # Update all hand tracking data
        self.update_hand_data()
        
        # Update headset visualization if available
        if not self.simulator.is_palm_fixed and "headset" in self.simulator.bodies:
            success = self.simulator.set_pose("headset", self.headset_pos, self.headset_ori)
            if success and self.debug:
                self.simulator.draw_body_frame("headset", axis_len=0.15)
        
        # Draw palm frames for both hands (only in debug mode)
        if self.debug:
            for hand_side in ['left', 'right']:
                palm_key = f'{hand_side}_PALM'
                if palm_key in self.simulator.bodies:
                    self.simulator.draw_body_frame(palm_key, axis_len=0.08)
            
        # Step the simulation
        self.simulator.step_simulation()

def setup_meta_quest_hand(is_palm_fixed: bool) -> MetaQuestHand:
    """Configure and initialize Meta Quest hand tracking interface
    
    Args:
        is_palm_fixed: Whether palm should be fixed
        
    Returns:
        MetaQuestHand: Configured Meta Quest hand tracking instance
    """
    meta_quest_hand = MetaQuestHand(
        use_headset_as_base_frame=not is_palm_fixed,
        is_palm_fixed=is_palm_fixed
    )
    
    # Configure scaling and offsets
    meta_quest_hand.set_pos_scale(1.5)
    
    if not is_palm_fixed:
        # When not fixed, set offsets for both hands
        meta_quest_hand.set_offset("left", "pos", np.array([0.0, 0.0, 0.0]))
        meta_quest_hand.set_offset("right", "pos", np.array([0.0, 0.0, 0.0]))
    else:
        # When fixed, hands are relative to palm
        meta_quest_hand.set_offset("left", "pos", np.array([-0.3, 0.0, 0.5]))
        meta_quest_hand.set_offset("right", "pos", np.array([0.3, 0.0, 0.5]))
    
    # Set default orientation offsets
    meta_quest_hand.set_offset("left", "ori", np.array([0.0, 0.0, 0.0, 1.0]))
    meta_quest_hand.set_offset("right", "ori", np.array([0.0, 0.0, 0.0, 1.0]))

    return meta_quest_hand

def main():
    """Main function to run the hand tracking visualization demo"""
    parser = argparse.ArgumentParser(description='Meta Quest Hand Tracking Visualization')
    parser.add_argument('--disable_palm_fixed', action='store_true', 
                        help='Disable palm fixed mode (enable headset visualization)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode (show body frames)')
    args = parser.parse_args()
    
    # Set is_palm_fixed based on the disable_palm_fixed argument
    is_palm_fixed = not args.disable_palm_fixed
    
    print(f"Initializing Hand Tracking Visualization Demo (Palm Fixed: {is_palm_fixed}, Debug: {args.debug})...")
    
    try:
        # Initialize PyBullet simulator
        simulator = PyBulletHandSimulator(gui=True, is_palm_fixed=is_palm_fixed)
        print("PyBullet hand simulator initialized")
        
        # Initialize Meta Quest hand tracking interface
        meta_quest_hand = setup_meta_quest_hand(is_palm_fixed)
        print("Meta Quest hand tracking interface initialized")
        
        # Create controller
        controller = MetaQuestHandController(simulator, meta_quest_hand, debug=args.debug)
        print("Hand tracking controller initialized")
        
        print("Starting main loop (Press Ctrl+C to stop)...")
        print("Move your hands in front of the Meta Quest headset to see tracking visualization")
        
        # Main simulation loop
        frame_count = 0
        start_time = time.time()
        
        while True:
            controller.update_simulation()
            time.sleep(1/240)  # Run at 240Hz
            
            frame_count += 1
            if frame_count % 2400 == 0:  # Print FPS every 10 seconds
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Running at {fps:.1f} FPS")
                
                # Print hand tracking status
                left_tracked = meta_quest_hand.hand_data["left"].is_tracked
                right_tracked = meta_quest_hand.hand_data["right"].is_tracked
                print(f"Hand tracking - Left: {'✓' if left_tracked else '✗'}, Right: {'✓' if right_tracked else '✗'}")
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulator.disconnect()
        print("Disconnected from PyBullet.")
        
if __name__ == "__main__":
    main()
