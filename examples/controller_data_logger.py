import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from quest_teleop.meta_quest_controller import MetaQuestController

def main():

    controller = MetaQuestController(is_mirror_control=False)
    print("Controller initialized")
    
    while True:
        print("\n" + "-"*50)
    
        # Headset data
        headset_pos = controller.meta_quest.get_headset_position()
        headset_ori = controller.meta_quest.get_headset_orientation()
        print(f"Headset pos: {headset_pos}")
        print(f"Headset ori: {headset_ori}")
        
        # Controller data for both hands
        for hand_side in ["left", "right"]:
            if hand_side in controller.meta_quest.controller_data:
                ctrl_data = controller.meta_quest.controller_data[hand_side]
                
                pos = controller.meta_quest.get_joystick_position(hand_side)
                ori = controller.meta_quest.get_joystick_orientation(hand_side)
                
                print(f"{hand_side} pos: {pos}")
                print(f"{hand_side} ori: {ori}")
                print(f"{hand_side} index_trigger: {ctrl_data.index_trigger:.3f}")
                print(f"{hand_side} hand_trigger: {ctrl_data.hand_trigger:.3f}")
                print(f"{hand_side} thumbstick: {ctrl_data.thumbstick.to_array()}")
                
                buttons = [name for name, pressed in ctrl_data.buttons.items() if pressed]
                print(f"{hand_side} buttons: {buttons}")
                
                unlocked = controller.is_joystick_unlocked(hand_side)
                print(f"{hand_side} unlocked: {unlocked}")
            else:
                print(f"{hand_side}: NOT CONNECTED")
                
        time.sleep(1.0)

if __name__ == "__main__":
    main()
