# Quest Teleop

A robotic arm teleoperation library based on Meta Quest VR controllers and hand tracking functionality. This library provides an intuitive VR control interface that converts Meta Quest input data into robot control commands, supporting multiple end-effectors and real-time simulation visualization.

## Core Features

- **VR Controller Support**: Full support for Meta Quest controller position, orientation, and button inputs
- **Hand Tracking**: Real-time hand tracking using Meta Quest cameras
- **Remote Control**: Real-time conversion of VR input data into robot motion commands
- **Multi End-Effector Compatibility**: Support for dexterous hands and parallel grippers
- **PyBullet Simulation**: Built-in physics simulation and real-time 3D visualization
- **Decoupled Architecture**: Complete separation of control logic and robot motion for system flexibility
- **Mirror Control**: Support for left-right hand mirror mapping, suitable for symmetric operations

## Installation Instructions

### System Requirements
- Python 3.8+
- Meta Quest 3 VR headset
- Windows/Linux
- PyBullet (for simulation visualization)

### Installation Steps

1. Clone the project locally:
```bash
git clone https://github.com/JaegerMai/quest_teleop.git
cd quest_teleop
```

2. Install dependencies:
```bash
pip install -e .
```


## Usage

### Step 1: Install Meta Quest APK
Follow the instructions in `apk/README.md` to install the data sender app to your Meta Quest headset and launch the application.

### Step 2: Configure Network
Set your PC's IP address to **172.16.33.142** or update the Quest application settings to match your actual IP address.

### Step 3: Run Visualization Examples

#### Hand Tracking Visualization
Run the hand tracking visualization to see real-time hand data from Meta Quest:
```bash
python examples/hand_visualize.py
```

![Hand Tracking Demo](media/hand.mp4)

#### Controller Joystick Visualization  
Run the joystick visualization to see real-time controller data:
```bash
python examples/joystick_visualize.py
```

![Controller Demo](media/joystick.mp4)

#### Controller Data Logger
For debugging and testing controller connectivity:
```bash
python examples/controller_data_logger.py
```

