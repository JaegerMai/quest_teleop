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
- Meta Quest VR headset and controllers
- Windows/Linux
- PyBullet (for simulation visualization)

### Installation Steps

1. Clone the project locally:
```bash
git clone <repository-url>
cd quest_teleop
```

2. Install dependencies:
```bash
pip install -e .
```

3. Ensure Meta Quest controllers are connected and working