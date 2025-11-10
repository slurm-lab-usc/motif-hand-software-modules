
# MOTIF Hand - Software Modules

[![arXiv](https://img.shields.io/badge/ArXiv-2506.19201-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2506.19201) [![web](https://img.shields.io/badge/Web-Motif_Hand-blue.svg?style=plastic)](https://slurm-lab-usc.github.io/motif-hand/) [![git](https://img.shields.io/badge/Github-Main_Page-orange.svg?style=plastic)](https://github.com/slurm-lab-usc/motif-hand-documents) [![license](https://img.shields.io/badge/LICENSE-Apache--2.0-white.svg?style=plastic)](./LICENSE)


## Features

* **Multi-modal Sensor Integration**: FSR (Force Sensitive Resistor), IMU (accelerometer, gyroscope, magnetometer), thermal cameras, and ToF sensors
* **Robotic Arm Control**: xArm7 integration with trajectory recording and playback
* **LEAP Hand Integration**: ROS2-based control for dexterous manipulation
* **Raspberry Pi Data Collection**: Network-based and standalone sensor data acquisition
* **Real-time Visualization**: MuJoCo-based tactile visualization and IMU data visualization
* **Coordinated Multi-System Operation**: Synchronized arm movement and sensor capture

## Project Structure

```
src/
├── LEAPHand_data_collector/   # ROS2 scripts for LEAP hand control with sensor recording
├── raspi_data_collector/      # Raspberry Pi sensor data collection (standalone & server modes)
├── utils/
│   ├── arm_control/          # xArm7 control utilities and trajectory management
│   ├── pcb_data/             # Tactile sensor data recording and loading
│   ├── sensor_postprocess/   # AHRS and data processing utilities
│   └── sensor_read/          # Sensor drivers (thermal, ToF, camera)
└── visualizer/               # Real-time tactile and IMU data visualization
```

## Installation

### Using UV (Recommended)

This project uses `uv` for package management. Install dependencies:

```bash
# Init the UV env
uv venv --python=python3.10 .venv
source .venv/bin/activate

# Install base dependencies
uv sync

# Install Raspberry Pi specific dependencies, only on the data collection device
uv sync --extra raspi

# Install robotic arm dependencies, only on the one directly conected to the XARM
uv sync --extra arm

# Install development tools
uv sync --group dev
```

## Using pyproject.toml

The file `pyproject. toml` defines the project configuration:

* **Dependencies**: Core packages required for all systems
* **Optional Dependencies**:
  * `raspi`: Raspberry Pi sensor libraries (Adafruit thermal cameras, GPIO)
  * `arm`: xArm Python SDK for robotic arm control
* **Development Tools**: Linters, formatters, and code quality tools

## Quick Start

### 1. LEAP Hand Data Collection

Control LEAP hand with ROS2 and record sensor data during flick motion:

```bash
# Test hand poses without recording
uv run -m MOTIFHand.LEAPHand_data_collector.hand_ros_flick_test

# Record sensor data during flick motions
uv run -m MOTIFHand.LEAPHand_data_collector.hand_ros_flick_recorder
```
### 2. Raspberry Pi Sensor Collection

**Standalone Mode** (local sensor control):

```bash
uv run -m MOTIFHand.raspi_data_collector.sensor_collector_standalone_raspi \
  --object test_object --captures [NUMS]
```
**Server Mode** (networked with arm control):

```bash
# On Raspberry Pi
uv run -m MOTIFHand.raspi_data_collector.sensor_collector_server_raspi --port 5556

# On Control PC
uv run -m MOTIFHand.utils.arm_control.xarm7.arm_client.arm_data_collector \
  --captures [NUMS] --object [my_object] --trajectory [recordings/scan.json]
```
### 3. Tactile Visualization

Real-time MuJoCo-based tactile sensor visualization:

```bash
uv run -m MOTIFHand.visualizer.tactile_visualizer
```
IMU and sensor data plotting:

```bash
uv run -m MOTIFHand.visualizer.IMU_data_viz
```
## System Architecture

```
  Control PC                   Raspberry Pi                Hand
┌─────────────┐              ┌──────────────┐         ┌────────────┐
│ Arm Control │─────ZMQ─────>│ Sensor Server│         │ ROS2 Node  │
│             │              │ - Thermal    │         │            │
│ Trajectory  │<────ZMQ──────│ - ToF        │<────────│ FSR/IMU    │
│ Recorder    │              │ - RGB Camera │  Serial │ Recording  │
└─────────────┘              └──────────────┘         └────────────┘
       │                            │                        │
       └────────────────────────────┴────────────────────────┘
                         Data Visualization
```
## Key Components

### Sensor Data Recorder (`utils/pcb_data/`)

* Multi-board FSR array (36 sensors per board)
* IMU data (accelerometer, gyroscope, magnetometer)
* Serial communication with error handling and interpolation
* JSON-based data export

### Arm Control (`utils/arm_control/`)

* xArm7 trajectory recording and playback
* Position and orientation control
* Integration with sensor systems
* ZMQ-based remote control

### Sensor Readers (`utils/sensor_read/`)

* Thermal cameras: MLX90640, Lepton35 (UVC and I2C)
* ToF distance sensor: TOF400F
* Raspberry Pi camera module
* Intel RealSense D405

### Visualizers (`visualizer/`)

* Real-time tactile grid visualization (6×6 per zone, 11 zones)
* IMU data plotting with AHRS processing
* MuJoCo physics integration

## Hardware Requirements

* **Control PC**: Python 3.10+, sufficient for visualization
* **Raspberry Pi**: 3B+ or later, with thermal camera (Lepton 3.5), ToF sensor, camera module
* **LEAP Hand**: ROS2-compatible installation
* **xArm7**: Network-accessible robotic arm (optional)
* **Custom PCB**: FSR arrays with IMU sensors (13 boards, RS485 communication)

## Network Configuration

Default IP addresses (configurable):

* Raspberry Pi: `192.168.0.110`
* ZMQ Port: `5556`

## Documentation

Detailed usage guides available in component directories:

* `src/LEAPHand_data_collector/Usage.md`
* `src/raspi_data_collector/Usage.md`
* `src/utils/arm_control/Usage.md`
