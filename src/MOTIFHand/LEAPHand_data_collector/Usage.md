# Hand Control Usage Guide

This folder contains ROS2-based scripts for controlling a LEAP hand with preset poses and sensor data recording.

## Overview

The hand control package provides two main scripts:

1. **hand_ros_flick_test.py** - Basic pose control for testing
2. **hand_ros_flick_recorder.py** - Pose control with integrated sensor data recording

## Prerequisites

### Dependencies

- ROS2 (Robot Operating System 2)
- Python 3.x
- Required Python packages:
  - `rclpy` (ROS2 Python client library)
  - `sensor_msgs` (ROS2 sensor message types)
  - `numpy`
- Sensor data collection module (`pcb_data.data_recorder`)
- Install the ROS package from [LEAP\_Hand\_API/ros2\_module](https://github.com/leap-hand/LEAP_Hand_API/tree/main)

## hand_ros_flick_test.py

### Description

A basic script for toggling between two preset hand poses. Useful for testing hand control without sensor integration.

### Usage

```bash
python3 hand_ros_flick_test.py
```

### Controls

* **Space**: Toggle between pose 1 and pose 2
* **q**: Quit the program

### Pose Configuration

The script defines two 4x4 matrices (16 joint angles total):

* **pose1**: Initial position (all joints at 0.0)
* **pose2**: Flick position (all joints at 0.0)

**Note**: Default poses have zero values. Modify the and arrays in the code to set actual joint angles. `self.<wbr/>pose1``self.<wbr/>pose2`

### Joint Layout

Each pose is a 4x4 matrix representing:

* Row 0: Thumb joints (4 joints)
* Row 1: Index finger joints (4 joints)
* Row 2: Middle finger joints (4 joints)
* Row 3: Ring finger joints (4 joints)

## hand\_ros\_flick\_recorder.py

### Description

An advanced script that executes a "flick" motion while recording sensor data. Includes automated timing and data collection.

### Usage

```
python3 hand_ros_flick_recorder.py
```

### Controls

* **Enter**: Execute flick motion and record sensor data
* **q**: Quit the program

### Workflow

1. **Initial State**: Hand starts in pose1 (relaxed position)
2. **Press Enter**: Initiates the following sequence:
   * Starts sensor recording
   * Waits 1.5 seconds
   * Executes pose2 (flick motion)
   * Records data for 3 seconds
   * Stops recording
   * Prepares for next recording
3. **Return to Initial**: Press Enter again (when in pose2) to return to pose1

### Preset Poses

**Pose 1** (Initial Position):

```
[0.0, 1.25, 1.40, 0.80]  # Thumb
[0.0, 0.0, 0.0, 0.0]  # Index
[0.0, 0.0, 0.0, 0.0]  # Middle
[0.0, 0.0, 0.0, 0.0]  # Ring
```

**Pose 2** (Flick Position):

```
[0.0, 1.25, 0.0, 0.0]  # Thumb
[0.0, 0.0, 0.0, 0.0]  # Index
[0.0, 0.0, 0.0, 0.0]  # Middle
[0.0, 0.0, 0.0, 0.0]  # Ring
```

### Recording Details

* Default recording label: "red"
* Recording duration: 3 seconds per flick
* Pre-flick delay: 1.5 seconds
* Data is saved automatically by the module `SensorDataCollector`

---

---

## Customization

### Modifying Poses

To change joint angles, edit the `pose1` and `pose2` arrays in either script:

```
self.pose1 = np.array([
    [thumb_j0, thumb_j1, thumb_j2, thumb_j3],
    [index_j0, index_j1, index_j2, index_j3],
    [middle_j0, middle_j1, middle_j2, middle_j3],
    [ring_j0, ring_j1, ring_j2, ring_j3]
])
```

### Changing Recording Parameters

In , modify these values: `hand_ros_flick_recorder.py`

```
time.sleep(1.5)  # Pre-flick delay
time.sleep(3)  # Recording duration
self.sensor_collector.prepare_recording("red")  # Label
```

### ROS2 Topic

If your hand uses a different topic, change the publisher initialization:

```
self.pub_hand = self.create_publisher(JointState, '/your_topic_name', 10)
```

---

---

## Troubleshooting

### Sensor Fails to Start

* Check sensor hardware connections
* Verify sensor permissions (may require `sudo`)
* Ensure module is properly installed `pcb_data.data_recorder`

### No Hand Movement

* Verify ROS2 is running: `ros2 topic list`
* Check topic exists `/cmd_allegro`
* Confirm hand controller node is active
* Verify joint angle values are within valid ranges

### Recording Errors

* Ensure sufficient disk space for data files
* Check write permissions in the recording directory
* Verify sensor initialization succeeded
