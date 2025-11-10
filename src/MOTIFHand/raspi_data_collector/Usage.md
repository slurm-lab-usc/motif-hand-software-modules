# Data Collector Usage Guide

This folder contains two sensor data collection scripts for the Raspberry Pi, each serving different purposes.

## File Overview

### 1. `sensor_data_collector.py` - Standalone Mode

**Location:** Raspberry Pi (Sensor System)
**Mode:** Standalone/Interactive
**Network:** Not required

#### Purpose

Direct, interactive control of sensors on the Raspberry Pi without network communication.

#### When to Use

- Testing sensors independently
- Manual data collection sessions
- Debugging sensor hardware
- Standalone object scanning without arm coordination

#### Usage

```bash
# On Raspberry Pi
python sensor_collector_standalone_raspi.py --object test_obj --captures 10
```

### 2. - ZMQ Server Mode `sensor_collector_server_raspi.py`

**Location:** Raspberry Pi (Sensor System)
**Mode:** Network Server (ZMQ REP socket)
**Network:** Required (receives commands from control PC)

#### Purpose

Acts as a remote sensor server that receives commands from the control PC running the arm controller.

#### When to Use

* Coordinated data collection with robotic arm
* Automated scanning workflows
* Remote sensor control from control PC
* Synchronized arm movement + sensor capture

#### Usage

```
# On Raspberry Pi - Start server
python sensor_data_collector_web.py --port 5556

# On Control PC - Run arm data collector (connects to this server)
cd arm_control/xarm7/arm_client
python arm_data_collector.py --captures 50 --object my_object
```

#### Architecture

```
Control PC                          Raspberry Pi
┌─────────────────────┐            ┌──────────────────────┐
│ arm_data_collector  │  ZMQ REQ   │ sensor_collector_    │
│                     │ ──────────>│ server_raspi         │
│ (Client)            │            │ (Server)             │
│                     │ <──────────│                      │
│                     │  ZMQ REP   │ Controls sensors:    │
└─────────────────────┘            │ - Thermal camera     │
                                   │ - ToF sensor         │
                                   │ - RPi camera         │
                                   └──────────────────────┘
```

#### Command Protocol

All communication via JSON over ZMQ:

**Initialize:**

```
{
  "type": "initialize",
  "parameters": {
    "object_name": "test_object",
    "output_dir": "./data",
    "camera_type": "Lepton35"
  }
}
```

**Collect Data:**

```
{
  "type": "collect_data",
  "capture_index": 1
}
```

**Record Video:**

```
{
  "type": "record_video",
  "duration": 30
}
```

**Convert Videos:**

```
{
  "type": "convert_videos"
}
```

**Stop Server:**

```
{
  "type": "stop_server"
}
```

---

---

## System Requirements

### Hardware

* Raspberry Pi (3B+ or later recommended)
* Thermal camera (MLX90640 or Lepton35)
* ToF distance sensor (TOF400F)
* Raspberry Pi camera module (v2 or HQ recommended)

### Software Dependencies

```
# Core packages
pip install numpy opencv-python pyzmq psutil

# Sensor libraries (project-specific)
# - sensor_read.Lepton35
# - sensor_read.MLX90640
# - sensor_read.ToF
# - sensor_read.camera
```

### Network Configuration

For ZMQ server mode:

* Raspberry Pi IP: `192.168.0.110` (default, configurable)
* ZMQ Port: `5556` (default, configurable)
* Ensure firewall allows ZMQ traffic

---

---

## Workflow Comparison

### Standalone Workflow

```
# 1. SSH into Raspberry Pi
ssh pi@192.168.0.110

# 2. Navigate to project
cd /path/to/project/data_collector

# 3. Run collector
python sensor_data_collector.py --object my_object --captures 10

# 4. Follow interactive prompts
# Press Enter to capture each position
```

### Network Server Workflow

```
# === On Raspberry Pi ===
# 1. Start server
python sensor_data_collector_web.py --port 5556

# Server now waits for commands...

# === On Control PC ===
# 2. Run arm data collector (automatically connects to server)
cd arm_control/xarm7/arm_client
python arm_data_collector.py \
  --captures 50 \
  --object my_object \
  --sensor-ip 192.168.0.110 \
  --trajectory recordings/MOTIF-SCAN.json

# 3. System automatically:
#    - Moves arm to position
#    - Sends collect_data command to Raspberry Pi
#    - Waits for response
#    - Moves to next position
```

---

---

## Data Output Structure

Both modes produce the same output structure:

```
data/
└── object_name_YYYYMMDD_HHMM/
    ├── thermal/              # Thermal images (PNG)
    ├── photo/                # RGB photos (JPG)
    ├── temperature/          # Temperature data (NPZ)
    ├── metadata.json         # Collection metadata
    └── video.h264/.mp4       # Optional video recording
```

### Metadata Format

```
{
  "object_name": "test_object",
  "capture_date": "20231015_1430",
  "total_captures": 50,
  "camera_parameters": {
    "thermal": {...},
    "rpi_camera": {...}
  },
  "captures": [
    {
      "capture_id": "cap01",
      "timestamp": "2023-10-15T14:30:15",
      "files": {
        "thermal_image": "thermal/test_object_20231015_1430_cap01_thermal.png",
        "photo": "photo/test_object_20231015_1430_cap01_photo.jpg",
        "temperature_data": "temperature/test_object_20231015_1430_cap01_temp.npz"
      },
      "data_validity": {
        "thermal": true,
        "photo": true,
        "tof": true
      },
      "sensor_readings": {
        "tof_distance_mm": 245.5
      }
    }
  ]
}
```

---

---

## Troubleshooting

### Standalone Mode Issues

**Sensors not detected:**

```
# Check I2C devices
i2cdetect -y 1

# Check serial ports
ls -l /dev/ttyAMA*

# Verify camera
libcamera-hello --list-cameras
```

**Camera initialization fails:**

* Ensure camera is enabled in `raspi-config`
* Check camera cable connection
* Verify no other process is using the camera

### Server Mode Issues

**Connection refused:**

* Verify server is running: `ps aux | grep sensor_data_collector_web`
* Check firewall: `sudo ufw status`
* Test connectivity: `ping 192.168.0.110`

**ZMQ timeout:**

* Increase timeout in client code
* Check network latency: `ping -c 10 192.168.0.110`
* Verify port is not blocked

**Commands not executing:**

* Check server logs for errors
* Verify command JSON format
* Ensure sensors initialized before collect\_data
