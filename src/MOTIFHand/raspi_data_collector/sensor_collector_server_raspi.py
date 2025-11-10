"""
Sensor Data Collector - ZMQ Server Mode (Raspberry Pi)
=======================================================

DEPLOYMENT: Raspberry Pi (Sensor System)
MODE: Network Server (ZMQ REP socket)

This script runs on the Raspberry Pi as a ZMQ server and receives commands
from the control PC running arm_data_collector.py.

ARCHITECTURE:
    Control PC (arm_data_collector.py)  <--ZMQ-->  Raspberry Pi (this script)
         [ZMQ REQ Client]                              [ZMQ REP Server]
         Sends commands                                Executes commands
                                                       Returns responses

USAGE:
    Start server on Raspberry Pi:

    python sensor_data_collector_web.py --port 5556

    The server listens for commands from the control PC and executes:
    - initialize: Set up sensors with parameters
    - collect_data: Capture sensor data at specified index
    - record_video: Start/stop video recording
    - convert_videos: Convert H.264 to MP4 format
    - stop_server: Graceful shutdown

COMMAND PROTOCOL:
    All commands sent/received as JSON over ZMQ:

    Request:  {"type": "command_name", "parameters": {...}}
    Response: {"status": "success/error", "message": "...", ...}

NETWORK REQUIREMENTS:
    - Raspberry Pi must be accessible from control PC
    - Default port: 5556
    - ZMQ library required on both systems

COORDINATION:
    This server is designed to work with:
    - arm_control/xarm7/arm_client/arm_data_collector.py (control PC client)

See sensor_data_collector.py for standalone operation without networking.
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import zmq

from MOTIFHand.utils.sensor_read.camera import \
    CameraController  # Import new camera controller
from MOTIFHand.utils.sensor_read.Lepton35 import ThermalStreamer
# Import sensor modules
from MOTIFHand.utils.sensor_read.MLX90640 import MLX90640Camera
from MOTIFHand.utils.sensor_read.ToF import TOF400FSensor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SensorDataCollector")

debug_thermal = True


class DataCollector:
    # Sensor parameters that can be configured
    CAMERA_PARAMS = {
        "MLX90640": {
            "resolution": [256, 192],
            "field_of_view": {"horizontal": 110, "vertical": 35},
        },
        "Lepton35": {
            "resolution": [160, 120],  # Lepton35 resolution
            "field_of_view": {
                "horizontal": 57,  # Lepton35 horizontal field of view
                "vertical": 45,  # Lepton35 vertical field of view
            },
        },
        "RPiCamera": {  # Using new camera parameters
            "photo_resolution": [3840, 2160],  # 4K resolution
            "video_resolution": [1920, 1080],  # 1080p resolution
            "video_framerate": 30,
        },
    }

    # Valid range for TOF sensor readings (in mm)
    TOF_VALID_RANGE = {"min": 1, "max": 4995}  # 1mm  # 50cm

    # Temperature data validation thresholds
    TEMP_THRESHOLDS = {
        "min_valid": -5.0,  # Celsius
        "max_valid": 90.0,  # Celsius
        "max_std_dev": 50.0,  # Max standard deviation as validity check
    }

    def __init__(
        self,
        object_name,
        output_dir="./data",
        tof_port="/dev/ttyAMA0",
        camera_type="MLX90640",
        camera_output_dir=None,
    ):
        """Initialize the data collector

        Args:
            object_name (str): Name of the object being scanned
            output_dir (str): Output directory for data
            tof_port (str): Serial port for the TOF sensor
            camera_type (str): Type of thermal camera
            camera_output_dir (str): Output directory for camera photos/videos
        """
        self.object_name = object_name
        self.output_dir = Path(output_dir)
        self.tof_port = tof_port
        self.camera_type = camera_type

        # Get current date and time as capture date (format: YYYYMMDD_HHMM)
        self.capture_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")

        # Create output directory
        self.base_dir = self.output_dir / f"{object_name}_{self.capture_date}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for thermal, photo and temperature data
        self.thermal_dir = self.base_dir / "thermal"
        self.photo_dir = self.base_dir / "photo"
        self.temperature_dir = self.base_dir / "temperature"
        self.thermal_dir.mkdir(exist_ok=True)
        self.photo_dir.mkdir(exist_ok=True)
        self.temperature_dir.mkdir(exist_ok=True)

        # Camera output directory (use base_dir if not specified)
        self.camera_output_dir = Path(camera_output_dir) if camera_output_dir else self.base_dir

        # Initialize sensors
        self.thermal_camera = None
        self.tof_sensor = None
        self.camera = None

        # Initialize camera controller
        if not self._init_camera():
            logger.warning("Failed to initialize camera")

        # Initialize thermal camera
        if self.camera_type == "MLX90640":
            self.thermal_camera = MLX90640Camera(display=False)
            if not self.thermal_camera.connect():
                logger.error("Failed to initialize MLX90640 camera")
                self.thermal_camera = None
        elif self.camera_type == "Lepton35":
            self.thermal_camera = ThermalStreamer(display=False)
            self.thermal_camera.start()
        else:
            logger.error(f"Unsupported camera type: {self.camera_type}")

        # Initialize TOF sensor
        self.tof_sensor = TOF400FSensor(port=self.tof_port)
        if not self.tof_sensor.connect():
            logger.error("Failed to initialize TOF sensor")
            self.tof_sensor = None

        # Metadata
        self.metadata = {
            "object_name": object_name,
            "capture_date": self.capture_date,
            "total_captures": 0,
            "camera_parameters": {
                "thermal": self.CAMERA_PARAMS.get(camera_type, {}),
                "rpi_camera": self.CAMERA_PARAMS.get("RPiCamera", {}),
            },
            "captures": [],
        }

        # Latest TOF distance reading
        self.last_tof_distance = None

        # Video recording data
        self.video_path = None
        self.recording_start_time = None
        self.is_recording = False

        # Track successful captures
        self.successful_captures = set()

        # Track recorded videos for later conversion
        self.recorded_videos = []

        # Check disk space
        self._check_disk_space()

        # Check for FFmpeg installation
        self._check_ffmpeg()

    def _check_disk_space(self, min_space_mb=500):  # Increased for video recording
        """Check if there's enough disk space available

        Args:
            min_space_mb (int): Minimum required space in MB

        Raises:
            IOError: If not enough disk space is available
        """
        disk_usage = psutil.disk_usage(self.output_dir)
        free_mb = disk_usage.free / (1024 * 1024)  # Convert to MB

        if free_mb < min_space_mb:
            raise OSError(
                f"Not enough disk space. Only {free_mb:.1f}MB available, {min_space_mb}MB required."
            )

        logger.info(f"Available disk space: {free_mb:.1f}MB")

    def _check_ffmpeg(self):
        """Check if FFmpeg is installed on the system"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True)
            if result.returncode == 0:
                logger.info("FFmpeg is available for video conversion")
                self.ffmpeg_available = True
            else:
                logger.warning("FFmpeg not found. MP4 conversion will not be available.")
                self.ffmpeg_available = False
        except FileNotFoundError:
            logger.warning("FFmpeg not found. MP4 conversion will not be available.")
            self.ffmpeg_available = False

    def _init_camera(self):
        """Initialize Raspberry Pi camera controller"""
        if self.camera is None:
            try:
                # Add a short delay before initialization to ensure resources
                # are free
                time.sleep(0.5)
                self.camera = CameraController.get_instance(str(self.camera_output_dir))
                logger.info("Camera initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Camera initialization failed: {str(e)}")
                return False
        return True

    def _release_camera(self):
        """Release camera resources"""
        if self.camera:
            try:
                # Camera controller is a singleton, no need to explicitly close
                self.camera = None
                logger.info("Camera reference released")
            except Exception as e:
                logger.error(f"Error releasing camera resources: {str(e)}")

    def setup_collection(self, num_captures):
        """Set up collection parameters

        Args:
            num_captures (int): Number of capture positions
        """
        self.metadata["total_captures"] = num_captures
        logger.info(f"Setting up collection: {num_captures} captures")

    def _get_capture_id(self, capture_index):
        """Generate capture ID (e.g., cap01, cap02, ...)

        Args:
            capture_index (int): Capture index (starting from 1)

        Returns:
            str: Formatted capture ID
        """
        return f"cap{capture_index:02d}"

    def _get_filename_base(self, capture_index):
        """Generate base filename

        Args:
            capture_index (int): Capture index (starting from 1)

        Returns:
            str: Base filename
        """

        capture_id = self._get_capture_id(capture_index)
        return f"{self.object_name}_{self.capture_date}_{capture_id}"

    def _validate_tof_reading(self, distance):
        """Validate TOF sensor reading

        Args:
            distance (float): Distance reading in mm

        Returns:
            bool: Whether the distance reading is valid
        """
        if distance is None:
            return False

        return self.TOF_VALID_RANGE["min"] <= distance <= self.TOF_VALID_RANGE["max"]

    def _validate_temperature_data(self, temp_data):
        """Validate temperature data

        Args:
            temp_data (dict): Temperature data dictionary

        Returns:
            bool: Whether the temperature data is valid
            str: Reason for invalidity (if invalid)
        """
        if temp_data is None:
            return False, "Missing temperature data"

        # Handle different camera types
        if self.camera_type == "MLX90640":
            if "data_array" not in temp_data:
                return False, "Missing temperature data array"
            data_array = temp_data["data_array"]
        elif self.camera_type == "Lepton35":
            if "temperature_data" not in temp_data:
                return False, "Missing temperature data array"
            data_array = temp_data["temperature_data"]
        else:
            return False, f"Unsupported camera type: {self.camera_type}"
        if debug_thermal:
            # Print temperature statistics
            min_temp = np.min(data_array)
            max_temp = np.max(data_array)
            avg_temp = np.mean(data_array)
            std_dev = np.std(data_array)
            print(f"\nTemperature Statistics:")
            print(f"Min Temperature: {min_temp:.2f}°C")
            print(f"Max Temperature: {max_temp:.2f}°C")
            print(f"Average Temperature: {avg_temp:.2f}°C")
            print(f"Standard Deviation: {std_dev:.2f}°C")
            print(f"Validation Thresholds:")
            print(f"Min Valid: {self.TEMP_THRESHOLDS['min_valid']}°C")
            print(f"Max Valid: {self.TEMP_THRESHOLDS['max_valid']}°C")
            print(f"Max Std Dev: {self.TEMP_THRESHOLDS['max_std_dev']}°C")

        # Check for NaN values
        if np.isnan(data_array).any():
            return False, "Temperature data contains NaN values"

        # Check temperature range
        if (data_array < self.TEMP_THRESHOLDS["min_valid"]).any() or (
            data_array > self.TEMP_THRESHOLDS["max_valid"]
        ).any():
            return (
                False,
                f"Temperature values outside valid range ({self.TEMP_THRESHOLDS['min_valid']}°C to {self.TEMP_THRESHOLDS['max_valid']}°C)",
            )

        # Check standard deviation (to detect noisy data)
        if std_dev > self.TEMP_THRESHOLDS["max_std_dev"]:
            return False, f"Temperature data too noisy (std dev: {std_dev:.2f})"

        return True, ""

    def _convert_h264_to_mp4(self, h264_path):
        """Convert H.264 file to MP4 format

        Args:
            h264_path (str): Path to the H.264 file

        Returns:
            str: Path to the converted MP4 file, or None if conversion failed
        """
        if not self.ffmpeg_available:
            logger.warning("FFmpeg not available, skipping conversion to MP4")
            return None

        try:
            # Create MP4 output path
            mp4_path = h264_path.replace(".h264", ".mp4")

            # Run FFmpeg to convert the file
            logger.info(f"Converting {h264_path} to MP4 format...")

            # Use FFmpeg to copy the video stream without re-encoding
            # This is much faster than re-encoding
            cmd = [
                "ffmpeg",
                "-i",
                h264_path,
                "-c:v",
                "copy",  # Copy video stream without re-encoding
                "-f",
                "mp4",
                mp4_path,
            ]

            result = subprocess.run(cmd, capture_output=True)

            if result.returncode == 0:
                logger.info(f"Successfully converted to MP4: {mp4_path}")

                # Check if original file should be deleted
                if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                    os.remove(h264_path)
                    logger.info(f"Removed original H.264 file: {h264_path}")

                return mp4_path
            else:
                error = result.stderr.decode("utf-8")
                logger.error(f"FFmpeg conversion failed: {error}")
                return None

        except Exception as e:
            logger.error(f"Error during video conversion: {str(e)}")
            return None

    def _convert_all_videos(self):
        """Convert all recorded H.264 videos to MP4 format"""
        if not self.recorded_videos or not self.ffmpeg_available:
            return

        convert_choice = input(
            "\nConvert all recorded videos to MP4 format? (y/n, default: n): "
        ).lower()
        if convert_choice != "y":
            print("Keeping all videos in original H.264 format.")
            print("You can convert them later using FFmpeg:")
            print("  ffmpeg -i input.h264 -c:v copy output.mp4")
            return

        print(f"\nConverting {len(self.recorded_videos)} video(s) to MP4 format...")

        converted_count = 0
        for video_info in self.recorded_videos:
            video_path = video_info["path"]
            print(f"Converting {os.path.basename(video_path)}...")

            mp4_path = self._convert_h264_to_mp4(video_path)
            if mp4_path:
                # Update the metadata
                video_info["path"] = mp4_path
                video_info["filename"] = os.path.basename(mp4_path)
                video_info["format"] = "mp4"
                converted_count += 1

                # Update the main metadata if this is the primary video
                if "video" in self.metadata and self.metadata["video"][
                    "filename"
                ] == os.path.basename(video_path):
                    self.metadata["video"]["filename"] = os.path.basename(mp4_path)
                    self.metadata["video"]["format"] = "mp4"

        # Save updated metadata
        self._save_metadata()

        print(
            f"Conversion complete: {converted_count}/{len(self.recorded_videos)} videos converted to MP4 format."
        )

    def record_video(self, duration=None):
        """Record video using Raspberry Pi camera

        Args:
            duration (int, optional): Recording duration in seconds. If None, user must stop manually.

        Returns:
            str: Path to the recorded video file
        """
        if not self._init_camera():
            return None

        try:
            # Generate filename for the video
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"{self.object_name}_{timestamp}_video.h264"
            video_path = os.path.join(str(self.base_dir), video_filename)

            # Add a small delay before starting recording to ensure camera is
            # ready
            time.sleep(0.5)

            # Start recording
            if duration is None:
                print("\nStarting video recording. Press 'q' to stop recording...")

                # Start recording video
                actual_path = self.camera.start_recording(
                    filename=video_filename, with_preview=False
                )

                # Give the camera a moment to start recording properly
                time.sleep(1)

                # Set recording state and start display thread
                self.is_recording = True
                self.recording_start_time = time.time()
                recording_thread = threading.Thread(target=self._display_recording_stats)
                recording_thread.daemon = True
                recording_thread.start()

                # Wait for user to press 'q'
                while self.is_recording:
                    if input("").lower() == "q":
                        break
                    time.sleep(0.1)

                # Add a small buffer time before stopping
                print("\nFinalizing recording...")
                time.sleep(2)
            else:
                print(f"\nRecording video for {duration} seconds...")

                # Start recording video
                actual_path = self.camera.start_recording(
                    filename=video_filename, with_preview=True
                )

                # Give the camera a moment to start recording properly
                time.sleep(1)

                # Set recording state and start display thread
                self.is_recording = True
                self.recording_start_time = time.time()
                recording_thread = threading.Thread(target=self._display_recording_stats)
                recording_thread.daemon = True
                recording_thread.start()

                # Wait for duration + a small buffer time
                actual_duration = duration + 2  # Adding 2 seconds buffer
                print(f"Recording will continue for {duration} seconds plus 2 seconds buffer...")
                time.sleep(actual_duration)

                print("Finalizing recording...")
                # Add a small buffer time at the end for finalization
                time.sleep(1)

            # Stop recording
            self.camera.stop_recording()
            self.is_recording = False
            record_duration = time.time() - self.recording_start_time

            # Wait a moment to ensure file is properly closed
            time.sleep(1)

            # Use actual path returned by camera controller
            if actual_path:
                video_path = actual_path

            # Update metadata
            video_filename = os.path.basename(video_path)
            video_metadata = {
                "type": "video",
                "filename": video_filename,
                "path": video_path,
                "format": "h264",
                "duration_seconds": record_duration,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "camera_parameters": {
                    "resolution": list(self.CAMERA_PARAMS["RPiCamera"]["video_resolution"]),
                    "framerate": self.CAMERA_PARAMS["RPiCamera"]["video_framerate"],
                },
            }

            # Store video metadata for later conversion
            self.recorded_videos.append(video_metadata)

            # Store in main metadata
            self.metadata["video"] = video_metadata
            self._save_metadata()

            print(f"\nVideo saved to: {video_path}")
            print(
                "Video will be available for conversion to MP4 after all data collection is completed."
            )

            return video_path

        except Exception as e:
            logger.error(f"Error recording video: {str(e)}")
            self.is_recording = False
            if self.camera and hasattr(self.camera, "is_recording") and self.camera.is_recording:
                try:
                    self.camera.stop_recording()
                    # Give it time to properly close resources
                    time.sleep(1)
                except Exception as stop_error:
                    logger.error(f"Error stopping recording after failure: {stop_error}")
            return None

    def _display_recording_stats(self):
        """Display recording statistics on a single line"""
        try:
            while self.is_recording:
                if self.recording_start_time:
                    elapsed = time.time() - self.recording_start_time
                    mins, secs = divmod(int(elapsed), 60)
                    # Clear line and update
                    sys.stdout.write(
                        f"\rRecording: {mins:02d}:{secs:02d} | Resolution: {self.CAMERA_PARAMS['RPiCamera']['video_resolution']} | Framerate: {self.CAMERA_PARAMS['RPiCamera']['video_framerate']}fps     "
                    )
                    sys.stdout.flush()
                time.sleep(0.5)
            sys.stdout.write("\r" + " " * 80 + "\r")  # Clear the line
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error displaying recording status: {str(e)}")

    def collect_data(self, capture_index, retry_count=3):
        """Collect sensor data for the specified capture

        Args:
            capture_index (int): Capture index (starting from 1)
            retry_count (int): Number of retries on failure

        Returns:
            bool: Whether data collection was successful
        """
        capture_id = self._get_capture_id(capture_index)

        logger.info(f"Collecting data for {capture_id}...")

        # Create capture metadata
        capture_metadata = {
            "capture_id": capture_id,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "files": {},
            "data_validity": {
                "tof_valid": False,
                "thermal_valid": False,
                "photo_valid": False,
            },
        }

        # Create TOF metadata
        tof_metadata = {
            "capture_id": capture_id,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "tof_distance_mm": None,
            "valid": False,
        }

        # Track sensor success statuses separately
        photo_success = False
        thermal_success = False
        tof_success = False

        # Step 1: Take photo with Raspberry Pi camera (if initialized and
        # available)
        photo_path = None

        if self.camera:
            try:
                # Generate filename for photo
                photo_filename = f"{capture_id}_photo.jpg"
                photo_path = self.photo_dir / photo_filename

                logger.info("Taking photo with Raspberry Pi camera...")

                # Allow camera to stabilize
                time.sleep(1)

                # Capture photo with explicit path (disable enhancements to
                # preserve colors)
                full_path = str(photo_path.absolute())
                logger.info(f"Capturing photo to {full_path}")
                self.camera.capture_photo(full_path, apply_enhancements=False)

                # Wait a moment for file to be written
                time.sleep(1)

                # Check if photo exists and is valid
                if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                    photo_success = True
                    capture_metadata["files"]["photo"] = photo_filename
                    capture_metadata["data_validity"]["photo_valid"] = True
                    logger.info(
                        f"Photo saved successfully to: {full_path} ({os.path.getsize(full_path)} bytes)"
                    )
                else:
                    file_exists = os.path.exists(full_path)
                    file_size = os.path.getsize(full_path) if file_exists else 0
                    logger.error(
                        f"Photo capture issue: File exists: {file_exists}, Size: {file_size} bytes"
                    )

            except Exception as e:
                logger.error(f"Error capturing photo: {str(e)}")

        # Add a delay after photo capture before proceeding with other sensors
        time.sleep(1)

        # Proceed with thermal and TOF data collection
        attempts = 0
        temp_data = None
        temp_valid = False
        temp_invalid_reason = ""

        while attempts < retry_count and not (thermal_success and tof_success):
            attempts += 1
            if attempts > 1:
                logger.info(f"Retry attempt {attempts}/{retry_count} for thermal and TOF...")

            try:
                # Step 2: Get thermal image data
                if not thermal_success and self.thermal_camera:
                    if self.camera_type == "MLX90640":
                        logger.info("Capturing thermal image with MLX90640...")
                        # Allow thermal camera to stabilize
                        time.sleep(1)

                        thermal_img, temp_data = self.thermal_camera.get_thermal_image(
                            with_colorbar=False
                        )

                        if thermal_img is not None and temp_data is not None:
                            # Validate temperature data
                            temp_valid, temp_invalid_reason = self._validate_temperature_data(
                                temp_data
                            )

                            # Save thermal image even if validation fails - we
                            # can use it for debugging
                            thermal_filename = f"{capture_id}_thermal.png"
                            thermal_path = self.thermal_dir / thermal_filename
                            cv2.imwrite(str(thermal_path), thermal_img)
                            capture_metadata["files"]["thermal"] = thermal_filename

                            # Save raw temperature data (in numpy format)
                            temp_raw_filename = f"{capture_id}_temperature.npy"
                            temp_raw_path = self.temperature_dir / temp_raw_filename
                            np.save(str(temp_raw_path), temp_data["data_array"])
                            capture_metadata["files"]["temperature"] = temp_raw_filename

                            logger.info(f"Saved thermal image: {thermal_path}")
                            logger.info(f"Saved temperature data: {temp_raw_path}")

                            thermal_success = True
                            capture_metadata["data_validity"]["thermal_valid"] = temp_valid

                            if not temp_valid:
                                logger.warning(
                                    f"Temperature data validation failed: {temp_invalid_reason}"
                                )
                        else:
                            logger.error("Failed to get thermal image data")

                        # Add a delay after thermal camera
                        time.sleep(1)
                    elif self.camera_type == "Lepton35":
                        logger.info("Capturing thermal image with Lepton35...")
                        # Allow thermal camera to stabilize
                        time.sleep(1)

                        # Set up timeout for data collection
                        start_time = time.time()
                        max_wait_time = 2.0  # Maximum 2 seconds for data collection

                        while time.time() - start_time < max_wait_time:
                            # Wait for a frame to be ready
                            if self.thermal_camera.frame_ready.wait(timeout=0.1):
                                self.thermal_camera.frame_ready.clear()

                                # Get the current frame and temperature data
                                with self.thermal_camera.frame_lock:
                                    if (
                                        self.thermal_camera.frame is not None
                                        and self.thermal_camera.temperature_data is not None
                                    ):
                                        thermal_img = self.thermal_camera.frame.copy()
                                        temp_data = {
                                            "temperature_data": self.thermal_camera.temperature_data.copy(),
                                            "min_temp": self.thermal_camera.min_temp,
                                            "max_temp": self.thermal_camera.max_temp,
                                        }

                                        # Validate temperature data
                                        temp_valid, temp_invalid_reason = (
                                            self._validate_temperature_data(temp_data)
                                        )

                                        if temp_valid:
                                            # Save thermal image
                                            thermal_filename = f"{capture_id}_thermal.png"
                                            thermal_path = self.thermal_dir / thermal_filename
                                            cv2.imwrite(str(thermal_path), thermal_img)
                                            capture_metadata["files"]["thermal"] = thermal_filename

                                            # Save raw temperature data (in
                                            # numpy format)
                                            temp_raw_filename = f"{capture_id}_temperature.npy"
                                            temp_raw_path = self.temperature_dir / temp_raw_filename
                                            np.save(
                                                str(temp_raw_path),
                                                temp_data["temperature_data"],
                                            )
                                            capture_metadata["files"][
                                                "temperature"
                                            ] = temp_raw_filename

                                            logger.info(f"Saved thermal image: {thermal_path}")
                                            logger.info(f"Saved temperature data: {temp_raw_path}")

                                            thermal_success = True
                                            capture_metadata["data_validity"][
                                                "thermal_valid"
                                            ] = True
                                            break
                                        else:
                                            logger.warning(
                                                f"Invalid temperature data: {temp_invalid_reason}"
                                            )
                                    else:
                                        logger.warning(
                                            "No valid frame or temperature data available"
                                        )

                            time.sleep(0.1)  # Short delay before next attempt

                        if not thermal_success:
                            logger.error("Failed to get valid thermal image within time limit")

                        # Add a delay after thermal camera
                        time.sleep(1)
                    else:
                        logger.error(f"Unsupported camera type: {self.camera_type}")

                # Step 3: Read TOF distance
                if not tof_success and self.tof_sensor:
                    logger.info("Reading distance with TOF sensor...")
                    # Allow TOF sensor to stabilize
                    time.sleep(1)

                    # Read TOF distance with multiple attempts to get a stable
                    # value
                    distance_readings = []

                    for _ in range(5):  # Try multiple readings
                        distance = self.tof_sensor.read_distance()
                        if self._validate_tof_reading(distance):
                            distance_readings.append(distance)
                        time.sleep(0.1)

                    # Use median of valid readings
                    if distance_readings:
                        self.last_tof_distance = float(np.median(distance_readings))
                        tof_success = True
                        capture_metadata["data_validity"]["tof_valid"] = True
                        tof_metadata["tof_distance_mm"] = self.last_tof_distance
                        tof_metadata["valid"] = True
                        logger.info(
                            f"TOF distance: {self.last_tof_distance} mm (valid: {tof_success})"
                        )
                    else:
                        logger.warning("Could not get valid TOF distance reading")

                    # Record TOF distance, even if invalid (for debugging)
                    tof_metadata["tof_distance_mm"] = self.last_tof_distance

                # If both thermal and TOF succeeded, break out of retry loop
                if thermal_success and tof_success:
                    break

            except Exception as e:
                logger.error(f"Error during sensor data collection: {str(e)}")
                if attempts < retry_count:
                    logger.info("Will retry...")
                    time.sleep(0.5)  # Wait before retry

        # Save metadata files
        info_path = self.base_dir / "info.json"
        tof_info_path = self.base_dir / "tof_data.json"

        # Prepare info data
        info_data = {
            "object_name": self.object_name,
            "capture_date": self.capture_date,
            "total_captures": self.metadata["total_captures"],
            "captures": [],
        }

        # Add capture data
        capture_data = {
            "capture_id": capture_id,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "sensors_status": {
                "photo": photo_success,
                "thermal": thermal_success,
                "tof": tof_success,
            },
        }

        # Add thermal data if available
        if thermal_success:
            capture_data["thermal"] = {
                "min_temp": float(temp_data["min_temp"]) if temp_data else None,
                "max_temp": float(temp_data["max_temp"]) if temp_data else None,
                "std_temp": (
                    float(
                        np.std(
                            temp_data[
                                (
                                    "temperature_data"
                                    if self.camera_type == "Lepton35"
                                    else "data_array"
                                )
                            ]
                        )
                    )
                    if temp_data
                    else None
                ),
                "valid": temp_valid,
                "validity_notes": temp_invalid_reason if not temp_valid else "",
            }

        # Add camera data if available
        if photo_success:
            capture_data["camera"] = {
                "valid": photo_success,
                "resolution": list(self.CAMERA_PARAMS["RPiCamera"]["photo_resolution"]),
            }

        # Load existing info data if it exists
        if info_path.exists():
            with open(info_path) as f:
                info_data = json.load(f)

        # Add new capture data
        info_data["captures"].append(capture_data)

        # Save info data
        with open(info_path, "w") as f:
            json.dump(info_data, f, indent=2)

        # Load existing TOF info data if it exists
        tof_info_data = {"captures": []}
        if tof_info_path.exists():
            with open(tof_info_path) as f:
                tof_info_data = json.load(f)

        # Add new TOF data
        tof_info_data["captures"].append(tof_metadata)

        # Save TOF info data
        with open(tof_info_path, "w") as f:
            json.dump(tof_info_data, f, indent=2)

        logger.info(f"Saved info data: {info_path}")
        logger.info(f"Saved TOF data: {tof_info_path}")

        # Consider the capture "successful" if at least the photo or
        # thermal+tof data was captured
        success = photo_success or (thermal_success and tof_success)

        if success:
            self.successful_captures.add(capture_index)
            logger.info(
                f"Capture {capture_index} completed with photo: {photo_success}, thermal: {thermal_success}, TOF: {tof_success}"
            )
        else:
            logger.error(f"Capture {capture_index} failed - no usable data collected")
        return success

    def _save_metadata(self):
        """Save metadata to JSON file"""
        metadata_path = self.base_dir / f"{self.object_name}_{self.capture_date}_metadata.json"
        # Calculate completion percentage
        if self.metadata["total_captures"] > 0:
            completion = len(self.successful_captures) / self.metadata["total_captures"] * 100

            self.metadata["completion_percentage"] = completion

        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Updated metadata: {metadata_path}")

    def __del__(self):
        """Cleanup method to ensure proper sensor shutdown"""
        if self.thermal_camera:
            if self.camera_type == "Lepton35":
                self.thermal_camera.close()
            elif self.camera_type == "MLX90640":
                self.thermal_camera.disconnect()
        if self.tof_sensor:
            self.tof_sensor.stop()
        if self.camera:
            self._release_camera()

    def run_collection(self, num_captures=None, start_from=1):
        """Run the data collection process

        Args:
            num_captures (int, optional): Number of captures
            start_from (int): Capture index to start from (for resuming)

        Returns:
            bool: Whether all collections were successful
        """
        print("\n===== Sensor Data Collector =====")
        print("Stage 1: Video Recording")
        print("Stage 2: Multi-sensor Data Collection")
        print("Stage 3: Video Format Conversion (Optional)")
        print("==========================\n")

        # First stage: Video recording
        print("Starting Stage 1: Video Recording")
        video_choice = input("Record video? (y/n): ").lower()

        if video_choice == "y":
            duration_choice = input(
                "Enter recording duration in seconds, or press Enter to record until you press 'q': "
            )
            duration = int(duration_choice) if duration_choice.isdigit() else None
            self.record_video(duration)

        # Second stage: Multi-sensor data collection
        print("\nStarting Stage 2: Multi-sensor Data Collection")

        if num_captures is None:
            try:
                num_captures = int(input("Enter number of captures: "))
            except ValueError:
                print("Invalid input, setting to default value 1")
                num_captures = 1

        self.setup_collection(num_captures)

        # Check if we're resuming a previous scan
        if start_from > 1:
            logger.info(f"Resuming from capture {start_from}/{num_captures}")
            # Load existing metadata if available
            metadata_path = self.base_dir / f"{self.object_name}_{self.capture_date}_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        self.metadata = json.load(f)
                    logger.info("Loaded existing metadata")
                    # Find successful captures
                    for cap_data in self.metadata.get("captures", []):
                        cap_id = cap_data.get("capture_id", "")
                        if cap_id.startswith("cap"):
                            try:
                                idx = int(cap_id[3:])
                                self.successful_captures.add(idx)

                            except ValueError:
                                pass
                except Exception as e:
                    logger.warning(f"Could not load existing metadata: {str(e)}")

        all_success = True

        for i in range(start_from, num_captures + 1):
            # Skip already completed captures
            if i in self.successful_captures:
                logger.info(f"Capture {i}/{num_captures} already completed, skipping")
                continue

            print(f"\n===== Capture {i}/{num_captures} =====")

            while True:
                choice = input(
                    "Press Enter to collect data, 'r' to retry last capture, 's' to skip this capture, 'q' to quit: "
                ).lower()

                if choice == "q":
                    print("Data collection aborted by user.")
                    self._release_camera()
                    return False
                elif choice == "s":
                    print(f"Skipping capture {i}...")
                    break
                elif choice == "r" and i > 1:
                    print(f"Retrying previous capture {i - 1}...")
                    success = self.collect_data(i - 1)
                    if success:
                        print(f"Successfully recollected data for capture {i - 1}!")
                    else:
                        print(f"Failed to recollect data for capture {i - 1}")
                    # Stay on current position after retry
                    continue
                else:
                    # Proceed with current capture
                    break

            if choice == "s":
                all_success = False
                continue

            print(f"Collecting data for capture {i}...")
            success = self.collect_data(i)

            if success:
                print(f"Capture {i} data collection complete!")
            else:
                print(f"Failed to collect data for capture {i}")
                all_success = False

                retry = input("Would you like to retry this capture? (y/n): ").lower() == "y"
                if retry:
                    print(f"Retrying capture {i}...")
                    success = self.collect_data(i)
                    if success:
                        print(f"Successfully recollected data for capture {i}!")
                        all_success = True
                    else:
                        print(f"Failed to recollect data for capture {i} again")

        completed = len(self.successful_captures)
        total = self.metadata["total_captures"]
        completion_percentage = (completed / total * 100) if total > 0 else 0

        print("\n===== Data Collection Summary =====")
        print(f"Completed captures: {completed}/{total} ({completion_percentage:.1f}%)")
        print(f"All data saved to: {self.base_dir}")

        if completion_percentage == 100:
            print("Data collection completed successfully!")
        else:
            print("Data collection completed with some missing captures.")

        # Third stage: Convert videos if needed
        if self.recorded_videos and self.ffmpeg_available:
            print("\nStarting Stage 3: Video Format Conversion")
            self._convert_all_videos()

        # Release camera resources
        self._release_camera()

        return all_success


class SensorDataCollectorWeb:
    def __init__(self, zmq_port=5556):
        """
        Initialize the web interface for sensor data collection

        Args:
            zmq_port: ZMQ communication port
        """
        self.zmq_port = zmq_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{zmq_port}")

        # Initialize data collector
        self.collector = None

        # Start message handling thread
        self.running = True
        self.thread = threading.Thread(target=self._handle_messages)
        self.thread.daemon = True
        self.thread.start()

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("SensorDataCollectorWeb")

    def _handle_messages(self):
        """Handle incoming ZMQ messages"""
        while self.running:
            try:
                # Wait for next request from client
                message = self.socket.recv_json()
                response = self._process_command(message)
                self.socket.send_json(response)
            except Exception as e:
                self.logger.error(f"Error handling message: {e}")
                self.socket.send_json({"status": "error", "message": str(e)})

    def _process_command(self, command):
        """
        Process incoming commands

        Args:
            command: Dictionary containing command details

        Returns:
            dict: Response message
        """
        cmd_type = command.get("type")

        if cmd_type == "stop_server":
            self.logger.info("Received stop signal from arm")
            self.running = False
            return {
                "status": "success",
                "message": "Server stopping",
                "completed": True,
            }

        elif cmd_type == "initialize":
            # Initialize data collector with parameters
            params = command.get("parameters", {})
            try:
                self.collector = DataCollector(
                    object_name=params.get("object_name", "default_object"),
                    output_dir=params.get("output_dir", "./data"),
                    tof_port=params.get("tof_port", "/dev/ttyAMA0"),
                    camera_type=params.get("camera_type", "Lepton35"),
                )
                return {
                    "status": "success",
                    "message": "Data collector initialized successfully",
                    "completed": True,
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to initialize data collector: {str(e)}",
                    "completed": False,
                }

        elif cmd_type == "collect_data":
            if not self.collector:
                return {
                    "status": "error",
                    "message": "Data collector not initialized",
                    "completed": False,
                }

            capture_index = command.get("capture_index")
            if capture_index is None:
                return {
                    "status": "error",
                    "message": "No capture index provided",
                    "completed": False,
                }

            try:
                # 获取当前进度信息
                total_captures = self.collector.metadata.get("total_captures", 0)
                successful_captures = len(self.collector.successful_captures)
                completion_percentage = (
                    (successful_captures / total_captures * 100) if total_captures > 0 else 0
                )

                # 发送进度信息
                progress_info = {
                    "total_captures": total_captures,
                    "successful_captures": successful_captures,
                    "completion_percentage": completion_percentage,
                    "current_capture": capture_index,
                }

                success = self.collector.collect_data(capture_index)

                # Update progress info
                if success:
                    successful_captures = len(self.collector.successful_captures)
                    completion_percentage = (
                        (successful_captures / total_captures * 100) if total_captures > 0 else 0
                    )
                    progress_info.update(
                        {
                            "successful_captures": successful_captures,
                            "completion_percentage": completion_percentage,
                        }
                    )

                return {
                    "status": "success" if success else "error",
                    "message": (
                        "Data collection completed" if success else "Data collection failed"
                    ),
                    "success": success,
                    "completed": True,
                    "capture_index": capture_index,
                    "progress": progress_info,
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error during data collection: {str(e)}",
                    "completed": False,
                }

        elif cmd_type == "record_video":
            if not self.collector:
                return {
                    "status": "error",
                    "message": "Data collector not initialized",
                    "completed": False,
                }

            duration = command.get("duration")  # Optional duration in seconds
            try:
                video_path = self.collector.record_video(duration)
                return {
                    "status": "success",
                    "message": "Video recording completed",
                    "video_path": video_path,
                    "completed": True,
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error during video recording: {str(e)}",
                    "completed": False,
                }

        elif cmd_type == "convert_videos":
            if not self.collector:
                return {
                    "status": "error",
                    "message": "Data collector not initialized",
                    "completed": False,
                }

            try:
                self.collector._convert_all_videos()
                return {
                    "status": "success",
                    "message": "Video conversion completed",
                    "completed": True,
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error during video conversion: {str(e)}",
                    "completed": False,
                }

        else:
            return {
                "status": "error",
                "message": f"Unknown command type: {cmd_type}",
                "completed": False,
            }

    def stop(self):
        """Stop the web interface"""
        self.running = False
        self.thread.join()
        self.socket.close()
        self.context.term()

        # Clean up data collector if it exists
        if self.collector:
            del self.collector


def main():
    """Main function to run the web interface"""


    parser = argparse.ArgumentParser(description="Web interface for sensor data collection")
    parser.add_argument("--port", "-p", type=int, default=5556, help="ZMQ communication port")

    args = parser.parse_args()

    collector = SensorDataCollectorWeb(zmq_port=args.port)
    print(f"Sensor data collector web interface started on port {args.port}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down web interface...")
        collector.stop()


if __name__ == "__main__":
    main()
