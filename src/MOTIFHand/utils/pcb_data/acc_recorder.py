import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

from MOTIFHand.utils.pcb_data.data_loader import RS485SensorReader, SensorData

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


@dataclass
class AccelerationRecord:
    """Data structure for acceleration recording"""

    timestamp: float
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    acc_magnitude: float


class AccelerationRecorder:
    """Records acceleration data with peak detection and JSON saving"""

    def __init__(
        self,
        buffer_duration: float = 2.0,  # Duration to keep in buffer (seconds)
        sampling_rate: float = 25.0,  # Expected sampling rate (Hz)
        peak_threshold: float = 5.0,  # Peak detection threshold (m/s^2)
        pre_peak_duration: float = 1.0,
        # Duration to record before peak (seconds)
        post_peak_duration: float = 3.0,
    ):  # Duration to record after peak (seconds)

        self.buffer_duration = buffer_duration
        self.sampling_rate = sampling_rate
        self.peak_threshold = peak_threshold
        self.pre_peak_duration = pre_peak_duration
        self.post_peak_duration = post_peak_duration

        # Calculate buffer size based on duration and sampling rate
        self.buffer_size = int(buffer_duration * sampling_rate)
        self.data_buffer = deque(maxlen=self.buffer_size)

        # Recording state
        self.is_recording = False
        self.recording_start_time = 0.0
        self.recording_data = []

        # Peak detection state
        self.last_peak_time = 0.0
        self.peak_cooldown = 1.0  # Minimum time between peaks (seconds)

    def _calculate_acc_magnitude(self, acc_x: float, acc_y: float, acc_z: float) -> float:
        """Calculate the magnitude of acceleration vector"""
        return math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    def _create_record(self, sensor_data: SensorData) -> AccelerationRecord:
        """Create an acceleration record from sensor data"""
        acc_magnitude = self._calculate_acc_magnitude(*sensor_data.acc)
        return AccelerationRecord(
            timestamp=time.time(),
            acc_x=sensor_data.acc[0],
            acc_y=sensor_data.acc[1],
            acc_z=sensor_data.acc[2],
            gyro_x=sensor_data.gyro[0],
            gyro_y=sensor_data.gyro[1],
            gyro_z=sensor_data.gyro[2],
            acc_magnitude=acc_magnitude,
        )

    def _detect_peak(self, record: AccelerationRecord) -> bool:
        """Detect if current acceleration is a peak"""
        current_time = time.time()

        # Check cooldown period
        if current_time - self.last_peak_time < self.peak_cooldown:
            return False

        # Check if magnitude exceeds threshold
        if record.acc_magnitude > self.peak_threshold:
            self.last_peak_time = current_time
            return True

        return False

    def update(self, sensor_data: SensorData) -> bool:
        """Update the recorder with new sensor data"""
        record = self._create_record(sensor_data)
        self.data_buffer.append(record)

        # Check for peak if not already recording
        if not self.is_recording and self._detect_peak(record):
            self._start_recording()
            return True

        # Continue recording if active
        if self.is_recording:
            self.recording_data.append(record)

            # Check if recording duration is complete
            if time.time() - self.recording_start_time >= self.post_peak_duration:
                self._stop_recording()
                return True

        return False

    def _start_recording(self):
        """Start recording data"""
        self.is_recording = True
        self.recording_start_time = time.time()
        self.recording_data = []

        # Add pre-peak data from buffer
        pre_peak_samples = int(self.pre_peak_duration * self.sampling_rate)
        for record in list(self.data_buffer)[-pre_peak_samples:]:
            self.recording_data.append(record)

    def _stop_recording(self):
        """Stop recording and save data"""
        self.is_recording = False

        # Convert records to dictionary format
        data_dict = {
            "records": [
                {
                    "timestamp": record.timestamp,
                    "acc_x": record.acc_x,
                    "acc_y": record.acc_y,
                    "acc_z": record.acc_z,
                    "gyro_x": record.gyro_x,
                    "gyro_y": record.gyro_y,
                    "gyro_z": record.gyro_z,
                    "acc_magnitude": record.acc_magnitude,
                }
                for record in self.recording_data
            ]
        }

        # Save to JSON file
        filename = f"acc_recording_{int(time.time())}.json"
        with open(filename, "w") as f:
            json.dump(data_dict, f, indent=2)

        print(f"Recording saved to {filename}")
        self.recording_data = []


def record_acceleration_data(
    port: str = "/dev/tty.usbmodem5A350015431",
    baudrate: int = 115200,
    peak_threshold: float = 5.0,
):
    """Main function to record acceleration data"""
    sensor = RS485SensorReader(port, baudrate)
    recorder = AccelerationRecorder(peak_threshold=peak_threshold)

    try:
        print("Starting acceleration recording...")
        if not sensor.start():
            print("Failed to start sensor interface")
            return

        last_print_time = time.time()

        while True:
            # Get data from all boards
            boards_data = sensor.get_all_boards_data()

            # Print debug information every second
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                print(f"\nCurrent boards data:")
                for board_id, data in boards_data.items():
                    print(f"Board {board_id}:")
                    print(f"  Acceleration: {data.acc}")
                    print(f"  Gyroscope: {data.gyro}")
                    print(f"  Temperature: {data.temp}")
                last_print_time = current_time

            # Process data from each board
            for board_id, data in boards_data.items():
                if recorder.update(data):
                    print(f"Peak detected on board {board_id}")

            time.sleep(0.001)  # Small delay to prevent CPU overload

    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        sensor.stop()


if __name__ == "__main__":
    record_acceleration_data()
