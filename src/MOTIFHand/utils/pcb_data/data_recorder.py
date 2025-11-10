import json
import os
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import serial

NUM_BOARDS = 3
FSR_COUNT = 36
PAYLOAD_SIZE = NUM_BOARDS * (1 + 12 + 12 + 16 + FSR_COUNT * 2)
BUFFER_SIZE = 10
INTERPOLATION_THRESHOLD = 3


@dataclass
class SensorData:
    """Sensor data structure"""

    id: int
    acc: tuple[float, float, float]
    gyro: tuple[float, float, float]
    mag: tuple[float, float, float]
    temp: float
    fsr: list[int]
    timestamp: float

    def is_valid(self) -> bool:
        """Check if data is valid (not all zeros)"""
        return not (
            all(x == 0 for x in self.acc)
            and all(x == 0 for x in self.gyro)
            and all(x == 0 for x in self.mag)
            and self.temp == 0
            and all(x == 0 for x in self.fsr)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "acc": list(self.acc),
            "gyro": list(self.gyro),
            "mag": list(self.mag),
            "temp": self.temp,
            "fsr": self.fsr,
            "timestamp": self.timestamp,
        }


class SensorModule:
    """Single sensor module class"""

    def __init__(self, module_id: int):
        self.module_id = module_id
        self.data_buffer = deque(maxlen=BUFFER_SIZE)
        self.current_data: SensorData | None = None
        self.zero_count = 0
        self.lock = threading.Lock()

    def add_data(self, data: SensorData):
        """Add new data"""
        with self.lock:
            if data.is_valid():
                self.data_buffer.append(data)
                self.current_data = data
                self.zero_count = 0
            else:
                self.zero_count += 1
                if self.zero_count <= INTERPOLATION_THRESHOLD and len(self.data_buffer) >= 2:
                    interpolated = self._interpolate_data()
                    if interpolated:
                        self.current_data = interpolated
                elif self.current_data is None and self.data_buffer:
                    self.current_data = self.data_buffer[-1]

    def _interpolate_data(self) -> SensorData | None:
        """Linear interpolation for data generation"""
        if len(self.data_buffer) < 2:
            return None

        data1 = self.data_buffer[-2]
        data2 = self.data_buffer[-1]

        alpha = 0.5

        interpolated_acc = tuple(
            data1.acc[i] * (1 - alpha) + data2.acc[i] * alpha for i in range(3)
        )
        interpolated_gyro = tuple(
            data1.gyro[i] * (1 - alpha) + data2.gyro[i] * alpha for i in range(3)
        )
        interpolated_mag = tuple(
            data1.mag[i] * (1 - alpha) + data2.mag[i] * alpha for i in range(3)
        )
        interpolated_temp = data1.temp * (1 - alpha) + data2.temp * alpha
        interpolated_fsr = [
            int(data1.fsr[i] * (1 - alpha) + data2.fsr[i] * alpha) for i in range(FSR_COUNT)
        ]

        return SensorData(
            id=data1.id,
            acc=interpolated_acc,
            gyro=interpolated_gyro,
            mag=interpolated_mag,
            temp=interpolated_temp,
            fsr=interpolated_fsr,
            timestamp=time.time(),
        )

    def get_current_data(self) -> SensorData | None:
        """Get current data"""
        with self.lock:
            return self.current_data

    def get_buffer_size(self) -> int:
        """Get buffer size"""
        with self.lock:
            return len(self.data_buffer)


class SensorDataCollector:
    """Sensor data collector with recording capabilities"""

    def __init__(self, port="/dev/tty.usbmodem5A350015431", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.modules = [SensorModule(i) for i in range(NUM_BOARDS)]
        self.running = False
        self.recording = False
        self.read_thread = None

        # Recording related attributes
        self.object_name = ""
        self.recorded_data = []
        self.recording_start_time = 0
        self.filename = ""
        self.data_lock = threading.Lock()

    def read_exact(self, ser, n):
        """Ensure complete reading of n bytes, otherwise return None"""
        buf = b""
        while len(buf) < n:
            chunk = ser.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def parse_payload(self, data):
        """Parse data packet"""
        offset = 0
        parsed_data = []

        for i in range(NUM_BOARDS):
            id_val = struct.unpack_from("B", data, offset)[0]
            offset += 1

            acc = struct.unpack_from("fff", data, offset)
            offset += 12

            gyro = struct.unpack_from("fff", data, offset)
            offset += 12

            mag = struct.unpack_from("fff", data, offset)
            offset += 12

            temp = struct.unpack_from("f", data, offset)[0]
            offset += 4

            fsr = list(struct.unpack_from(f"{FSR_COUNT}H", data, offset))
            offset += FSR_COUNT * 2

            sensor_data = SensorData(
                id=id_val,
                acc=acc,
                gyro=gyro,
                mag=mag,
                temp=temp,
                fsr=fsr,
                timestamp=time.time(),
            )

            parsed_data.append(sensor_data)

        return parsed_data

    def _read_data(self):
        """Data reading thread"""
        ser = None
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=0.5)

            while self.running:
                head1 = ser.read(1)
                if head1 != b"\xa1":
                    continue

                ser.read(1)

                length_bytes = self.read_exact(ser, 2)
                if length_bytes is None:
                    continue

                payload_len = struct.unpack("<H", length_bytes)[0]

                if payload_len != PAYLOAD_SIZE:
                    self.read_exact(ser, payload_len + 1)
                    continue

                payload = self.read_exact(ser, payload_len)
                if payload is None:
                    continue

                frame_tail = self.read_exact(ser, 1)
                if frame_tail != b"\x55":
                    continue

                # Parse and distribute data
                try:
                    parsed_data = self.parse_payload(payload)
                    for i, data in enumerate(parsed_data):
                        if i < len(self.modules):
                            self.modules[i].add_data(data)

                            # Record data if recording is active
                            if self.recording and data.is_valid():
                                with self.data_lock:
                                    recording_data = {
                                        "module_id": i,
                                        "relative_timestamp": data.timestamp
                                        - self.recording_start_time,
                                        "absolute_timestamp": data.timestamp,
                                        "acc": list(data.acc),
                                        "gyro": list(data.gyro),
                                        "mag": list(data.mag),
                                        "temp": data.temp,
                                    }
                                    self.recorded_data.append(recording_data)

                except Exception as e:
                    print(f"Parse error: {e}")
                    continue

        except Exception as e:
            print(f"Read error: {e}")
        finally:
            if ser:
                ser.close()

    def start_sensor(self) -> bool:
        """Start sensor reading and check if sensors are ready"""
        if self.running:
            return True

        print(f"Starting sensor reader on {self.port}...")
        self.running = True

        # Start reading thread
        self.read_thread = threading.Thread(target=self._read_data)
        self.read_thread.daemon = True
        self.read_thread.start()

        # Wait and check if sensors are providing valid data
        print("Checking sensor validity...")
        time.sleep(2)  # Wait for data to stabilize

        valid_modules = 0
        for i, module in enumerate(self.modules):
            data = module.get_current_data()
            if data and data.is_valid():
                valid_modules += 1
                print(f"Module {i}: OK")
            else:
                print(f"Module {i}: No valid data")

        if valid_modules > 0:
            print(f"Sensor ready! {valid_modules}/{NUM_BOARDS} modules active")
            return True
        else:
            print("Warning: No valid sensor data detected")
            return False

    def prepare_recording(self, object_name: str) -> bool:
        """Prepare for recording with object name"""
        if not self.running:
            print("Error: Sensor not started. Call start_sensor() first.")
            return False

        if self.recording:
            print("Error: Already recording. Stop current recording first.")
            return False

        # Validate object name
        if not object_name or not object_name.strip():
            print("Error: Object name cannot be empty")
            return False

        self.object_name = object_name.strip()

        # Create a data_collected directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_collected")
        os.makedirs(data_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(data_dir, f"{self.object_name}_{timestamp}.json")

        print(f"Ready to record data for object: {self.object_name}")
        print(f"Data will be saved to: {self.filename}")
        return True

    def start_recording(self) -> bool:
        """Start data recording"""
        if not self.running:
            print("Error: Sensor not started")
            return False

        if self.recording:
            print("Error: Already recording")
            return False

        if not self.object_name:
            print("Error: No object name set. Call prepare_recording() first")
            return False

        # Check if sensors are still providing valid data
        valid_count = sum(
            1
            for module in self.modules
            if module.get_current_data() and module.get_current_data().is_valid()
        )

        if valid_count == 0:
            print("Error: No valid sensor data available")
            return False

        with self.data_lock:
            self.recorded_data = []
            self.recording_start_time = time.time()
            self.recording = True

        print(f"Recording started for {self.object_name}")
        print(f"Valid sensors: {valid_count}/{NUM_BOARDS}")
        return True

    def stop_recording(self) -> bool:
        """Stop recording and save data"""
        if not self.recording:
            print("Error: Not currently recording")
            return False

        self.recording = False

        with self.data_lock:
            if not self.recorded_data:
                print("Warning: No data recorded")
                return False

            # Prepare data for saving
            recording_info = {
                "object_name": self.object_name,
                "recording_start_time": self.recording_start_time,
                "recording_duration": time.time() - self.recording_start_time,
                "total_samples": len(self.recorded_data),
                "filename": self.filename,
                "created_at": datetime.now().isoformat(),
                "sensor_data": self.recorded_data,
            }

            # Save to JSON file
            try:
                with open(self.filename, "w", encoding="utf-8") as f:
                    json.dump(recording_info, f, indent=2, ensure_ascii=False)

                print(f"Recording stopped and saved successfully!")
                print(f"File: {self.filename}")
                print(f"Duration: {recording_info['recording_duration']:.2f} seconds")
                print(f"Total samples: {recording_info['total_samples']}")

                # Reset recording state
                self.object_name = ""
                self.recorded_data = []
                self.filename = ""

                return True

            except Exception as e:
                print(f"Error saving file: {e}")
                return False

    def stop_sensor(self):
        """Stop sensor reading"""
        self.recording = False
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=1)
        print("Sensor stopped")

    def get_current_status(self) -> dict[str, Any]:
        """Get current system status"""
        valid_modules = []
        for i, module in enumerate(self.modules):
            data = module.get_current_data()
            valid_modules.append(
                {
                    "module_id": i,
                    "has_data": data is not None,
                    "is_valid": data.is_valid() if data else False,
                    "buffer_size": module.get_buffer_size(),
                }
            )

        return {
            "sensor_running": self.running,
            "recording": self.recording,
            "object_name": self.object_name,
            "filename": self.filename,
            "recorded_samples": len(self.recorded_data) if self.recording else 0,
            "modules": valid_modules,
        }


# Example usage and testing
def main():
    """Main function for testing"""
    collector = SensorDataCollector()

    try:
        # Start sensor
        if not collector.start_sensor():
            print("Failed to start sensor")
            return

        # Wait for user input
        object_name = input("Enter object name: ").strip()
        if not object_name:
            print("No object name provided")
            return

        # Prepare recording
        if not collector.prepare_recording(object_name):
            return

        input("Press Enter to start recording...")

        # Start recording
        if not collector.start_recording():
            return

        input("Press Enter to stop recording...")

        # Stop recording
        collector.stop_recording()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        collector.stop_sensor()


if __name__ == "__main__":
    main()
