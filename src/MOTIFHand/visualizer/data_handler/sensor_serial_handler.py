import os
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

import serial

NUM_BOARDS = 3
NUM_FINGERS = 4
FSR_COUNT = 36
PAYLOAD_SIZE = (
    NUM_FINGERS * NUM_BOARDS * (1 + 12 + 12 + 16 + FSR_COUNT * 2)
)  # Each float is 4 bytes
BUFFER_SIZE = 10  # Buffer size
INTERPOLATION_THRESHOLD = 3  # Interpolate when consecutive zero values exceed this count


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


class SensorModule:
    """Single sensor module class"""

    def __init__(self, finger_id: int, board_id: int):
        self.finger_id = finger_id
        self.board_id = board_id
        self.data_buffer = deque(maxlen=BUFFER_SIZE)
        self.current_data: SensorData | None = None
        self.zero_count = 0  # Consecutive zero value count
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
                # If consecutive zeros exceed threshold and have historical
                # data, interpolate
                if self.zero_count <= INTERPOLATION_THRESHOLD and len(self.data_buffer) >= 2:
                    interpolated = self._interpolate_data()
                    if interpolated:
                        self.current_data = interpolated
                # If too many zeros, keep the last valid data
                elif self.current_data is None and self.data_buffer:
                    self.current_data = self.data_buffer[-1]

    def _interpolate_data(self) -> SensorData | None:
        """Linear interpolation to generate data"""
        if len(self.data_buffer) < 2:
            return None

        # Use the last two valid data points for interpolation
        data1 = self.data_buffer[-2]
        data2 = self.data_buffer[-1]

        # Simple linear interpolation
        alpha = 0.5  # Interpolation coefficient

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


class RS485MultiSensorReader:
    """RS485 multi-sensor reader main class"""

    _instance = None
    _lock = threading.Lock()
    _ref_count = 0
    _serial_lock = threading.Lock()
    _serial = None

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self, port="/dev/tty.usbmodem5A350015431", baudrate=115200
    ):  # 460800 COM9: big board COM10: small board
        with self._lock:
            if self._initialized:
                return

            self.port = port
            self.baudrate = baudrate
            # Create modules for each finger-board combination
            self.modules = {}
            for finger_id in range(NUM_FINGERS):
                for board_id in range(NUM_BOARDS):
                    module_key = f"{finger_id}-{board_id}"
                    self.modules[module_key] = SensorModule(finger_id, board_id)

            self.running = False
            self.read_thread = None
            self.display_thread = None
            self._initialized = True
            self._ref_count = 0

    def _get_serial(self):
        """Get shared serial instance"""
        with self._serial_lock:
            if self._serial is None or not self._serial.is_open:
                try:
                    self._serial = serial.Serial(self.port, self.baudrate, timeout=1)
                except Exception as e:
                    print(f"Failed to open serial port: {e}")
                    return None
            return self._serial

    def _close_serial(self):
        """Close serial connection"""
        with self._serial_lock:
            if self._serial and self._serial.is_open:
                self._serial.close()
                self._serial = None

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
        """Parse data payload"""
        offset = 0
        parsed_data = []

        for finger_id in range(NUM_FINGERS):
            for board_id in range(NUM_BOARDS):
                # id_str = struct.unpack_from('8s', data, offset)[0].decode('ascii').strip('\x00')
                struct.unpack_from("8s", data, offset)[0]
                # id_str = id_bytes.decode('utf-8', errors='ignore').strip('\x00')
                id_val = struct.unpack_from("B", data, offset)[0]  # Unpack uint8_t
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

                parsed_data.append((finger_id, board_id, sensor_data))

        return parsed_data

    def _read_data(self):
        """Data reading thread"""
        while self.running:
            ser = self._get_serial()
            if ser is None:
                time.sleep(1)  # Wait for serial port to be available
                continue

            try:
                head1 = ser.read(1)
                if head1 != b"\xa1":
                    continue

                ser.read(1)

                length_bytes = self.read_exact(ser, 2)
                if length_bytes is None:
                    print("[!] Timeout reading length bytes")
                    continue

                payload_len = struct.unpack("<H", length_bytes)[0]

                if payload_len != PAYLOAD_SIZE:
                    print(f"[!] Unexpected payload size: {payload_len}, expected: {PAYLOAD_SIZE}")
                    # Clear remaining frame: payload + tail
                    self.read_exact(ser, payload_len + 1)
                    continue

                payload = self.read_exact(ser, payload_len)
                if payload is None:
                    print("[!] Incomplete payload, skipping.")
                    continue

                frame_tail = self.read_exact(ser, 1)
                if frame_tail != b"\x55":
                    continue

                # Parse and distribute data
                try:
                    parsed_data = self.parse_payload(payload)
                    for finger_id, board_id, data in parsed_data:
                        module_key = f"{finger_id}-{board_id}"
                        if module_key in self.modules:
                            self.modules[module_key].add_data(data)
                except Exception as e:
                    print(f"Parse error: {e}")
                    continue

            except Exception as e:
                print(f"Read error: {e}")
                time.sleep(0.1)  # Brief wait when error occurs

    def _display_data(self):
        """Command line display thread"""
        while self.running:
            # Clear screen and move cursor to top
            if os.name == "nt":  # Windows
                os.system("cls")
            else:  # Unix/Linux/MacOS
                print("\033[2J\033[H", end="")

            print("=" * 80)
            print(f"RS485 Multi-Sensor Data Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

            for finger_id in range(NUM_FINGERS):
                for board_id in range(NUM_BOARDS):
                    module_key = f"{finger_id}-{board_id}"
                    module = self.modules[module_key]
                    data = module.get_current_data()
                    buffer_size = module.get_buffer_size()

                    print(
                        f"\n--- Finger {finger_id} Board {board_id} (Buffer: {buffer_size}/{BUFFER_SIZE}) ---"
                    )

                    if data:
                        print(f"ID: {data.id}")
                        print(f"Acc:  ({data.acc[0]:8.5f}, {data.acc[1]:8.5f}, {data.acc[2]:8.5f})")
                        print(
                            f"Gyro: ({data.gyro[0]:8.5f}, {data.gyro[1]:8.5f}, {data.gyro[2]:8.5f})"
                        )
                        print(f"Mag:  ({data.mag[0]:8.5f}, {data.mag[1]:8.5f}, {data.mag[2]:8.5f})")
                        print(f"Temp: {data.temp:8.5f}")

                        # Display FSR data (6 groups, 6 each)
                        for group in range(6):
                            start_idx = group * 6
                            end_idx = start_idx + 6
                            fsr_group = data.fsr[start_idx:end_idx]
                            print(f"FSR[{group}]: {fsr_group}")

                        age = time.time() - data.timestamp
                        print(f"Data age: {age:.2f}s")
                    else:
                        print("No data available")

                print("=" * 40)

            print("\n" + "=" * 80)
            print("Press Ctrl+C to stop")

            time.sleep(0.01)  # Update frequency

    def start(self):
        """Start reader"""
        with self._lock:
            if self._ref_count == 0:
                print(f"Starting RS485 multi-sensor reader on {self.port}...")
                self.running = True

                # Start reading thread
                self.read_thread = threading.Thread(target=self._read_data)
                self.read_thread.daemon = True
                self.read_thread.start()

                # Start display thread
                self.display_thread = threading.Thread(target=self._display_data)
                self.display_thread.daemon = True
                self.display_thread.start()

            self._ref_count += 1

    def stop(self):
        """Stop reader"""
        with self._lock:
            if self._ref_count > 0:
                self._ref_count -= 1

                if self._ref_count == 0:
                    self.running = False
                    if self.read_thread:
                        self.read_thread.join(timeout=1)
                    if self.display_thread:
                        self.display_thread.join(timeout=1)
                    self._close_serial()  # Close serial connection
                    print("Stopped.")

    def get_module_data(self, finger_id: int, board_id: int) -> SensorData | None:
        """External interface: Get current data for specified finger-board module"""
        module_key = f"{finger_id}-{board_id}"
        if module_key in self.modules:
            return self.modules[module_key].get_current_data()
        return None

    def get_all_modules_data(self) -> dict:
        """External interface: Get current data for all modules"""
        return {key: module.get_current_data() for key, module in self.modules.items()}


def main():
    """Main function"""
    reader = RS485MultiSensorReader()

    try:
        reader.start()

        # Main thread keeps running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
        reader.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
