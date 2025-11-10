import re
import threading
import time

import numpy as np
import serial


class SensorData:
    """Class to hold sensor data"""

    def __init__(self):
        self.matrix = np.zeros((6, 6))
        self.acc = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.gyro = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.mag = {"x": 0.0, "y": 0.0, "z": 0.0, "temp": 0.0}
        self.timestamp = None


class SensorDataReader:
    """Singleton class to read and manage sensor data"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._current_data = SensorData()
        self._running = False
        self._thread = None
        self._data_lock = threading.Lock()
        self._port = None

    def start(self, port_name: str, baud_rate: int = 115200) -> bool:
        """Start the sensor data reading thread"""
        if self._running:
            return False

        try:
            self._port = serial.Serial(port_name, baud_rate, timeout=1)
            self._running = True
            self._thread = threading.Thread(target=self._read_data_thread, daemon=True)
            self._thread.start()
            return True
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            return False

    def stop(self) -> None:
        """Stop the sensor data reading thread"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._port and self._port.is_open:
            self._port.close()

    def get_data(self) -> SensorData:
        """Get the latest sensor data"""
        with self._data_lock:
            # Return a copy to prevent external modification
            data_copy = SensorData()
            data_copy.matrix = self._current_data.matrix.copy()
            data_copy.acc = dict(self._current_data.acc)
            data_copy.gyro = dict(self._current_data.gyro)
            data_copy.mag = dict(self._current_data.mag)
            data_copy.timestamp = self._current_data.timestamp
            return data_copy

    def _validate_line(self, line: str, pattern: str) -> bool:
        """Validate if a line matches the expected pattern"""
        return bool(re.match(pattern, line))

    def _validate_matrix_row(self, line: str) -> bool:
        """Validate a matrix row line"""
        pattern = r"^\s*\d+\.\d+\s*,\s*\d+\.\d+\s*,\s*\d+\.\d+\s*,\s*\d+\.\d+\s*,\s*\d+\.\d+\s*,\s*\d+\.\d+\s*$"
        return self._validate_line(line, pattern)

    def _validate_acc_gyro(self, line: str) -> bool:
        """Validate accelerometer and gyroscope data line"""
        pattern = r"^\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*$"
        return self._validate_line(line, pattern)

    def _validate_mag(self, line: str) -> bool:
        """Validate magnetometer data line"""
        pattern = r"^\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*$"
        return self._validate_line(line, pattern)

    def _parse_data(self, lines: list) -> None:
        """Parse data and update the current sensor data"""
        try:
            # Parse matrix
            for i in range(6):
                self._current_data.matrix[i] = [float(val.strip()) for val in lines[i].split(",")]

            # Parse accelerometer and gyroscope data
            acc_gyro = [float(val.strip()) for val in lines[6].split(",")]
            self._current_data.acc["x"] = acc_gyro[0]
            self._current_data.acc["y"] = acc_gyro[1]
            self._current_data.acc["z"] = acc_gyro[2]
            self._current_data.gyro["x"] = acc_gyro[3]
            self._current_data.gyro["y"] = acc_gyro[4]
            self._current_data.gyro["z"] = acc_gyro[5]

            # Parse magnetometer data
            mag_data = [float(val.strip()) for val in lines[7].split(",")]
            self._current_data.mag["x"] = mag_data[0]
            self._current_data.mag["y"] = mag_data[1]
            self._current_data.mag["z"] = mag_data[2]
            self._current_data.mag["temp"] = mag_data[3]

            # Update timestamp
            self._current_data.timestamp = time.time()
        except Exception as e:
            print(f"Error parsing data: {e}")

    def _read_data_thread(self) -> None:
        """Thread function to continuously read sensor data"""
        buffer = []
        valid_rows = 0
        in_sequence = False

        while self._running:
            try:
                # Read a line of data
                raw_line = self._port.readline()

                # Try to decode the line
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    try:
                        line = raw_line.decode("gbk").strip()
                    except UnicodeDecodeError:
                        # Skip invalid data
                        in_sequence = False
                        valid_rows = 0
                        buffer = []
                        continue

                if not line:
                    continue

                # Check for start of a new valid sequence
                if not in_sequence and self._validate_matrix_row(line):
                    in_sequence = True
                    valid_rows = 1
                    buffer = [line]
                    continue

                # Continue building a sequence
                if in_sequence:
                    if valid_rows < 6 and self._validate_matrix_row(line):
                        buffer.append(line)
                        valid_rows += 1
                    elif valid_rows == 6 and self._validate_acc_gyro(line):
                        buffer.append(line)
                        valid_rows += 1
                    elif valid_rows == 7 and self._validate_mag(line):
                        buffer.append(line)
                        valid_rows += 1

                        # Complete data set received
                        if valid_rows == 8:
                            with self._data_lock:
                                self._parse_data(buffer)

                            # Reset for next sequence
                            in_sequence = False
                            valid_rows = 0
                            buffer = []
                    else:
                        # Sequence interrupted
                        in_sequence = False
                        valid_rows = 0
                        buffer = []

                        # Check if this line could start a new sequence
                        if self._validate_matrix_row(line):
                            in_sequence = True
                            valid_rows = 1
                            buffer = [line]

            except Exception as e:
                print(f"Error in read thread: {e}")
                time.sleep(0.1)  # Avoid CPU thrashing on error


# Example usage
if __name__ == "__main__":
    # Create singleton instance
    reader = SensorDataReader()

    # Start reading from serial port
    port = "/dev/tty.usbserial-B003A1FK"  # Change to your port
    if reader.start(port):
        print(f"Started reading from {port}")

        try:
            # Main thread can now do other work while reader runs in background
            while True:
                # Get latest data
                data = reader.get_data()

                print(f"Matrix: {data.matrix}")
                # print(f"Acc: x={data.acc['x']:.2f}, y={data.acc['y']:.2f}, z={data.acc['z']:.2f}")
                # print(f"Gyro: x={data.gyro['x']:.2f}, y={data.gyro['y']:.2f}, z={data.gyro['z']:.2f}")
                # print(f"Mag: x={data.mag['x']:.2f}, y={data.mag['y']:.2f}, z={data.mag['z']:.2f}")
                time.sleep(0.05)  # Wait a second between reads

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            reader.stop()
            print("Reader stopped")
