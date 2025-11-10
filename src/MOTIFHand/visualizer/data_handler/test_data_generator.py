import math
import threading
import time

import numpy as np

from .sensor_serial_handler import (FSR_COUNT, NUM_BOARDS, NUM_FINGERS,
                                    SensorData)


class TestDataGenerator:
    """Test data generator for simulating sensor data"""

    def __init__(self, frequency: float = 100.0):
        """
        Initialize test data generator

        Args:
            frequency: Data generation frequency in Hz
        """
        self.frequency = frequency
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Store current data for each module
        self.current_data: dict[str, SensorData] = {}

        # Simulation parameters
        self.time = 0.0
        self.dt = 1.0 / frequency

        # Motion simulation parameters
        self.motion_amplitude = 0.5
        self.motion_frequency = 0.5  # Hz
        self.noise_level = 0.1

        # Initialize data for all modules
        self._init_modules()

    def _init_modules(self):
        """Initialize data for all sensor modules"""
        for finger_id in range(NUM_FINGERS):
            for board_id in range(NUM_BOARDS):
                module_key = f"{finger_id}-{board_id}"
                self.current_data[module_key] = self._generate_initial_data(finger_id, board_id)

    def _generate_initial_data(self, finger_id: int, board_id: int) -> SensorData:
        """Generate initial data for a module"""
        # Add some variation based on finger and board position
        base_offset = finger_id * 0.1 + board_id * 0.05

        return SensorData(
            id=board_id,
            acc=(0.0, 0.0, -9.81 + base_offset),  # Gravity + small offset
            gyro=(0.0, 0.0, 0.0),
            mag=(0.0, 0.0, 0.0),
            temp=25.0 + base_offset,
            fsr=[100 + int(base_offset * 1000)] * FSR_COUNT,
            timestamp=time.time(),
        )

    def _generate_sensor_data(self, finger_id: int, board_id: int) -> SensorData:
        """Generate realistic sensor data for a module"""
        # Add some variation based on finger and board position
        base_offset = finger_id * 0.1 + board_id * 0.05
        phase_offset = finger_id * 0.5 + board_id * 0.25

        # Time-based motion simulation
        motion_x = self.motion_amplitude * math.sin(
            2 * math.pi * self.motion_frequency * self.time + phase_offset
        )
        motion_y = (
            self.motion_amplitude
            * 0.5
            * math.cos(2 * math.pi * self.motion_frequency * self.time + phase_offset)
        )
        motion_z = (
            self.motion_amplitude
            * 0.3
            * math.sin(4 * math.pi * self.motion_frequency * self.time + phase_offset)
        )

        # Add noise
        noise_x = np.random.normal(0, self.noise_level)
        noise_y = np.random.normal(0, self.noise_level)
        noise_z = np.random.normal(0, self.noise_level)

        # Accelerometer data (gravity + motion + noise)
        acc_x = motion_x + noise_x
        acc_y = motion_y + noise_y
        acc_z = -9.81 + motion_z + noise_z + base_offset

        # Gyroscope data (angular velocity from motion)
        gyro_x = 2 * math.pi * self.motion_frequency * self.motion_amplitude * math.cos(
            2 * math.pi * self.motion_frequency * self.time + phase_offset
        ) + np.random.normal(0, 0.1)
        gyro_y = -math.pi * self.motion_frequency * self.motion_amplitude * math.sin(
            2 * math.pi * self.motion_frequency * self.time + phase_offset
        ) + np.random.normal(0, 0.1)
        gyro_z = 4 * math.pi * self.motion_frequency * self.motion_amplitude * math.cos(
            4 * math.pi * self.motion_frequency * self.time + phase_offset
        ) + np.random.normal(0, 0.1)

        # Magnetometer data (simulate Earth's magnetic field with some
        # variation)
        mag_x = 0.2 * math.cos(2 * math.pi * 0.1 * self.time + phase_offset) + np.random.normal(
            0, 0.05
        )
        mag_y = 0.2 * math.sin(2 * math.pi * 0.1 * self.time + phase_offset) + np.random.normal(
            0, 0.05
        )
        mag_z = (
            0.5
            + 0.1 * math.sin(2 * math.pi * 0.05 * self.time + phase_offset)
            + np.random.normal(0, 0.05)
        )

        # Temperature (slow variation)
        temp = (
            25.0
            + 5.0 * math.sin(2 * math.pi * 0.01 * self.time + phase_offset)
            + base_offset
            + np.random.normal(0, 0.1)
        )

        # FSR data (pressure simulation)
        base_pressure = 100 + int(base_offset * 1000)
        pressure_variation = 50 * math.sin(2 * math.pi * 0.2 * self.time + phase_offset)
        fsr_data = []
        for i in range(FSR_COUNT):
            # Add some spatial variation to FSR sensors
            spatial_factor = math.sin(i * 0.1 + phase_offset)
            pressure = base_pressure + pressure_variation * spatial_factor + np.random.normal(0, 10)
            fsr_data.append(max(0, int(pressure)))

        return SensorData(
            id=board_id,
            acc=(acc_x, acc_y, acc_z),
            gyro=(gyro_x, gyro_y, gyro_z),
            mag=(mag_x, mag_y, mag_z),
            temp=temp,
            fsr=fsr_data,
            timestamp=time.time(),
        )

    def _generate_data_thread(self):
        """Data generation thread"""
        while self.running:
            start_time = time.time()

            with self.lock:
                # Generate data for all modules
                for finger_id in range(NUM_FINGERS):
                    for board_id in range(NUM_BOARDS):
                        module_key = f"{finger_id}-{board_id}"
                        self.current_data[module_key] = self._generate_sensor_data(
                            finger_id, board_id
                        )

                # Update simulation time
                self.time += self.dt

            # Sleep to maintain frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, self.dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self):
        """Start test data generation"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._generate_data_thread)
            self.thread.daemon = True
            self.thread.start()
            print(f"Test data generator started at {self.frequency} Hz")

    def stop(self):
        """Stop test data generation"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        print("Test data generator stopped")

    def get_module_data(self, finger_id: int, board_id: int) -> SensorData | None:
        """Get current data for specified module"""
        module_key = f"{finger_id}-{board_id}"
        with self.lock:
            return self.current_data.get(module_key)

    def get_all_modules_data(self) -> dict[str, SensorData]:
        """Get current data for all modules"""
        with self.lock:
            return self.current_data.copy()

    def set_frequency(self, frequency: float):
        """Set data generation frequency"""
        self.frequency = frequency
        self.dt = 1.0 / frequency

    def set_motion_parameters(
        self, amplitude: float = None, frequency: float = None, noise: float = None
    ):
        """Set motion simulation parameters"""
        if amplitude is not None:
            self.motion_amplitude = amplitude
        if frequency is not None:
            self.motion_frequency = frequency
        if noise is not None:
            self.noise_level = noise

    def get_status(self) -> dict[str, any]:
        """Get generator status"""
        return {
            "running": self.running,
            "frequency": self.frequency,
            "motion_amplitude": self.motion_amplitude,
            "motion_frequency": self.motion_frequency,
            "noise_level": self.noise_level,
            "modules_count": len(self.current_data),
        }


def main():
    """Test the data generator"""
    generator = TestDataGenerator(frequency=50.0)

    try:
        generator.start()

        print("Test data generator running...")
        print("Press Ctrl+C to stop")

        while True:
            time.sleep(1)

            # Display some sample data
            sample_data = generator.get_module_data(0, 0)
            if sample_data:
                print(f"\nSample data from F0-B0:")
                print(
                    f"Acc: ({sample_data.acc[0]:.3f}, {sample_data.acc[1]:.3f}, {sample_data.acc[2]:.3f})"
                )
                print(
                    f"Gyro: ({sample_data.gyro[0]:.3f}, {sample_data.gyro[1]:.3f}, {sample_data.gyro[2]:.3f})"
                )
                print(
                    f"Mag: ({sample_data.mag[0]:.3f}, {sample_data.mag[1]:.3f}, {sample_data.mag[2]:.3f})"
                )
                print(f"Temp: {sample_data.temp:.2f}")

    except KeyboardInterrupt:
        print("\nStopping...")
        generator.stop()
        print("Stopped")


if __name__ == "__main__":
    main()
