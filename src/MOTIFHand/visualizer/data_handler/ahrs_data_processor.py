import threading
import time
from dataclasses import dataclass

import numpy as np

from .sensor_serial_handler import RS485MultiSensorReader

try:
    from ahrs.filters import EKF, Madgwick, Mahony

    AHRS_AVAILABLE = True
except ImportError:
    print("Warning: ahrs package is not installed, please run: pip install ahrs")
    AHRS_AVAILABLE = False


@dataclass
class Quaternion:
    """Quaternion data structure"""

    w: float
    x: float
    y: float
    z: float

    def __post_init__(self):
        # Normalize quaternion
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 0:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm

    def to_euler(self) -> tuple[float, float, float]:
        """Convert to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            # use 90 degrees if out of range
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


class AHRSFilter:
    """AHRS filter wrapper class"""

    def __init__(self, filter_type: str = "madgwick", **kwargs):
        """
        Initialize AHRS filter

        Args:
            filter_type: Filter type ('madgwick', 'mahony', 'ekf')
            **kwargs: Filter parameters
        """
        if not AHRS_AVAILABLE:
            raise ImportError("ahrs package is not installed, please run: pip install ahrs")

        self.filter_type = filter_type.lower()

        # Default parameters
        default_params = {
            "madgwick": {"frequency": 100.0, "beta": 0.1},
            "mahony": {"frequency": 100.0, "Kp": 1.0, "Ki": 0.3},
            "ekf": {"frequency": 100.0},
        }

        # Merge parameters
        params = default_params.get(self.filter_type, {})
        params.update(kwargs)

        # Create filter
        if self.filter_type == "madgwick":
            self.filter = Madgwick(**params)
        elif self.filter_type == "mahony":
            self.filter = Mahony(**params)
        elif self.filter_type == "ekf":
            self.filter = EKF(**params)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        # Initialize quaternion
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.initialized = False

    def update(
        self,
        gyro: tuple[float, float, float],
        accel: tuple[float, float, float],
        mag: tuple[float, float, float],
    ) -> np.ndarray:
        """
        Update AHRS filter

        Args:
            gyro: Gyroscope data (gx, gy, gz)
            accel: Accelerometer data (ax, ay, az)
            mag: Magnetometer data (mx, my, mz)

        Returns:
            Quaternion [w, x, y, z]
        """
        # Convert to numpy arrays
        gyro_array = np.array(gyro)
        accel_array = np.array(accel)
        mag_array = np.array(mag)

        # Update filter - use updateMARG method (includes magnetometer data)
        if not self.initialized:
            # First update, use initial quaternion
            self.q = np.array([1.0, 0.0, 0.0, 0.0])
            self.initialized = True

        # Update quaternion using updateMARG method
        self.q = self.filter.updateMARG(self.q, gyro_array, accel_array, mag_array)

        return self.q

    def get_quaternion(self) -> np.ndarray:
        """Get current quaternion"""
        return self.q.copy()


class AHRSDataProcessor:
    """AHRS data processor main class"""

    def __init__(
        self,
        sensor_reader: RS485MultiSensorReader | None = None,
        filter_type: str = "madgwick",
        **filter_params,
    ):
        """
        Initialize AHRS data processor

        Args:
            sensor_reader: Sensor reader
            filter_type: AHRS filter type ('madgwick', 'mahony', 'ekf')
            **filter_params: Filter parameters
        """
        self.sensor_reader = sensor_reader
        self.filter_type = filter_type
        self.filter_params = filter_params

        self.ahrs_modules: dict[str, AHRSFilter] = {}
        self.quaternion_data: dict[str, Quaternion] = {}
        self.lock = threading.Lock()
        self.running = False
        self.process_thread = None

        # Initialize AHRS for each sensor module
        self._init_ahrs_modules()

    def _init_ahrs_modules(self):
        """Initialize AHRS for all sensor modules"""
        from MOTIFHand.visualizer.data_handler.sensor_serial_handler import NUM_BOARDS, NUM_FINGERS

        for finger_id in range(NUM_FINGERS):
            for board_id in range(NUM_BOARDS):
                module_key = f"{finger_id}-{board_id}"
                try:
                    self.ahrs_modules[module_key] = AHRSFilter(
                        filter_type=self.filter_type, **self.filter_params
                    )
                    self.quaternion_data[module_key] = Quaternion(1.0, 0.0, 0.0, 0.0)
                except ImportError as e:
                    print(f"Warning: Module {module_key} initialization failed: {e}")
                    # Use simple unit quaternion as fallback
                    self.quaternion_data[module_key] = Quaternion(1.0, 0.0, 0.0, 0.0)

    def set_sensor_reader(self, sensor_reader: RS485MultiSensorReader):
        """Set sensor reader"""
        self.sensor_reader = sensor_reader

    def _process_sensor_data(self):
        """Thread for processing sensor data"""
        while self.running:
            if self.sensor_reader is None:
                time.sleep(0.01)
                continue

            # Get data from all modules
            all_data = self.sensor_reader.get_all_modules_data()

            with self.lock:
                for module_key, sensor_data in all_data.items():
                    if sensor_data is None:
                        continue

                    if module_key in self.ahrs_modules:
                        try:
                            ahrs_filter = self.ahrs_modules[module_key]

                            # Update AHRS algorithm
                            q = ahrs_filter.update(
                                gyro=sensor_data.gyro,
                                accel=sensor_data.acc,
                                mag=sensor_data.mag,
                            )

                            # Update quaternion data
                            self.quaternion_data[module_key] = Quaternion(q[0], q[1], q[2], q[3])

                        except Exception as e:
                            print(f"Module {module_key} AHRS update failed: {e}")
                            # Keep previous quaternion data

            time.sleep(0.01)  # 100Hz processing frequency

    def start(self):
        """Start AHRS processor"""
        if not self.running:
            self.running = True
            self.process_thread = threading.Thread(target=self._process_sensor_data)
            self.process_thread.daemon = True
            self.process_thread.start()
            print(f"AHRS data processor started (filter: {self.filter_type})")

    def stop(self):
        """Stop AHRS processor"""
        self.running = False
        if self.process_thread:
            self.process_thread.join(timeout=1)
        print("AHRS data processor stopped")

    def get_quaternion(self, finger_id: int, board_id: int) -> Quaternion | None:
        """Get quaternion for specified sensor module"""
        module_key = f"{finger_id}-{board_id}"
        with self.lock:
            return self.quaternion_data.get(module_key)

    def get_all_quaternions(self) -> dict[str, Quaternion]:
        """Get quaternions for all sensor modules"""
        with self.lock:
            return self.quaternion_data.copy()

    def get_euler_angles(
        self, finger_id: int, board_id: int
    ) -> tuple[float, float, float] | None:
        """Get Euler angles for specified sensor module (roll, pitch, yaw)"""
        quat = self.get_quaternion(finger_id, board_id)
        if quat:
            return quat.to_euler()
        return None

    def get_all_euler_angles(self) -> dict[str, tuple[float, float, float]]:
        """Get Euler angles for all sensor modules"""
        with self.lock:
            return {key: quat.to_euler() for key, quat in self.quaternion_data.items()}

    def get_filter_info(self) -> dict[str, str]:
        """Get filter information"""
        return {
            "filter_type": self.filter_type,
            "filter_params": str(self.filter_params),
            "ahrs_available": AHRS_AVAILABLE,
        }


def main():
    """Main function - demonstrate AHRS processor"""
    if not AHRS_AVAILABLE:
        print("Error: ahrs package is not installed")
        print("Please run: pip install ahrs")
        return

    # Create sensor reader
    sensor_reader = RS485MultiSensorReader()

    # Create AHRS processor - can choose different filters
    # ahrs_processor = AHRSDataProcessor(sensor_reader, filter_type='madgwick', beta=0.1)
    ahrs_processor = AHRSDataProcessor(sensor_reader, filter_type="mahony", Kp=1.0, Ki=0.3)
    # ahrs_processor = AHRSDataProcessor(sensor_reader, filter_type='madgwick')

    try:
        # Start sensor reader
        sensor_reader.start()

        # Start AHRS processor
        ahrs_processor.start()

        print("AHRS data fusion system running...")
        print("Press Ctrl+C to stop")

        # Display filter information
        filter_info = ahrs_processor.get_filter_info()
        print(f"Filter type: {filter_info['filter_type']}")
        print(f"Filter parameters: {filter_info['filter_params']}")

        # Main loop - display quaternions and Euler angles
        while True:
            time.sleep(1)

            # Display quaternions for all modules
            quaternions = ahrs_processor.get_all_quaternions()
            euler_angles = ahrs_processor.get_all_euler_angles()

            print("\n" + "=" * 60)
            print(f"AHRS Data - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)

            for module_key, quat in quaternions.items():
                euler = euler_angles[module_key]
                print(f"\nModule {module_key}:")
                print(
                    f"  Quaternion: w={quat.w:.4f}, x={quat.x:.4f}, y={quat.y:.4f}, z={quat.z:.4f}"
                )
                print(
                    f"  Euler angles: Roll={np.degrees(euler[0]):.2f}°, Pitch={np.degrees(euler[1]):.2f}°, Yaw={np.degrees(euler[2]):.2f}°"
                )

    except KeyboardInterrupt:
        print("\nStopping...")
        ahrs_processor.stop()
        sensor_reader.stop()
        print("Stopped")


if __name__ == "__main__":
    main()
