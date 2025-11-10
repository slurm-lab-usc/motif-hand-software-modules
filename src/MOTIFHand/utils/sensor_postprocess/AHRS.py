import math
import time

import numpy as np
from MOTIFHand.utils.sensor_postprocess.data_serial_handler import SensorDataReader


class EKF_AHRS:
    """Simplified Extended Kalman Filter-based AHRS with magnetometer calibration"""

    def __init__(self):
        """Initialize the AHRS system"""
        # Sensor data reader
        self.reader = SensorDataReader()

        # State vector: [q0, q1, q2, q3, bx, by, bz]
        # q* = quaternion, b* = gyro bias
        self.state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # State covariance matrix
        self.P = np.eye(7) * 0.1

        # Process and measurement noise parameters
        self.Q = np.eye(7) * 1e-5  # Process noise
        self.R_accel = np.eye(3) * 0.1  # Accelerometer noise
        self.R_mag = np.eye(3) * 0.2  # Magnetometer noise

        # Magnetometer calibration parameters
        self.mag_offset = np.zeros(3)  # Hard iron correction
        self.mag_transform = np.eye(3)  # Soft iron correction
        self.mag_calibrated = False

        # Reference vectors
        self.g_ref = np.array([0, 0, 1])  # Gravity reference
        self.b_ref = np.array([1, 0, 0])  # Magnetic field reference

        # Timing
        self.last_time = None
        self.dt = 0.01  # Default time step (100Hz)

        # Initial bias corrections
        self.accel_bias = np.zeros(3)
        self.is_initialized = False

    def start(self, port_name: str, baud_rate: int = 115200) -> bool:
        """Start the sensor data reader"""
        return self.reader.start(port_name, baud_rate)

    def stop(self) -> None:
        """Stop the sensor data reader"""
        self.reader.stop()

    def calibrate_magnetometer(self, num_samples: int = 200) -> None:
        """
        Perform simple magnetometer calibration

        Args:
            num_samples: Number of samples for calibration
        """
        print("Magnetometer Calibration")
        print("Move the sensor in a figure-8 pattern for about 10 seconds...")

        # Collect samples
        samples = []
        for i in range(num_samples):
            data = self.reader.get_data()
            samples.append([data.mag["x"], data.mag["y"], data.mag["z"]])
            time.sleep(0.05)

        samples = np.array(samples)

        # Hard iron correction (center the ellipsoid)
        self.mag_offset = (np.max(samples, axis=0) + np.min(samples, axis=0)) / 2

        # Soft iron correction (normalize the ellipsoid to a sphere)
        centered = samples - self.mag_offset

        # Get the transformation matrix from ellipsoid to sphere
        try:
            # Calculate covariance matrix
            cov = np.cov(centered.T)

            # Perform eigendecomposition
            eigenvals, eigenvecs = np.linalg.eig(cov)

            # Compute scaling factors
            scales = np.sqrt(np.max(eigenvals) / eigenvals)

            # Compute transformation matrix
            D = np.diag(scales)
            self.mag_transform = eigenvecs @ D @ eigenvecs.T

            self.mag_calibrated = True
            print("Magnetometer calibration complete")
        except np.linalg.LinAlgError:
            print("Calibration failed - insufficient rotation data")
            self.mag_transform = np.eye(3)

    def initialize(self, num_samples: int = 50) -> None:
        """
        Initialize the filter with static sensor data

        Args:
            num_samples: Number of samples to average
        """
        print("Initializing... Keep the sensor stationary")

        # Collect samples
        acc_samples = []
        gyro_samples = []

        for i in range(num_samples):
            data = self.reader.get_data()
            acc_samples.append([data.acc["x"], data.acc["y"], data.acc["z"]])
            gyro_samples.append([data.gyro["x"], data.gyro["y"], data.gyro["z"]])
            time.sleep(0.02)

        acc_samples = np.array(acc_samples)
        gyro_samples = np.array(gyro_samples)

        # Compute biases
        acc_mean = np.mean(acc_samples, axis=0)
        acc_norm = np.linalg.norm(acc_mean)
        self.accel_bias = acc_mean - (acc_mean / acc_norm)

        gyro_bias = np.mean(gyro_samples, axis=0)

        # Initialize state with proper orientation
        acc_normalized = acc_mean / acc_norm

        # Set initial quaternion from accelerometer reading
        if abs(acc_normalized[2] - 1.0) < 0.05:
            # Sensor is approximately level
            self.state[0:4] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Compute rotation from [0,0,1] to acceleration vector
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, acc_normalized)
            rotation_axis_norm = np.linalg.norm(rotation_axis)

            if rotation_axis_norm > 1e-6:
                rotation_axis /= rotation_axis_norm
                angle = np.arccos(np.dot(z_axis, acc_normalized))

                q0 = np.cos(angle / 2)
                q_vec = rotation_axis * np.sin(angle / 2)

                self.state[0:4] = np.array([q0, q_vec[0], q_vec[1], q_vec[2]])
                self.state[0:4] /= np.linalg.norm(self.state[0:4])

        # Set initial gyro bias
        self.state[4:7] = gyro_bias

        # Reset covariance
        self.P = np.eye(7) * 0.1

        self.is_initialized = True
        print("Initialization complete")

    def update(self) -> dict:
        """
        Update orientation estimate with latest sensor data

        Returns:
            Dictionary with orientation data
        """
        # Get current sensor data
        data = self.reader.get_data()

        # Calculate time step
        current_time = time.time()
        if self.last_time is not None:
            self.dt = current_time - self.last_time
            # Sanity check on dt
            if self.dt > 0.5 or self.dt < 0.0001:
                self.dt = 0.01  # Default to 100Hz if time step is unreasonable
        self.last_time = current_time

        # Initialize if needed
        if not self.is_initialized:
            self.initialize()
            return self._get_orientation_output()

        # Extract current quaternion and gyro bias
        q = self.state[0:4]
        bias = self.state[4:7]

        # Get accelerometer data (correct and normalize)
        acc = np.array([data.acc["x"], data.acc["y"], data.acc["z"]]) - self.accel_bias
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 0:
            acc_normalized = acc / acc_norm
        else:
            acc_normalized = np.array([0, 0, 1])

        # Get gyroscope data (in rad/s)
        gyro = np.array(
            [
                math.radians(data.gyro["x"]),
                math.radians(data.gyro["y"]),
                math.radians(data.gyro["z"]),
            ]
        )

        # Get magnetometer data
        mag = np.array([data.mag["x"], data.mag["y"], data.mag["z"]])

        # Apply magnetometer calibration if available
        if self.mag_calibrated:
            mag = self.mag_transform @ (mag - self.mag_offset)

        # Normalize magnetometer
        mag_norm = np.linalg.norm(mag)
        if mag_norm > 0:
            mag_normalized = mag / mag_norm
        else:
            mag_normalized = np.array([1, 0, 0])

        # 1. Prediction step
        # Correct gyro with estimated bias
        gyro_corrected = gyro - bias

        # Compute quaternion derivative
        q0, q1, q2, q3 = q

        # Simplified quaternion derivative from angular velocity
        q_dot = 0.5 * np.array(
            [
                -q1 * gyro_corrected[0] - q2 * gyro_corrected[1] - q3 * gyro_corrected[2],
                q0 * gyro_corrected[0] + q2 * gyro_corrected[2] - q3 * gyro_corrected[1],
                q0 * gyro_corrected[1] - q1 * gyro_corrected[2] + q3 * gyro_corrected[0],
                q0 * gyro_corrected[2] + q1 * gyro_corrected[1] - q2 * gyro_corrected[0],
            ]
        )

        # Predict quaternion
        q_pred = q + q_dot * self.dt
        q_pred /= np.linalg.norm(q_pred)

        # Compute simplified process model Jacobian
        F = np.eye(7)

        # Predict state and covariance
        self.state[0:4] = q_pred
        self.state[4:7] = bias  # Assume constant bias
        self.P = F @ self.P @ F.T + self.Q

        # 2. Update with accelerometer if not in high dynamic motion
        if abs(acc_norm - 9.81) < 2.0:  # Within ~2g of expected gravity
            self._update_with_accelerometer(acc_normalized)

        # 3. Update with magnetometer if calibrated
        if self.mag_calibrated:
            self._update_with_magnetometer(mag_normalized)

        # Return orientation data
        return self._get_orientation_output()

    def _update_with_accelerometer(self, acc: np.ndarray) -> None:
        """
        Update step using accelerometer measurements

        Args:
            acc: Normalized accelerometer readings [ax, ay, az]
        """
        # Extract current quaternion
        q = self.state[0:4]
        q0, q1, q2, q3 = q

        # Compute expected gravity direction in body frame
        exp_acc = self._rotate_vector_by_quaternion(self.g_ref, q, inverse=True)

        # Innovation (measurement - prediction)
        y = acc - exp_acc

        # Simplified measurement Jacobian
        H = np.zeros((3, 7))

        # Only need accelerometer-to-quaternion Jacobian
        # This is a simplified approximation
        H[0, 0] = 2 * q2
        H[0, 1] = 2 * q3
        H[0, 2] = 2 * q0
        H[0, 3] = 2 * q1

        H[1, 0] = -2 * q1
        H[1, 1] = -2 * q0
        H[1, 2] = 2 * q3
        H[1, 3] = 2 * q2

        H[2, 0] = 0
        H[2, 1] = -4 * q1
        H[2, 2] = -4 * q2
        H[2, 3] = 0

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_accel

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(7)
        self.P = (I - K @ H) @ self.P

        # Normalize quaternion
        self.state[0:4] /= np.linalg.norm(self.state[0:4])

    def _update_with_magnetometer(self, mag: np.ndarray) -> None:
        """
        Update step using magnetometer measurements

        Args:
            mag: Normalized magnetometer readings [mx, my, mz]
        """
        # Extract current quaternion
        q = self.state[0:4]

        # Update magnetic reference vector if needed
        # (ensuring we only use mag for yaw)
        if np.linalg.norm(self.b_ref - np.array([1, 0, 0])) < 1e-6:
            # Rotate mag to world frame
            mag_world = self._rotate_vector_by_quaternion(mag, q)

            # Project to horizontal plane
            mag_world[2] = 0

            # Normalize
            mag_world_norm = np.linalg.norm(mag_world)
            if mag_world_norm > 0:
                self.b_ref = mag_world / mag_world_norm

        # Compute expected magnetic field in body frame
        exp_mag = self._rotate_vector_by_quaternion(self.b_ref, q, inverse=True)

        # Innovation (measurement - prediction)
        y = mag - exp_mag

        # Simplified measurement Jacobian - magnetometer only affects yaw
        H = np.zeros((3, 7))

        q0, q1, q2, q3 = q  # 解包四元数分量
        # Only update quaternion components with magnetometer
        H[0, 0] = 2 * q3
        H[0, 3] = 2 * q0

        H[1, 0] = 2 * q2
        H[1, 2] = 2 * q0

        H[2, 1] = -2 * q2
        H[2, 2] = -2 * q1

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_mag

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(7)
        self.P = (I - K @ H) @ self.P

        # Normalize quaternion
        self.state[0:4] /= np.linalg.norm(self.state[0:4])

    def _rotate_vector_by_quaternion(
        self, v: np.ndarray, q: np.ndarray, inverse: bool = False
    ) -> np.ndarray:
        """
        Rotate a vector by a quaternion

        Args:
            v: Vector to rotate [x, y, z]
            q: Quaternion [w, x, y, z]
            inverse: If True, use conjugate quaternion (inverse rotation)

        Returns:
            Rotated vector
        """
        q0, q1, q2, q3 = q

        if inverse:
            # Use conjugate quaternion for inverse rotation
            q1, q2, q3 = -q1, -q2, -q3

        # Quaternion rotation formula: q * v * q
        x = v[0]
        y = v[1]
        z = v[2]

        # Quaternion multiplication
        vx = (
            (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) * x
            + 2 * (q1 * q2 - q0 * q3) * y
            + 2 * (q1 * q3 + q0 * q2) * z
        )
        vy = (
            2 * (q1 * q2 + q0 * q3) * x
            + (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) * y
            + 2 * (q2 * q3 - q0 * q1) * z
        )
        vz = (
            2 * (q1 * q3 - q0 * q2) * x
            + 2 * (q2 * q3 + q0 * q1) * y
            + (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3) * z
        )

        return np.array([vx, vy, vz])

    def _quaternion_to_euler(self, q: np.ndarray) -> tuple[float, float, float]:
        """
        Convert quaternion to Euler angles

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        w, x, y, z = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            # Use 90 degrees if out of range
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (roll, pitch, yaw)

    def _get_orientation_output(self) -> dict:
        """Format orientation data for output"""
        q = self.state[0:4]
        euler_rad = self._quaternion_to_euler(q)
        euler_deg = tuple(math.degrees(angle) for angle in euler_rad)

        return {
            "quaternion": {"w": q[0], "x": q[1], "y": q[2], "z": q[3]},
            "euler_degrees": {
                "roll": euler_deg[0],
                "pitch": euler_deg[1],
                "yaw": euler_deg[2],
            },
            "euler_radians": {
                "roll": euler_rad[0],
                "pitch": euler_rad[1],
                "yaw": euler_rad[2],
            },
            "gyro_bias": {"x": self.state[4], "y": self.state[5], "z": self.state[6]},
        }


# Example usage
if __name__ == "__main__":
    # Create AHRS instance
    ahrs = EKF_AHRS()

    # Start serial connection
    port = "/dev/tty.usbserial-B003A1FK"  # Change to your port
    if ahrs.start(port):
        print(f"Started AHRS on {port}")

        try:
            # Calibrate magnetometer
            ahrs.calibrate_magnetometer()

            # Initialize with stationary data
            ahrs.initialize()

            # Main loop
            while True:
                # Update and get orientation
                orientation = ahrs.update()

                # Print orientation
                roll = orientation["euler_degrees"]["roll"]
                pitch = orientation["euler_degrees"]["pitch"]
                yaw = orientation["euler_degrees"]["yaw"]

                print(f"Roll: {roll:.1f}°, Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°")

                time.sleep(0.05)  # Update at ~20Hz

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            ahrs.stop()
            print("AHRS stopped")
