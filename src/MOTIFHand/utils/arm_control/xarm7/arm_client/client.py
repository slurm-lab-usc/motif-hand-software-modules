import math
import threading
import time
import warnings
from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI


class ControlMode(Enum):
    """Control mode enum for the robot"""

    JOINT = 0
    CARTESIAN = 1


class XArmController:
    _instances = {}  # Store instances for different IPs

    def __new__(cls, ip="192.168.1.209"):
        """Singleton pattern implementation
        Args:
            ip: Robot's IP address
        """
        if ip not in cls._instances:
            cls._instances[ip] = super().__new__(cls)
        return cls._instances[ip]

    def __init__(self, ip="192.168.1.209"):
        """Initialize robot controller
        Args:
            ip: Robot's IP address
        """
        # Ensure initialization only once
        if not hasattr(self, "initialized"):
            self.ip = ip
            self.arm = XArmAPI(ip)
            self.arm.motion_enable(True)
            self.arm.set_mode(0)
            self.arm.set_state(state=0)

            # Control thread related
            self.control_thread = None
            self.running = False
            self.target_position = None
            self.target_joint_angles = None
            self.control_mode = ControlMode.CARTESIAN
            self.speed = 100
            self.acc = 60
            self.control_frequency = 100  # Hz
            self.control_period = 1.0 / self.control_frequency
            self.last_update_time = time.time()

            self.initialized = True

    def _process_pose(self, position, orientation, is_radian):
        """Process position and orientation input
        Args:
            position: [x, y, z] position (mm)
            orientation: orientation, can be:
                - [roll, pitch, yaw] Euler angles (degrees or radians)
                - [w, x, y, z] quaternion
            is_radian: whether the orientation is in radians (only applies to Euler angles)
        Returns:
            [x, y, z, roll, pitch, yaw] complete pose in degrees
        """
        # Check if position values might be in meters instead of millimeters
        position = np.array(position)
        if np.any(np.abs(position) < 1.0):  # If any value is less than 1mm
            warnings.warn(
                "Small position values detected. Please confirm if you should convert from meters to millimeters. "
                "Current input values: {} mm".format(position),
                UserWarning,
            )

        if len(orientation) == 3:  # Euler angles
            euler = np.array(orientation)
            if is_radian:
                euler = np.degrees(euler)  # Convert radians to degrees
            else:
                # Check if the angle values are within a reasonable range
                if np.any(np.abs(euler) > 2 * np.pi):
                    warnings.warn(
                        "Large angle values detected. Please confirm if you should use radians (is_radian=True). "
                        "Current input values: {}".format(euler),
                        UserWarning,
                    )
        else:  # Quaternion
            r = Rotation.from_quat([orientation[1], orientation[2], orientation[3], orientation[0]])
            euler = r.as_euler("xyz", degrees=True)

        return position.tolist() + euler.tolist()

    def set_ee_pose(self, position, orientation, is_radian=False, speed=None, acc=None, wait=True):
        """Set end-effector pose
        Args:
            position: [x, y, z] position (mm)
            orientation: orientation, can be:
                - [roll, pitch, yaw] Euler angles (degrees or radians)
                - [w, x, y, z] quaternion
            is_radian: whether the orientation is in radians (only applies to Euler angles)
            speed: movement speed (mm/s)
            acc: acceleration
            wait: whether to wait for movement completion
        """
        pose = self._process_pose(position, orientation, is_radian)

        if speed is None:
            speed = self.speed
        if acc is None:
            acc = self.acc

        if wait:
            # Direct movement execution
            self.arm.set_position(
                x=pose[0],
                y=pose[1],
                z=pose[2],
                roll=pose[3],
                pitch=pose[4],
                yaw=pose[5],
                speed=speed,
                mvacc=acc,
                wait=True,
            )
        else:
            # Set target position for control thread execution
            self.target_position = pose
            self.control_mode = ControlMode.CARTESIAN
            self.speed = speed
            self.acc = acc
            self.last_update_time = time.time()

    def get_ee_pose(self, is_radian=False):
        """Get current end-effector pose
        Args:
            is_radian: whether to return values in radians
        Returns:
            [x, y, z, roll, pitch, yaw] current pose
        """
        pose = self.arm.get_position(is_radian=is_radian)
        return pose[0:3] + pose[3:6]

    def set_joint_angle(self, servo_id=None, angle=None, speed=None, is_radian=False, wait=True):
        """Set joint angle(s)
        Args:
            servo_id: joint ID (1-7), if None then angle must be a list of 7 joint angles
            angle: target angle, can be single angle or list of 7 joint angles
            speed: movement speed
            is_radian: whether angle is in radians
            wait: whether to wait for movement completion
        """
        if speed is None:
            speed = self.speed

        if servo_id is not None:
            # Set single joint angle
            if wait:
                self.arm.set_servo_angle(
                    servo_id=servo_id,
                    angle=angle,
                    speed=speed,
                    is_radian=is_radian,
                    wait=wait,
                )
            else:
                # Get current joint angles
                current_angles = self.get_joint_angle(is_radian=is_radian)
                # Update the specified joint
                current_angles[servo_id - 1] = angle
                # Set as target for control thread
                self.target_joint_angles = current_angles
                self.control_mode = ControlMode.JOINT
                self.speed = speed
                self.last_update_time = time.time()
        else:
            # Set all joint angles
            if wait:
                self.arm.set_servo_angle(angle=angle, speed=speed, is_radian=is_radian, wait=wait)
            else:
                # Convert to radians if needed for consistency
                if not is_radian:
                    self.target_joint_angles = np.radians(angle).tolist()
                else:
                    self.target_joint_angles = angle
                self.control_mode = ControlMode.JOINT
                self.speed = speed
                self.last_update_time = time.time()

    def get_joint_angle(self, is_radian=False):
        """Get current joint angles
        Args:
            is_radian: whether to return values in radians
        Returns:
            List of 7 joint angles
        """
        return self.arm.get_servo_angle(is_radian=is_radian)

    def set_pause_time(self, time):
        """Set pause time
        Args:
            time: pause time in seconds
        """
        self.arm.set_pause_time(time)

    def go_home(self):
        """Move to home position"""
        self.arm.set_servo_angle(angle=[0, -36, 0, 13, 0, 49, 0], speed=10, wait=True)

    def disconnect(self):
        """Disconnect from robot"""
        self.stop_control()
        self.arm.disconnect()
        # Remove from instances dictionary
        if self.ip in XArmController._instances:
            del XArmController._instances[self.ip]

    def start_control(self):
        """Start control thread"""
        if self.control_thread is None or not self.control_thread.is_alive():
            self.running = True
            self.control_thread = threading.Thread(target=self._control_loop)
            self.control_thread.daemon = True  # Make thread exit when main program exits
            self.control_thread.start()

    def stop_control(self):
        """Stop control thread"""
        self.running = False
        if self.control_thread is not None:
            self.control_thread.join(timeout=1.0)  # Wait up to 1 second
            self.control_thread = None

    def _control_loop(self):
        """Control loop that runs at specified frequency"""
        while self.running:
            start_time = time.time()

            # Execute latest command based on control mode
            if self.control_mode == ControlMode.CARTESIAN and self.target_position is not None:
                pos = self.target_position
                self.arm.set_position(
                    x=pos[0],
                    y=pos[1],
                    z=pos[2],
                    roll=pos[3],
                    pitch=pos[4],
                    yaw=pos[5],
                    speed=self.speed,
                    mvacc=self.acc,
                    wait=False,
                )

            elif self.control_mode == ControlMode.JOINT and self.target_joint_angles is not None:
                self.arm.set_servo_angle(
                    angle=self.target_joint_angles,
                    speed=self.speed,
                    is_radian=True,
                    wait=False,
                )

            # Calculate sleep time to maintain control frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, self.control_period - elapsed)
            time.sleep(sleep_time)

    def set_control_frequency(self, frequency):
        """Set control loop frequency
        Args:
            frequency: control frequency in Hz
        """
        self.control_frequency = frequency
        self.control_period = 1.0 / frequency

    def __del__(self):
        """Destructor, ensure thread is properly closed"""
        self.disconnect()

    @classmethod
    def get_instance(cls, ip="192.168.1.209"):
        """Get controller instance for specified IP
        Args:
            ip: Robot's IP address
        Returns:
            XArmController: Controller instance
        """
        return cls(ip)


# Example of how to use the continuous control features
if __name__ == "__main__":
    # Create controller instance
    controller = XArmController.get_instance("192.168.1.216")

    try:
        # Start continuous control thread (must be started to use non-blocking
        # commands)
        controller.start_control()
        controller.set_control_frequency(100)  # Set to 100Hz

        # Example of continuous Cartesian control
        print("Starting continuous Cartesian control...")
        for i in range(10):
            # Set new target position (no waiting)
            x = 500 + 50 * math.sin(i * 0.5)
            y = 50 * math.cos(i * 0.5)
            z = 500
            controller.set_ee_pose([x, y, z], [math.pi, 0, 0], is_radian=True, wait=False)
            time.sleep(0.5)  # Wait a bit between commands

        time.sleep(1)

        # Example of continuous Joint control
        print("Starting continuous Joint control...")
        for i in range(10):
            # Set new joint angles (no waiting)
            angles = [0, -36 + 5 * math.sin(i * 0.5), 0, 13, 0, 49, 0]
            controller.set_joint_angle(angle=angles, wait=False)
            time.sleep(1)  # Wait a bit between commands

    finally:
        # Clean up
        controller.stop_control()
        controller.go_home()
        time.sleep(2)
        controller.disconnect()
