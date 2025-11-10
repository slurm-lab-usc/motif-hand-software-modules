#!/usr/bin/env python3
# xArm7 ROS2 Control Script - Joint Position and Cartesian Position Control

import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from xarm_msgs.srv import (GetFloat32List, MoveCartesian, MoveHome, MoveJoint,
                           SetInt16, SetInt16ById)

# Add custom path
sys.path.append("${COLON_WS}/install/xarm_msgs/share/xarm_msgs")


# Import correct messages and services


class XArm7Controller(Node):
    def __init__(self, namespace="/xarm"):
        """
        Initialize xArm7 controller node

        Args:
            namespace: xArm ROS2 namespace
        """
        super().__init__("xarm7_controller")

        # Save namespace
        self.namespace = namespace

        # Define joint names
        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        # Define default home position (degrees)
        self.home_joint_positions = [0.0, -36.0, 0.0, 13.0, 0.0, 49.0, 0.0]

        # Create service clients for the services that actually exist
        self.get_servo_angle_client = self.create_client(
            GetFloat32List, f"{namespace}/get_servo_angle"
        )

        self.set_position_client = self.create_client(MoveCartesian, f"{namespace}/set_position")

        self.set_servo_angle_client = self.create_client(MoveJoint, f"{namespace}/set_servo_angle")

        self.set_servo_angle_j_client = self.create_client(
            MoveJoint, f"{namespace}/set_servo_angle_j"
        )

        self.set_mode_client = self.create_client(SetInt16, f"{namespace}/set_mode")

        self.set_state_client = self.create_client(SetInt16, f"{namespace}/set_state")

        self.motion_enable_client = self.create_client(SetInt16ById, f"{namespace}/motion_enable")

        self.move_home_client = self.create_client(MoveHome, f"{namespace}/move_gohome")

        # Wait for services to be available with timeout
        self.get_logger().info("Waiting for services (with 5 second timeout)...")
        self.wait_for_services(timeout_sec=5.0)

        # Store current joint positions
        self.current_joint_positions = None

        self.get_logger().info("xArm7 controller initialized successfully!")

    def wait_for_services(self, timeout_sec=5.0):
        """
        Wait for all service clients to be available with timeout

        Args:
            timeout_sec: Timeout in seconds for each service
        """
        services = [
            (self.get_servo_angle_client, "get_servo_angle"),
            (self.set_position_client, "set_position"),
            (self.set_servo_angle_client, "set_servo_angle"),
            (self.set_servo_angle_j_client, "set_servo_angle_j"),
            (self.set_mode_client, "set_mode"),
            (self.set_state_client, "set_state"),
            (self.motion_enable_client, "motion_enable"),
            (self.move_home_client, "move_gohome"),
        ]

        for client, name in services:
            self.get_logger().info(f"Waiting for {name} service...")
            available = client.wait_for_service(timeout_sec=timeout_sec)
            if available:
                self.get_logger().info(f"{name} service available")
            else:
                self.get_logger().warn(f"{name} service not available after {timeout_sec} seconds")

    def enable_robot(self, enable=True):
        """
        Enable or disable robot motion

        Args:
            enable: True to enable, False to disable

        Returns:
            Response code, 0 means success
        """
        request = SetInt16ById.Request()
        request.id = 8  # All joints
        request.data = 1 if enable else 0

        self.get_logger().info(f'{"Enabling" if enable else "Disabling"} robot motion...')
        future = self.motion_enable_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result.ret == 0:
            self.get_logger().info(
                f'Successfully {"enabled" if enable else "disabled"} robot motion'
            )
        else:
            self.get_logger().error(
                f'Failed to {"enable" if enable else "disable"} robot motion, error code: {result.ret}'
            )

        return result.ret

    def set_robot_mode(self, mode):
        """
        Set robot control mode

        Args:
            mode: 0-position control, 1-servo control, 2-zero force control

        Returns:
            Response code, 0 means success
        """
        if not self.set_mode_client.service_is_ready():
            self.get_logger().error("set_mode service not available")
            return -1

        request = SetInt16.Request()
        request.data = mode
        future = self.set_mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result.ret == 0:
            self.get_logger().info(f"Successfully set mode to {mode}")
        else:
            self.get_logger().error(f"Failed to set mode, error code: {result.ret}")

        return result.ret

    def set_robot_state(self, state):
        """
        Set robot state

        Args:
            state: 0-ready, 4-stop, 1-pause

        Returns:
            Response code, 0 means success
        """
        if not self.set_state_client.service_is_ready():
            self.get_logger().error("set_state service not available")
            return -1

        request = SetInt16.Request()
        request.data = state
        future = self.set_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result.ret == 0:
            self.get_logger().info(f"Successfully set state to {state}")
        else:
            self.get_logger().error(f"Failed to set state, error code: {result.ret}")

        return result.ret

    def get_current_joint_positions(self):
        """
        Get current joint positions

        Returns:
            List of joint positions
        """
        if not self.get_servo_angle_client.service_is_ready():
            self.get_logger().error("get_servo_angle service not available")
            return None

        request = GetFloat32List.Request()
        future = self.get_servo_angle_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result.ret == 0:
            self.get_logger().info(f"Current joint positions: {result.datas}")
            return result.datas
        else:
            self.get_logger().error(f"Failed to get joint positions, error code: {result.ret}")
            return None

    def move_to_joint_positions(
        self,
        positions,
        speed=25.0,
        acceleration=10.0,
        is_radian=False,
        use_angle_j=False,
    ):
        """
        Move to specified joint positions

        Args:
            positions: List of 7 joint positions
            speed: Motion speed percentage (1-100) as float
            acceleration: Acceleration percentage (1-100) as float
            is_radian: Use radians (True) or degrees (False)
            use_angle_j: Use set_servo_angle_j service instead of set_servo_angle

        Returns:
            True for success, False for failure
        """
        # Safety check
        if len(positions) != 7:
            self.get_logger().error(
                f"Joint positions must be a list of 7 elements, got {len(positions)}"
            )
            return False

        client = self.set_servo_angle_j_client if use_angle_j else self.set_servo_angle_client

        if not client.service_is_ready():
            service_name = "set_servo_angle_j" if use_angle_j else "set_servo_angle"
            self.get_logger().error(f"{service_name} service not available")
            return False

        request = MoveJoint.Request()

        # IMPORTANT: Convert from degrees to radians if needed
        # This is critical for robot safety
        angles_to_use = positions
        if not is_radian:
            self.get_logger().info(
                f"Converting angles from degrees to radians (REQUIRED FOR SAFETY): {positions}"
            )
            # Convert each angle from degrees to radians
            angles_to_use = [math.radians(angle) for angle in positions]

        request.angles = angles_to_use
        request.speed = float(speed)  # Ensure we're using float type
        request.acc = float(acceleration)  # Ensure we're using float type
        request.mvtime = 0.0  # 0 means automatic calculation of motion time
        request.wait = False  # Wait for motion to complete
        # Note: is_radian attribute doesn't exist in the request

        self.get_logger().info(
            f"Moving to joint positions: {positions} degrees or {angles_to_use} radians"
        )
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(f"Moving to joint positions: {positions}")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result.ret == 0:
            self.get_logger().info("Joint motion completed!")
            return True
        else:
            self.get_logger().error(f"Joint motion failed, error code: {result.ret}")
            return False

    def move_to_cartesian_pose(self, position, speed=25.0, acceleration=10.0, is_radian=False):
        """
        Move end-effector to Cartesian pose via linear motion

        Args:
            position: [x, y, z, roll, pitch, yaw] position and orientation list
            speed: Motion speed percentage (1-100) as float
            acceleration: Acceleration percentage (1-100) as float
            is_radian: Use radians (True) or degrees (False) for orientation

        Returns:
            True for success, False for failure
        """
        # Safety check
        if len(position) != 6:
            self.get_logger().error(
                f"Cartesian pose must be a list of 6 elements, got {len(position)}"
            )
            return False

        if not self.set_position_client.service_is_ready():
            self.get_logger().error("set_position service not available")
            return False

        # Create a copy of position to avoid modifying the original
        position_copy = position.copy()

        # IMPORTANT: Convert orientation from degrees to radians if needed
        # This is critical for robot safety
        if not is_radian:
            self.get_logger().info(
                f"Converting orientation from degrees to radians (REQUIRED FOR SAFETY): {position}"
            )
            # Only convert the orientation part (last 3 elements)
            position_copy[3] = math.radians(position[3])  # roll
            position_copy[4] = math.radians(position[4])  # pitch
            position_copy[5] = math.radians(position[5])  # yaw

        request = MoveCartesian.Request()
        request.pose = position_copy
        request.speed = float(speed)  # Ensure we're using float type
        request.acc = float(acceleration)  # Ensure we're using float type
        request.mvtime = 0.0  # 0 means automatic calculation of motion time
        request.wait = False  # Wait for motion to complete
        # Note: is_radian attribute doesn't exist in the request

        self.get_logger().info(
            f"Moving to Cartesian pose: {position} (degrees) or {position_copy} (with radians for orientation)"
        )
        future = self.set_position_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(f"Moving to Cartesian pose: {position}")
        future = self.set_position_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result.ret == 0:
            self.get_logger().info("Cartesian motion completed!")
            return True
        else:
            self.get_logger().error(f"Cartesian motion failed, error code: {result.ret}")
            return False

    def move_to_home_position(self, speed=25.0, acceleration=10.0):
        """
        Move to predefined home position: [0, -36, 0, 13, 0, 49, 0]

        Args:
            speed: Motion speed percentage (1-100) as float
            acceleration: Acceleration percentage (1-100) as float

        Returns:
            True for success, False for failure
        """
        self.get_logger().info("Moving to home position [0, -36, 0, 13, 0, 49, 0]")
        return self.move_to_joint_positions(
            self.home_joint_positions,
            speed=float(speed),
            acceleration=float(acceleration),
            is_radian=False,  # Use degrees instead of radians
        )

    def run_joint_control_demo(self, speed=20.0, acc=10.0):
        """
        Run joint position control demo

        Args:
            speed: Motion speed percentage as float
            acc: Acceleration percentage as float
        """
        # First enable the robot and set it to position control mode
        self.enable_robot(True)
        self.set_robot_mode(0)
        self.set_robot_state(0)

        # First move to home position
        self.move_to_home_position(speed=float(speed), acceleration=float(acc))
        time.sleep(1.0)

        # Define a series of joint positions (degrees)
        joint_positions_1 = [10.0, -30.0, 5.0, 20.0, 5.0, 40.0, 0.0]
        joint_positions_2 = [-10.0, -40.0, -5.0, 30.0, -5.0, 55.0, 0.0]

        # Execute joint motion sequence
        self.get_logger().info("Starting joint position control demo...")

        self.move_to_joint_positions(
            joint_positions_1,
            speed=float(speed),
            acceleration=float(acc),
            is_radian=False,
        )
        time.sleep(1.0)

        self.move_to_joint_positions(
            joint_positions_2,
            speed=float(speed),
            acceleration=float(acc),
            is_radian=False,
        )
        time.sleep(1.0)

        # Return to home position
        self.move_to_home_position(speed=float(speed), acceleration=float(acc))

        self.get_logger().info("Joint position control demo completed!")

    def run_cartesian_control_demo(self, speed=20.0, acc=10.0):
        """
        Run Cartesian position control demo

        Args:
            speed: Motion speed percentage as float
            acc: Acceleration percentage as float
        """
        # First ensure robot is in correct mode and state
        self.enable_robot(True)
        self.set_robot_mode(0)
        self.set_robot_state(0)

        # First move to home position
        self.move_to_home_position(speed=float(speed), acceleration=float(acc))
        time.sleep(1.0)

        # Get current joint positions
        current_joints = self.get_current_joint_positions()
        if not current_joints:
            self.get_logger().error("Unable to get current joint positions, demo cancelled")
            return

        # We'll do relative motions from home position for safety
        # Define Cartesian positions near the home position
        # These should be determined based on the actual robot kinematics
        # For demonstration, using approximate values near home position
        home_tcp = [
            30.0,
            0.0,
            30.0,
            180.0,
            0.0,
            0.0,
        ]  # Example TCP position when at home position

        # Define relative motions (in meters for position, degrees for orientation)
        # Move forward
        pose_1 = [
            home_tcp[0] + 0.05,
            home_tcp[1],
            home_tcp[2],
            home_tcp[3],
            home_tcp[4],
            home_tcp[5],
        ]
        # Move right
        pose_2 = [
            home_tcp[0],
            home_tcp[1] + 0.05,
            home_tcp[2],
            home_tcp[3],
            home_tcp[4],
            home_tcp[5],
        ]
        # Move up
        pose_3 = [
            home_tcp[0],
            home_tcp[1],
            home_tcp[2] + 0.05,
            home_tcp[3],
            home_tcp[4],
            home_tcp[5],
        ]
        # Change orientation slightly
        pose_4 = [
            home_tcp[0],
            home_tcp[1],
            home_tcp[2],
            home_tcp[3],
            home_tcp[4] + 10.0,
            home_tcp[5],
        ]

        # Execute Cartesian motion sequence
        self.get_logger().info("Starting Cartesian position control demo...")

        self.move_to_cartesian_pose(pose_1, speed=speed, acceleration=acc, is_radian=False)
        time.sleep(1.0)

        self.move_to_cartesian_pose(pose_2, speed=speed, acceleration=acc, is_radian=False)
        time.sleep(1.0)

        self.move_to_cartesian_pose(pose_3, speed=speed, acceleration=acc, is_radian=False)
        time.sleep(1.0)

        self.move_to_cartesian_pose(pose_4, speed=speed, acceleration=acc, is_radian=False)
        time.sleep(1.0)

        # Return to home position
        self.move_to_home_position(speed=speed, acceleration=acc)

        self.get_logger().info("Cartesian position control demo completed!")


def main(args=None):
    """Main function"""
    # Initialize ROS2
    rclpy.init(args=args)

    # Create controller
    controller = XArm7Controller()

    try:
        # Safety settings
        low_speed = 10.0  # Low speed mode, 10% speed
        low_acc = 5.0  # Low acceleration, 5%

        # Print available ROS2 services
        controller.get_logger().info("Available ROS2 services:")
        os.system("ros2 service list")

        # Ask user which demo to run
        controller.get_logger().info("\nSelect an option:")
        controller.get_logger().info("1. Run joint position control demo")
        controller.get_logger().info("2. Run Cartesian position control demo")
        controller.get_logger().info("3. Move to home position")
        controller.get_logger().info("4. Enable robot")
        controller.get_logger().info("5. Disable robot")
        controller.get_logger().info("0. Exit")

        choice = input("Enter choice (0-5): ")

        if choice == "1":
            controller.run_joint_control_demo(speed=low_speed, acc=low_acc)
        elif choice == "2":
            controller.run_cartesian_control_demo(speed=low_speed, acc=low_acc)
        elif choice == "3":
            controller.move_to_home_position(speed=low_speed, acceleration=low_acc)
        elif choice == "4":
            controller.enable_robot(True)
        elif choice == "5":
            controller.enable_robot(False)
        elif choice == "0":
            controller.get_logger().info("Exiting program.")
        else:
            controller.get_logger().error("Invalid choice. Exiting program.")

    except Exception as e:
        controller.get_logger().error(f"Error during execution: {str(e)}")
        # Try to stop robot on error
        try:
            controller.set_robot_state(4)  # Stop state
        except BaseException:
            pass

    finally:
        # Cleanup and shutdown
        controller.get_logger().info("Shutting down node.")
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
