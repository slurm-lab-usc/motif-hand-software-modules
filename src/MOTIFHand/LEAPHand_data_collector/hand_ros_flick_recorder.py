#!/usr/bin/env python3

import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from MOTIFHand.utils.pcb_data.data_recorder import SensorDataCollector


class HandFlickControl(Node):
    def __init__(self):
        super().__init__("hand_flick_control")
        self.pub_hand = self.create_publisher(JointState, "/cmd_allegro", 10)

        # Two preset poses as 4x4 matrices
        # First pose: initial position
        self.pose1 = np.array(
            [
                [0.0, 1.25, 1.40, 0.80],  # Thumb joints
                [0.0, 0.0, 0.0, 0.0],  # Index finger joints
                [0.0, 0.0, 0.0, 0.0],  # Middle finger joints
                [0.0, 0.0, 0.0, 0.0],  # Ring finger joints
            ]
        )

        # Second pose: flick position
        self.pose2 = np.array(
            [
                [0.0, 1.25, 0.0, 0.0],  # Thumb joints
                [0.0, 0.0, 0.0, 0.0],  # Index finger joints
                [0.0, 0.0, 0.0, 0.0],  # Middle finger joints
                [0.0, 0.0, 0.0, 0.0],  # Ring finger joints
            ]
        )

        self.current_pose = 1  # Currently at pose1

        # Initialize sensor collector
        self.sensor_collector = SensorDataCollector()
        if not self.sensor_collector.start_sensor():
            self.get_logger().error("Failed to start sensor!")
            return

        # Prepare recording with default name
        self.sensor_collector.prepare_recording("red")

    def execute_pose(self, pose_data):
        # Flatten the 4x4 matrix into 16 joint angles
        flat_pose = pose_data.flatten()

        # Create and publish joint state
        state = JointState()
        state.position = flat_pose.tolist()
        self.pub_hand.publish(state)
        print(f"Executing pose {self.current_pose}")

    def toggle_pose(self):
        if self.current_pose == 1:
            # Start recording before executing pose2
            print("Starting recording...")
            if not self.sensor_collector.start_recording():
                print("Failed to start recording!")
                return

            # Wait a moment before executing the flick
            time.sleep(1.5)

            # Execute pose2 (flick)
            self.execute_pose(self.pose2)
            self.current_pose = 2

            # Wait for 3 seconds
            print("Waiting for 3 seconds...")
            time.sleep(3)

            # Stop recording
            print("Stopping recording...")
            self.sensor_collector.stop_recording()

            # Prepare for next recording
            self.sensor_collector.prepare_recording("red")
        else:
            self.execute_pose(self.pose1)
            self.current_pose = 1


def main(args=None):
    rclpy.init(args=args)
    hand_control = HandFlickControl()

    print("Controls:")
    print("Enter: Execute flick motion and record data")
    print("q: Quit")

    while True:
        try:
            # Wait for user input
            cmd = input("\nPress Enter to execute flick, q to quit: ").lower()
            if cmd == "" or cmd == "q":
                if cmd == "q":
                    break
                hand_control.toggle_pose()
        except KeyboardInterrupt:
            break

    # Cleanup
    hand_control.sensor_collector.stop_sensor()
    hand_control.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
