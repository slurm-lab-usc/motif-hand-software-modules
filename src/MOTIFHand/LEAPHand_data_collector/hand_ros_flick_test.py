#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class HandFlickControl(Node):
    def __init__(self):
        super().__init__("hand_flick_control")
        self.pub_hand = self.create_publisher(JointState, "/cmd_allegro", 10)

        # Two preset poses as 4x4 matrices
        # First pose: initial position
        self.pose1 = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],  # Thumb joints
                [0.0, 0.0, 0.0, 0.0],  # Index finger joints
                [0.0, 0.0, 0.0, 0.0],  # Middle finger joints
                [0.0, 0.0, 0.0, 0.0],  # Ring finger joints
            ]
        )

        # Second pose: flick position
        self.pose2 = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],  # Thumb joints
                [0.0, 0.0, 0.0, 0.0],  # Index finger joints
                [0.0, 0.0, 0.0, 0.0],  # Middle finger joints
                [0.0, 0.0, 0.0, 0.0],  # Ring finger joints
            ]
        )

        self.current_pose = 1  # Currently at pose1

    def execute_pose(self, pose_data):
        # Flatten the 4x4 matrix into 16 joint angles
        flat_pose = pose_data.flatten()

        # Create and publish a joint state
        state = JointState()
        state.position = flat_pose.tolist()
        self.pub_hand.publish(state)
        print(f"Executing pose {self.current_pose}")

    def toggle_pose(self):
        if self.current_pose == 1:
            self.execute_pose(self.pose2)
            self.current_pose = 2
        else:
            self.execute_pose(self.pose1)
            self.current_pose = 1


def main(args=None):
    rclpy.init(args=args)
    hand_control = HandFlickControl()

    print("Controls:")
    print("Space: Toggle between poses")
    print("q: Quit")

    while True:
        try:
            # Wait for user input
            cmd = input("\nPress space to toggle pose, q to quit: ").lower()
            if cmd == " ":
                hand_control.toggle_pose()
            elif cmd == "q":
                break
        except KeyboardInterrupt:
            break

    hand_control.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
