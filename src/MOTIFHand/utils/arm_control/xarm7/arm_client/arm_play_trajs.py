import argparse
import json
import os
import re

from MOTIFHand.utils.arm_control.xarm7.arm_client.client import XArmController


def list_available_recordings(directory="recordings"):
    """
    List all available recording files

    Args:
        directory: Directory containing recording files

    Returns:
        list: List of recording file paths
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []

    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not files:
        print(f"No recording files found in {directory}.")
        return []

    print(f"Found {len(files)} recording files:")
    for i, file in enumerate(files):
        file_path = os.path.join(directory, file)
        try:
            # Try to read metadata to show more information
            with open(file_path) as f:
                data = json.load(f)
                metadata = data.get("metadata", {})
                count = metadata.get("record_count", "Unknown")
                recorded_at = metadata.get("recorded_at", "Unknown")
                print(f"{i + 1}. {file} - {count} points, recorded at {recorded_at}")
        except BaseException:
            print(f"{i + 1}. {file}")

    return [os.path.join(directory, f) for f in files]


class ArmPlayer:
    """
    Plays back recorded robot arm trajectories from JSON files
    created by the ArmRecorder.
    """

    def __init__(self, ip="192.168.1.239", default_speed=20):
        """
        Initialize the player

        Args:
            ip: Robot's IP address
            default_speed: Default movement speed (1-100)
        """
        self.controller = XArmController.get_instance(ip)
        self.trajectory_data = None
        self.metadata = None
        self.speed = default_speed  # Default to a lower speed for safety

    def load_trajectory(self, filepath):
        """
        Load trajectory data from a JSON file

        Args:
            filepath: Path to the recording JSON file

        Returns:
            bool: True if loading was successful
        """
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Extract metadata and recordings
            self.metadata = data.get("metadata", {})
            self.trajectory_data = data.get("recordings", [])

            if not self.trajectory_data:
                print("Error: No trajectory data found in the file.")
                return False

            print(f"Loaded trajectory with {len(self.trajectory_data)} points.")
            print(f"Metadata: {self.metadata}")
            return True

        except Exception as e:
            print(f"Error loading trajectory file: {e}")
            return False

    def play_point(self, point_index):
        """
        Play a specific point from the trajectory

        Args:
            point_index: Index of the point to play

        Returns:
            bool: True if playback was successful
        """
        if not self.trajectory_data:
            print("No trajectory data loaded. Please load a trajectory first.")
            return False

        if point_index < 0 or point_index >= len(self.trajectory_data):
            print(f"Invalid point index. Must be between 0 and {len(self.trajectory_data) - 1}.")
            return False

        point = self.trajectory_data[point_index]
        print(f"Playing point {point_index + 1}/{len(self.trajectory_data)} at speed {self.speed}")

        # Check what data is available
        has_joint_angles = self.metadata.get("recorded_joint_angles", False)
        has_ee_pose = self.metadata.get("recorded_ee_pose", False)

        try:
            # Use joint angles if available (and were recorded), otherwise use
            # EE pose
            if has_joint_angles and "joint_angles" in point:
                joint_angles = point["joint_angles"]

                # Fix: Check if joint_angles is a nested list and extract the
                # inner list
                if (
                    isinstance(joint_angles, list)
                    and len(joint_angles) == 2
                    and isinstance(joint_angles[1], list)
                ):
                    # This handles the case where joint_angles is [0,
                    # [...actual angles...]]
                    joint_angles = joint_angles[1]

                # Ensure we have a flat list of angles
                if not isinstance(joint_angles, list) or not all(
                    isinstance(x, (int, float)) for x in joint_angles
                ):
                    print(f"Warning: Invalid joint angles format: {joint_angles}")
                    return False

                print(f"Setting joint angles: {joint_angles}")
                # Use the current speed setting
                self.controller.set_joint_angle(angle=joint_angles, speed=self.speed, wait=True)
                return True

            elif has_ee_pose and "ee_pose" in point:
                ee_pose = point["ee_pose"]

                # Fix: Check if ee_pose is a nested list and extract the inner
                # list
                if isinstance(ee_pose, list) and len(ee_pose) == 2 and isinstance(ee_pose[1], list):
                    ee_pose = ee_pose[1]

                # Ensure we have a flat list with 6 values
                if not isinstance(ee_pose, list) or len(ee_pose) != 6:
                    print(f"Warning: Invalid EE pose format: {ee_pose}")
                    return False

                position = ee_pose[:3]
                orientation = ee_pose[3:]
                print(f"Setting EE pose: {ee_pose}")
                # Use the current speed setting
                self.controller.set_ee_pose(position, orientation, speed=self.speed, wait=True)
                return True

            else:
                print("No valid control data found for this point.")
                return False

        except Exception as e:
            print(f"Error during playback: {e}")
            return False

    def play_trajectory(self, speed=None):
        """
        Play back the loaded trajectory with manual confirmation for each point

        Args:
            speed: Initial speed setting (1-100)
        """
        if not self.trajectory_data:
            print("No trajectory data loaded. Please load a trajectory first.")
            return

        # Set initial speed if provided
        if speed is not None:
            self.speed = speed

        print("Preparing to play trajectory...")
        print(f"Current speed setting: {self.speed} (range: 1-100)")

        # Check what data is available
        has_joint_angles = self.metadata.get("recorded_joint_angles", False)
        has_ee_pose = self.metadata.get("recorded_ee_pose", False)

        if not (has_joint_angles or has_ee_pose):
            print("Error: Trajectory doesn't contain joint angles or EE pose data.")
            return

        print(f"Available data: Joint angles: {has_joint_angles}, EE pose: {has_ee_pose}")
        print(f"Total trajectory points: {len(self.trajectory_data)}")
        print("Commands:")
        print("  Press Enter - play next point")
        print("  'jjXX' - jump to point XX (e.g., 'jj5' for point 5)")
        print("  'sXX' - set speed to XX (e.g., 's10' for speed 10)")
        print("  'q' or 'exit' - quit playback")

        # Confirm with user
        input("Press Enter to start playback (make sure the area around the robot is clear)...")

        print("Starting trajectory playback...")

        try:
            current_index = 0

            while current_index < len(self.trajectory_data):
                # Display current point information
                print(
                    f"\nCurrent position: Point {current_index + 1}/{len(self.trajectory_data)}, Speed: {self.speed}"
                )

                # Get user input
                user_input = input("Enter command: ")

                # Check for exit command
                if user_input.lower() in ["q", "quit", "exit"]:
                    print("Playback terminated by user.")
                    break

                # Check for jump command
                jump_match = re.match(r"jj(\d+)", user_input)
                if jump_match:
                    try:
                        jump_to = int(jump_match.group(1))
                        if 1 <= jump_to <= len(self.trajectory_data):
                            jump_index = jump_to - 1  # Convert to 0-based index
                            print(f"Jumping to point {jump_to}/{len(self.trajectory_data)}")

                            # Play the selected point
                            if self.play_point(jump_index):
                                current_index = jump_index
                            else:
                                print("Failed to jump to selected point.")
                        else:
                            print(
                                f"Invalid point number. Must be between 1 and {len(self.trajectory_data)}."
                            )
                    except ValueError:
                        print("Invalid jump command format. Use 'jjXX' where XX is a number.")
                    continue

                # Check for speed command
                speed_match = re.match(r"s(\d+)", user_input)
                if speed_match:
                    try:
                        new_speed = int(speed_match.group(1))
                        if 1 <= new_speed <= 100:
                            self.speed = new_speed
                            print(f"Speed set to {self.speed}")
                        else:
                            print("Invalid speed value. Must be between 1 and 100.")
                    except ValueError:
                        print("Invalid speed command format. Use 'sXX' where XX is a number.")
                    continue

                # Process normal progression (Enter key)
                if user_input == "":
                    # Play the current point
                    if self.play_point(current_index):
                        current_index += 1
                    else:
                        print("Failed to play current point. Try again or jump to next point.")
                else:
                    print(
                        "Unknown command. Press Enter to continue, 'jjXX' to jump, 'sXX' to set speed, or 'q' to quit."
                    )

            if current_index >= len(self.trajectory_data):
                print("\nTrajectory playback completed.")

        except KeyboardInterrupt:
            print("\nPlayback interrupted. Stopping robot...")
            self.controller.stop_control()

        except Exception as e:
            print(f"Error during playback: {e}")
            self.controller.stop_control()


def main():
    """
    Main function to run the player
    """
    parser = argparse.ArgumentParser(description="Play back recorded robot arm trajectories")
    parser.add_argument("--file", "-f", help="Path to trajectory file")
    parser.add_argument(
        "--speed", "-s", type=int, default=20, help="Initial movement speed (1-100)"
    )
    parser.add_argument("--ip", "-i", default="192.168.1.239", help="Robot IP address")
    args = parser.parse_args()

    # Validate speed argument
    if args.speed < 1 or args.speed > 100:
        print("Speed must be between 1 and 100. Using default speed of 20.")
        args.speed = 20

    player = ArmPlayer(ip=args.ip, default_speed=args.speed)

    # If no file specified, list available recordings
    if not args.file:
        files = list_available_recordings()
        if files:
            selection = input("Enter the number of the file to play (or 'q' to quit): ")
            if selection.lower() == "q":
                return
            try:
                file_index = int(selection) - 1
                if 0 <= file_index < len(files):
                    args.file = files[file_index]
                else:
                    print("Invalid selection.")
                    return
            except BaseException:
                print("Invalid input.")
                return
        else:
            return

    # Load and play trajectory
    if player.load_trajectory(args.file):
        player.play_trajectory(speed=args.speed)


if __name__ == "__main__":
    main()
