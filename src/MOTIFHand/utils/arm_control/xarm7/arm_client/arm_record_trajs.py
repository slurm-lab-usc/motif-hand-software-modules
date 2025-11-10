import json
import os
import time
from datetime import datetime

from MOTIFHand.utils.arm_control.xarm7.arm_client.client import XArmController

# Configuration flags
RECORD_JOINT_ANGLES = True  # Set to True to record joint angles
RECORD_EE_POSE = True  # Set to True to record end-effector pose


class ArmRecorder:
    """
    Records the robot arm state (joint angles and/or end-effector pose)
    when user presses Enter key.
    """

    def __init__(self, ip="192.168.1.216", output_dir="recordings"):
        """
        Initialize the recorder

        Args:
            ip: Robot's IP address
            output_dir: Directory to save recording files
        """
        self.controller = XArmController.get_instance(ip)
        self.recorded_data = {"joint_angles": [], "ee_poses": [], "timestamps": []}
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def record_current_state(self):
        """
        Record the current state of the robot arm
        """
        global joint_angles, ee_pose
        timestamp = time.time()
        formatted_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")

        # Record joint angles if enabled
        if RECORD_JOINT_ANGLES:
            joint_angles = self.controller.get_joint_angle(is_radian=False)
            self.recorded_data["joint_angles"].append(joint_angles)

        # Record end-effector pose if enabled
        if RECORD_EE_POSE:
            ee_pose = self.controller.get_ee_pose(is_radian=False)
            self.recorded_data["ee_poses"].append(ee_pose)

        # Record timestamp
        self.recorded_data["timestamps"].append(formatted_time)

        # Print information about what was recorded
        print(f"Recording #{len(self.recorded_data['timestamps'])} at {formatted_time}")
        if RECORD_JOINT_ANGLES:
            print(f"Joint angles: {joint_angles}")
        if RECORD_EE_POSE:
            print(f"EE pose: {ee_pose}")

    def save_to_json(self, filename=None):
        """
        Save recorded data to JSON file

        Args:
            filename: Optional filename, if None a timestamp will be used
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arm_recording_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        # Create a formatted JSON object
        output_data = {
            "metadata": {
                "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "record_count": len(self.recorded_data["timestamps"]),
                "recorded_joint_angles": RECORD_JOINT_ANGLES,
                "recorded_ee_pose": RECORD_EE_POSE,
            },
            "recordings": [],
        }

        # Combine data for each timestamp
        for i in range(len(self.recorded_data["timestamps"])):
            record = {"timestamp": self.recorded_data["timestamps"][i]}

            if RECORD_JOINT_ANGLES:
                record["joint_angles"] = self.recorded_data["joint_angles"][i]

            if RECORD_EE_POSE:
                record["ee_pose"] = self.recorded_data["ee_poses"][i]

            output_data["recordings"].append(record)

        # Write to file
        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved recording to {filepath}")
        return filepath


def main():
    """
    Main function to run the recorder
    """
    print("=== XArm State Recorder ===")
    print(f"Recording joint angles: {RECORD_JOINT_ANGLES}")
    print(f"Recording EE pose: {RECORD_EE_POSE}")
    print("Press Enter to record the current state, or type 'q' to save and quit.")

    # Initialize recorder
    recorder = ArmRecorder(ip="192.168.1.239", output_dir="recordings")
    print("Recorder initialized. Press Enter to start recording.")
    print("Press 'q' to quit and save the recording.")

    # Main loop
    try:
        while True:
            user_input = input("> ")
            if user_input.lower() == "q":
                break

            # Record on empty input (just pressing Enter)
            recorder.record_current_state()

    except KeyboardInterrupt:
        print("\nRecording interrupted.")

    finally:
        # Save recording to file
        if len(recorder.recorded_data["timestamps"]) > 0:
            filepath = recorder.save_to_json()
            print(f"Recording saved to {filepath}")
        else:
            print("No data recorded.")


if __name__ == "__main__":
    main()
