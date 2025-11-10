import argparse
import logging
import time

import zmq
from MOTIFHand.utils.arm_control.xarm7.arm_client.arm_play_trajs import ArmPlayer


class ArmDataCollector:
    def __init__(self, sensor_ip="192.168.0.110", sensor_port=5556):
        """
        Initialize the arm data collector

        Args:
            sensor_ip: IP address of the Raspberry Pi
            sensor_port: ZMQ port for sensor data collection
        """
        # Initialize ZMQ context and socket for sensor communication
        self.context = zmq.Context()
        self.sensor_socket = self.context.socket(zmq.REQ)
        self.sensor_socket.connect(f"tcp://{sensor_ip}:{sensor_port}")

        # Initialize arm player
        self.arm_player = None

        # Add mode control
        self.auto_mode = False  # Default is manual mode

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ArmDataCollector")

    def _send_command(self, command):
        """
        Send a command to the sensor server and wait for response

        Args:
            command: Command dictionary to send

        Returns:
            dict: Response from server
        """
        self.sensor_socket.send_json(command)
        response = self.sensor_socket.recv_json()
        return response

    def initialize(self, arm_ip="192.168.1.239", sensor_params=None):
        """
        Initialize both the arm and sensor collector

        Args:
            arm_ip: Robot's IP address
            sensor_params: Parameters for sensor initialization

        Returns:
            bool: Whether initialization was successful
        """
        # Initialize arm player
        try:
            self.arm_player = ArmPlayer(ip=arm_ip)
            self.logger.info("Arm player initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize arm player: {e}")
            return False

        # Initialize sensor collector
        if sensor_params is None:
            sensor_params = {
                "object_name": "default_object",
                "output_dir": "./data",
                "camera_type": "Lepton35",
            }

        command = {"type": "initialize", "parameters": sensor_params}

        response = self._send_command(command)
        if response["status"] == "success":
            self.logger.info("Sensor data collector initialized successfully")
            return True
        else:
            self.logger.error(f"Failed to initialize sensor data collector: {response['message']}")
            return False

    def load_trajectory(self, filepath):
        """
        Load a trajectory file for the arm

        Args:
            filepath: Path to the trajectory file

        Returns:
            bool: Whether loading was successful
        """
        if not self.arm_player:
            self.logger.error("Arm player not initialized")
            return False

        success = self.arm_player.load_trajectory(filepath)
        if success:
            self.logger.info("Trajectory loaded successfully")
            return True
        else:
            self.logger.error("Failed to load trajectory")
            return False

    def set_speed(self, speed):
        """
        Set the arm movement speed

        Args:
            speed: Speed value (1-100)

        Returns:
            bool: Whether setting speed was successful
        """
        if not self.arm_player:
            self.logger.error("Arm player not initialized")
            return False

        if 1 <= speed <= 100:
            self.arm_player.speed = speed
            self.logger.info(f"Arm speed set to {speed}")
            return True
        else:
            self.logger.error("Speed must be between 1 and 100")
            return False

    def move_and_collect(self, point_index, capture_index):
        """
        Move arm to position and collect data

        Args:
            point_index: Index of the point in trajectory
            capture_index: Index for data capture

        Returns:
            bool: Whether the operation was successful
        """
        if not self.arm_player:
            self.logger.error("Arm player not initialized")
            return False

        # Move arm to position
        self.logger.info(f"Moving arm to point {point_index}")
        if not self.arm_player.play_point(point_index):
            self.logger.error(f"Failed to move arm to point {point_index}")
            return False

        # Wait for arm to stabilize
        time.sleep(2)

        # Collect data and wait for completion
        command = {"type": "collect_data", "capture_index": capture_index}

        response = self._send_command(command)
        if response["status"] == "success" and response.get("completed", False):
            # Handle progress information
            progress = response.get("progress", {})
            if progress:
                total = progress.get("total_captures", 0)
                successful = progress.get("successful_captures", 0)
                percentage = progress.get("completion_percentage", 0)
                current = progress.get("current_capture", 0)

                self.logger.info(
                    f"Progress: {successful}/{total} captures completed ({percentage:.1f}%)"
                )
                self.logger.info(f"Current capture: {current}")

            self.logger.info(f"Data collection completed for capture {capture_index}")
            return response.get("success", False)
        else:
            self.logger.error(f"Failed to collect data: {response['message']}")
            return False

    def record_video(self, duration=None):
        """
        Record a video

        Args:
            duration: Optional duration in seconds

        Returns:
            bool: Whether recording was successful
        """
        command = {"type": "record_video", "duration": duration}

        response = self._send_command(command)
        if response["status"] == "success":
            self.logger.info("Video recording completed")
            return True
        else:
            self.logger.error(f"Failed to record video: {response['message']}")
            return False

    def convert_videos(self):
        """
        Convert recorded videos to MP4 format

        Returns:
            bool: Whether conversion was successful
        """
        command = {"type": "convert_videos"}

        response = self._send_command(command)
        if response["status"] == "success":
            self.logger.info("Video conversion completed")
            return True
        else:
            self.logger.error(f"Failed to convert videos: {response['message']}")
            return False

    def set_mode(self, auto_mode=False):
        """
        Set control mode

        Args:
            auto_mode: True for automatic mode, False for manual mode
        """
        self.auto_mode = auto_mode
        mode_str = "Automatic" if auto_mode else "Manual"
        self.logger.info(f"Switched to {mode_str} mode")

    def stop_sensor_server(self):
        """
        Send stop signal to Raspberry Pi
        """
        command = {"type": "stop_server"}
        try:
            response = self._send_command(command)
            if response["status"] == "success":
                self.logger.info("Successfully sent stop signal to sensor server")
            else:
                self.logger.warning(f"Failed to send stop signal: {response['message']}")
        except Exception as e:
            self.logger.error(f"Error sending stop signal: {e}")

    def run_collection(self, trajectory_file, num_captures, object_name, output_dir="./data"):
        """
        Run the data collection process

        Args:
            trajectory_file: Path to the trajectory file
            num_captures: Number of captures to perform
            object_name: Name of the object being scanned
            output_dir: Output directory for data
        """
        # Initialize systems
        if not self.initialize(
            sensor_params={
                "object_name": object_name,
                "output_dir": output_dir,
                "camera_type": "Lepton35",
            }
        ):
            self.logger.error("Failed to initialize systems")
            return

        # Load trajectory
        if not self.load_trajectory(trajectory_file):
            self.logger.error("Failed to load trajectory")
            return

        # Set initial speed
        self.set_speed(20)  # Start with a safe speed

        # Record initial video
        self.logger.info("Recording initial video...")
        # self.record_video()

        # Perform captures
        i = 1
        while i <= num_captures:
            self.logger.info(f"Current Pose: {i}/{num_captures}")

            if not self.auto_mode:
                # Wait for user confirmation in manual mode
                while True:
                    choice = input(
                        "\nPlease select an option:\n"
                        "1. Continue to next position\n"
                        "2. Switch mode\n"
                        "3. Exit collection\n"
                        "Enter option (1/2/3): "
                    )

                    if choice == "1":
                        break
                    elif choice == "2":
                        self.auto_mode = not self.auto_mode
                        mode_str = "Automatic" if self.auto_mode else "Manual"
                        print(f"Switched to {mode_str} mode")
                        if self.auto_mode:
                            break
                    elif choice == "3":
                        self.logger.info("Collection terminated by user")
                        self.stop_sensor_server()  # Send stop signal
                        return
                    else:
                        print("Invalid option, please try again")

            # Move arm and collect data
            if not self.move_and_collect(i - 1, i):  # 0-based index for points
                self.logger.error(f"Failed to complete capture {i}")
                if not self.auto_mode:
                    retry = input("Retry current position? (y/n): ").lower() == "y"
                    if retry:
                        continue
                i += 1
                continue

            self.logger.info(f"Completed capture {i}")
            i += 1

        # Convert videos at the end
        # self.logger.info("Converting videos...")
        # self.convert_videos()

        self.logger.info("Data collection completed")

        # Send stop signal to Raspberry Pi
        self.stop_sensor_server()

    def close(self):
        """Close all connections"""
        self.sensor_socket.close()
        self.context.term()


def main():
    """Main function to run the collector"""
    parser = argparse.ArgumentParser(description="Arm data collector")
    parser.add_argument(
        "--sensor-ip", default="192.168.0.110", help="IP address of the Raspberry Pi"
    )
    parser.add_argument(
        "--sensor-port",
        type=int,
        default=5556,
        help="ZMQ port for sensor data collection",
    )
    parser.add_argument("--arm-ip", default="192.168.1.239", help="Robot IP address")
    parser.add_argument(
        "--trajectory",
        default="/home/hanyang/Projects/fullhandtactile/arm_control/xarm7/arm_client/recordings/MOTIF-SCAN.json",
        help="Path to the trajectory file",
    )
    parser.add_argument("--captures", type=int, required=True, help="Number of captures to perform")
    parser.add_argument("--object", required=True, help="Name of the object being scanned")
    parser.add_argument("--output", default="./data", help="Output directory for data")
    parser.add_argument("--auto", action="store_true", help="Enable automatic mode")

    args = parser.parse_args()

    collector = ArmDataCollector(sensor_ip=args.sensor_ip, sensor_port=args.sensor_port)

    # Set initial mode
    collector.set_mode(args.auto)

    try:
        collector.run_collection(
            trajectory_file=args.trajectory,
            num_captures=args.captures,
            object_name=args.object,
            output_dir=args.output,
        )
    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
