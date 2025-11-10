import json
import os
import time
import traceback
from contextlib import contextmanager
from datetime import datetime

# Try to import required dependencies
try:
    import cv2
    import numpy as np
    import open3d as o3d
    import pyrealsense2 as rs
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    print("Please install required packages: pip install pyrealsense2 numpy opencv-python open3d")
    exit(1)


class RGBDDataCollector:
    """
    A class to collect RGBD and point cloud data from Intel RealSense D405 camera
    at different viewpoints around an object.
    """

    def __init__(self):
        """Initialize the RealSense camera pipeline and configuration."""
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable color and depth streams with lower resolution
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Camera parameters (will be updated after starting pipeline)
        self.depth_scale = 0.001  # Default depth scale for RealSense cameras
        self.min_depth = 0.3  # Minimum depth in meters
        self.max_depth = 3.0  # Maximum depth in meters

        # Create align object to align depth frames to color frames
        self.align = None

        # Data storage variables
        self.object_name = None
        self.capture_date = None
        self.total_positions = None
        self.data_dir = None
        self.metadata = {}

        # Camera intrinsics (will be set after starting pipeline)
        self.depth_intrin = None
        self.color_intrin = None

        # Profile
        self.profile = None

    def start(self):
        """Start the RealSense pipeline and get camera parameters."""
        print("Starting RealSense camera...")
        try:
            self.profile = self.pipeline.start(self.config)

            # Get depth scale
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            # Get camera intrinsics
            self.depth_intrin = (
                self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            )
            self.color_intrin = (
                self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            )

            # Create alignment object
            self.align = rs.align(rs.stream.color)

            # Set some reasonable depth settings
            # Find depth sensor and set options
            depth_sensor = self.profile.get_device().first_depth_sensor()
            if depth_sensor.supports(rs.option.exposure):
                depth_sensor.set_option(rs.option.exposure, 43000)  # Set exposure value

            # Allow auto-exposure to stabilize
            print("Warming up camera (2 seconds)...")
            start_time = time.time()
            while time.time() - start_time < 2:
                self.pipeline.wait_for_frames()

            print("Camera started successfully.")
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            traceback.print_exc()
            return False

    def stop(self):
        """Stop the RealSense pipeline."""
        try:
            self.pipeline.stop()
            print("Camera stopped.")
        except Exception as e:
            print(f"Error stopping camera: {e}")

    @contextmanager
    def camera_session(self):
        """Context manager for camera session."""
        success = self.start()
        if not success:
            raise RuntimeError("Failed to start camera")
        try:
            yield
        finally:
            self.stop()

    def create_directories(self):
        """Create the directory structure for data storage."""
        try:
            self.data_dir = os.path.join("data", f"{self.object_name}_{self.capture_date}")
            os.makedirs(os.path.join(self.data_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(self.data_dir, "depth"), exist_ok=True)
            os.makedirs(os.path.join(self.data_dir, "pointcloud"), exist_ok=True)

            print(f"Created data directories at {self.data_dir}")
            return True
        except Exception as e:
            print(f"Error creating directories: {e}")
            return False

    def capture_frame(self, timeout_ms=5000):
        """
        Capture a single frame of RGB and depth data.

        Args:
            timeout_ms: Timeout in milliseconds for frame capture

        Returns:
            Tuple of (color_image, depth_image) or (None, None) if capture failed
        """
        tries = 0
        max_tries = 3

        while tries < max_tries:
            try:
                # Wait for a coherent pair of frames
                frames = self.pipeline.wait_for_frames(timeout_ms)
                if not frames:
                    print("No frames received, retrying...")
                    tries += 1
                    continue

                # Align depth to color frame
                aligned_frames = self.align.process(frames)

                # Get aligned frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    print("Invalid frames received, retrying...")
                    tries += 1
                    continue

                # Convert frames to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                return color_image, depth_image

            except Exception as e:
                print(f"Error capturing frame: {e}")
                tries += 1

        print("Failed to capture valid frames after multiple attempts")
        return None, None

    def create_point_cloud(self, color_image, depth_image):
        """
        Create a point cloud from RGB and depth images.

        Args:
            color_image: RGB image as numpy array
            depth_image: Depth image as numpy array

        Returns:
            Open3D point cloud object or None if creation fails
        """
        try:
            # Ensure we have valid intrinsics
            if not self.color_intrin or not self.depth_intrin:
                print("Error: Camera intrinsics not available")
                return None

            # Create Open3D images
            o3d_color = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            o3d_depth = o3d.geometry.Image(depth_image)

            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color,
                o3d_depth,
                depth_scale=1.0 / self.depth_scale,
                depth_trunc=self.max_depth,
                convert_rgb_to_intensity=False,
            )

            # Create intrinsic parameters for Open3D
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                self.color_intrin.width,
                self.color_intrin.height,
                self.color_intrin.fx,
                self.color_intrin.fy,
                self.color_intrin.ppx,
                self.color_intrin.ppy,
            )

            # Create point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

            return pcd

        except Exception as e:
            print(f"Error creating point cloud: {e}")
            traceback.print_exc()
            return None

    def save_data(self, position_id, color_image, depth_image, point_cloud):
        """
        Save RGB, depth, and point cloud data to files.

        Args:
            position_id: ID of the current position (e.g., 'pos01')
            color_image: RGB image as numpy array
            depth_image: Depth image as numpy array
            point_cloud: Open3D point cloud object

        Returns:
            Dict with metadata about saved files, or None if saving fails
        """
        try:
            # Check if inputs are valid
            if color_image is None or depth_image is None or point_cloud is None:
                print(f"Error: Cannot save data for position {position_id}, invalid inputs")
                return None

            # Save RGB image
            rgb_path = os.path.join(self.data_dir, "rgb", f"{position_id}.png")
            cv2.imwrite(rgb_path, color_image)

            # Save visualization of depth image (normalized for visualization)
            # Scale depth for better visualization
            normalized_depth = np.uint8(cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX))
            depth_vis_path = os.path.join(self.data_dir, "depth", f"{position_id}_vis.png")
            cv2.imwrite(depth_vis_path, normalized_depth)

            # Save raw depth data as 16-bit PNG
            depth_png_path = os.path.join(self.data_dir, "depth", f"{position_id}.png")
            cv2.imwrite(depth_png_path, depth_image)

            # Also save raw depth data as numpy array for maximum precision
            raw_depth_path = os.path.join(self.data_dir, "depth", f"{position_id}.npy")
            np.save(raw_depth_path, depth_image)

            # Save point cloud
            pcd_path = os.path.join(self.data_dir, "pointcloud", f"{position_id}.pcd")
            success = o3d.io.write_point_cloud(pcd_path, point_cloud)
            if not success:
                print(f"Warning: Failed to write point cloud for {position_id}")

            print(f"Saved data for position {position_id}")

            # Calculate angle based on position
            try:
                angle = (int(position_id[3:]) - 1) * (360 / self.total_positions)
            except BaseException:
                angle = 0
                print(f"Warning: Could not calculate angle for {position_id}")

            return {
                "position_id": position_id,
                "angle": angle,
                "timestamp": datetime.now().isoformat(),
                "rgb_path": rgb_path,
                "depth_png_path": depth_png_path,
                "depth_vis_path": depth_vis_path,
                "raw_depth_path": raw_depth_path,
                "pointcloud_path": pcd_path,
            }

        except Exception as e:
            print(f"Error saving data: {e}")
            traceback.print_exc()
            return None

    def save_metadata(self):
        """Save collection metadata to a JSON file."""
        try:
            metadata_path = os.path.join(self.data_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            print(f"Saved metadata to {metadata_path}")
            return True
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return False

    def get_user_input(self):
        """Get and validate user input for data collection parameters."""
        try:
            # Get object name
            while True:
                self.object_name = input("Enter object name: ").strip()
                if self.object_name:
                    break
                print("Error: Object name cannot be empty")

            # Get number of positions
            while True:
                positions_input = input(
                    "Enter number of capture positions around the object (e.g., 4 for every 90°): "
                ).strip()
                try:
                    self.total_positions = int(positions_input)
                    if self.total_positions > 0:
                        break
                    print("Error: Number of positions must be greater than 0")
                except ValueError:
                    print("Error: Please enter a valid number")

            # Set capture date
            self.capture_date = datetime.now().strftime("%Y%m%d")

            return True
        except Exception as e:
            print(f"Error getting user input: {e}")
            return False

    def capture_multiple_frames(self, num_frames=10, delay=0.1):
        """
        Capture multiple frames and average them for better quality.

        Args:
            num_frames: Number of frames to capture and average
            delay: Delay between frames in seconds

        Returns:
            Tuple of (averaged_color_image, averaged_depth_image) or (None, None) if capture failed
        """
        print(f"Capturing {num_frames} frames for averaging...")

        color_frames = []
        depth_frames = []

        for i in range(num_frames):
            print(f"Capturing frame {i+1}/{num_frames}...", end="\r")
            color, depth = self.capture_frame()

            if color is None or depth is None:
                print(f"\nWarning: Failed to capture frame {i+1}")
                continue

            color_frames.append(color)
            depth_frames.append(depth)
            time.sleep(delay)

        print()  # New line after progress

        # Check if we have enough frames
        if len(color_frames) < 1 or len(depth_frames) < 1:
            print("Error: Failed to capture enough valid frames for averaging")
            return None, None

        # Average frames
        # For color, convert to float, average, then back to uint8
        color_frames_float = [frame.astype(np.float32) for frame in color_frames]
        avg_color = np.mean(color_frames_float, axis=0).astype(np.uint8)

        # For depth, we'll use median instead of mean to reduce noise
        # This works better for depth data than simple averaging
        avg_depth = np.median(np.array(depth_frames), axis=0).astype(np.uint16)

        return avg_color, avg_depth

    def collect_data(self):
        """Main method to collect data from multiple viewpoints."""
        # Get user input for collection parameters
        if not self.get_user_input():
            return False

        # Create directory structure
        if not self.create_directories():
            return False

        # Initialize metadata with camera parameters (we'll update depth_scale
        # after camera starts)
        self.metadata = {
            "object_name": self.object_name,
            "capture_date": self.capture_date,
            "total_positions": self.total_positions,
            "angular_step": 360 / self.total_positions,
            "camera_parameters": {
                "resolution": [640, 480],
                "depth_scale": self.depth_scale,  # This will be updated after camera starts
                "min_depth": self.min_depth,
                "max_depth": self.max_depth,
            },
            "positions": [],
        }

        # Use context manager to ensure camera is properly stopped
        with self.camera_session():
            # Update metadata with actual depth scale
            self.metadata["camera_parameters"]["depth_scale"] = self.depth_scale

            # Loop through each position
            for pos in range(1, self.total_positions + 1):
                position_id = f"pos{pos:02d}"
                angle = (pos - 1) * (360 / self.total_positions)

                # Prompt user to confirm capture
                input(
                    f"\nPosition {pos}/{self.total_positions} ({angle}°): Place camera and press Enter to continue..."
                )

                # Show preview
                print("Previewing current view (press 'c' to capture or 'q' to quit)...")

                preview_active = True
                while preview_active:
                    color_image, depth_image = self.capture_frame()

                    if color_image is None or depth_image is None:
                        print("Failed to get preview frames, retrying...")
                        time.sleep(0.5)
                        continue

                    # Create a colorized depth map for visualization
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                    )

                    # Resize images if they're too large for screen
                    scale = min(1.0, 1280 / (color_image.shape[1] + depth_colormap.shape[1]))
                    if scale < 1.0:
                        width = int(color_image.shape[1] * scale)
                        height = int(color_image.shape[0] * scale)
                        color_image_display = cv2.resize(color_image, (width, height))
                        depth_colormap_display = cv2.resize(depth_colormap, (width, height))
                    else:
                        color_image_display = color_image.copy()
                        depth_colormap_display = depth_colormap.copy()

                    # Show preview
                    preview = np.hstack((color_image_display, depth_colormap_display))
                    cv2.namedWindow("Preview (c=capture, q=quit)", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("Preview (c=capture, q=quit)", preview)

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord("c"):
                        preview_active = False
                    elif key & 0xFF == ord("q"):
                        print("Data collection canceled by user")
                        cv2.destroyAllWindows()
                        return False

                cv2.destroyAllWindows()

                # Capture data
                print(f"Capturing position {position_id}...")

                # Get multiple frames and average them for better quality
                color_image, depth_image = self.capture_multiple_frames(num_frames=10, delay=0.1)

                if color_image is None or depth_image is None:
                    print(f"Failed to capture data for position {position_id}, skipping...")
                    continue

                # Create point cloud
                print("Generating point cloud...")
                point_cloud = self.create_point_cloud(color_image, depth_image)

                if point_cloud is None:
                    print(f"Failed to create point cloud for position {position_id}, skipping...")
                    continue

                # Save data
                position_data = self.save_data(position_id, color_image, depth_image, point_cloud)

                if position_data:
                    # Add position to metadata
                    self.metadata["positions"].append(position_data)
                    print(f"Successfully captured position {pos}/{self.total_positions}")
                else:
                    print(f"Failed to save data for position {position_id}")

        # Save metadata
        self.save_metadata()

        print(f"\nData collection complete! Data saved to {self.data_dir}")
        return True


def main():
    """Main function to run the data collection process."""
    print("=" * 50)
    print("RGBD Data Collection Tool")
    print("=" * 50)
    print("This script will help you collect RGBD and point cloud data")
    print("from a RealSense D405 camera at different viewpoints around an object.")
    print("=" * 50)

    # Check if the required libraries are available
    missing_libraries = []
    try:
        pass
    except ImportError:
        missing_libraries.append("pyrealsense2")
    try:
        pass
    except ImportError:
        missing_libraries.append("numpy")
    try:
        pass
    except ImportError:
        missing_libraries.append("opencv-python")
    try:
        pass
    except ImportError:
        missing_libraries.append("open3d")

    if missing_libraries:
        print("Error: The following required libraries are missing:")
        for lib in missing_libraries:
            print(f"  - {lib}")
        print("\nPlease install them using pip:")
        print(f"  pip install {' '.join(missing_libraries)}")
        return

    try:
        collector = RGBDDataCollector()
        collector.collect_data()
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    except Exception as e:
        print(f"Error during data collection: {e}")
        traceback.print_exc()

    print("Program finished.")


if __name__ == "__main__":
    main()
