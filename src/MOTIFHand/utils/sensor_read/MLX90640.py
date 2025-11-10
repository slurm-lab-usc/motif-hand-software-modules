import argparse
import atexit
import logging
import sys
import time
from contextlib import contextmanager

import adafruit_mlx90640
import board
import busio
import cv2
import numpy as np
from scipy.ndimage import zoom

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MLX90640")

# Configuration parameters
INTERPOLATION_FACTOR = 8
ORIGINAL_HEIGHT = 24
ORIGINAL_WIDTH = 32
INTERPOLATED_HEIGHT = ORIGINAL_HEIGHT * INTERPOLATION_FACTOR  # 192
INTERPOLATED_WIDTH = ORIGINAL_WIDTH * INTERPOLATION_FACTOR  # 256
REFRESH_RATE = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
MAX_RETRIES = 5
RETRY_DELAY = 0.1
WINDOW_NAME = "MLX90640 Thermal Camera"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


class MLX90640Camera:
    def __init__(self, display=True, interpolation_factor=INTERPOLATION_FACTOR):
        """Initialize the MLX90640 thermal camera.

        Args:
            display (bool): Whether to display the thermal image in a window
            interpolation_factor (int): Factor for interpolating the low-res thermal image
        """
        self.i2c = None
        self.mlx = None
        self.frame = np.zeros((ORIGINAL_HEIGHT * ORIGINAL_WIDTH,))
        self.t_array = []
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.running = False
        self.colorbar = None
        self.last_min_temp = None
        self.last_max_temp = None
        self.display = display
        self.interpolation_factor = interpolation_factor

    def connect(self):
        """Connect to the MLX90640 sensor via I2C.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
            self.mlx.refresh_rate = REFRESH_RATE

            logger.info("MLX90640 sensor connected successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MLX90640 sensor: {e}")
            return False

    def initialize(self):
        """Initialize the camera system, including display if enabled.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.connect():
            return False

        if self.display:
            # Create a window for the thermal image
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Register cleanup function
        atexit.register(self.cleanup)

        logger.info("MLX90640 camera initialized successfully")
        return True

    def interpolate_thermal_data(self, data_array):
        """Performs bicubic interpolation on the thermal data array.

        Args:
            data_array (numpy.ndarray): The original thermal data array

        Returns:
            numpy.ndarray: The interpolated thermal data array
        """
        # Use scipy's zoom for bicubic interpolation (order=3)
        interpolated = zoom(data_array, self.interpolation_factor, order=3)
        return interpolated

    def create_colorbar(self, height, min_temp, max_temp, width=30):
        """Creates a colorbar image to display alongside the thermal image.

        Args:
            height (int): Height of the colorbar
            min_temp (float): Minimum temperature for the scale
            max_temp (float): Maximum temperature for the scale
            width (int): Width of the colorbar

        Returns:
            numpy.ndarray: The colorbar image
        """
        # Check if we can reuse existing colorbar (optimization)
        if (
            self.colorbar is not None
            and self.last_min_temp == min_temp
            and self.last_max_temp == max_temp
            and self.colorbar.shape[0] == height
        ):
            return self.colorbar

        # Create new colorbar
        colorbar = np.zeros((height, width, 3), dtype=np.uint8)

        # Create gradient
        for i in range(height):
            # Normalize position from bottom (hot) to top (cold)
            normalized_position = 1 - (i / height)
            # Convert to 0-255 for color mapping
            value = int(255 * normalized_position)
            # Apply same colormap as thermal image
            color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            colorbar[i, :] = color

        # Add temperature labels
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        # Max temperature label
        cv2.putText(colorbar, f"{max_temp:.1f}°C", (2, 15), font, font_scale, text_color, 1)
        # Min temperature label
        cv2.putText(
            colorbar,
            f"{min_temp:.1f}°C",
            (2, height - 5),
            font,
            font_scale,
            text_color,
            1,
        )
        # Middle temperature label
        mid_temp = (min_temp + max_temp) / 2
        cv2.putText(
            colorbar,
            f"{mid_temp:.1f}°C",
            (2, height // 2),
            font,
            font_scale,
            text_color,
            1,
        )

        # Save for later reuse
        self.colorbar = colorbar
        self.last_min_temp = min_temp
        self.last_max_temp = max_temp

        return colorbar

    def get_frame(self):
        """Get a frame from the thermal camera with error handling.

        Returns:
            bool: True if frame was successfully captured, False otherwise
        """
        if not self.mlx:
            logger.error("Sensor not initialized")
            return False

        retry_count = 0

        while retry_count < MAX_RETRIES:
            try:
                # Get frame data from sensor
                self.mlx.getFrame(self.frame)
                return True
            except ValueError:
                retry_count += 1
                logger.debug(f"Value error, retrying {retry_count}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY)
            except RuntimeError as e:
                retry_count += 1
                logger.warning(f"Runtime error, retrying {retry_count}/{MAX_RETRIES}: {e}")
                time.sleep(RETRY_DELAY)

        # If we reached this point, all retries failed
        logger.error(f"Failed to get frame after {MAX_RETRIES} retries")
        return False

    def process_frame(self):
        """Process the current frame and return the display image and temperature data.

        Returns:
            tuple: (display_img, min_temp, max_temp, avg_temp, data_array)
                - display_img: The processed image ready for display
                - min_temp: Minimum temperature in the frame
                - max_temp: Maximum temperature in the frame
                - avg_temp: Average temperature in the frame
                - data_array: The raw temperature data array
        """
        # Reshape to original dimensions
        data_array = np.reshape(self.frame, (ORIGINAL_HEIGHT, ORIGINAL_WIDTH))

        # Flip left-right as in the original code
        data_array = np.fliplr(data_array)

        # Get temperature range
        min_temp = np.min(data_array)
        max_temp = np.max(data_array)
        avg_temp = np.mean(data_array)

        # Apply interpolation
        interpolated_data = self.interpolate_thermal_data(data_array)

        # Normalize the data to 0-255 range for visualization
        normalized_data = cv2.normalize(interpolated_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Apply colormap (jet is similar to the default matplotlib colormap)
        thermal_img = cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET)

        display_img = None
        if self.display:
            # Create colorbar - reusing if possible
            colorbar = self.create_colorbar(thermal_img.shape[0], min_temp, max_temp)

            # Combine thermal image and colorbar
            display_img = np.hstack((thermal_img, colorbar))

            # Update FPS calculation
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()

            # Add information overlay
            info_text = f"Min: {min_temp:.1f}°C  Avg: {avg_temp:.1f}°C  Max: {max_temp:.1f}°C  FPS: {self.fps:.1f}"
            cv2.putText(
                display_img,
                info_text,
                (10, thermal_img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Add title
            title = f"MLX90640 Thermal Camera - {self.interpolation_factor}x Interpolated"
            title += f" ({thermal_img.shape[1]}x{thermal_img.shape[0]})"
            cv2.putText(
                display_img,
                title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return display_img, min_temp, max_temp, avg_temp, data_array

    def read_temperatures(self):
        """Read a single frame of temperature data from the sensor.

        Returns:
            tuple: (data_array, min_temp, max_temp, avg_temp) or None if read failed
                - data_array: 2D numpy array of temperature values
                - min_temp: Minimum temperature in the frame
                - max_temp: Maximum temperature in the frame
                - avg_temp: Average temperature in the frame
        """
        if not self.get_frame():
            return None

        # Reshape to original dimensions
        data_array = np.reshape(self.frame, (ORIGINAL_HEIGHT, ORIGINAL_WIDTH))

        # Flip left-right
        data_array = np.fliplr(data_array)

        # Get temperature range
        min_temp = np.min(data_array)
        max_temp = np.max(data_array)
        avg_temp = np.mean(data_array)

        return data_array, min_temp, max_temp, avg_temp

    def get_thermal_image(self, with_colorbar=False):
        """Capture and process a thermal image, returning it as a NumPy array.

        This method captures a single frame from the thermal camera, processes it,
        and returns the processed image as a NumPy array (BGR format for OpenCV).

        Args:
            with_colorbar (bool): Whether to include the temperature colorbar in the image

        Returns:
            tuple: (image_array, temp_data) or (None, None) if capture failed
                - image_array: NumPy array (BGR format) of the processed thermal image
                - temp_data: Dictionary with temperature data:
                    - min_temp: Minimum temperature in the frame
                    - max_temp: Maximum temperature in the frame
                    - avg_temp: Average temperature in the frame
                    - data_array: The raw temperature array (24x32)
        """
        if not self.get_frame():
            return None, None

        # Process the frame
        data_array = np.reshape(self.frame, (ORIGINAL_HEIGHT, ORIGINAL_WIDTH))
        data_array = np.fliplr(data_array)

        # Get temperature range
        min_temp = np.min(data_array)
        max_temp = np.max(data_array)
        avg_temp = np.mean(data_array)

        # Apply interpolation
        interpolated_data = self.interpolate_thermal_data(data_array)

        # Normalize the data to 0-255 range for visualization
        normalized_data = cv2.normalize(interpolated_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Apply colormap (jet is similar to the default matplotlib colormap)
        thermal_img = cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET)

        # Prepare the output image
        if with_colorbar:
            # Create colorbar
            colorbar = self.create_colorbar(thermal_img.shape[0], min_temp, max_temp)

            # Combine thermal image and colorbar
            output_img = np.hstack((thermal_img, colorbar))
        else:
            output_img = thermal_img

        # Package temperature data
        temp_data = {
            "min_temp": min_temp,
            "max_temp": max_temp,
            "avg_temp": avg_temp,
            "data_array": data_array,
        }

        return output_img, temp_data

    def run(self, callback=None, interval=0.05):
        """Main camera loop with optional callback for temperature data.

        Args:
            callback (callable): Optional callback function that will receive
                                (data_array, min_temp, max_temp, avg_temp) for each frame
            interval (float): Time interval between frame captures in seconds
        """
        if not self.initialize():
            logger.error("Failed to initialize. Exiting.")
            return

        self.running = True

        logger.info(
            f"Starting thermal camera. {'Press q to quit.' if self.display else 'Use Ctrl+C to stop.'}"
        )

        try:
            while self.running:
                t1 = time.perf_counter()

                # Get frame from sensor
                if not self.get_frame():
                    # If failed to get frame, wait and continue
                    time.sleep(1)  # Longer delay after failure
                    continue

                # Process the frame and get display image and temperature data
                display_img, min_temp, max_temp, avg_temp, data_array = self.process_frame()

                # If there's a callback, send the temperature data
                if callback:
                    callback(data_array, min_temp, max_temp, avg_temp)

                # Display the image if display is enabled
                if self.display and display_img is not None:
                    cv2.imshow(WINDOW_NAME, display_img)

                    # Break the loop if 'q' is pressed
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("User requested exit")
                        break

                # Calculate frame processing time
                frame_time = time.perf_counter() - t1

                # Adaptive sleep based on frame processing time and desired
                # interval
                sleep_time = max(0.01, interval - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the camera and clean up resources."""
        self.running = False
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if hasattr(cv2, "destroyAllWindows"):
            cv2.destroyAllWindows()
        logger.info("Camera resources released")


@contextmanager
def camera_context(display=True, interpolation_factor=INTERPOLATION_FACTOR):
    """Context manager for the MLX90640 thermal camera to ensure proper cleanup.

    Args:
        display (bool): Whether to display the thermal image in a window
        interpolation_factor (int): Factor for interpolating the low-res thermal image

    Yields:
        MLX90640Camera: The camera object if initialization was successful, None otherwise
    """
    camera = MLX90640Camera(display=display, interpolation_factor=interpolation_factor)
    try:
        if camera.initialize():
            yield camera
        else:
            logger.error("Failed to initialize camera")
            yield None
    finally:
        camera.stop()


def temperature_callback(data_array, min_temp, max_temp, avg_temp):
    """Example callback function to handle temperature data.

    Args:
        data_array (numpy.ndarray): 2D array of temperature values
        min_temp (float): Minimum temperature in the frame
        max_temp (float): Maximum temperature in the frame
        avg_temp (float): Average temperature in the frame
    """
    # You could integrate with other systems here
    # For example, save to a database, trigger an alarm based on temperature,
    # etc.


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description="MLX90640 Thermal Camera Interface")

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without displaying the thermal image",
    )
    parser.add_argument(
        "--interpolation",
        type=int,
        default=INTERPOLATION_FACTOR,
        help=f"Interpolation factor (default: {INTERPOLATION_FACTOR})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Reading interval in seconds (default: 0.05)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def main():
    """Main function to run the thermal camera."""
    args = parse_arguments()

    # Set logging level based on arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Use context manager to ensure proper cleanup
    with camera_context(
        display=not args.no_display, interpolation_factor=args.interpolation
    ) as camera:
        if camera:
            try:
                # Start camera with example callback
                camera.run(callback=temperature_callback, interval=args.interval)
            except KeyboardInterrupt:
                logger.info("Program stopped by user")


if __name__ == "__main__":
    main()
