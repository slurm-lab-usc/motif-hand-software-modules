import argparse
import logging
import sys
import threading
import time
from contextlib import contextmanager

import cv2
import numpy as np
from pylepton.Lepton3 import Lepton3

# 配置参数
SCALE_FACTOR = 8
COLORMAP = cv2.COLORMAP_JET
WINDOW_NAME = "FLIR Lepton Thermal Stream"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MAX_RETRIES = 5
RETRY_DELAY = 0.1

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Lepton35")


class ThermalStreamer:
    """
    High-performance thermal imaging streamer optimized for 30 FPS
    """

    def __init__(self, scale_factor=SCALE_FACTOR, colormap=COLORMAP, display=True):
        self.scale_factor = scale_factor
        self.colormap = colormap
        self.display = display
        self.running = False
        self.frame = None
        self.raw_data = None
        self.min_temp = 0
        self.max_temp = 0
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = 0
        self.temperature_data = None

        # Pre-allocate arrays for better performance
        self.temp_scale_cache = {}

        # Performance optimization flags
        self.show_temperature_scale = True
        self.show_fps_counter = False

        # Threading
        self.capture_thread = None
        self.process_thread = None
        self.frame_ready = threading.Event()
        self.data_ready = threading.Event()
        self.frame_lock = threading.Lock()

        logger.info(
            "ThermalStreamer initialized with scale_factor=%d, display=%s",
            scale_factor,
            display,
        )

    def start(self):
        """Start thermal streaming"""
        self.running = True
        self.last_fps_time = time.time()

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()

    def stop(self):
        """Stop thermal streaming"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
        cv2.destroyAllWindows()

    def close(self):
        """Close the thermal camera and its resources"""
        self.stop()
        # Additional cleanup if needed
        self.capture_thread = None
        self.process_thread = None
        logger.info("Thermal camera resources closed")

    def _capture_loop(self):
        """Thread dedicated to capturing frames from Lepton sensor"""
        # This approach keeps the with-context active for the entire loop,
        # which is necessary for Lepton3 to work properly
        with Lepton3() as lepton:
            logger.info("Lepton3 sensor initialized successfully")
            while self.running:
                try:
                    # Capture raw data
                    raw_data, _ = lepton.capture()

                    # Signal that new data is available
                    with self.frame_lock:
                        self.raw_data = raw_data.copy()
                    self.data_ready.set()

                    # Performance measurement
                    current_time = time.time()
                    self.frame_count += 1
                    elapsed = current_time - self.last_fps_time
                    if elapsed >= 1.0:  # Update FPS once per second
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.last_fps_time = current_time
                        logger.debug("Capture thread FPS: %.1f", self.fps)

                    # Tight timing loop for maximum frame rate
                    # No sleep here to maximize frame rate
                except Exception as e:
                    logger.error("Capture error: %s", e)
                    time.sleep(0.1)

    def _process_loop(self):
        """Thread dedicated to processing frames"""
        while self.running:
            # Wait for new data with short timeout for responsive shutdown
            if self.data_ready.wait(timeout=0.03):  # ~30fps timeout
                self.data_ready.clear()

                try:
                    # Get raw data safely
                    with self.frame_lock:
                        if self.raw_data is None:
                            continue
                        raw_data = self.raw_data.copy()

                    # Process the frame - use faster processing methods
                    temperature_data = raw_data / 100.0 - 273.15
                    min_temp = np.min(temperature_data)
                    max_temp = np.max(temperature_data)

                    # Fast normalization with pre-allocated arrays when
                    # possible
                    normalized_temp = np.clip(
                        (temperature_data - min_temp) / (max_temp - min_temp + 1e-6),
                        0,
                        1,
                    )
                    normalized_temp_8bit = (normalized_temp * 255).astype(np.uint8)
                    colored_img = cv2.applyColorMap(normalized_temp_8bit, self.colormap)

                    # Fast resize using LINEAR interpolation for speed
                    enlarged_img = cv2.resize(
                        colored_img,
                        (
                            colored_img.shape[1] * self.scale_factor,
                            colored_img.shape[0] * self.scale_factor,
                        ),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    # Add FPS counter if enabled
                    if self.show_fps_counter:
                        cv2.putText(
                            enlarged_img,
                            f"FPS: {self.fps:.1f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    # Update the frame without temperature scale
                    with self.frame_lock:
                        self.frame = enlarged_img
                        self.min_temp = min_temp
                        self.max_temp = max_temp
                        self.temperature_data = temperature_data

                    # Signal that a new frame is ready
                    self.frame_ready.set()

                except Exception as e:
                    print(f"Processing error: {e}")
            else:
                # If no data is available, check if we should continue running
                if not self.running:
                    break

    def _add_temp_scale_fast(self, img, min_temp, max_temp):
        """
        Fast version of temperature scale overlay
        Uses caching to avoid regenerating the color bar every frame
        """
        # Check if we have a cached scale
        cache_key = (min_temp, max_temp)
        if cache_key in self.temp_scale_cache:
            scale_overlay = self.temp_scale_cache[cache_key]
            result = img.copy()
            h, w = img.shape[:2]
            result[h - scale_overlay.shape[0] : h, 0 : scale_overlay.shape[1]] = scale_overlay
            return result

        # Create scale if not cached (simplified for speed)
        h, w = img.shape[:2]
        scale_height = 40
        scale_img = np.zeros((scale_height, w, 3), dtype=np.uint8)

        # Create color gradient
        bar_width = w - 100
        bar_x = 50
        bar_y = 10
        bar_height = 20

        # Draw the bar
        for i in range(bar_width):
            color_value = int(255 * i / bar_width)
            color = cv2.applyColorMap(np.array([[color_value]], dtype=np.uint8), self.colormap)[
                0, 0
            ]
            cv2.line(
                scale_img,
                (bar_x + i, bar_y),
                (bar_x + i, bar_y + bar_height),
                (int(color[0]), int(color[1]), int(color[2])),
                1,
            )

        # Add min/max text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            scale_img,
            f"{min_temp:.1f}°C",
            (bar_x - 5, bar_y + bar_height + 15),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            scale_img,
            f"{max_temp:.1f}°C",
            (bar_x + bar_width - 30, bar_y + bar_height + 15),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Cache the scale
        self.temp_scale_cache[cache_key] = scale_img
        if len(self.temp_scale_cache) > 20:  # Limit cache size
            self.temp_scale_cache.pop(next(iter(self.temp_scale_cache)))

        # Combine with image
        result = img.copy()
        result[h - scale_height : h, 0:w] = scale_img
        return result

    def display(self):
        """
        Display the thermal stream in a window
        Returns True if continuing, False if should stop
        """
        if self.frame_ready.wait(timeout=0.016):  # ~60fps timeout to ensure smooth display
            self.frame_ready.clear()

            with self.frame_lock:
                if self.frame is not None:
                    cv2.imshow(WINDOW_NAME, self.frame)

                    # Check for quit with minimal wait time
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return False

            return True
        return True

    def get_current_frame(self):
        """Get the current processed frame"""
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def get_temperature_data(self):
        """Get the current temperature data"""
        with self.frame_lock:
            if self.temperature_data is not None:
                return self.temperature_data.copy(), self.min_temp, self.max_temp
        return None, 0, 0

    def set_options(self, scale_factor=None, colormap=None, show_scale=None, show_fps=None):
        """Update streaming options"""
        if scale_factor is not None:
            self.scale_factor = scale_factor
        if colormap is not None:
            self.colormap = colormap
        if show_scale is not None:
            self.show_temperature_scale = show_scale
        if show_fps is not None:
            self.show_fps_counter = show_fps

    def run(self, callback=None):
        """Run the thermal stream with optional callback

        Args:
            callback (callable): Callback function that receives (temperature_data, min_temp, max_temp)
        """
        self.start()
        try:
            while self.running:
                if self.frame_ready.wait(timeout=0.016):  # ~60fps
                    self.frame_ready.clear()

                    with self.frame_lock:
                        if self.frame is not None and callback:
                            callback(self.temperature_data, self.min_temp, self.max_temp)

                    if self.display and self.frame is not None:
                        cv2.imshow(WINDOW_NAME, self.frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        except KeyboardInterrupt:
            logger.info("Program interrupted by user")
        finally:
            self.stop()

    def get_thermal_image(self, with_colorbar=False):
        """Get the current thermal image

        Args:
            with_colorbar (bool): Whether to include temperature colorbar

        Returns:
            tuple: (image, temp_data) or (None, None) if failed
                - image: Processed thermal image
                - temp_data: Temperature data dictionary
                    - min_temp: Minimum temperature
                    - max_temp: Maximum temperature
                    - temperature_data: Raw temperature data
        """
        with self.frame_lock:
            if self.frame is None:
                return None, None

            image = self.frame.copy()
            temp_data = {
                "min_temp": self.min_temp,
                "max_temp": self.max_temp,
                "temperature_data": (
                    self.temperature_data.copy() if self.temperature_data is not None else None
                ),
            }

            if with_colorbar:
                image = self._add_temp_scale_fast(image, self.min_temp, self.max_temp)

            return image, temp_data

    def read_temperatures(self):
        """Read current temperature data

        Returns:
            tuple: (temperature_data, min_temp, max_temp) or (None, 0, 0) if failed
        """
        with self.frame_lock:
            if self.temperature_data is not None:
                return self.temperature_data.copy(), self.min_temp, self.max_temp
        return None, 0, 0


def save_thermal_image(streamer, output_filename="thermal_color.jpg"):
    """
    Capture and save a single thermal image from the streamer
    """
    # Get the current frame
    frame = streamer.get_current_frame()
    if frame is not None:
        # Save the frame
        cv2.imwrite(output_filename, frame)
        print(f"Saved thermal image to {output_filename}")
        return True
    return False


@contextmanager
def camera_context(display=True, scale_factor=SCALE_FACTOR, colormap=COLORMAP):
    """Context manager for proper resource cleanup

    Args:
        display (bool): Whether to display the image
        scale_factor (int): Image scale factor
        colormap: Color mapping
    """
    camera = ThermalStreamer(display=display, scale_factor=scale_factor, colormap=colormap)
    try:
        yield camera
    finally:
        camera.stop()


def parse_arguments():
    """Parse command line arguments

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Lepton35 Thermal Camera Interface")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without displaying the thermal image",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=SCALE_FACTOR,
        help=f"Image scale factor (default: {SCALE_FACTOR})",
    )
    parser.add_argument(
        "--colormap",
        type=int,
        default=COLORMAP,
        help=f"Color mapping (default: {COLORMAP})",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def temperature_callback(temperature_data, min_temp, max_temp):
    """Example callback function for temperature data processing

    Args:
        temperature_data (numpy.ndarray): Temperature data array
        min_temp (float): Minimum temperature
        max_temp (float): Maximum temperature
    """
    # Add custom temperature data processing logic here
    # For example: save to database, trigger alarms, etc.
    logger.debug("Temperature range: %.1f°C - %.1f°C", min_temp, max_temp)


def main():
    """Main function"""
    args = parse_arguments()

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting Lepton35 thermal camera")
    logger.info(
        "Configuration: display=%s, scale_factor=%d, colormap=%d",
        not args.no_display,
        args.scale_factor,
        args.colormap,
    )

    # Use context manager
    with camera_context(
        display=not args.no_display,
        scale_factor=args.scale_factor,
        colormap=args.colormap,
    ) as streamer:
        if streamer:
            try:
                # Set display options
                streamer.set_options(show_scale=True, show_fps=True)

                # Run with callback
                streamer.run(callback=temperature_callback)
            except KeyboardInterrupt:
                logger.info("Program interrupted by user")
            finally:
                logger.info("Program ended")


if __name__ == "__main__":
    main()
