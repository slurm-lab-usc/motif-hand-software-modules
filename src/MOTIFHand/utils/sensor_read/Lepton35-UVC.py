import logging
import sys
import threading
import time
from contextlib import contextmanager
from queue import Queue

import cv2
import numpy as np
from MOTIFHand.utils.sensor_read.uvctypes import *

# Configuration parameters
SCALE_FACTOR = 8
COLORMAP = cv2.COLORMAP_JET
WINDOW_NAME = "FLIR Lepton Thermal Stream"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MAX_RETRIES = 5
RETRY_DELAY = 0.1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Lepton35-UVC")


class ThermalStreamer:
    """
    High-performance thermal imaging streamer optimized for UVC interface
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

        # UVC specific
        self.BUF_SIZE = 2
        self.q = Queue(self.BUF_SIZE)
        self.PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(
            self.py_frame_callback
        )
        self.ctx = POINTER(uvc_context)()
        self.dev = POINTER(uvc_device)()
        self.devh = POINTER(uvc_device_handle)()
        self.ctrl = uvc_stream_ctrl()

        logger.info(
            "ThermalStreamer initialized with scale_factor=%d, display=%s",
            scale_factor,
            display,
        )

    def py_frame_callback(self, frame, userptr):
        """Callback function for UVC frame capture"""
        array_pointer = cast(
            frame.contents.data,
            POINTER(c_uint16 * (frame.contents.width * frame.contents.height)),
        )
        data = np.frombuffer(array_pointer.contents, dtype=np.dtype(np.uint16)).reshape(
            frame.contents.height, frame.contents.width
        )
        if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
            return

        if not self.q.full():
            self.q.put(data)

    def start(self):
        """Start thermal streaming"""
        self.running = True
        self.last_fps_time = time.time()

        # Initialize UVC camera
        self.init_thermal_data_frames()

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
        self.cleanup()

    def close(self):
        """Close the thermal camera and its resources"""
        self.stop()
        self.capture_thread = None
        self.process_thread = None
        logger.info("Thermal camera resources closed")

    def _capture_loop(self):
        """Thread dedicated to capturing frames from UVC device"""
        res = libuvc.uvc_start_streaming(
            self.devh, byref(self.ctrl), self.PTR_PY_FRAME_CALLBACK, None, 0
        )
        if res < 0:
            logger.error("uvc_start_streaming failed: %d", res)
            return

        try:
            while self.running:
                try:
                    data = self.q.get(True, 0.5)
                    if data is not None:
                        with self.frame_lock:
                            self.raw_data = data.copy()
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
                except BaseException:
                    # Timeout is normal, just continue
                    pass
        except Exception as e:
            logger.error("Capture error: %s", e)
        finally:
            logger.info("Stopping stream...")
            libuvc.uvc_stop_streaming(self.devh)

    def _process_loop(self):
        """Thread dedicated to processing frames"""
        while self.running:
            if self.data_ready.wait(timeout=0.03):  # ~30fps timeout
                self.data_ready.clear()

                try:
                    with self.frame_lock:
                        if self.raw_data is None:
                            continue
                        raw_data = self.raw_data.copy()
                    # print(raw_data)
                    # Process the frame - match the original implementation
                    cv2.normalize(raw_data, raw_data, 0, 65535, cv2.NORM_MINMAX)
                    np.right_shift(raw_data, 8, raw_data)
                    # raw_data = raw_data/100.00 - 273.15
                    # print(raw_data)
                    colored_img = cv2.applyColorMap(np.uint8(raw_data), self.colormap)

                    # Fast resize
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

                    # Update the frame
                    with self.frame_lock:
                        self.frame = enlarged_img
                        self.min_temp = np.min(raw_data)
                        self.max_temp = np.max(raw_data)
                        self.temperature_data = raw_data

                    self.frame_ready.set()

                except Exception as e:
                    logger.error("Processing error: %s", e)

    def init_thermal_data_frames(self):
        """Initialize the thermal camera"""
        res = libuvc.uvc_init(byref(self.ctx), 0)
        if res < 0:
            logger.error("uvc_init error")
            raise RuntimeError("Failed to initialize UVC context")

        try:
            res = libuvc.uvc_find_device(self.ctx, byref(self.dev), PT_USB_VID, PT_USB_PID, 0)
            if res < 0:
                logger.error("uvc_find_device error")
                raise RuntimeError("Failed to find device")

            try:
                res = libuvc.uvc_open(self.dev, byref(self.devh))
                if res < 0:
                    logger.error("uvc_open error")
                    raise RuntimeError("Failed to open device")

                logger.info("Device opened successfully!")

                # Print device information
                print_device_info(self.devh)
                print_device_formats(self.devh)

                # Setup FFC mode
                set_manual_ffc(self.devh)

                # Get Y16 format
                frame_formats = uvc_get_frame_formats_by_guid(self.devh, VS_FMT_GUID_YUYV)
                if len(frame_formats) == 0:
                    logger.error("Device does not support YUYV format")
                    raise RuntimeError("Y16 format not supported")

                # Find the best format (prefer 160x120)
                valid_format = None
                for fmt in frame_formats:
                    logger.info(
                        "Available format: %dx%d @ %dfps",
                        fmt.wWidth,
                        fmt.wHeight,
                        int(1e7 / fmt.dwDefaultFrameInterval),
                    )
                    if fmt.wWidth == 160 and fmt.wHeight == 120:
                        valid_format = fmt
                        break

                if not valid_format:
                    valid_format = frame_formats[0]

                logger.info(
                    "Using format: %dx%d @ %dfps",
                    valid_format.wWidth,
                    valid_format.wHeight,
                    int(1e7 / valid_format.dwDefaultFrameInterval),
                )

                # Setup stream control
                res = libuvc.uvc_get_stream_ctrl_format_size(
                    self.devh,
                    byref(self.ctrl),
                    UVC_FRAME_FORMAT_UYVY,
                    valid_format.wWidth,
                    valid_format.wHeight,
                    int(1e7 / valid_format.dwDefaultFrameInterval),
                )

                if res < 0:
                    logger.error("uvc_get_stream_ctrl_format_size failed")
                    raise RuntimeError("Failed to setup stream control")

                # Verify the control
                res = libuvc.uvc_probe_stream_ctrl(self.devh, byref(self.ctrl))
                if res < 0:
                    logger.error("uvc_probe_stream_ctrl failed")
                    raise RuntimeError("Failed to verify stream control")

            except Exception as e:
                logger.error("Error while setting up camera: %s", e)
                libuvc.uvc_close(self.devh)
                libuvc.uvc_unref_device(self.dev)
                raise

        except Exception as e:
            logger.error("Error while opening camera: %s", e)
            libuvc.uvc_exit(self.ctx)
            raise

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")

        if hasattr(self, "devh") and self.devh:
            libuvc.uvc_close(self.devh)

        if hasattr(self, "dev") and self.dev:
            libuvc.uvc_unref_device(self.dev)

        if hasattr(self, "ctx") and self.ctx:
            libuvc.uvc_exit(self.ctx)

        logger.info("Cleanup complete")

    def perform_ffc(self):
        """Perform Flat Field Correction"""
        perform_manual_ffc(self.devh)

    def print_shutter_info(self):
        """Print shutter information"""
        print_shutter_info(self.devh)

    def set_manual_ffc(self):
        """Set manual FFC mode"""
        set_manual_ffc(self.devh)

    def set_auto_ffc(self):
        """Set automatic FFC mode"""
        set_auto_ffc(self.devh)

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


def main():
    """Main function"""
    logger.info("Starting Lepton35-UVC thermal camera")

    # Use context manager
    with camera_context() as streamer:
        if streamer:
            try:
                # Set display options
                streamer.set_options(show_scale=True, show_fps=True)

                # Run with callback
                streamer.run()
            except KeyboardInterrupt:
                logger.info("Program interrupted by user")
            finally:
                logger.info("Program ended")


if __name__ == "__main__":
    main()
