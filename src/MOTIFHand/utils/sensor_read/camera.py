#!/usr/bin/env python3
"""
Raspberry Pi Camera Controller using picamera2
This module provides a singleton controller class for camera operations including photos and videos
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput


class CameraController:
    """
    Singleton Camera Controller class for Raspberry Pi camera operations
    Provides functionality for photo and video capture with configurable settings
    """

    _instance = None

    def __init__(self, output_dir="./output"):
        """
        Initialize the camera controller with specified output directory

        Args:
            output_dir (str): Directory to save photos and videos
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        self.picam2 = Picamera2()
        self.is_recording = False

        try:
            # Configure camera
            self.photo_config = self.picam2.create_still_configuration(
                main={"size": (3840, 2160)},  # 4K resolution for photos
                lores={"size": (640, 480)},
                display="lores",
                buffer_count=3,
                controls={
                    "AfMode": 1,  # 启用自动对焦
                    "AfRange": 1,  # 正常对焦范围
                    "AfSpeed": 1,
                },  # 正常对焦速度
            )

            self.video_config = self.picam2.create_video_configuration(
                main={"size": (1920, 1080)},  # 1080p for videos
                lores={"size": (640, 480)},
                display="lores",
            )

            # Default to photo mode
            self.picam2.configure(self.photo_config)
            self.picam2.start()

            # Initialize encoder for video
            self.encoder = H264Encoder(10000000)  # 10Mbps bitrate
            self.current_mode = "photo"
            self.current_video_path = None

            print("Camera controller initialized successfully")

        except Exception as e:
            print(f"Error initializing camera: {str(e)}")
            raise

    @classmethod
    def get_instance(cls, output_dir="./output"):
        """
        Get or create the singleton instance of the camera controller

        Args:
            output_dir (str): Directory to save photos and videos

        Returns:
            CameraController: The singleton instance
        """
        if cls._instance is None:
            cls._instance = CameraController(output_dir)
        return cls._instance

    def _switch_mode(self, mode):
        """
        Switch camera between photo and video modes

        Args:
            mode (str): "photo" or "video"
        """
        if self.current_mode == mode:
            return

        self.picam2.stop()

        if mode == "photo":
            self.picam2.configure(self.photo_config)
        else:  # video
            self.picam2.configure(self.video_config)

        self.picam2.start()
        self.current_mode = mode
        print(f"Switched to {mode} mode")

    def _apply_image_enhancements(self, image):
        """
        Apply OpenCV image enhancements

        Args:
            image (numpy.ndarray): The image to enhance

        Returns:
            numpy.ndarray: Enhanced image
        """
        # Apply auto white balance correction
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Apply sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # Apply slight saturation increase
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.2)
        hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return enhanced

    def capture_photo(self, filename=None, apply_enhancements=False):
        """
        Capture a photo and save it to the output directory

        Args:
            filename (str, optional): Custom filename for the photo. If None, a timestamp-based name is used
            apply_enhancements (bool): Whether to apply OpenCV enhancements to the photo

        Returns:
            str: Path to the saved photo
        """
        # Switch to photo mode if needed
        self._switch_mode("photo")

        if filename is not None:
            # 如果是 Path 对象，转换为字符串
            filename = str(filename)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"

        # Ensure filename has correct extension
        if not filename.lower().endswith((".jpg", ".jpeg")):
            filename += ".jpg"

        # Create full path
        filepath = os.path.join(self.output_dir, filename)

        # Wait before capturing
        time.sleep(1)

        # Capture the image
        if apply_enhancements:
            # Capture as array for processing
            image = self.picam2.capture_array()
            enhanced = self._apply_image_enhancements(image)
            cv2.imwrite(filepath, enhanced)
            print(f"Enhanced photo saved to {filepath}")
        else:
            # Direct capture to file
            self.picam2.capture_file(filepath)
            print(f"Photo saved to {filepath}")

        return filepath

    def start_recording(self, filename=None, with_preview=False):
        """
        Start recording a video

        Args:
            filename (str, optional): Custom filename for the video. If None, a timestamp-based name is used
            with_preview (bool): Whether to display a preview window during recording

        Returns:
            str: Path to the video file
        """
        # Check if already recording
        if self.is_recording:
            print("Already recording. Stop current recording first.")
            return self.current_video_path

        # Switch to video mode
        self._switch_mode("video")

        if filename is not None:
            # 如果是 Path 对象，转换为字符串
            filename = str(filename)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}.h264"

        # Ensure filename has correct extension
        if not filename.lower().endswith(".h264"):
            filename += ".h264"

        # Create full path
        filepath = os.path.join(self.output_dir, filename)
        self.current_video_path = filepath

        try:
            # Start recording
            output = FileOutput(filepath)
            self.encoder = H264Encoder(10000000)  # Reinitialize encoder

            if with_preview:
                self.picam2.start_preview(show_preview=True)
                # 给预览一点时间初始化
                time.sleep(0.5)

            # 启动编码器
            self.picam2.start_encoder(self.encoder, output)
            self.is_recording = True

            print(f"Started recording to {filepath}")
            return filepath

        except Exception as e:
            print(f"Error starting recording: {str(e)}")
            if hasattr(self, "picam2") and self.picam2:
                try:
                    self.picam2.stop_preview()
                except BaseException:
                    pass
            return None

    def stop_recording(self):
        """
        Stop the current video recording

        Returns:
            str: Path to the recorded video file or None if not recording
        """
        if not self.is_recording:
            print("Not currently recording")
            return None

        # Stop recording
        self.picam2.stop_encoder()
        self.is_recording = False

        # Close preview if open
        self.picam2.stop_preview()

        print(f"Recording stopped. Video saved to {self.current_video_path}")
        return self.current_video_path

    def __del__(self):
        """Clean up resources when the object is destroyed"""
        if hasattr(self, "picam2") and self.picam2:
            if self.is_recording:
                self.stop_recording()
            self.picam2.stop()
            self.picam2.close()
            print("Camera resources released")


# Main execution for testing when script is run directly
if __name__ == "__main__":
    # Test the camera controller
    camera = CameraController.get_instance()

    print("\n=== Testing Photo Capture ===")
    # Test auto-named photo without enhancements
    photo1 = camera.capture_photo()
    print(f"Auto-named photo: {photo1}")

    # Test named photo with enhancements
    photo2 = camera.capture_photo("enhanced_test.jpg", apply_enhancements=True)
    print(f"Named enhanced photo: {photo2}")

    print("\n=== Testing Video Recording ===")
    # Test auto-named video recording
    video1 = camera.start_recording()
    print("Recording for 5 seconds...")
    time.sleep(5)
    camera.stop_recording()

    # Test named video with preview
    print("\nStarting named video with preview...")
    video2 = camera.start_recording("test_preview.h264", with_preview=True)
    print("Recording for 5 seconds with preview...")
    time.sleep(5)
    camera.stop_recording()

    print("\n=== All tests completed ===")
