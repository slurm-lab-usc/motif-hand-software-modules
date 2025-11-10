#!/usr/bin/env python3
"""
AHRS data fusion usage example
Demonstrates how to use AHRSDataProcessor to get the quaternion of the sensor fusion
Support multiple AHRS filters: Madgwick, Mahony, EKF
"""

import time

import numpy as np
from MOTIFHand.visualizer.data_handler.ahrs_data_processor import AHRSDataProcessor, Quaternion
from MOTIFHand.visualizer.data_handler.sensor_serial_handler import RS485MultiSensorReader


def example_basic_usage():
    """Basic usage example - Madgwick filter"""
    print("=== AHRS data fusion basic usage example (Madgwick) ===")

    # 1. Create sensor reader
    sensor_reader = RS485MultiSensorReader()

    # 2. Create AHRS processor - using Madgwick filter
    ahrs_processor = AHRSDataProcessor(
        sensor_reader, filter_type="madgwick", frequency=100.0, beta=0.1
    )

    try:
        # 3. Start system
        sensor_reader.start()
        ahrs_processor.start()

        print("System started, waiting for data...")
        time.sleep(2)  # Wait for system to stabilize

        # 4. Get quaternion data
        for i in range(5):  # Get data 5 times
            print(f"\n--- Data acquisition {i+1} ---")

            # Get quaternion data from a specific sensor module
            quat = ahrs_processor.get_quaternion(finger_id=0, board_id=0)
            if quat:
                print(
                    f"Sensor 0-0 quaternion: w={quat.w:.4f}, x={quat.x:.4f}, y={quat.y:.4f}, z={quat.z:.4f}"
                )

                # Convert to Euler angles
                roll, pitch, yaw = quat.to_euler()
                print(
                    f"Euler angles: Roll={np.degrees(roll):.2f}°, Pitch={np.degrees(pitch):.2f}°, Yaw={np.degrees(yaw):.2f}°"
                )
            else:
                print("Sensor 0-0 has no data")

            time.sleep(1)

    finally:
        # 5. Stop system
        ahrs_processor.stop()
        sensor_reader.stop()
        print("System stopped")


def example_mahony_filter():
    """Mahony filter example"""
    print("\n=== Mahony filter example ===")

    sensor_reader = RS485MultiSensorReader()

    # Use Mahony filter
    ahrs_processor = AHRSDataProcessor(
        sensor_reader,
        filter_type="mahony",
        frequency=100.0,
        Kp=1.0,  # Proportional gain
        Ki=0.3,  # Integral gain
    )

    try:
        sensor_reader.start()
        ahrs_processor.start()

        print("Mahony filter system started, waiting for data...")
        time.sleep(2)

        # Get filter information
        filter_info = ahrs_processor.get_filter_info()
        print(f"Filter type: {filter_info['filter_type']}")
        print(f"Filter parameters: {filter_info['filter_params']}")

        # Get data
        for i in range(3):
            print(f"\n--- Mahony data {i+1} ---")
            quat = ahrs_processor.get_quaternion(finger_id=0, board_id=0)
            if quat:
                roll, pitch, yaw = quat.to_euler()
                print(f"Quaternion: w={quat.w:.4f}, x={quat.x:.4f}, y={quat.y:.4f}, z={quat.z:.4f}")
                print(
                    f"Euler angles: Roll={np.degrees(roll):.2f}°, Pitch={np.degrees(pitch):.2f}°, Yaw={np.degrees(yaw):.2f}°"
                )
            time.sleep(1)

    finally:
        ahrs_processor.stop()
        sensor_reader.stop()


def example_ekf_filter():
    """EKF filter example"""
    print("\n=== EKF filter example ===")

    sensor_reader = RS485MultiSensorReader()

    # Use EKF filter
    ahrs_processor = AHRSDataProcessor(sensor_reader, filter_type="ekf", frequency=100.0)

    try:
        sensor_reader.start()
        ahrs_processor.start()

        print("EKF filter system started, waiting for data...")
        time.sleep(2)

        # Get filter information
        filter_info = ahrs_processor.get_filter_info()
        print(f"Filter type: {filter_info['filter_type']}")
        print(f"Filter parameters: {filter_info['filter_params']}")

        # Get data
        for i in range(3):
            print(f"\n--- EKF data {i+1} ---")
            quat = ahrs_processor.get_quaternion(finger_id=0, board_id=0)
            if quat:
                roll, pitch, yaw = quat.to_euler()
                print(f"Quaternion: w={quat.w:.4f}, x={quat.x:.4f}, y={quat.y:.4f}, z={quat.z:.4f}")
                print(
                    f"Euler angles: Roll={np.degrees(roll):.2f}°, Pitch={np.degrees(pitch):.2f}°, Yaw={np.degrees(yaw):.2f}°"
                )
            time.sleep(1)

    finally:
        ahrs_processor.stop()
        sensor_reader.stop()


def example_get_all_data():
    """Example of getting data from all sensors"""
    print("\n=== Get data from all sensors example ===")

    sensor_reader = RS485MultiSensorReader()
    ahrs_processor = AHRSDataProcessor(sensor_reader, filter_type="madgwick")

    try:
        sensor_reader.start()
        ahrs_processor.start()

        print("System started, waiting for data...")
        time.sleep(2)

        # Get quaternion data from all sensors
        all_quaternions = ahrs_processor.get_all_quaternions()
        all_euler_angles = ahrs_processor.get_all_euler_angles()

        print(f"\nGot data from {len(all_quaternions)} sensors:")

        for module_key, quat in all_quaternions.items():
            euler = all_euler_angles[module_key]
            print(f"\nModule {module_key}:")
            print(f"   Quaternion: w={quat.w:.4f}, x={quat.x:.4f}, y={quat.y:.4f}, z={quat.z:.4f}")
            print(
                f"   Euler angles: Roll={np.degrees(euler[0]):.2f}°, Pitch={np.degrees(euler[1]):.2f}°, Yaw={np.degrees(euler[2]):.2f}°"
            )

    finally:
        ahrs_processor.stop()
        sensor_reader.stop()


def example_continuous_monitoring():
    """Continuous monitoring example"""
    print("\n=== Continuous monitoring example ===")

    sensor_reader = RS485MultiSensorReader()
    ahrs_processor = AHRSDataProcessor(sensor_reader, filter_type="madgwick")

    try:
        sensor_reader.start()
        ahrs_processor.start()

        print("Starting continuous monitoring, press Ctrl+C to stop...")

        # Continuous monitoring for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            # Get all quaternions
            quaternions = ahrs_processor.get_all_quaternions()

            # Calculate average attitude (simple example)
            if quaternions:
                avg_w = sum(q.w for q in quaternions.values()) / len(quaternions)
                avg_x = sum(q.x for q in quaternions.values()) / len(quaternions)
                avg_y = sum(q.y for q in quaternions.values()) / len(quaternions)
                avg_z = sum(q.z for q in quaternions.values()) / len(quaternions)

                avg_quat = Quaternion(avg_w, avg_x, avg_y, avg_z)
                roll, pitch, yaw = avg_quat.to_euler()

                print(
                    f"Time: {time.strftime('%H:%M:%S')} - Average attitude: "
                    f"Roll={np.degrees(roll):.1f}°, "
                    f"Pitch={np.degrees(pitch):.1f}°, "
                    f"Yaw={np.degrees(yaw):.1f}°"
                )

            time.sleep(0.5)  # Update every 0.5 seconds

    except KeyboardInterrupt:
        print("\nMonitoring stopped")
    finally:
        ahrs_processor.stop()
        sensor_reader.stop()


def example_quaternion_operations():
    """Quaternion operations example"""
    print("\n=== Quaternion operations example ===")

    # Create two quaternions
    q1 = Quaternion(1.0, 0.0, 0.0, 0.0)  # Unit quaternion
    q2 = Quaternion(0.707, 0.0, 0.707, 0.0)  # Rotate 90 degrees around Y axis

    print(f"Quaternion 1: w={q1.w:.4f}, x={q1.x:.4f}, y={q1.y:.4f}, z={q1.z:.4f}")
    print(f"Quaternion 2: w={q2.w:.4f}, x={q2.x:.4f}, y={q2.y:.4f}, z={q2.z:.4f}")

    # Convert to Euler angles
    roll1, pitch1, yaw1 = q1.to_euler()
    roll2, pitch2, yaw2 = q2.to_euler()

    print(
        f"Quaternion 1 Euler angles: Roll={np.degrees(roll1):.2f}°, Pitch={np.degrees(pitch1):.2f}°, Yaw={np.degrees(yaw1):.2f}°"
    )
    print(
        f"Quaternion 2 Euler angles: Roll={np.degrees(roll2):.2f}°, Pitch={np.degrees(pitch2):.2f}°, Yaw={np.degrees(yaw2):.2f}°"
    )


def example_filter_comparison():
    """Filter comparison example"""
    print("\n=== Filter comparison example ===")

    sensor_reader = RS485MultiSensorReader()

    # Create different filter processors
    madgwick_processor = AHRSDataProcessor(sensor_reader, filter_type="madgwick", beta=0.1)
    mahony_processor = AHRSDataProcessor(sensor_reader, filter_type="mahony", Kp=1.0, Ki=0.3)

    try:
        sensor_reader.start()
        madgwick_processor.start()
        mahony_processor.start()

        print("Filter comparison system started...")
        time.sleep(2)

        # Compare results from different filters
        for i in range(3):
            print(f"\n--- Comparison data {i+1} ---")

            # Madgwick result
            madgwick_quat = madgwick_processor.get_quaternion(finger_id=0, board_id=0)
            if madgwick_quat:
                madgwick_euler = madgwick_quat.to_euler()
                print(
                    f"Madgwick: Roll={np.degrees(madgwick_euler[0]):.2f}°, "
                    f"Pitch={np.degrees(madgwick_euler[1]):.2f}°, "
                    f"Yaw={np.degrees(madgwick_euler[2]):.2f}°"
                )

            # Mahony result
            mahony_quat = mahony_processor.get_quaternion(finger_id=0, board_id=0)
            if mahony_quat:
                mahony_euler = mahony_quat.to_euler()
                print(
                    f"Mahony:  Roll={np.degrees(mahony_euler[0]):.2f}°, "
                    f"Pitch={np.degrees(mahony_euler[1]):.2f}°, "
                    f"Yaw={np.degrees(mahony_euler[2]):.2f}°"
                )

            time.sleep(1)

    finally:
        madgwick_processor.stop()
        mahony_processor.stop()
        sensor_reader.stop()


if __name__ == "__main__":
    # Run all examples
    try:
        example_basic_usage()
        example_mahony_filter()
        example_ekf_filter()
        example_get_all_data()
        example_continuous_monitoring()
        example_quaternion_operations()
        example_filter_comparison()

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Program error: {e}")

    print("\nAll examples completed")
