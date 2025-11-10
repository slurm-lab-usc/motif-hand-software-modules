import argparse
import atexit
import binascii
import logging
import time
from contextlib import contextmanager

import serial

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TOF050FSensor")

# Default configuration (now accessible as constants)
DEFAULT_SERIAL_PORT = "/dev/ttyAMA0"
DEFAULT_BAUD_RATE = 115200
DEFAULT_TIMEOUT = 1
DEFAULT_READ_INTERVAL = 0.05
MAX_ERROR_COUNT = 10
SIGNIFICANT_CHANGE_THRESHOLD = 2
REPORT_INTERVAL = 0.5


class TOF400FSensor:
    def __init__(
        self,
        port=DEFAULT_SERIAL_PORT,
        baudrate=DEFAULT_BAUD_RATE,
        timeout=DEFAULT_TIMEOUT,
    ):
        """Initialize the TOF400F sensor with serial connection."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_port = None
        self.running = False

        # Lookup table for hexadecimal conversion (much faster than if-else
        # chain)
        self.hex_lookup = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "a": 10,
            "b": 11,
            "c": 12,
            "d": 13,
            "e": 14,
            "f": 15,
        }

        # Register cleanup function
        atexit.register(self.close)

    def connect(self):
        """Connect to the serial port."""
        try:
            self.serial_port = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=self.timeout
            )

            if self.serial_port.isOpen():
                logger.info(f"Serial port {self.port} is open")
                return True
            else:
                logger.error(f"Serial port {self.port} failed to open")
                return False
        except Exception as e:
            logger.error(f"Error connecting to serial port: {e}")
            return False

    def _hex_to_int(self, hex_char):
        """Convert a hex character to its integer value using lookup table."""
        return self.hex_lookup.get(hex_char.lower(), 0)

    def _parse_distance_data(self, hex_data):
        """Parse the hexadecimal data to extract distance value in mm."""
        try:
            if len(hex_data) <= 8:
                logger.debug(f"Hex data too short: {hex_data}")
                return None

            # Validate format (basic sanity check)
            # Find data pattern (can be improved based on exact protocol)
            if "b" not in hex_data:
                logger.debug("No valid data pattern found")
                return None

            # Extract relevant bytes for distance calculation
            # Use more robust indexing with boundary checks
            if len(hex_data) >= 12:
                byte1_low = hex_data[9:10]
                byte2_high = hex_data[10:11]
                byte2_low = hex_data[11:12]

                # Calculate distance value
                distance_value = (
                    self._hex_to_int(byte2_low)
                    + self._hex_to_int(byte2_high) * 16
                    + self._hex_to_int(byte1_low) * 256
                )

                # Sanity check the value (typical range check)
                if 0 <= distance_value <= 4000:  # Adjust range as per sensor specs
                    return distance_value
                else:
                    logger.debug(f"Distance value out of expected range: {distance_value}")
                    return None
            else:
                logger.debug("Hex data not long enough for parsing")
                return None

        except (IndexError, ValueError) as e:
            logger.warning(f"Error parsing distance data: {e}, hex_data: {hex_data}")
            return None

    def read_distance(self):
        """Read a single distance measurement from the sensor."""
        if not self.serial_port or not self.serial_port.isOpen():
            logger.error("Serial port not open")
            return None

        self.serial_port.flushInput()  # Clear input buffer first
        time.sleep(0.01)

        if self.serial_port.in_waiting:
            try:
                raw_data = self.serial_port.read(self.serial_port.in_waiting)
                hex_data = str(binascii.b2a_hex(raw_data))

                distance = self._parse_distance_data(hex_data)
                return distance
            except Exception as e:
                logger.error(f"Error reading data: {e}")
                return None
        return None

    def reset_connection(self):
        """Reset the serial connection if needed."""
        logger.info("Resetting serial connection...")
        try:
            self.close()
            time.sleep(0.5)
            success = self.connect()
            if success:
                logger.info("Serial connection reset successfully")
            else:
                logger.error("Failed to reset serial connection")
        except Exception as e:
            logger.error(f"Error during connection reset: {e}")

    def start_continuous_reading(self, callback=None, read_interval=DEFAULT_READ_INTERVAL):
        """Start continuous reading with optional callback function."""
        if not self.serial_port or not self.serial_port.isOpen():
            if not self.connect():
                logger.error("Failed to connect. Cannot start continuous reading.")
                return

        self.running = True
        last_distance = 0
        last_output_time = time.perf_counter()
        last_reset_time = time.perf_counter()
        error_count = 0

        logger.info("Starting continuous reading. Press Ctrl+C to stop.")

        try:
            while self.running:
                distance = self.read_distance()
                current_time = time.perf_counter()

                if distance is not None:
                    # Calculate rate of change in mm/s if we have previous
                    # reading
                    time_diff = current_time - last_output_time
                    should_output = False

                    # Determine if we should output (based on change or time
                    # interval)
                    if (
                        last_distance > 0
                        and abs(distance - last_distance) > SIGNIFICANT_CHANGE_THRESHOLD
                    ) or (current_time - last_output_time) > REPORT_INTERVAL:
                        should_output = True

                    if should_output:
                        if time_diff > 0 and last_distance > 0:
                            rate = (distance - last_distance) / time_diff
                            logger.info(f"Distance: {distance} mm | Rate: {rate:.1f} mm/s")
                        else:
                            logger.info(f"Distance: {distance} mm")
                        last_output_time = current_time

                    # If callback provided, pass the distance to it
                    if callback:
                        callback(distance)

                    # Update last values
                    last_distance = distance
                    error_count = 0
                else:
                    error_count += 1
                    if error_count >= MAX_ERROR_COUNT:
                        logger.warning(
                            f"Multiple failed readings ({error_count}). Attempting recovery..."
                        )

                        # Check if enough time has passed since last reset
                        if current_time - last_reset_time > 30:  # 30 seconds between resets
                            self.reset_connection()
                            last_reset_time = current_time

                        error_count = 0

                time.sleep(read_interval)

        except KeyboardInterrupt:
            logger.info("Sensor reading stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous reading: {e}")
        finally:
            self.running = False

    def stop(self):
        """Stop the continuous reading."""
        self.running = False
        logger.info("Stopping continuous reading")

    def close(self):
        """Close the serial port."""
        if self.serial_port and self.serial_port.isOpen():
            self.serial_port.close()
            logger.info("Serial port closed")
            self.serial_port = None


@contextmanager
def sensor_context(port=DEFAULT_SERIAL_PORT, baudrate=DEFAULT_BAUD_RATE, timeout=DEFAULT_TIMEOUT):
    """Context manager for the sensor to ensure proper cleanup."""
    sensor = TOF400FSensor(port, baudrate, timeout)
    try:
        if sensor.connect():
            yield sensor
        else:
            logger.error("Failed to connect to sensor")
            yield None
    finally:
        sensor.stop()
        sensor.close()


def distance_callback(distance):
    """Example callback function to handle distance data."""
    # You could integrate with other systems here
    # For example, save to a database, trigger an alarm, etc.


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="TOF400F Sensor Interface")

    parser.add_argument(
        "--port",
        type=str,
        default=DEFAULT_SERIAL_PORT,
        help=f"Serial port (default: {DEFAULT_SERIAL_PORT})",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DEFAULT_BAUD_RATE,
        help=f"Baud rate (default: {DEFAULT_BAUD_RATE})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Serial timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_READ_INTERVAL,
        help=f"Reading interval in seconds (default: {DEFAULT_READ_INTERVAL})",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def main():
    """Main function to run the sensor interface."""
    args = parse_arguments()

    # Set logging level based on arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Use context manager to ensure proper cleanup
    with sensor_context(args.port, args.baudrate, args.timeout) as sensor:
        if sensor:
            try:
                # Start continuous reading
                sensor.start_continuous_reading(
                    callback=distance_callback, read_interval=args.interval
                )
            except KeyboardInterrupt:
                logger.info("Program stopped by user")


if __name__ == "__main__":
    main()
