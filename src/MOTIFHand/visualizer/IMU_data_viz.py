import sys
import tkinter as tk
from collections import deque
from tkinter import messagebox, ttk

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# # Ensure the current directory is in the path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)  # Go up one level to the workspace root
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# Import the sensor handlers
try:
    from MOTIFHand.visualizer.data_handler.ahrs_data_processor import \
        AHRSDataProcessor
    from MOTIFHand.visualizer.data_handler.sensor_serial_handler import \
        RS485MultiSensorReader
    from MOTIFHand.visualizer.data_handler.test_data_generator import \
        TestDataGenerator
except ImportError as e:
    print(f"Error importing sensor handlers: {e}")
    sys.exit(1)

matplotlib.use("TkAgg")  # Set matplotlib backend


class MultiSensorVisualizer:
    def __init__(self, root):
        """
        Initialize the Multi-Sensor Data Visualizer

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Multi-Sensor Data Visualizer")
        self.root.geometry("1600x1000")  # Increased from 1400x900

        # Data history for plotting
        self.history_length = 200  # Number of data points to display
        self.time_history = deque(range(self.history_length), maxlen=self.history_length)

        # Sensor data storage
        self.sensor_data = {}  # Store current sensor data for each module
        self.ahrs_data = {}  # Store AHRS quaternion data for each module

        # Mode control
        self.current_mode = "sensor"  # "sensor" or "ahrs"
        self.expanded_view = False  # Whether showing detailed waveform view
        self.selected_module = None  # Currently selected module for detailed view

        # Initialize sensor handlers
        self.sensor_reader = RS485MultiSensorReader()
        self.ahrs_processor = AHRSDataProcessor(self.sensor_reader)
        self.test_data_generator = TestDataGenerator()

        # Create separate AHRS processor for test data
        self.test_ahrs_processor = AHRSDataProcessor(self.test_data_generator)

        # Current data source
        self.current_data_source = "Serial"

        # Create UI components
        self.create_ui()

        # Setup plots
        self.setup_plots()

        # Start animation for real-time plotting
        self.animation = FuncAnimation(
            self.fig, self.update_plots, interval=50, cache_frame_data=False
        )

    def create_ui(self):
        """Create the user interface components"""
        # Create frames
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Mode selection
        ttk.Label(control_frame, text="Display Mode:").grid(row=0, column=0, padx=5, pady=5)
        self.mode_var = tk.StringVar(value="Sensor Data")
        self.mode_combobox = ttk.Combobox(
            control_frame,
            textvariable=self.mode_var,
            values=["Sensor Data", "AHRS Fusion"],
            width=15,
            state="readonly",
        )
        self.mode_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.mode_combobox.bind("<<ComboboxSelected>>", self.on_mode_change)

        # Data source selection
        ttk.Label(control_frame, text="Data Source:").grid(row=0, column=2, padx=5, pady=5)
        self.data_source_var = tk.StringVar(value="Serial")
        self.data_source_combobox = ttk.Combobox(
            control_frame,
            textvariable=self.data_source_var,
            values=["Serial", "Test Data", "Simulation"],
            width=12,
            state="readonly",
        )
        self.data_source_combobox.grid(row=0, column=3, padx=5, pady=5)
        self.data_source_combobox.bind("<<ComboboxSelected>>", self.on_data_source_change)

        # Serial port control
        ttk.Label(control_frame, text="Port:").grid(row=0, column=4, padx=5, pady=5)
        self.port_entry = ttk.Entry(control_frame, width=15)
        self.port_entry.insert(0, "/dev/tty.usbmodem5A350015431")  # Default port
        self.port_entry.grid(row=0, column=5, padx=5, pady=5)

        ttk.Label(control_frame, text="Baud Rate:").grid(row=0, column=6, padx=5, pady=5)
        self.baud_entry = ttk.Entry(control_frame, width=10)
        self.baud_entry.insert(0, "115200")  # Default baud rate
        self.baud_entry.grid(row=0, column=7, padx=5, pady=5)

        self.connect_button = ttk.Button(
            control_frame, text="Connect", command=self.toggle_connection
        )
        self.connect_button.grid(row=0, column=8, padx=5, pady=5)

        # AHRS filter selection (only visible in AHRS mode)
        ttk.Label(control_frame, text="AHRS Filter:").grid(row=0, column=9, padx=5, pady=5)
        self.filter_var = tk.StringVar(value="madgwick")
        self.filter_combobox = ttk.Combobox(
            control_frame,
            textvariable=self.filter_var,
            values=["madgwick", "mahony", "ekf"],
            width=10,
            state="readonly",
        )
        self.filter_combobox.grid(row=0, column=10, padx=5, pady=5)
        self.filter_combobox.bind("<<ComboboxSelected>>", self.on_filter_change)

        # Control buttons
        self.reset_button = ttk.Button(control_frame, text="Reset View", command=self.reset_view)
        self.reset_button.grid(row=0, column=11, padx=5, pady=5)

        self.back_button = ttk.Button(control_frame, text="Back to Grid", command=self.back_to_grid)
        self.back_button.grid(row=0, column=12, padx=5, pady=5)
        self.back_button.config(state="disabled")  # Initially disabled

        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Disconnected")
        self.status_label.grid(row=1, column=0, columnspan=13, padx=5, pady=5)

        # Create figures and canvas
        self.fig = plt.figure(figsize=(18, 12))  # Increased from (14, 9)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind click events
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

    def setup_plots(self):
        """Setup the grid layout plots"""
        self.fig.clear()

        if self.expanded_view:
            self.setup_detailed_view()
        else:
            self.setup_grid_view()

        self.fig.tight_layout()

    def setup_grid_view(self):
        """Setup the 4x3 grid layout for overview (11 total coordinate systems)"""
        # Create a 4x3 grid layout with better spacing and horizontal aspect
        # ratio
        gs = self.fig.add_gridspec(
            4,
            4,
            hspace=0.5,
            wspace=0.5,  # Increased spacing
            # Left labels, then 3 coordinate systems
            width_ratios=[0.3, 1, 1, 1],
            height_ratios=[1, 1, 1, 1],
        )

        # Initialize coordinate system plots
        self.coord_plots = {}

        # Create layout (11 total coordinate systems)
        # Fingers 0-3, but skip F0-B0, so we have:
        # F0-B1, F0-B2, (empty)
        # F1-B0, F1-B1, F1-B2
        # F2-B0, F2-B1, F2-B2
        # F3-B0, F3-B1, F3-B2

        # First row: F0-B1, F0-B2, empty
        for board in range(1, 3):
            ax = self.fig.add_subplot(gs[0, board + 1], projection="3d")  # Use columns 2 and 3
            ax.set_xlim([-2.0, 2.0])  # Increased range
            ax.set_ylim([-1.5, 1.5])  # Increased range
            ax.set_zlim([-1.5, 1.5])  # Increased range

            # Create coordinate arrows (larger, horizontal aspect ratio)
            (x_arrow,) = ax.plot([0, 1.8], [0, 0], [0, 0], "r-", linewidth=4)  # Increased size
            (y_arrow,) = ax.plot([0, 0], [0, 1.5], [0, 0], "g-", linewidth=4)  # Increased size
            (z_arrow,) = ax.plot([0, 0], [0, 0], [0, 1.5], "b-", linewidth=4)  # Increased size

            # Add arrowheads (larger)
            x_head = ax.scatter([1.8], [0], [0], color="r", s=120)  # Increased size
            y_head = ax.scatter([0], [1.5], [0], color="g", s=120)  # Increased size
            z_head = ax.scatter([0], [0], [1.5], color="b", s=120)  # Increased size

            # Add IMU body (larger)
            imu_body = ax.scatter([0], [0], [0], color="black", s=150)  # Increased size

            # Store plot elements
            module_key = f"0-{board}"
            self.coord_plots[module_key] = {
                "ax": ax,
                "x_arrow": x_arrow,
                "y_arrow": y_arrow,
                "z_arrow": z_arrow,
                "x_head": x_head,
                "y_head": y_head,
                "z_head": z_head,
                "imu_body": imu_body,
            }

        # Add label for first row on the left
        ax_label = self.fig.add_subplot(gs[0, 0])
        ax_label.text(
            0.5,
            0.5,
            "F0",
            fontsize=18,
            ha="center",
            va="center",  # Increased font size
            transform=ax_label.transAxes,
            weight="bold",
        )
        ax_label.axis("off")

        # Rows 1-3: F1-B0, F1-B1, F1-B2; F2-B0, F2-B1, F2-B2; F3-B0, F3-B1,
        # F3-B2
        for finger in range(1, 4):
            for board in range(3):
                ax = self.fig.add_subplot(
                    gs[finger, board + 1], projection="3d"
                )  # Use columns 2, 3, 4
                ax.set_xlim([-2.0, 2.0])  # Increased range
                ax.set_ylim([-1.5, 1.5])  # Increased range
                ax.set_zlim([-1.5, 1.5])  # Increased range

                # Create coordinate arrows (larger, horizontal aspect ratio)
                (x_arrow,) = ax.plot([0, 1.8], [0, 0], [0, 0], "r-", linewidth=4)  # Increased size
                (y_arrow,) = ax.plot([0, 0], [0, 1.5], [0, 0], "g-", linewidth=4)  # Increased size
                (z_arrow,) = ax.plot([0, 0], [0, 0], [0, 1.5], "b-", linewidth=4)  # Increased size

                # Add arrowheads (larger)
                x_head = ax.scatter([1.8], [0], [0], color="r", s=120)  # Increased size
                y_head = ax.scatter([0], [1.5], [0], color="g", s=120)  # Increased size
                z_head = ax.scatter([0], [0], [1.5], color="b", s=120)  # Increased size

                # Add IMU body (larger)
                imu_body = ax.scatter([0], [0], [0], color="black", s=150)  # Increased size

                # Store plot elements
                module_key = f"{finger}-{board}"
                self.coord_plots[module_key] = {
                    "ax": ax,
                    "x_arrow": x_arrow,
                    "y_arrow": y_arrow,
                    "z_arrow": z_arrow,
                    "x_head": x_head,
                    "y_head": y_head,
                    "z_head": z_head,
                    "imu_body": imu_body,
                }

        # Add labels for rows 1-3 on the left
        for finger in range(1, 4):
            ax_label = self.fig.add_subplot(gs[finger, 0])
            ax_label.text(
                0.5,
                0.5,
                f"F{finger}",
                fontsize=18,
                ha="center",
                va="center",  # Increased font size
                transform=ax_label.transAxes,
                weight="bold",
            )
            ax_label.axis("off")

        # Add FX-BX labels on the left side of each coordinate system
        # First row labels
        for board in range(1, 3):
            ax = self.coord_plots[f"0-{board}"]["ax"]
            ax.text2D(
                -0.25,
                0.5,
                f"B{board}",
                fontsize=14,
                ha="center",
                va="center",  # Increased font size
                transform=ax.transAxes,
                weight="bold",
                rotation=90,
            )

        # Rows 1-3 labels
        for finger in range(1, 4):
            for board in range(3):
                ax = self.coord_plots[f"{finger}-{board}"]["ax"]
                ax.text2D(
                    -0.25,
                    0.5,
                    f"B{board}",
                    fontsize=14,
                    ha="center",
                    va="center",  # Increased font size
                    transform=ax.transAxes,
                    weight="bold",
                    rotation=90,
                )

        # Add a title for the entire grid
        self.fig.suptitle(
            "Multi-Sensor Orientation Display", fontsize=20, y=0.98
        )  # Increased font size

    def setup_detailed_view(self):
        """Setup detailed waveform view for selected module"""
        if self.selected_module is None:
            return

        if self.current_mode == "sensor":
            # 3x3 grid for sensor data (acc, gyro, mag)
            gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # Initialize data history for this module
            if self.selected_module not in self.sensor_data:
                self.sensor_data[self.selected_module] = {
                    "acc_x": deque([0] * self.history_length, maxlen=self.history_length),
                    "acc_y": deque([0] * self.history_length, maxlen=self.history_length),
                    "acc_z": deque([0] * self.history_length, maxlen=self.history_length),
                    "gyro_x": deque([0] * self.history_length, maxlen=self.history_length),
                    "gyro_y": deque([0] * self.history_length, maxlen=self.history_length),
                    "gyro_z": deque([0] * self.history_length, maxlen=self.history_length),
                    "mag_x": deque([0] * self.history_length, maxlen=self.history_length),
                    "mag_y": deque([0] * self.history_length, maxlen=self.history_length),
                    "mag_z": deque([0] * self.history_length, maxlen=self.history_length),
                }

            # Create plots
            self.detail_plots = {}

            # Acceleration plots
            ax_acc_x = self.fig.add_subplot(gs[0, 0])
            ax_acc_x.set_title("Acceleration X")
            (acc_x_line,) = ax_acc_x.plot(
                self.time_history, self.sensor_data[self.selected_module]["acc_x"], "r-"
            )

            ax_acc_y = self.fig.add_subplot(gs[0, 1])
            ax_acc_y.set_title("Acceleration Y")
            (acc_y_line,) = ax_acc_y.plot(
                self.time_history, self.sensor_data[self.selected_module]["acc_y"], "g-"
            )

            ax_acc_z = self.fig.add_subplot(gs[0, 2])
            ax_acc_z.set_title("Acceleration Z")
            (acc_z_line,) = ax_acc_z.plot(
                self.time_history, self.sensor_data[self.selected_module]["acc_z"], "b-"
            )

            # Gyroscope plots
            ax_gyro_x = self.fig.add_subplot(gs[1, 0])
            ax_gyro_x.set_title("Gyroscope X")
            (gyro_x_line,) = ax_gyro_x.plot(
                self.time_history,
                self.sensor_data[self.selected_module]["gyro_x"],
                "r-",
            )

            ax_gyro_y = self.fig.add_subplot(gs[1, 1])
            ax_gyro_y.set_title("Gyroscope Y")
            (gyro_y_line,) = ax_gyro_y.plot(
                self.time_history,
                self.sensor_data[self.selected_module]["gyro_y"],
                "g-",
            )

            ax_gyro_z = self.fig.add_subplot(gs[1, 2])
            ax_gyro_z.set_title("Gyroscope Z")
            (gyro_z_line,) = ax_gyro_z.plot(
                self.time_history,
                self.sensor_data[self.selected_module]["gyro_z"],
                "b-",
            )

            # Magnetometer plots
            ax_mag_x = self.fig.add_subplot(gs[2, 0])
            ax_mag_x.set_title("Magnetometer X")
            (mag_x_line,) = ax_mag_x.plot(
                self.time_history, self.sensor_data[self.selected_module]["mag_x"], "r-"
            )

            ax_mag_y = self.fig.add_subplot(gs[2, 1])
            ax_mag_y.set_title("Magnetometer Y")
            (mag_y_line,) = ax_mag_y.plot(
                self.time_history, self.sensor_data[self.selected_module]["mag_y"], "g-"
            )

            ax_mag_z = self.fig.add_subplot(gs[2, 2])
            ax_mag_z.set_title("Magnetometer Z")
            (mag_z_line,) = ax_mag_z.plot(
                self.time_history, self.sensor_data[self.selected_module]["mag_z"], "b-"
            )

            # Store plot elements
            self.detail_plots = {
                "acc_x": (ax_acc_x, acc_x_line),
                "acc_y": (ax_acc_y, acc_y_line),
                "acc_z": (ax_acc_z, acc_z_line),
                "gyro_x": (ax_gyro_x, gyro_x_line),
                "gyro_y": (ax_gyro_y, gyro_y_line),
                "gyro_z": (ax_gyro_z, gyro_z_line),
                "mag_x": (ax_mag_x, mag_x_line),
                "mag_y": (ax_mag_y, mag_y_line),
                "mag_z": (ax_mag_z, mag_z_line),
            }

        else:  # AHRS mode
            # 2x2 grid for quaternion components
            gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

            # Initialize quaternion history for this module
            if self.selected_module not in self.ahrs_data:
                self.ahrs_data[self.selected_module] = {
                    "w": deque([1.0] * self.history_length, maxlen=self.history_length),
                    "x": deque([0.0] * self.history_length, maxlen=self.history_length),
                    "y": deque([0.0] * self.history_length, maxlen=self.history_length),
                    "z": deque([0.0] * self.history_length, maxlen=self.history_length),
                }

            # Create quaternion plots
            ax_w = self.fig.add_subplot(gs[0, 0])
            ax_w.set_title("Quaternion W")
            (w_line,) = ax_w.plot(
                self.time_history, self.ahrs_data[self.selected_module]["w"], "r-"
            )

            ax_x = self.fig.add_subplot(gs[0, 1])
            ax_x.set_title("Quaternion X")
            (x_line,) = ax_x.plot(
                self.time_history, self.ahrs_data[self.selected_module]["x"], "g-"
            )

            ax_y = self.fig.add_subplot(gs[1, 0])
            ax_y.set_title("Quaternion Y")
            (y_line,) = ax_y.plot(
                self.time_history, self.ahrs_data[self.selected_module]["y"], "b-"
            )

            ax_z = self.fig.add_subplot(gs[1, 1])
            ax_z.set_title("Quaternion Z")
            (z_line,) = ax_z.plot(
                self.time_history, self.ahrs_data[self.selected_module]["z"], "m-"
            )

            # Store plot elements
            self.detail_plots = {
                "w": (ax_w, w_line),
                "x": (ax_x, x_line),
                "y": (ax_y, y_line),
                "z": (ax_z, z_line),
            }

    def update_plots(self, frame):
        """Update all plots with current data"""
        if self.expanded_view:
            self.update_detailed_view()
        else:
            self.update_grid_view()

        return []

    def update_grid_view(self):
        """Update the grid view coordinate systems"""
        # Get current data based on data source
        if self.current_data_source == "Serial":
            if self.current_mode == "sensor":
                all_data = self.sensor_reader.get_all_modules_data()
            else:  # AHRS mode
                all_data = self.ahrs_processor.get_all_quaternions()
        elif self.current_data_source in ["Test Data", "Simulation"]:
            if self.current_mode == "sensor":
                all_data = self.test_data_generator.get_all_modules_data()
            else:  # AHRS mode - use proper AHRS processor for test data
                all_data = self.test_ahrs_processor.get_all_quaternions()
        else:
            # Fallback
            all_data = {}

        # Update each coordinate system
        for module_key, plot_elements in self.coord_plots.items():
            if module_key in all_data and all_data[module_key] is not None:
                if self.current_mode == "sensor":
                    self.update_sensor_coordinate_system(
                        module_key, all_data[module_key], plot_elements
                    )
                else:
                    # For AHRS mode, data should be quaternions
                    data = all_data[module_key]
                    if hasattr(data, "w"):  # Quaternion
                        self.update_ahrs_coordinate_system(module_key, data, plot_elements)
                    else:  # Fallback to sensor data if quaternion not available
                        self.update_sensor_coordinate_system(module_key, data, plot_elements)
            else:
                # Reset to default position if no data
                self.reset_coordinate_system(plot_elements)

    def update_sensor_coordinate_system(self, module_key, sensor_data, plot_elements):
        """Update coordinate system based on sensor data"""
        # Simple orientation calculation from accelerometer
        acc = sensor_data.acc
        acc_magnitude = np.linalg.norm(acc)

        if acc_magnitude > 0.1:
            # Normalize acceleration
            acc_normalized = np.array(acc) / acc_magnitude

            # Create rotation matrix based on gravity direction
            # This is a simplified approach - in practice you'd use proper
            # sensor fusion
            gravity = np.array([0, 0, -1])  # Expected gravity direction
            rotation_axis = np.cross(acc_normalized, gravity)
            rotation_angle = np.arccos(np.dot(acc_normalized, gravity))

            if np.linalg.norm(rotation_axis) > 0.001:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

                # Create rotation matrix using Rodrigues' rotation formula
                K = np.array(
                    [
                        [0, -rotation_axis[2], rotation_axis[1]],
                        [rotation_axis[2], 0, -rotation_axis[0]],
                        [-rotation_axis[1], rotation_axis[0], 0],
                    ]
                )
                R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
            else:
                R = np.eye(3)
        else:
            R = np.eye(3)

        # Apply rotation to coordinate vectors (horizontal aspect ratio)
        x_base = np.array([1.8, 0, 0])  # Longer X-axis
        y_base = np.array([0, 1.5, 0])
        z_base = np.array([0, 0, 1.5])

        x_rotated = R @ x_base
        y_rotated = R @ y_base
        z_rotated = R @ z_base

        # Update arrows
        plot_elements["x_arrow"].set_data_3d(
            [0, x_rotated[0]], [0, x_rotated[1]], [0, x_rotated[2]]
        )
        plot_elements["y_arrow"].set_data_3d(
            [0, y_rotated[0]], [0, y_rotated[1]], [0, y_rotated[2]]
        )
        plot_elements["z_arrow"].set_data_3d(
            [0, z_rotated[0]], [0, z_rotated[1]], [0, z_rotated[2]]
        )

        # Update arrowheads
        plot_elements["x_head"]._offsets3d = (
            [x_rotated[0]],
            [x_rotated[1]],
            [x_rotated[2]],
        )
        plot_elements["y_head"]._offsets3d = (
            [y_rotated[0]],
            [y_rotated[1]],
            [y_rotated[2]],
        )
        plot_elements["z_head"]._offsets3d = (
            [z_rotated[0]],
            [z_rotated[1]],
            [z_rotated[2]],
        )

    def update_ahrs_coordinate_system(self, module_key, quaternion, plot_elements):
        """Update coordinate system based on AHRS quaternion"""
        # Convert quaternion to rotation matrix
        w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z

        R = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * w * z,
                    2 * x * z + 2 * w * y,
                ],
                [
                    2 * x * y + 2 * w * z,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * w * x,
                ],
                [
                    2 * x * z - 2 * w * y,
                    2 * y * z + 2 * w * x,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )

        # Apply rotation to coordinate vectors (horizontal aspect ratio)
        x_base = np.array([1.8, 0, 0])  # Longer X-axis
        y_base = np.array([0, 1.5, 0])
        z_base = np.array([0, 0, 1.5])

        x_rotated = R @ x_base
        y_rotated = R @ y_base
        z_rotated = R @ z_base

        # Update arrows
        plot_elements["x_arrow"].set_data_3d(
            [0, x_rotated[0]], [0, x_rotated[1]], [0, x_rotated[2]]
        )
        plot_elements["y_arrow"].set_data_3d(
            [0, y_rotated[0]], [0, y_rotated[1]], [0, y_rotated[2]]
        )
        plot_elements["z_arrow"].set_data_3d(
            [0, z_rotated[0]], [0, z_rotated[1]], [0, z_rotated[2]]
        )

        # Update arrowheads
        plot_elements["x_head"]._offsets3d = (
            [x_rotated[0]],
            [x_rotated[1]],
            [x_rotated[2]],
        )
        plot_elements["y_head"]._offsets3d = (
            [y_rotated[0]],
            [y_rotated[1]],
            [y_rotated[2]],
        )
        plot_elements["z_head"]._offsets3d = (
            [z_rotated[0]],
            [z_rotated[1]],
            [z_rotated[2]],
        )

    def reset_coordinate_system(self, plot_elements):
        """Reset coordinate system to default position"""
        # Reset arrows to default position (horizontal aspect ratio)
        plot_elements["x_arrow"].set_data_3d([0, 1.8], [0, 0], [0, 0])  # Longer X-axis
        plot_elements["y_arrow"].set_data_3d([0, 0], [0, 1.5], [0, 0])
        plot_elements["z_arrow"].set_data_3d([0, 0], [0, 0], [0, 1.5])

        # Reset arrowheads
        plot_elements["x_head"]._offsets3d = ([1.8], [0], [0])  # Longer X-axis
        plot_elements["y_head"]._offsets3d = ([0], [1.5], [0])
        plot_elements["z_head"]._offsets3d = ([0], [0], [1.5])

    def update_detailed_view(self):
        """Update detailed waveform view"""
        if self.selected_module is None:
            return

        if self.current_mode == "sensor":
            # Update sensor data history based on data source
            if self.current_data_source == "Serial":
                sensor_data = self.sensor_reader.get_module_data(
                    *map(int, self.selected_module.split("-"))
                )
            elif self.current_data_source in ["Test Data", "Simulation"]:
                sensor_data = self.test_data_generator.get_module_data(
                    *map(int, self.selected_module.split("-"))
                )
            else:
                sensor_data = None

            if sensor_data:
                self.sensor_data[self.selected_module]["acc_x"].append(sensor_data.acc[0])
                self.sensor_data[self.selected_module]["acc_y"].append(sensor_data.acc[1])
                self.sensor_data[self.selected_module]["acc_z"].append(sensor_data.acc[2])
                self.sensor_data[self.selected_module]["gyro_x"].append(sensor_data.gyro[0])
                self.sensor_data[self.selected_module]["gyro_y"].append(sensor_data.gyro[1])
                self.sensor_data[self.selected_module]["gyro_z"].append(sensor_data.gyro[2])
                self.sensor_data[self.selected_module]["mag_x"].append(sensor_data.mag[0])
                self.sensor_data[self.selected_module]["mag_y"].append(sensor_data.mag[1])
                self.sensor_data[self.selected_module]["mag_z"].append(sensor_data.mag[2])

            # Update plot lines
            for key, (ax, line) in self.detail_plots.items():
                line.set_ydata(self.sensor_data[self.selected_module][key])
                # Auto-adjust y limits
                data = self.sensor_data[self.selected_module][key]
                if len(data) > 0:
                    min_val, max_val = min(data), max(data)
                    padding = (max_val - min_val) * 0.1 if max_val != min_val else 1.0
                    ax.set_ylim([min_val - padding, max_val + padding])

        else:  # AHRS mode
            # Update quaternion history based on data source
            if self.current_data_source == "Serial":
                quaternion = self.ahrs_processor.get_quaternion(
                    *map(int, self.selected_module.split("-"))
                )
            elif self.current_data_source in ["Test Data", "Simulation"]:
                # Use proper AHRS processor for test data
                quaternion = self.test_ahrs_processor.get_quaternion(
                    *map(int, self.selected_module.split("-"))
                )
            else:
                quaternion = None

            if quaternion:
                self.ahrs_data[self.selected_module]["w"].append(quaternion.w)
                self.ahrs_data[self.selected_module]["x"].append(quaternion.x)
                self.ahrs_data[self.selected_module]["y"].append(quaternion.y)
                self.ahrs_data[self.selected_module]["z"].append(quaternion.z)

            # Update plot lines
            for key, (ax, line) in self.detail_plots.items():
                line.set_ydata(self.ahrs_data[self.selected_module][key])
                # Auto-adjust y limits
                data = self.ahrs_data[self.selected_module][key]
                if len(data) > 0:
                    min_val, max_val = min(data), max(data)
                    padding = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
                    ax.set_ylim([min_val - padding, max_val + padding])

    def on_plot_click(self, event):
        """Handle plot click events"""
        if event.inaxes is None:
            return

        if not self.expanded_view:
            # Find which module was clicked
            for module_key, plot_elements in self.coord_plots.items():
                if plot_elements["ax"] == event.inaxes:
                    self.selected_module = module_key
                    self.expanded_view = True
                    self.setup_plots()
                    self.back_button.config(state="normal")
                    self.status_label.config(text=f"Status: Detailed view - Module {module_key}")
                    break

    def on_mode_change(self, event=None):
        """Handle mode change between Sensor Data and AHRS Fusion"""
        mode = self.mode_var.get()
        if mode == "Sensor Data":
            self.current_mode = "sensor"
        else:
            self.current_mode = "ahrs"

        # Reset view when changing modes
        if self.expanded_view:
            self.back_to_grid()

        self.status_label.config(text=f"Status: Mode changed to {mode}")

    def on_filter_change(self, event=None):
        """Handle AHRS filter change"""
        if self.current_mode == "ahrs":
            filter_type = self.filter_var.get()
            # Reinitialize AHRS processor with new filter
            self.ahrs_processor = AHRSDataProcessor(self.sensor_reader, filter_type=filter_type)
            if self.sensor_reader.running:
                self.ahrs_processor.start()

            self.status_label.config(text=f"Status: AHRS filter changed to {filter_type}")

    def toggle_connection(self):
        """Connect or disconnect from the sensor system"""
        data_source = self.data_source_var.get()

        if data_source == "Serial":
            if not self.sensor_reader.running:
                # Get port and baud rate from UI
                port = self.port_entry.get()
                try:
                    baud_rate = int(self.baud_entry.get())
                except ValueError:
                    messagebox.showerror("Input Error", "Baud rate must be a number")
                    return

                # Update sensor reader port
                self.sensor_reader.port = port
                self.sensor_reader.baudrate = baud_rate

                # Start sensor reader and AHRS processor
                try:
                    self.sensor_reader.start()
                    self.ahrs_processor.start()
                    self.connect_button.config(text="Disconnect")
                    self.status_label.config(text="Status: Connected to Serial")
                except Exception as e:
                    messagebox.showerror("Connection Error", f"Failed to connect: {e}")
            else:
                # Disconnect
                self.sensor_reader.stop()
                self.ahrs_processor.stop()
                self.connect_button.config(text="Connect")
                self.status_label.config(text="Status: Disconnected from Serial")

        elif data_source == "Test Data":
            if not self.test_data_generator.running:
                # Start test data generator
                try:
                    self.test_data_generator.start()
                    self.test_ahrs_processor.start()  # Start AHRS processor for test data
                    self.connect_button.config(text="Disconnect")
                    self.status_label.config(text="Status: Test Data Running")
                except Exception as e:
                    messagebox.showerror("Test Data Error", f"Failed to start test data: {e}")
            else:
                # Stop test data generator
                self.test_data_generator.stop()
                self.test_ahrs_processor.stop()  # Stop AHRS processor for test data
                self.connect_button.config(text="Connect")
                self.status_label.config(text="Status: Test Data Stopped")

        elif data_source == "Simulation":
            if not self.test_data_generator.running:
                # Start simulation (using test data generator with different
                # parameters)
                try:
                    self.test_data_generator.set_motion_parameters(
                        amplitude=1.0, frequency=1.0, noise=0.05
                    )
                    self.test_data_generator.start()
                    self.test_ahrs_processor.start()  # Start AHRS processor for simulation
                    self.connect_button.config(text="Disconnect")
                    self.status_label.config(text="Status: Simulation Running")
                except Exception as e:
                    messagebox.showerror("Simulation Error", f"Failed to start simulation: {e}")
            else:
                # Stop simulation
                self.test_data_generator.stop()
                self.test_ahrs_processor.stop()  # Stop AHRS processor for simulation
                self.connect_button.config(text="Connect")
                self.status_label.config(text="Status: Simulation Stopped")

    def back_to_grid(self):
        """Return to grid view"""
        self.expanded_view = False
        self.selected_module = None
        self.setup_plots()
        self.back_button.config(state="disabled")
        self.status_label.config(text="Status: Grid view")

    def reset_view(self):
        """Reset the view"""
        if self.expanded_view:
            self.back_to_grid()

        # Clear all data history
        self.sensor_data.clear()
        self.ahrs_data.clear()

        self.status_label.config(text="Status: View reset")

    def on_data_source_change(self, event=None):
        """Handle data source change"""
        new_source = self.data_source_var.get()

        # Disconnect current source if connected
        if self.sensor_reader.running:
            self.sensor_reader.stop()
        if self.ahrs_processor.running:
            self.ahrs_processor.stop()
        if self.test_data_generator.running:
            self.test_data_generator.stop()
        if self.test_ahrs_processor.running:
            self.test_ahrs_processor.stop()

        self.current_data_source = new_source
        self.connect_button.config(text="Connect")
        self.status_label.config(text=f"Status: Data source changed to {new_source}")

        # Update UI based on data source
        if new_source == "Serial":
            self.port_entry.config(state="normal")
            self.baud_entry.config(state="normal")
        else:
            self.port_entry.config(state="disabled")
            self.baud_entry.config(state="disabled")


def main():
    """Main function to start the multi-sensor visualizer"""
    root = tk.Tk()
    MultiSensorVisualizer(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()


if __name__ == "__main__":
    main()
