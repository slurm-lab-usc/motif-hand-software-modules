# python
import os
import time

import mujoco
import numpy as np

from MOTIFHand.visualizer.data_handler.sensor_serial_handler import \
    RS485MultiSensorReader


class TactileVisualizer:
    """Real-time tactile sensor visualization system"""

    def __init__(self, xml_path: str):
        """Initialize the tactile visualizer

        Args:
            xml_path: Path to the MuJoCo XML model
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Definitions for 11 tactile zones
        self.tactile_zones = [
            # Index finger - 3 zones
            "if_md",
            "if_px",
            "if_bs",
            # Middle finger - 3 zones
            "mf_md",
            "mf_px",
            "mf_bs",
            # Ring finger - 3 zones
            "rf_md",
            "rf_px",
            "rf_bs",
            # Thumb - 2 zones
            "th_px",
            "th_bs",
        ]

        # Tactile grid parameters
        self.grid_size = 6  # 6x6 grid

        # Prebuild geom_id mapping for all tactile cells
        self.tactile_geom_ids = self._build_geom_id_mapping()

        # Color definitions
        self.colors = {
            "inactive": [0, 0, 1, 0.3],  # blue, no contact
            "low": [0, 1, 0, 0.5],  # green, low pressure
            "medium": [1, 1, 0, 0.7],  # yellow, medium pressure
            "high": [1, 0.5, 0, 0.8],  # orange, high pressure
            "max": [1, 0, 0, 1.0],  # red, maximum pressure
        }

        print(f"✓ Tactile visualizer initialization complete")
        print(f"✓ Found {len(self.tactile_geom_ids)} tactile cells")
        self._print_zone_summary()

    def _build_geom_id_mapping(self) -> dict[str, dict[tuple[int, int], int]]:
        """Build a mapping from tactile zone and grid coordinates to geom_id

        Returns:
            Nested dictionary {zone_name: {(i, j): geom_id}}
        """
        geom_mapping = {}
        total_found = 0

        for zone in self.tactile_zones:
            geom_mapping[zone] = {}
            zone_count = 0

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    geom_name = f"{zone}_tactile_{i}_{j}"
                    try:
                        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                        if geom_id >= 0:
                            geom_mapping[zone][(i, j)] = geom_id
                            zone_count += 1
                            total_found += 1
                    except BaseException:
                        pass  # ignore missing geom

            if zone_count > 0:
                print(f"  {zone}: {zone_count} cells")

        return geom_mapping

    def _print_zone_summary(self):
        """Print a summary of tactile zones"""
        print(f"\nTactile zone distribution:")
        finger_zones = {
            "Index": ["if_md", "if_px", "if_bs"],
            "Middle": ["mf_md", "mf_px", "mf_bs"],
            "Ring": ["rf_md", "rf_px", "rf_bs"],
            "Thumb": ["th_px", "th_bs"],
        }

        for finger, zones in finger_zones.items():
            zone_counts = [len(self.tactile_geom_ids.get(zone, {})) for zone in zones]
            print(f"  {finger}: {zones} -> {zone_counts} cells")

    def update_tactile_colors_by_zone(self, tactile_data: dict[str, np.ndarray]):
        """Update each zone's colors based on tactile data

        Args:
            tactile_data: dictionary {zone_name: 6x6_pressure_array}
                          pressure values range [0.0, 1.0]
        """
        updated_count = 0

        for zone_name, pressure_grid in tactile_data.items():
            if zone_name not in self.tactile_geom_ids:
                continue

            # Ensure pressure_grid is 6x6
            if pressure_grid.shape != (self.grid_size, self.grid_size):
                print(
                    f"Warning: {zone_name} data has wrong shape {pressure_grid.shape}, expected ({self.grid_size}, {self.grid_size})"
                )
                continue

            # Update colors for all cells in this zone
            for (i, j), geom_id in self.tactile_geom_ids[zone_name].items():
                pressure = pressure_grid[i, j]
                color = self._pressure_to_color(pressure)
                self.model.geom_rgba[geom_id] = color
                updated_count += 1

        return updated_count

    def update_tactile_colors_by_coords(
        self, tactile_data: dict[str, dict[tuple[int, int], float]]
    ):
        """Update single cell colors by coordinates

        Args:
            tactile_data: nested dict {zone_name: {(i,j): pressure_value}}
        """
        updated_count = 0

        for zone_name, coords_data in tactile_data.items():
            if zone_name not in self.tactile_geom_ids:
                continue

            for (i, j), pressure in coords_data.items():
                if (i, j) in self.tactile_geom_ids[zone_name]:
                    geom_id = self.tactile_geom_ids[zone_name][(i, j)]
                    color = self._pressure_to_color(pressure)
                    self.model.geom_rgba[geom_id] = color
                    updated_count += 1

        return updated_count

    def _pressure_to_color(self, pressure: float) -> list[float]:
        """Convert pressure value to color

        Args:
            pressure: pressure value [0.0, 1.0]

        Returns:
            RGBA color list
        """
        # pressure = np.clip(pressure, 0.0, 1.0)

        if pressure < 3.0:
            return self.colors["inactive"]
        elif pressure < 12.0:
            return self.colors["low"]
        elif pressure < 35.0:
            return self.colors["medium"]
        elif pressure < 50.0:
            return self.colors["high"]
        else:
            return self.colors["max"]

    def reset_all_colors(self, color_type: str = "inactive"):
        """Reset all tactile cell colors

        Args:
            color_type: 'inactive', 'low', 'medium', 'high', 'max'
        """
        color = self.colors.get(color_type, self.colors["inactive"])
        updated_count = 0

        for zone_name, coords_dict in self.tactile_geom_ids.items():
            for geom_id in coords_dict.values():
                self.model.geom_rgba[geom_id] = color
                updated_count += 1

        print(f"✓ Reset {updated_count} cells to {color_type} color")

    def simulate_random_tactile_data(self) -> dict[str, np.ndarray]:
        """Generate random tactile data for testing

        Returns:
            Dictionary of random tactile data
        """
        tactile_data = {}

        for zone in self.tactile_zones:
            if len(self.tactile_geom_ids.get(zone, {})) > 0:
                # Generate random 6x6 pressure data
                pressure_grid = np.random.random((self.grid_size, self.grid_size))
                # Add some hotspots
                pressure_grid[2:4, 2:4] = np.random.uniform(0.7, 1.0, (2, 2))
                tactile_data[zone] = pressure_grid

        return tactile_data

    def get_tactile_data_from_sensor(self, reader: RS485MultiSensorReader) -> dict:
        """Read the latest data from the sensor and convert it to visualizer format"""
        tactile_data = {}

        ZONE_MAPPING = {
            (1, 0): "if_md",
            (1, 1): "if_px",
            (3, 2): "if_bs",
            (2, 0): "mf_md",
            (2, 1): "mf_px",
            (2, 2): "mf_bs",
            (3, 0): "rf_md",
            (3, 1): "rf_px",
            (1, 2): "rf_bs",
            (0, 1): "th_px",
            (0, 2): "th_bs",
        }

        for (finger_id, board_id), zone_name in ZONE_MAPPING.items():
            sensor_data = reader.get_module_data(finger_id, board_id)
            if sensor_data is None:
                continue
            fsr_values = sensor_data.fsr  # list of length 36
            if len(fsr_values) != 36:
                continue
            # Convert to 6x6 np.array and normalize to [0,1] (assume max 1023)
            fsr_array = np.array(fsr_values, dtype=np.float32).reshape((6, 6))  # / 71.0
            tactile_data[zone_name] = fsr_array

        return tactile_data

    def get_zone_info(self, zone_name: str) -> dict:
        """Get detailed info for a specified zone

        Args:
            zone_name: zone name, e.g. 'if_md'

        Returns:
            Zone info dictionary
        """
        if zone_name not in self.tactile_geom_ids:
            return {"error": f"Zone {zone_name} not found"}

        coords_dict = self.tactile_geom_ids[zone_name]

        info = {
            "zone_name": zone_name,
            "grid_size": self.grid_size,
            "total_cells": len(coords_dict),
            "available_coords": list(coords_dict.keys()),
            "geom_ids": list(coords_dict.values()),
        }

        return info

    def test_visualization(self, duration: float = 5.0):
        """Test the visualization effect

        Args:
            duration: test duration in seconds
        """
        print(f"\nStarting tactile visualization test for {duration} seconds...")

        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < duration:
            # Generate random tactile data
            tactile_data = self.simulate_random_tactile_data()

            # Update colors
            updated = self.update_tactile_colors_by_zone(tactile_data)

            # Step simulation
            mujoco.mj_step(self.model, self.data)

            frame_count += 1

            # Update every 0.1 s
            time.sleep(0.1)

        print(
            f"✓ Test complete: {frame_count} frames, updated {updated} cells on average per frame"
        )

        # Reset colors
        self.reset_all_colors()

    def show_random_tactile_in_viewer(self, duration: float = None):
        """Display colored cells in the MuJoCo viewer in real time

        Args:
            duration: duration in seconds; if None, run until viewer closes
        """
        import mujoco.viewer

        if duration is not None:
            print(f"\nStarting tactile visualization viewer demo for {duration} seconds...")
        else:
            print(
                f"\nStarting infinite tactile visualization viewer demo (close window to exit)..."
            )

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            while viewer.is_running() and (
                duration is None or (time.time() - start_time < duration)
            ):
                # Generate random tactile data
                tactile_data = self.simulate_random_tactile_data()
                # Update all cell colors
                self.update_tactile_colors_by_zone(tactile_data)
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                # Refresh viewer
                viewer.sync()
                # Control refresh rate
                time.sleep(0.01)
        print("✓ Viewer demo ended")

    def show_sensor_tactile_in_viewer(self, reader, duration: float = None):
        """Display real-time sensor tactile data in the MuJoCo viewer

        Args:
            reader: RS485MultiSensorReader instance
            duration: duration in seconds; if None, run until viewer closes
        """
        import mujoco.viewer

        if duration is not None:
            print(f"\nStarting sensor real-time tactile visualization for {duration} seconds...")
        else:
            print(
                f"\nStarting infinite sensor real-time tactile visualization (close window to exit)..."
            )

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            while viewer.is_running() and (
                duration is None or (time.time() - start_time < duration)
            ):
                tactile_data = self.get_tactile_data_from_sensor(reader)
                self.update_tactile_colors_by_zone(tactile_data)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.001)

        print("✓ Viewer demo ended")


# Usage example
def main():
    """Main function - demonstrate usage of the tactile visualizer"""

    # 1. Initialize visualizer
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_dir, "assets/scene_left.xml")
    reader = RS485MultiSensorReader()
    reader.start()

    visualizer = TactileVisualizer(xml_path)

    visualizer.show_sensor_tactile_in_viewer(reader)


if __name__ == "__main__":
    main()
