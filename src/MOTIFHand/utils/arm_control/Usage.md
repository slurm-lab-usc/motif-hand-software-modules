**Features:**

- Switches between two predefined poses
- Press **Enter**: Move to alternate position
- Type `q` + **Enter**: Quit

## Typical Workflow

### Recording a New Trajectory

1. **Position the arm manually** using the teach pendant or programming interface
2. **Run the recorder:**
   ```bash
   python xarm7/arm_client/arm_record_trajs.py
   ```
3. **Record waypoints** by pressing Enter at each desired position
4. **Save** by typing `s` and pressing Enter
5. All the trajectories will be stored under `recordings`

### Playing Back a Trajectory

1. **Review available recordings:**
   ```bash
   python xarm7/arm_client/arm_play_trajs.py
   ```
2. **Select the desired recording** from the list
3. **Choose playback mode** and follow prompts

### Collecting Data with Sensor

1. **Record a trajectory** (see above)
2. **Ensure Raspberry Pi sensor system is running**
3. **Run data collector:**
   ```bash
   python xarm7/arm_client/arm_data_collector.py \
     --captures 50 \
     --object my_object \
     --trajectory recordings/my_scan.json
   ```
4. **Data is saved** to the specified output directory

## Important Notes

- **Safety First**: Always start with low speeds (20-30) and increase gradually
- **Position Units**: Use millimeters for position values (not meters)
- **Orientation**: Can use either Euler angles or quaternions
- **IP Addresses**: Update IP addresses to match your network configuration
- **Trajectories**: Stored in `xarm7/arm_client/recordings/` directory

## Troubleshooting

**Connection Issues:**

- Verify robot IP address is correct
- Ensure robot is powered on and network accessible
- Check that motion is enabled on the robot

**Movement Issues:**

- Reduce speed parameter if movements are erratic
- Verify position values are in millimeters
- Check for workspace limits and collision zones

**Data Collection Issues:**

- Ensure Raspberry Pi sensor system is running
- Verify network connectivity between systems
- Check ZMQ port is not blocked by firewall
