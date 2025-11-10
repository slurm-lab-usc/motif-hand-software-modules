import time

from MOTIFHand.utils.arm_control.xarm7.arm_client.client import XArmController


def main():
    # Create robot controller instance
    controller = XArmController.get_instance(
        "192.168.1.239"
    )  # Please modify IP address according to your setup

    # Define two target positions
    pose1 = {
        "position": [500, 0, 200],  # [x, y, z] in mm
        "orientation": [3.14159, 0, 0],  # [roll, pitch, yaw] in radians
    }

    pose2 = {
        "position": [500, 0, 400],  # [x, y, z] in mm
        "orientation": [3.14159, 0, 0],  # [roll, pitch, yaw] in radians
    }

    current_pose = 1  # Current position index

    print("Program started!")
    print("Press Enter to switch positions")
    print("Type 'q' and press Enter to quit")

    try:
        # First move to position 1
        controller.set_ee_pose(
            position=pose1["position"],
            orientation=pose1["orientation"],
            is_radian=True,
            speed=100,
            wait=True,
        )

        while True:
            user_input = input()

            if user_input.lower() == "q":
                print("Program exiting")
                break

            # Switch positions
            if current_pose == 1:
                print("Moving to position 2")
                controller.set_ee_pose(
                    position=pose2["position"],
                    orientation=pose2["orientation"],
                    is_radian=True,
                    speed=100,
                    wait=True,
                )
                current_pose = 2
            else:
                print("Moving to position 1")
                controller.set_ee_pose(
                    position=pose1["position"],
                    orientation=pose1["orientation"],
                    is_radian=True,
                    speed=100,
                    wait=True,
                )
                current_pose = 1

    finally:
        # Cleanup
        # controller.go_home()
        time.sleep(2)
        controller.disconnect()


if __name__ == "__main__":
    main()
