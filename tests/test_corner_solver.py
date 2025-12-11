"""
Test CornerSolver with a simple double wishbone suspension.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.corner_solver import CornerSolver
from pysuspension.control_arm import ControlArm
from pysuspension.suspension_link import SuspensionLink
from pysuspension.attachment_point import AttachmentPoint
from pysuspension.joint_types import JointType


def test_corner_solver():
    """Test CornerSolver with simple double wishbone suspension."""
    print("=" * 70)
    print("CORNER SOLVER TEST - Simple Double Wishbone")
    print("=" * 70)

    # Create a simple double wishbone suspension
    # Coordinate system: X = forward, Y = outboard, Z = up

    print("\n--- Creating Suspension Components ---")

    # Upper control arm (shorter, at top)
    upper_arm = ControlArm("upper_control_arm")
    upper_front_link = SuspensionLink(
        endpoint1=[1400, 0, 600],     # Chassis mount (front)
        endpoint2=[1400, 650, 580],   # Ball joint
        name="upper_front",
        unit='mm'
    )
    upper_rear_link = SuspensionLink(
        endpoint1=[1200, 0, 600],     # Chassis mount (rear)
        endpoint2=[1400, 650, 580],   # Ball joint (shared)
        name="upper_rear",
        unit='mm'
    )
    upper_arm.add_link(upper_front_link)
    upper_arm.add_link(upper_rear_link)

    # Lower control arm (longer, at bottom)
    lower_arm = ControlArm("lower_control_arm")
    lower_front_link = SuspensionLink(
        endpoint1=[1500, 0, 300],     # Chassis mount (front)
        endpoint2=[1400, 700, 200],   # Ball joint
        name="lower_front",
        unit='mm'
    )
    lower_rear_link = SuspensionLink(
        endpoint1=[1100, 0, 300],     # Chassis mount (rear)
        endpoint2=[1400, 700, 200],   # Ball joint (shared)
        name="lower_rear",
        unit='mm'
    )
    lower_arm.add_link(lower_front_link)
    lower_arm.add_link(lower_rear_link)

    # Wheel center (on knuckle, between ball joints)
    wheel_center = AttachmentPoint("wheel_center", [1400, 750, 390], unit='mm')

    print(f"Upper arm: {len(upper_arm.links)} links")
    print(f"Lower arm: {len(lower_arm.links)} links")
    print(f"Wheel center: {wheel_center.position} mm")

    print("\n--- Building CornerSolver ---")
    solver = CornerSolver("double_wishbone")

    # Mark chassis mount points
    solver.chassis_mounts.extend([
        upper_front_link.endpoint1,
        upper_rear_link.endpoint1,
        lower_front_link.endpoint1,
        lower_rear_link.endpoint1
    ])

    # Add upper control arm links
    solver.add_link(upper_front_link, end1_mount_point=upper_front_link.endpoint1)
    solver.add_link(upper_rear_link, end1_mount_point=upper_rear_link.endpoint1)

    # Add lower control arm links
    solver.add_link(lower_front_link, end1_mount_point=lower_front_link.endpoint1)
    solver.add_link(lower_rear_link, end1_mount_point=lower_rear_link.endpoint1)

    # Ball joints connect control arms to knuckle
    upper_ball_joint = upper_front_link.endpoint2
    lower_ball_joint = lower_front_link.endpoint2

    # Connect wheel center to ball joints with rigid links
    # (This simulates the knuckle connecting the ball joints to the wheel)
    # NOTE: We're NOT using explicit joints here - the geometry alone constrains the system
    upper_to_wheel = SuspensionLink(
        upper_ball_joint,
        wheel_center,
        name="upper_knuckle_link",
        unit='mm'
    )
    lower_to_wheel = SuspensionLink(
        lower_ball_joint,
        wheel_center,
        name="lower_knuckle_link",
        unit='mm'
    )
    solver.add_link(upper_to_wheel)
    solver.add_link(lower_to_wheel)

    # Set wheel center for heave calculations
    solver.set_wheel_center(wheel_center)

    print(solver)
    solver.save_initial_state()

    print("\n--- Solving Initial Configuration ---")
    result = solver.solve()
    print(result)
    print(f"RMS error: {result.get_rms_error():.6f} mm")

    print("\n--- Testing Heave Travel ---")
    heave_values = [-50, -25, 0, 25, 50]  # mm

    print(f"\n{'Heave (mm)':<12} {'Wheel Z (mm)':<15} {'Camber (deg)':<15} {'RMS Error (mm)':<15}")
    print("-" * 60)

    for heave in heave_values:
        # Reset to initial
        solver.reset_to_initial_state()

        # Solve for this heave position
        result = solver.solve_for_heave(heave, unit='mm')

        wheel_z = result.get_position("wheel_center")[2]
        camber = solver.get_camber(upper_ball_joint, lower_ball_joint, unit='deg')
        rms_error = result.get_rms_error()

        print(f"{heave:<12.1f} {wheel_z:<15.3f} {camber:<15.3f} {rms_error:<15.6f}")

    print("\n--- Final Positions at +50mm Heave ---")
    print(f"Upper ball joint: {result.get_position('upper_front_endpoint2')}")
    print(f"Lower ball joint: {result.get_position('lower_front_endpoint2')}")
    print(f"Wheel center: {result.get_position('wheel_center')}")

    print("\n✓ CornerSolver test completed successfully!")


if __name__ == "__main__":
    try:
        test_corner_solver()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
