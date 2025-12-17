"""
Debug script to check link geometry and constraints.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.kinematic_solver import KinematicSolver
from tests.test_kinematic_solver import setup_simple_suspension


def debug_link_geometry():
    """Debug link geometry and endpoint freedom."""
    print("=" * 70)
    print("DEBUG: Link Geometry Analysis")
    print("=" * 70)

    chassis, knuckle_name = setup_simple_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    print("\nAnalyzing upper_front link:")
    upper_front = solver.registry.links['upper_front']

    print(f"Link length: {upper_front.length:.3f} mm")
    print(f"Endpoint1 (chassis side): {upper_front.endpoint1.position}")
    print(f"Endpoint2 (knuckle side): {upper_front.endpoint2.position}")

    # Check if endpoints are in solver state
    ep1_name = upper_front.endpoint1.name
    ep2_name = upper_front.endpoint2.name

    ep1_free = ep1_name in solver.state.free_points
    ep2_free = ep2_name in solver.state.free_points

    print(f"\nEndpoint1 '{ep1_name}': {'FREE' if ep1_free else 'FIXED'}")
    print(f"Endpoint2 '{ep2_name}': {'FREE' if ep2_free else 'FIXED'}")

    # Check what joints these endpoints belong to
    print(f"\nEndpoint1 joint: {upper_front.endpoint1.joint.name if upper_front.endpoint1.joint else 'None'}")
    print(f"Endpoint2 joint: {upper_front.endpoint2.joint.name if upper_front.endpoint2.joint else 'None'}")

    # Get the knuckle
    knuckle = solver.get_knuckle(knuckle_name)

    # Calculate geometric limits
    # If chassis point is fixed and link length is constant,
    # endpoint2 must lie on a sphere
    chassis_point = np.array([1400, 0, 600])  # Fixed chassis bushing location
    link_length = upper_front.length

    print(f"\n{'='*70}")
    print("GEOMETRIC ANALYSIS:")
    print(f"{'='*70}")
    print(f"\nIf chassis bushing stays at {chassis_point}")
    print(f"And link length is {link_length:.3f} mm")
    print(f"Then endpoint2 must be on sphere of radius {link_length:.3f} mm")

    # Initial knuckle ball joint position
    initial_ball = knuckle.attachment_points[0].position.copy()
    print(f"\nInitial knuckle ball joint: {initial_ball}")
    initial_dist = np.linalg.norm(initial_ball - chassis_point)
    print(f"Initial distance chassis->ball: {initial_dist:.3f} mm")
    print(f"Link length: {link_length:.3f} mm")
    print(f"Match: {abs(initial_dist - link_length) < 0.1}")

    # After moving knuckle down 30mm
    target_ball = initial_ball.copy()
    target_ball[2] -= 30
    print(f"\nTarget ball joint after -30mm heave: {target_ball}")
    target_dist = np.linalg.norm(target_ball - chassis_point)
    print(f"Target distance chassis->ball: {target_dist:.3f} mm")
    print(f"Link length: {link_length:.3f} mm")
    print(f"Distance change: {target_dist - initial_dist:.3f} mm")

    print(f"\n{'='*70}")
    print("PROBLEM DIAGNOSIS:")
    print(f"{'='*70}")

    if abs(target_dist - link_length) > 1.0:
        print(f"\n⚠️  GEOMETRIC CONFLICT DETECTED!")
        print(f"The target knuckle position is {target_dist:.3f} mm from chassis")
        print(f"But the link length is only {link_length:.3f} mm")
        print(f"Difference: {abs(target_dist - link_length):.3f} mm")
        print(f"\nWith a rigid link and fixed chassis point, this geometry is IMPOSSIBLE!")
        print(f"The solver must choose between:")
        print(f"  1. Violating link length constraint")
        print(f"  2. Not achieving target heave displacement")
        print(f"  3. Allowing massive bushing deflection ← Currently doing this")
    else:
        print(f"\n✓ Geometry is compatible")


if __name__ == "__main__":
    debug_link_geometry()
