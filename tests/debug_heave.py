"""
Debug script to investigate heave test failure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.kinematic_solver import KinematicSolver
from pysuspension.chassis import Chassis
from pysuspension.chassis_corner import ChassisCorner
from pysuspension.suspension_knuckle import SuspensionKnuckle
from pysuspension.control_arm import ControlArm
from pysuspension.suspension_link import SuspensionLink
from pysuspension.coil_spring import CoilSpring
from pysuspension.suspension_joint import SuspensionJoint
from pysuspension.attachment_point import AttachmentPoint
from pysuspension.joint_types import JointType


def setup_simple_suspension():
    """Set up a double-wishbone suspension."""
    chassis = Chassis("test_chassis")

    # Create chassis corner with attachment points
    corner = ChassisCorner("front_left")
    uf_chassis = corner.add_attachment_point("upper_front", [1400, 0, 600], unit='mm')
    ur_chassis = corner.add_attachment_point("upper_rear", [1200, 0, 600], unit='mm')
    lf_chassis = corner.add_attachment_point("lower_front", [1500, 0, 300], unit='mm')
    lr_chassis = corner.add_attachment_point("lower_rear", [1100, 0, 300], unit='mm')
    chassis.add_corner(corner)

    # Create suspension knuckle with attachment points
    knuckle = SuspensionKnuckle(
        tire_center_x=1400,
        tire_center_y=750,
        rolling_radius=390,
        toe_angle=0.0,
        camber_angle=-1.0,
        unit='mm',
        name='front_left_knuckle'
    )

    # Add attachment points to knuckle
    upper_ball = knuckle.add_attachment_point("upper_ball_joint", [1400, 650, 580], unit='mm')
    lower_ball = knuckle.add_attachment_point("lower_ball_joint", [1400, 700, 200], unit='mm')
    tie_rod_knuckle = knuckle.add_attachment_point("tie_rod", [1400, 650, 390], unit='mm')

    # Create upper A-arm links
    upper_front_link = SuspensionLink(
        endpoint1=[1400, 0, 600],
        endpoint2=[1400, 650, 580],
        name="upper_front",
        unit='mm'
    )

    upper_rear_link = SuspensionLink(
        endpoint1=[1200, 0, 600],
        endpoint2=[1400, 650, 580],
        name="upper_rear",
        unit='mm'
    )

    # Create lower A-arm links
    lower_front_link = SuspensionLink(
        endpoint1=[1500, 0, 300],
        endpoint2=[1400, 700, 200],
        name="lower_front",
        unit='mm'
    )

    lower_rear_link = SuspensionLink(
        endpoint1=[1100, 0, 300],
        endpoint2=[1400, 700, 200],
        name="lower_rear",
        unit='mm'
    )

    # Create joints connecting components
    uf_bushing = SuspensionJoint("uf_bushing", JointType.BUSHING_SOFT)
    uf_bushing.add_attachment_point(uf_chassis)
    uf_bushing.add_attachment_point(upper_front_link.endpoint1)

    ur_bushing = SuspensionJoint("ur_bushing", JointType.BUSHING_SOFT)
    ur_bushing.add_attachment_point(ur_chassis)
    ur_bushing.add_attachment_point(upper_rear_link.endpoint1)

    upper_ball_joint = SuspensionJoint("upper_ball", JointType.BALL_JOINT)
    upper_ball_joint.add_attachment_point(upper_front_link.endpoint2)
    upper_ball_joint.add_attachment_point(upper_rear_link.endpoint2)
    upper_ball_joint.add_attachment_point(upper_ball)

    lf_bushing = SuspensionJoint("lf_bushing", JointType.BUSHING_SOFT)
    lf_bushing.add_attachment_point(lf_chassis)
    lf_bushing.add_attachment_point(lower_front_link.endpoint1)

    lr_bushing = SuspensionJoint("lr_bushing", JointType.BUSHING_SOFT)
    lr_bushing.add_attachment_point(lr_chassis)
    lr_bushing.add_attachment_point(lower_rear_link.endpoint1)

    lower_ball_joint = SuspensionJoint("lower_ball", JointType.BALL_JOINT)
    lower_ball_joint.add_attachment_point(lower_front_link.endpoint2)
    lower_ball_joint.add_attachment_point(lower_rear_link.endpoint2)
    lower_ball_joint.add_attachment_point(lower_ball)

    # Register all components with chassis
    chassis.add_component(knuckle)
    chassis.add_joint(uf_bushing)
    chassis.add_joint(ur_bushing)
    chassis.add_joint(upper_ball_joint)
    chassis.add_joint(lf_bushing)
    chassis.add_joint(lr_bushing)
    chassis.add_joint(lower_ball_joint)

    return chassis, 'front_left_knuckle'


def debug_heave():
    """Debug the heave test."""
    print("=" * 70)
    print("DEBUG: Heave Test Investigation")
    print("=" * 70)

    chassis, knuckle_name = setup_simple_suspension()

    print("\n1. Creating solver from chassis...")
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    print(f"\n{solver}")
    print(f"\nSolver state: {solver.state}")

    print(f"\nFree points ({len(solver.state.free_points)}):")
    for point_id in sorted(solver.state.free_points):
        point = solver.state.points[point_id]
        print(f"  {point_id}: {point.position}")

    print(f"\nFixed points ({len(solver.state.fixed_points)}):")
    for point_id in sorted(solver.state.fixed_points):
        point = solver.state.points[point_id]
        print(f"  {point_id}: {point.position}")

    print(f"\n2. Checking initial constraint errors...")
    constraints = solver._generate_constraints_from_components()
    print(f"Total constraints: {len(constraints)}")

    # Check initial errors
    print("\nInitial constraint errors (top 10):")
    errors = []
    for c in constraints:
        error = c.get_physical_error()
        errors.append((c.name, error))
    errors.sort(key=lambda x: x[1], reverse=True)

    for name, error in errors[:10]:
        print(f"  {name}: {error:.6f} mm")

    print(f"\nTotal initial error: {sum(e for _, e in errors):.6f} mm")
    print(f"RMS initial error: {np.sqrt(np.mean([e**2 for _, e in errors])):.6f} mm")

    # Get knuckle position
    knuckle = solver.get_knuckle(knuckle_name)
    initial_contact_patch = knuckle.get_tire_contact_patch(unit='mm')
    print(f"\n3. Initial knuckle tire contact patch: {initial_contact_patch}")
    print(f"   Knuckle tire_center: {knuckle.tire_center}")

    # Check knuckle attachment points
    print(f"\nKnuckle attachment points:")
    for ap in knuckle.attachment_points:
        is_free = ap.name in solver.state.free_points
        is_fixed = ap.name in solver.state.fixed_points
        print(f"  {ap.name}: {ap.position} - {'FREE' if is_free else 'FIXED' if is_fixed else 'UNKNOWN'}")

    # Now try to solve for heave
    print(f"\n4. Solving for heave displacement: -30 mm")
    heave_displacement = -30.0

    result = solver.solve_for_heave(knuckle_name, heave_displacement, unit='mm')

    print(f"\nResult: {result}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"RMS error: {result.get_rms_error():.6f} mm")

    # Check final position
    final_contact_patch = knuckle.get_tire_contact_patch(unit='mm')
    actual_displacement = final_contact_patch[2] - initial_contact_patch[2]

    print(f"\nFinal tire contact patch: {final_contact_patch}")
    print(f"Actual Z displacement: {actual_displacement:.2f} mm (target: {heave_displacement:.2f} mm)")
    print(f"Error: {abs(actual_displacement - heave_displacement):.2f} mm")

    # Check final constraint errors
    print(f"\nFinal constraint errors (top 10 largest):")
    final_errors = list(result.constraint_errors.items())
    final_errors.sort(key=lambda x: x[1], reverse=True)
    for name, error in final_errors[:10]:
        print(f"  {name}: {error:.6f} mm")

    # Check if knuckle points moved
    print(f"\nKnuckle attachment point movements:")
    for ap in knuckle.attachment_points:
        print(f"  {ap.name}: {ap.position}")


if __name__ == "__main__":
    debug_heave()
