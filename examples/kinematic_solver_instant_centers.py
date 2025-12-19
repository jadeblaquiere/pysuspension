"""
Example: Instant Center Calculations with KinematicSolver

This example demonstrates how to use the KinematicSolver to calculate
instant centers for roll and pitch motion. Instant centers are critical
parameters in suspension design that affect handling characteristics.

Key concepts:
- Roll instant center: Determines roll stiffness and camber gain
- Pitch instant center: Affects anti-dive and anti-squat characteristics
- Circle fitting: Used to find centers from contact patch trajectory
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension import (
    Chassis, ChassisCorner, SuspensionKnuckle, SuspensionLink,
    SuspensionJoint, KinematicSolver
)
from pysuspension.joint_types import JointType


def create_double_wishbone_suspension():
    """
    Create a simple double-wishbone front suspension for demonstration.

    Returns:
        Tuple of (chassis, knuckle_name)
    """
    print("Creating double-wishbone suspension geometry...")

    # Create chassis
    chassis = Chassis("example_chassis")

    # Create chassis corner with mounting points
    corner = ChassisCorner("front_left")
    uf_chassis = corner.add_attachment_point("upper_front", [1400, 0, 600], unit='mm')
    ur_chassis = corner.add_attachment_point("upper_rear", [1200, 0, 600], unit='mm')
    lf_chassis = corner.add_attachment_point("lower_front", [1500, 0, 300], unit='mm')
    lr_chassis = corner.add_attachment_point("lower_rear", [1100, 0, 300], unit='mm')
    tr_chassis = corner.add_attachment_point("chassis_tr", [1150, 0, 400], unit='mm')
    chassis.add_corner(corner)

    # Create suspension knuckle with tire geometry
    knuckle = SuspensionKnuckle(
        tire_center_x=1400,      # mm
        tire_center_y=750,       # mm - outboard position
        rolling_radius=390,      # mm - tire outer radius
        toe_angle=0.0,           # degrees - neutral toe
        camber_angle=-1.0,       # degrees - negative camber
        unit='mm',
        name='front_left_knuckle'
    )

    # Add knuckle attachment points for control arms and tie rod
    upper_ball = knuckle.add_attachment_point("upper_ball_joint", [1400, 650, 580], unit='mm')
    lower_ball = knuckle.add_attachment_point("lower_ball_joint", [1400, 700, 200], unit='mm')
    tie_rod_knuckle = knuckle.add_attachment_point("tie_rod", [1400, 650, 390], unit='mm')

    # Create upper A-arm links
    upper_front_link = SuspensionLink(
        endpoint1=[1400, 0, 600],    # Chassis mount
        endpoint2=[1400, 650, 580],  # Upper ball joint
        name="upper_front",
        unit='mm'
    )

    upper_rear_link = SuspensionLink(
        endpoint1=[1200, 0, 600],    # Chassis mount
        endpoint2=[1400, 650, 580],  # Upper ball joint (same location)
        name="upper_rear",
        unit='mm'
    )

    # Create lower A-arm links
    lower_front_link = SuspensionLink(
        endpoint1=[1500, 0, 300],    # Chassis mount
        endpoint2=[1400, 700, 200],  # Lower ball joint
        name="lower_front",
        unit='mm'
    )

    lower_rear_link = SuspensionLink(
        endpoint1=[1100, 0, 300],    # Chassis mount
        endpoint2=[1400, 700, 200],  # Lower ball joint (same location)
        name="lower_rear",
        unit='mm'
    )

    # Create tie rod link
    tie_rod_link = SuspensionLink(
        endpoint1=[1150, 0, 400],    # Chassis mount (steering rack)
        endpoint2=[1400, 650, 390],  # Tie rod outer ball joint
        name="tr_outer",
        unit='mm'
    )

    # Create joints to connect components
    # Upper bushings (chassis to upper A-arm)
    uf_bushing = SuspensionJoint("uf_bushing", JointType.SPHERICAL_BEARING)
    uf_bushing.add_attachment_point(uf_chassis)
    uf_bushing.add_attachment_point(upper_front_link.endpoint1)

    ur_bushing = SuspensionJoint("ur_bushing", JointType.SPHERICAL_BEARING)
    ur_bushing.add_attachment_point(ur_chassis)
    ur_bushing.add_attachment_point(upper_rear_link.endpoint1)

    # Upper ball joint (upper A-arm to knuckle)
    upper_ball_joint = SuspensionJoint("upper_ball", JointType.BALL_JOINT)
    upper_ball_joint.add_attachment_point(upper_front_link.endpoint2)
    upper_ball_joint.add_attachment_point(upper_rear_link.endpoint2)
    upper_ball_joint.add_attachment_point(upper_ball)

    # Lower bushings (chassis to lower A-arm)
    lf_bushing = SuspensionJoint("lf_bushing", JointType.SPHERICAL_BEARING)
    lf_bushing.add_attachment_point(lf_chassis)
    lf_bushing.add_attachment_point(lower_front_link.endpoint1)

    lr_bushing = SuspensionJoint("lr_bushing", JointType.SPHERICAL_BEARING)
    lr_bushing.add_attachment_point(lr_chassis)
    lr_bushing.add_attachment_point(lower_rear_link.endpoint1)

    # Lower ball joint (lower A-arm to knuckle)
    lower_ball_joint = SuspensionJoint("lower_ball", JointType.BALL_JOINT)
    lower_ball_joint.add_attachment_point(lower_front_link.endpoint2)
    lower_ball_joint.add_attachment_point(lower_rear_link.endpoint2)
    lower_ball_joint.add_attachment_point(lower_ball)

    # Tie rod joints
    outer_tr_joint = SuspensionJoint("outer_tr_ball", JointType.BALL_JOINT)
    outer_tr_joint.add_attachment_point(tie_rod_link.endpoint2)
    outer_tr_joint.add_attachment_point(tie_rod_knuckle)

    inner_tr_joint = SuspensionJoint("inner_tr_ball", JointType.BALL_JOINT)
    inner_tr_joint.add_attachment_point(tie_rod_link.endpoint1)
    inner_tr_joint.add_attachment_point(tr_chassis)

    # Register all components with chassis
    chassis.add_component(knuckle)
    chassis.add_component(upper_front_link)
    chassis.add_component(upper_rear_link)
    chassis.add_component(lower_front_link)
    chassis.add_component(lower_rear_link)
    chassis.add_component(tie_rod_link)

    chassis.add_joint(uf_bushing)
    chassis.add_joint(ur_bushing)
    chassis.add_joint(upper_ball_joint)
    chassis.add_joint(lf_bushing)
    chassis.add_joint(lr_bushing)
    chassis.add_joint(lower_ball_joint)
    chassis.add_joint(outer_tr_joint)
    chassis.add_joint(inner_tr_joint)

    print("✓ Suspension created with:")
    print(f"  - 1 knuckle with tire geometry")
    print(f"  - 5 links (2 upper, 2 lower, 1 tie rod)")
    print(f"  - 8 joints (4 bushings, 4 ball joints)")

    return chassis, 'front_left_knuckle'


def example_basic_instant_centers():
    """Example 1: Basic instant center calculation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Instant Center Calculation")
    print("=" * 70)

    # Create suspension geometry
    chassis, knuckle_name = create_double_wishbone_suspension()

    # Create solver from chassis
    print("\nCreating KinematicSolver...")
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])
    print(f"✓ Solver created with {len(solver.registry.links)} links and {len(solver.registry.joints)} joints")

    # Calculate instant centers with default settings
    print("\nCalculating instant centers...")
    print("Using default z_offsets: [0, 5, 10, -5, -10] mm")

    result = solver.calculate_instant_centers(knuckle_name)

    # Display results
    print("\n" + "-" * 70)
    print("INSTANT CENTER RESULTS")
    print("-" * 70)

    roll_center = result['roll_center']
    pitch_center = result['pitch_center']

    print(f"\nRoll Instant Center:")
    print(f"  Position: [{roll_center[0]:.2f}, {roll_center[1]:.2f}, {roll_center[2]:.2f}] mm")
    print(f"  Height above ground: {roll_center[2]:.2f} mm")
    print(f"  Lateral offset from centerline: {roll_center[1]:.2f} mm")
    print(f"  Circle radius: {result['roll_radius']:.2f} mm")
    print(f"  Fit quality: {result['roll_fit_quality']:.4f} ({result['roll_fit_quality']*100:.2f}%)")

    print(f"\nPitch Instant Center:")
    print(f"  Position: [{pitch_center[0]:.2f}, {pitch_center[1]:.2f}, {pitch_center[2]:.2f}] mm")
    print(f"  Height above ground: {pitch_center[2]:.2f} mm")
    print(f"  Longitudinal position: {pitch_center[0]:.2f} mm")
    print(f"  Circle radius: {result['pitch_radius']:.2f} mm")
    print(f"  Fit quality: {result['pitch_fit_quality']:.4f} ({result['pitch_fit_quality']*100:.2f}%)")

    # Analysis of convergence
    print(f"\nSolver Convergence:")
    max_error = max(result['solve_errors'])
    avg_error = sum(result['solve_errors']) / len(result['solve_errors'])
    print(f"  Max solve error: {max_error:.4f} mm")
    print(f"  Avg solve error: {avg_error:.4f} mm")
    print(f"  All solves converged: {'✓' if max_error < 1.0 else '✗'}")


def example_custom_travel_range():
    """Example 2: Custom travel range for instant centers."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Travel Range")
    print("=" * 70)

    chassis, knuckle_name = create_double_wishbone_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    # Use larger travel range for more accurate circle fitting
    z_offsets = [-40, -30, -20, -10, 0, 10, 20, 30, 40]

    print(f"\nCalculating instant centers over ±40mm travel range")
    print(f"Using {len(z_offsets)} measurement points: {z_offsets} mm")

    result = solver.calculate_instant_centers(knuckle_name, z_offsets=z_offsets)

    # Show how instant center varies with more data points
    print(f"\nResults with extended travel:")
    print(f"  Roll center height: {result['roll_center'][2]:.2f} mm")
    print(f"  Roll fit quality: {result['roll_fit_quality']:.4f} (better with more points)")
    print(f"  Pitch center height: {result['pitch_center'][2]:.2f} mm")
    print(f"  Pitch fit quality: {result['pitch_fit_quality']:.4f}")

    # Show contact point trajectory
    print(f"\nContact Point Trajectory (Z positions):")
    for i, cp in enumerate(result['contact_points']):
        print(f"  Offset {z_offsets[i]:4.0f} mm: contact Z = {cp[2]:.2f} mm")


def example_instant_center_migration():
    """Example 3: Track instant center migration through travel."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Instant Center Migration Analysis")
    print("=" * 70)

    chassis, knuckle_name = create_double_wishbone_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    # Calculate instant centers at different ride heights
    ride_height_offsets = [-40, -20, 0, 20, 40]

    print("\nAnalyzing instant center migration through suspension travel:")
    print("This shows how roll center height changes with ride height\n")

    for offset in ride_height_offsets:
        # Solve to new ride height
        solver.solve_for_heave(knuckle_name, offset, unit='mm')

        # Calculate instant centers at this position
        # Use small range around current position
        local_offsets = [-5, 0, 5]
        result = solver.calculate_instant_centers(knuckle_name, z_offsets=local_offsets)

        rc_height = result['roll_center'][2]
        pc_height = result['pitch_center'][2]

        print(f"Ride height {offset:+3.0f} mm:")
        print(f"  Roll center height:  {rc_height:6.2f} mm")
        print(f"  Pitch center height: {pc_height:6.2f} mm")


def example_unit_conversion():
    """Example 4: Using different units."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Unit Conversion")
    print("=" * 70)

    chassis, knuckle_name = create_double_wishbone_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    # Calculate in millimeters
    z_offsets_mm = [0, 5, 10, -5, -10]
    result_mm = solver.calculate_instant_centers(knuckle_name, z_offsets=z_offsets_mm, unit='mm')

    # Calculate in meters (with equivalent offsets)
    z_offsets_m = [z / 1000.0 for z in z_offsets_mm]
    result_m = solver.calculate_instant_centers(knuckle_name, z_offsets=z_offsets_m, unit='m')

    print("\nInstant centers calculated in different units:")
    print(f"\nMillimeters:")
    print(f"  Roll center:  [{result_mm['roll_center'][0]:.2f}, {result_mm['roll_center'][1]:.2f}, {result_mm['roll_center'][2]:.2f}] mm")
    print(f"  Pitch center: [{result_mm['pitch_center'][0]:.2f}, {result_mm['pitch_center'][1]:.2f}, {result_mm['pitch_center'][2]:.2f}] mm")

    print(f"\nMeters:")
    print(f"  Roll center:  [{result_m['roll_center'][0]:.6f}, {result_m['roll_center'][1]:.6f}, {result_m['roll_center'][2]:.6f}] m")
    print(f"  Pitch center: [{result_m['pitch_center'][0]:.6f}, {result_m['pitch_center'][1]:.6f}, {result_m['pitch_center'][2]:.6f}] m")

    # Verify conversion
    conversion_check = np.allclose(result_mm['roll_center'] / 1000.0, result_m['roll_center'])
    print(f"\nUnit conversion verified: {'✓' if conversion_check else '✗'}")


def example_design_interpretation():
    """Example 5: Interpreting results for suspension design."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Design Interpretation")
    print("=" * 70)

    chassis, knuckle_name = create_double_wishbone_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    result = solver.calculate_instant_centers(knuckle_name)

    roll_center = result['roll_center']
    pitch_center = result['pitch_center']

    print("\nSuspension Geometry Analysis:")
    print("-" * 70)

    # Roll center height analysis
    rc_height = roll_center[2]
    print(f"\nRoll Center Height: {rc_height:.2f} mm")
    if rc_height < 50:
        print("  → Low roll center: Better ride quality, less jacking")
    elif rc_height < 150:
        print("  → Medium roll center: Balanced handling")
    else:
        print("  → High roll center: More responsive turn-in, risk of jacking")

    # Roll center lateral offset
    rc_lateral = roll_center[1]
    print(f"\nRoll Center Lateral Offset: {rc_lateral:.2f} mm")
    if abs(rc_lateral) < 100:
        print("  → Near centerline: Symmetric roll behavior")
    else:
        print("  → Offset from centerline: Asymmetric roll stiffness")

    # Pitch center height
    pc_height = pitch_center[2]
    print(f"\nPitch Center Height: {pc_height:.2f} mm")
    if pc_height < 100:
        print("  → Low pitch center: Less anti-dive")
    else:
        print("  → Elevated pitch center: More anti-dive effect")

    # Fit quality assessment
    print(f"\nFit Quality Assessment:")
    roll_quality = result['roll_fit_quality']
    pitch_quality = result['pitch_fit_quality']

    print(f"  Roll fit quality: {roll_quality:.4f}")
    if roll_quality < 0.01:
        print("  → Excellent: Suspension follows near-circular arc")
    elif roll_quality < 0.05:
        print("  → Good: Acceptable approximation")
    else:
        print("  → Poor: Instant center may vary significantly through travel")

    print(f"\n  Pitch fit quality: {pitch_quality:.4f}")
    if pitch_quality < 0.01:
        print("  → Excellent: Pitch motion is near-circular")
    elif pitch_quality < 0.05:
        print("  → Good: Acceptable for most applications")
    else:
        print("  → Poor: Significant deviation from circular motion")


if __name__ == "__main__":
    print("=" * 70)
    print("KINEMATIC SOLVER - INSTANT CENTER EXAMPLES")
    print("=" * 70)

    # Run all examples
    example_basic_instant_centers()
    example_custom_travel_range()
    example_instant_center_migration()
    example_unit_conversion()
    example_design_interpretation()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Instant centers describe the geometric roll and pitch behavior")
    print("2. Roll center height affects roll stiffness and jacking forces")
    print("3. Pitch center height influences anti-dive characteristics")
    print("4. Fit quality indicates how well the suspension follows a circular arc")
    print("5. Instant centers can migrate through suspension travel")
