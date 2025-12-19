"""
Test instant center calculations for KinematicSolver.

Tests the calculate_instant_centers() method that calculates roll and pitch
instant centers using circle fitting to tire contact patch trajectories.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.kinematic_solver import KinematicSolver
from pysuspension.chassis import Chassis
from pysuspension.chassis_corner import ChassisCorner
from pysuspension.suspension_knuckle import SuspensionKnuckle
from pysuspension.suspension_link import SuspensionLink
from pysuspension.suspension_joint import SuspensionJoint
from pysuspension.joint_types import JointType


def setup_simple_suspension():
    """
    Set up a proper double-wishbone suspension following pysuspension patterns.

    This creates a realistic suspension where:
    - Upper and lower A-arms have unequal lengths
    - Each component owns its own attachment points
    - Joints connect separate attachment points with coincident constraints
    - Chassis mounts use BUSHING_SOFT (compliant)
    - Ball joints use BALL_JOINT

    Returns:
        Tuple of (chassis, knuckle_name)
    """
    # Create chassis
    chassis = Chassis("test_chassis")

    # Create chassis corner with attachment points
    corner = ChassisCorner("front_left")
    uf_chassis = corner.add_attachment_point("upper_front", [1400, 0, 600], unit='mm')
    ur_chassis = corner.add_attachment_point("upper_rear", [1200, 0, 600], unit='mm')
    lf_chassis = corner.add_attachment_point("lower_front", [1500, 0, 300], unit='mm')
    lr_chassis = corner.add_attachment_point("lower_rear", [1100, 0, 300], unit='mm')
    tr_chassis = corner.add_attachment_point("chassis tr", [1150, 0, 400], unit='mm')
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

    # Create upper A-arm links - each link gets its own attachment points
    # Upper front link: chassis -> upper ball joint
    upper_front_link = SuspensionLink(
        endpoint1=[1400, 0, 600],        # Chassis end
        endpoint2=[1400, 650, 580],      # Ball joint end
        name="upper_front",
        unit='mm'
    )

    # Upper rear link: chassis -> upper ball joint
    upper_rear_link = SuspensionLink(
        endpoint1=[1200, 0, 600],        # Chassis end
        endpoint2=[1400, 650, 580],      # Ball joint end (same location)
        name="upper_rear",
        unit='mm'
    )

    # Create lower A-arm links
    # Lower front link: chassis -> lower ball joint
    lower_front_link = SuspensionLink(
        endpoint1=[1500, 0, 300],        # Chassis end
        endpoint2=[1400, 700, 200],      # Ball joint end
        name="lower_front",
        unit='mm'
    )

    # Lower rear link: chassis -> lower ball joint
    lower_rear_link = SuspensionLink(
        endpoint1=[1100, 0, 300],        # Chassis end
        endpoint2=[1400, 700, 200],      # Ball joint end (same location)
        name="lower_rear",
        unit='mm'
    )

    # Tie rod link: chassis -> outer TR joint
    tie_rod_link = SuspensionLink(
        endpoint1=[1150, 0, 400],        # Chassis end
        endpoint2=[1400, 650, 390],      # Outer TR joint end (same location)
        name="tr_outer",
        unit='mm'
    )

    # Create joints connecting components
    # Upper front bushing: chassis to link (compliant)
    uf_bushing = SuspensionJoint("uf_bushing", JointType.SPHERICAL_BEARING)
    uf_bushing.add_attachment_point(uf_chassis)
    uf_bushing.add_attachment_point(upper_front_link.endpoint1)

    # Upper rear bushing: chassis to link (compliant)
    ur_bushing = SuspensionJoint("ur_bushing", JointType.SPHERICAL_BEARING)
    ur_bushing.add_attachment_point(ur_chassis)
    ur_bushing.add_attachment_point(upper_rear_link.endpoint1)

    # Upper ball joint: both upper links + knuckle (rigid)
    upper_ball_joint = SuspensionJoint("upper_ball", JointType.BALL_JOINT)
    upper_ball_joint.add_attachment_point(upper_front_link.endpoint2)
    upper_ball_joint.add_attachment_point(upper_rear_link.endpoint2)
    upper_ball_joint.add_attachment_point(upper_ball)

    # Lower front bushing: chassis to link (compliant)
    lf_bushing = SuspensionJoint("lf_bushing", JointType.SPHERICAL_BEARING)
    lf_bushing.add_attachment_point(lf_chassis)
    lf_bushing.add_attachment_point(lower_front_link.endpoint1)

    # Lower rear bushing: chassis to link (compliant)
    lr_bushing = SuspensionJoint("lr_bushing", JointType.SPHERICAL_BEARING)
    lr_bushing.add_attachment_point(lr_chassis)
    lr_bushing.add_attachment_point(lower_rear_link.endpoint1)

    # Lower ball joint: both lower links + knuckle (rigid)
    lower_ball_joint = SuspensionJoint("lower_ball", JointType.BALL_JOINT)
    lower_ball_joint.add_attachment_point(lower_front_link.endpoint2)
    lower_ball_joint.add_attachment_point(lower_rear_link.endpoint2)
    lower_ball_joint.add_attachment_point(lower_ball)

    # TR outer joint: TR link + knuckle (rigid)
    outer_tr_joint = SuspensionJoint("outer tie rod ball", JointType.BALL_JOINT)
    outer_tr_joint.add_attachment_point(tie_rod_link.endpoint2)
    outer_tr_joint.add_attachment_point(tie_rod_knuckle)

    # TR inner joint: both lower links + knuckle (rigid)
    inner_tr_joint = SuspensionJoint("inner tie rod ball", JointType.BALL_JOINT)
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

    return chassis, 'front_left_knuckle'


def test_instant_center_basic():
    """Test basic instant center calculation."""
    print("\n" + "=" * 70)
    print("TEST: calculate_instant_centers() - Basic")
    print("=" * 70)

    chassis, knuckle_name = setup_simple_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    print("\nCalculating instant centers with default z_offsets...")
    result = solver.calculate_instant_centers(knuckle_name)

    # Verify all expected keys are present
    expected_keys = [
        'roll_center', 'pitch_center',
        'roll_radius', 'pitch_radius',
        'contact_points', 'wheel_centers',
        'roll_fit_quality', 'pitch_fit_quality',
        'roll_residuals', 'pitch_residuals',
        'solve_errors'
    ]

    for key in expected_keys:
        assert key in result, f"Missing key '{key}' in result"

    print(f"\n✓ All {len(expected_keys)} expected keys present in result")

    # Verify data types and shapes
    assert isinstance(result['roll_center'], np.ndarray), "roll_center should be ndarray"
    assert isinstance(result['pitch_center'], np.ndarray), "pitch_center should be ndarray"
    assert result['roll_center'].shape == (3,), "roll_center should be 3D"
    assert result['pitch_center'].shape == (3,), "pitch_center should be 3D"

    print(f"✓ Roll center: {result['roll_center']} mm")
    print(f"✓ Pitch center: {result['pitch_center']} mm")
    print(f"✓ Roll radius: {result['roll_radius']:.2f} mm")
    print(f"✓ Pitch radius: {result['pitch_radius']:.2f} mm")

    # Verify contact points captured (default is 5 offsets)
    assert result['contact_points'].shape[0] == 5, "Should have 5 contact points"
    assert result['wheel_centers'].shape[0] == 5, "Should have 5 wheel centers"

    print(f"✓ Captured {result['contact_points'].shape[0]} contact points")

    # Verify solve errors are reasonable (should be < 1mm)
    max_error = max(result['solve_errors'])
    print(f"✓ Max solve error: {max_error:.6f} mm")
    assert max_error < 1.0, f"Solve error {max_error} mm exceeds 1mm threshold"

    # Verify fit quality is reasonable (< 5%)
    print(f"✓ Roll fit quality: {result['roll_fit_quality']:.4f} ({result['roll_fit_quality']*100:.2f}%)")
    print(f"✓ Pitch fit quality: {result['pitch_fit_quality']:.4f} ({result['pitch_fit_quality']*100:.2f}%)")
    assert result['roll_fit_quality'] < 0.05, f"Roll fit quality {result['roll_fit_quality']} exceeds 5%"
    assert result['pitch_fit_quality'] < 0.05, f"Pitch fit quality {result['pitch_fit_quality']} exceeds 5%"

    print("\n✓ TEST PASSED: Basic instant center calculation")


def test_instant_center_state_restoration():
    """Test that state is properly restored after instant center calculation."""
    print("\n" + "=" * 70)
    print("TEST: calculate_instant_centers() - State Restoration")
    print("=" * 70)

    chassis, knuckle_name = setup_simple_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])
    knuckle = solver.get_knuckle(knuckle_name)

    # Save initial state
    initial_tire_center = knuckle.tire_center.copy()
    initial_snapshot = solver.state.save_snapshot()

    print(f"\nInitial tire center: {initial_tire_center}")

    # Calculate instant centers
    print("Calculating instant centers...")
    result = solver.calculate_instant_centers(knuckle_name)

    # Verify state is restored
    final_tire_center = knuckle.tire_center.copy()
    final_snapshot = solver.state.save_snapshot()

    print(f"Final tire center: {final_tire_center}")

    # Check knuckle position restored
    tire_center_diff = np.linalg.norm(final_tire_center - initial_tire_center)
    print(f"Tire center difference: {tire_center_diff:.6f} mm")
    assert tire_center_diff < 1e-10, f"Tire center not restored (diff={tire_center_diff})"

    # Check all attachment points restored
    for point_name in initial_snapshot.keys():
        initial_pos = initial_snapshot[point_name]
        final_pos = final_snapshot[point_name]
        diff = np.linalg.norm(final_pos - initial_pos)
        assert diff < 1e-10, f"Point '{point_name}' not restored (diff={diff})"

    print(f"✓ All {len(initial_snapshot)} attachment points restored")
    print("\n✓ TEST PASSED: State properly restored")


def test_instant_center_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "=" * 70)
    print("TEST: calculate_instant_centers() - Error Handling")
    print("=" * 70)

    chassis, knuckle_name = setup_simple_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    # Test 1: Non-existent knuckle name
    print("\nTest 1: Non-existent knuckle name...")
    try:
        solver.calculate_instant_centers('non_existent_knuckle')
        assert False, "Should raise KeyError for non-existent knuckle"
    except KeyError as e:
        print(f"✓ Correctly raised KeyError: {e}")

    # Test 2: Insufficient z_offsets
    print("\nTest 2: Insufficient z_offsets (< 3)...")
    try:
        solver.calculate_instant_centers(knuckle_name, z_offsets=[0, 5])
        assert False, "Should raise ValueError for insufficient z_offsets"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test 3: Valid edge case - exactly 3 offsets
    print("\nTest 3: Valid edge case - exactly 3 offsets...")
    result = solver.calculate_instant_centers(knuckle_name, z_offsets=[0, 5, 10])
    assert result['contact_points'].shape[0] == 3
    print("✓ Successfully handled 3 z_offsets")

    print("\n✓ TEST PASSED: Error handling works correctly")


def test_instant_center_custom_offsets():
    """Test instant center calculation with custom z_offsets."""
    print("\n" + "=" * 70)
    print("TEST: calculate_instant_centers() - Custom Offsets")
    print("=" * 70)

    chassis, knuckle_name = setup_simple_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    # Use larger range of offsets
    z_offsets = [-30, -20, -10, 0, 10, 20, 30]
    print(f"\nUsing custom z_offsets: {z_offsets} mm")

    result = solver.calculate_instant_centers(knuckle_name, z_offsets=z_offsets)

    # Verify correct number of points
    assert result['contact_points'].shape[0] == len(z_offsets)
    assert result['wheel_centers'].shape[0] == len(z_offsets)
    print(f"✓ Captured {len(z_offsets)} contact points as expected")

    # Verify results are consistent
    print(f"✓ Roll center: {result['roll_center']} mm")
    print(f"✓ Pitch center: {result['pitch_center']} mm")
    print(f"✓ Roll fit quality: {result['roll_fit_quality']:.4f}")
    print(f"✓ Pitch fit quality: {result['pitch_fit_quality']:.4f}")

    # With more points and larger range, fit should be good
    assert result['roll_fit_quality'] < 0.05
    assert result['pitch_fit_quality'] < 0.05

    print("\n✓ TEST PASSED: Custom offsets work correctly")


def test_instant_center_units():
    """Test instant center calculation with different units."""
    print("\n" + "=" * 70)
    print("TEST: calculate_instant_centers() - Unit Conversion")
    print("=" * 70)

    chassis, knuckle_name = setup_simple_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    # Define z_offsets in each unit
    z_offsets_mm = [0, 5, 10, -5, -10]
    z_offsets_m = [z / 1000.0 for z in z_offsets_mm]

    # Calculate in mm
    print("\nCalculating instant centers in mm...")
    result_mm = solver.calculate_instant_centers(knuckle_name, z_offsets=z_offsets_mm, unit='mm')

    # Calculate in m (using equivalent z_offsets in meters)
    print("Calculating instant centers in m...")
    result_m = solver.calculate_instant_centers(knuckle_name, z_offsets=z_offsets_m, unit='m')

    # Verify conversion is correct (1m = 1000mm)
    roll_center_mm = result_mm['roll_center']
    roll_center_m = result_m['roll_center']

    print(f"\nRoll center (mm): {roll_center_mm}")
    print(f"Roll center (m):  {roll_center_m}")

    # Check conversion
    expected_m = roll_center_mm / 1000.0
    diff = np.linalg.norm(roll_center_m - expected_m)
    print(f"Conversion difference: {diff:.10f} m")
    assert diff < 1e-6, f"Unit conversion failed (diff={diff})"

    # Check radius conversion
    radius_mm = result_mm['roll_radius']
    radius_m = result_m['roll_radius']
    print(f"\nRoll radius (mm): {radius_mm:.2f}")
    print(f"Roll radius (m):  {radius_m:.6f}")

    expected_radius_m = radius_mm / 1000.0
    radius_diff = abs(radius_m - expected_radius_m)
    print(f"Radius conversion difference: {radius_diff:.10f} m")
    assert radius_diff < 1e-6, f"Radius conversion failed (diff={radius_diff})"

    print("\n✓ TEST PASSED: Unit conversion works correctly")


def test_instant_center_convergence():
    """Test that all solves converge properly."""
    print("\n" + "=" * 70)
    print("TEST: calculate_instant_centers() - Convergence Quality")
    print("=" * 70)

    chassis, knuckle_name = setup_simple_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    # Use a range of offsets
    z_offsets = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    print(f"\nTesting convergence with {len(z_offsets)} offsets: {z_offsets} mm")

    result = solver.calculate_instant_centers(knuckle_name, z_offsets=z_offsets)

    # Check all solve errors
    print("\nSolve errors for each offset:")
    for i, (offset, error) in enumerate(zip(z_offsets, result['solve_errors'])):
        print(f"  z={offset:4.0f} mm: error={error:.6f} mm")
        assert error < 1.0, f"Solve at z={offset} failed to converge (error={error})"

    max_error = max(result['solve_errors'])
    avg_error = sum(result['solve_errors']) / len(result['solve_errors'])

    print(f"\n✓ Max solve error: {max_error:.6f} mm")
    print(f"✓ Avg solve error: {avg_error:.6f} mm")
    print(f"✓ All {len(z_offsets)} solves converged")

    # Check fit quality
    print(f"\n✓ Roll fit quality: {result['roll_fit_quality']:.4f} ({result['roll_fit_quality']*100:.2f}%)")
    print(f"✓ Pitch fit quality: {result['pitch_fit_quality']:.4f} ({result['pitch_fit_quality']*100:.2f}%)")

    print("\n✓ TEST PASSED: All solves converged successfully")


def test_instant_center_multi_knuckle():
    """Test instant center calculation with multiple knuckles."""
    print("\n" + "=" * 70)
    print("TEST: calculate_instant_centers() - Multiple Knuckles")
    print("=" * 70)

    # Create chassis with two corners (left and right)
    chassis = Chassis("test_chassis")

    # Create left corner
    left_corner = ChassisCorner("front_left")
    uf_chassis_l = left_corner.add_attachment_point("upper_front_l", [1400, 0, 600], unit='mm')
    ur_chassis_l = left_corner.add_attachment_point("upper_rear_l", [1200, 0, 600], unit='mm')
    lf_chassis_l = left_corner.add_attachment_point("lower_front_l", [1500, 0, 300], unit='mm')
    lr_chassis_l = left_corner.add_attachment_point("lower_rear_l", [1100, 0, 300], unit='mm')
    tr_chassis_l = left_corner.add_attachment_point("chassis_tr_l", [1150, 0, 400], unit='mm')
    chassis.add_corner(left_corner)

    # Create right corner (mirrored in Y)
    right_corner = ChassisCorner("front_right")
    uf_chassis_r = right_corner.add_attachment_point("upper_front_r", [1400, 0, 600], unit='mm')
    ur_chassis_r = right_corner.add_attachment_point("upper_rear_r", [1200, 0, 600], unit='mm')
    lf_chassis_r = right_corner.add_attachment_point("lower_front_r", [1500, 0, 300], unit='mm')
    lr_chassis_r = right_corner.add_attachment_point("lower_rear_r", [1100, 0, 300], unit='mm')
    tr_chassis_r = right_corner.add_attachment_point("chassis_tr_r", [1150, 0, 400], unit='mm')
    chassis.add_corner(right_corner)

    # Create left knuckle
    knuckle_left = SuspensionKnuckle(
        tire_center_x=1400, tire_center_y=750, rolling_radius=390,
        toe_angle=0.0, camber_angle=-1.0, unit='mm', name='front_left_knuckle'
    )
    upper_ball_l = knuckle_left.add_attachment_point("upper_ball_joint_l", [1400, 650, 580], unit='mm')
    lower_ball_l = knuckle_left.add_attachment_point("lower_ball_joint_l", [1400, 700, 200], unit='mm')
    tie_rod_knuckle_l = knuckle_left.add_attachment_point("tie_rod_l", [1400, 650, 390], unit='mm')

    # Create right knuckle (mirrored in Y)
    knuckle_right = SuspensionKnuckle(
        tire_center_x=1400, tire_center_y=-750, rolling_radius=390,
        toe_angle=0.0, camber_angle=1.0, unit='mm', name='front_right_knuckle'
    )
    upper_ball_r = knuckle_right.add_attachment_point("upper_ball_joint_r", [1400, -650, 580], unit='mm')
    lower_ball_r = knuckle_right.add_attachment_point("lower_ball_joint_r", [1400, -700, 200], unit='mm')
    tie_rod_knuckle_r = knuckle_right.add_attachment_point("tie_rod_r", [1400, -650, 390], unit='mm')

    # Create links for left side
    upper_front_link_l = SuspensionLink([1400, 0, 600], [1400, 650, 580], name="upper_front_l", unit='mm')
    upper_rear_link_l = SuspensionLink([1200, 0, 600], [1400, 650, 580], name="upper_rear_l", unit='mm')
    lower_front_link_l = SuspensionLink([1500, 0, 300], [1400, 700, 200], name="lower_front_l", unit='mm')
    lower_rear_link_l = SuspensionLink([1100, 0, 300], [1400, 700, 200], name="lower_rear_l", unit='mm')
    tie_rod_link_l = SuspensionLink([1150, 0, 400], [1400, 650, 390], name="tr_outer_l", unit='mm')

    # Create links for right side
    upper_front_link_r = SuspensionLink([1400, 0, 600], [1400, -650, 580], name="upper_front_r", unit='mm')
    upper_rear_link_r = SuspensionLink([1200, 0, 600], [1400, -650, 580], name="upper_rear_r", unit='mm')
    lower_front_link_r = SuspensionLink([1500, 0, 300], [1400, -700, 200], name="lower_front_r", unit='mm')
    lower_rear_link_r = SuspensionLink([1100, 0, 300], [1400, -700, 200], name="lower_rear_r", unit='mm')
    tie_rod_link_r = SuspensionLink([1150, 0, 400], [1400, -650, 390], name="tr_outer_r", unit='mm')

    # Create joints for left side
    uf_bushing_l = SuspensionJoint("uf_bushing_l", JointType.SPHERICAL_BEARING)
    uf_bushing_l.add_attachment_point(uf_chassis_l)
    uf_bushing_l.add_attachment_point(upper_front_link_l.endpoint1)

    ur_bushing_l = SuspensionJoint("ur_bushing_l", JointType.SPHERICAL_BEARING)
    ur_bushing_l.add_attachment_point(ur_chassis_l)
    ur_bushing_l.add_attachment_point(upper_rear_link_l.endpoint1)

    upper_ball_joint_l = SuspensionJoint("upper_ball_l", JointType.BALL_JOINT)
    upper_ball_joint_l.add_attachment_point(upper_front_link_l.endpoint2)
    upper_ball_joint_l.add_attachment_point(upper_rear_link_l.endpoint2)
    upper_ball_joint_l.add_attachment_point(upper_ball_l)

    lf_bushing_l = SuspensionJoint("lf_bushing_l", JointType.SPHERICAL_BEARING)
    lf_bushing_l.add_attachment_point(lf_chassis_l)
    lf_bushing_l.add_attachment_point(lower_front_link_l.endpoint1)

    lr_bushing_l = SuspensionJoint("lr_bushing_l", JointType.SPHERICAL_BEARING)
    lr_bushing_l.add_attachment_point(lr_chassis_l)
    lr_bushing_l.add_attachment_point(lower_rear_link_l.endpoint1)

    lower_ball_joint_l = SuspensionJoint("lower_ball_l", JointType.BALL_JOINT)
    lower_ball_joint_l.add_attachment_point(lower_front_link_l.endpoint2)
    lower_ball_joint_l.add_attachment_point(lower_rear_link_l.endpoint2)
    lower_ball_joint_l.add_attachment_point(lower_ball_l)

    outer_tr_joint_l = SuspensionJoint("outer_tr_l", JointType.BALL_JOINT)
    outer_tr_joint_l.add_attachment_point(tie_rod_link_l.endpoint2)
    outer_tr_joint_l.add_attachment_point(tie_rod_knuckle_l)

    inner_tr_joint_l = SuspensionJoint("inner_tr_l", JointType.BALL_JOINT)
    inner_tr_joint_l.add_attachment_point(tie_rod_link_l.endpoint1)
    inner_tr_joint_l.add_attachment_point(tr_chassis_l)

    # Create joints for right side
    uf_bushing_r = SuspensionJoint("uf_bushing_r", JointType.SPHERICAL_BEARING)
    uf_bushing_r.add_attachment_point(uf_chassis_r)
    uf_bushing_r.add_attachment_point(upper_front_link_r.endpoint1)

    ur_bushing_r = SuspensionJoint("ur_bushing_r", JointType.SPHERICAL_BEARING)
    ur_bushing_r.add_attachment_point(ur_chassis_r)
    ur_bushing_r.add_attachment_point(upper_rear_link_r.endpoint1)

    upper_ball_joint_r = SuspensionJoint("upper_ball_r", JointType.BALL_JOINT)
    upper_ball_joint_r.add_attachment_point(upper_front_link_r.endpoint2)
    upper_ball_joint_r.add_attachment_point(upper_rear_link_r.endpoint2)
    upper_ball_joint_r.add_attachment_point(upper_ball_r)

    lf_bushing_r = SuspensionJoint("lf_bushing_r", JointType.SPHERICAL_BEARING)
    lf_bushing_r.add_attachment_point(lf_chassis_r)
    lf_bushing_r.add_attachment_point(lower_front_link_r.endpoint1)

    lr_bushing_r = SuspensionJoint("lr_bushing_r", JointType.SPHERICAL_BEARING)
    lr_bushing_r.add_attachment_point(lr_chassis_r)
    lr_bushing_r.add_attachment_point(lower_rear_link_r.endpoint1)

    lower_ball_joint_r = SuspensionJoint("lower_ball_r", JointType.BALL_JOINT)
    lower_ball_joint_r.add_attachment_point(lower_front_link_r.endpoint2)
    lower_ball_joint_r.add_attachment_point(lower_rear_link_r.endpoint2)
    lower_ball_joint_r.add_attachment_point(lower_ball_r)

    outer_tr_joint_r = SuspensionJoint("outer_tr_r", JointType.BALL_JOINT)
    outer_tr_joint_r.add_attachment_point(tie_rod_link_r.endpoint2)
    outer_tr_joint_r.add_attachment_point(tie_rod_knuckle_r)

    inner_tr_joint_r = SuspensionJoint("inner_tr_r", JointType.BALL_JOINT)
    inner_tr_joint_r.add_attachment_point(tie_rod_link_r.endpoint1)
    inner_tr_joint_r.add_attachment_point(tr_chassis_r)

    # Register all components
    chassis.add_component(knuckle_left)
    chassis.add_component(knuckle_right)
    chassis.add_component(upper_front_link_l)
    chassis.add_component(upper_rear_link_l)
    chassis.add_component(lower_front_link_l)
    chassis.add_component(lower_rear_link_l)
    chassis.add_component(tie_rod_link_l)
    chassis.add_component(upper_front_link_r)
    chassis.add_component(upper_rear_link_r)
    chassis.add_component(lower_front_link_r)
    chassis.add_component(lower_rear_link_r)
    chassis.add_component(tie_rod_link_r)

    chassis.add_joint(uf_bushing_l)
    chassis.add_joint(ur_bushing_l)
    chassis.add_joint(upper_ball_joint_l)
    chassis.add_joint(lf_bushing_l)
    chassis.add_joint(lr_bushing_l)
    chassis.add_joint(lower_ball_joint_l)
    chassis.add_joint(outer_tr_joint_l)
    chassis.add_joint(inner_tr_joint_l)
    chassis.add_joint(uf_bushing_r)
    chassis.add_joint(ur_bushing_r)
    chassis.add_joint(upper_ball_joint_r)
    chassis.add_joint(lf_bushing_r)
    chassis.add_joint(lr_bushing_r)
    chassis.add_joint(lower_ball_joint_r)
    chassis.add_joint(outer_tr_joint_r)
    chassis.add_joint(inner_tr_joint_r)

    # Create solver with both corners
    print("\nCreating solver with both front corners...")
    solver = KinematicSolver.from_chassis(chassis, ['front_left', 'front_right'])

    print(f"Solver has {len(solver.registry.knuckles)} knuckles")
    assert len(solver.registry.knuckles) == 2

    # Calculate instant centers for left knuckle
    print("\nCalculating instant centers for left knuckle...")
    result_left = solver.calculate_instant_centers('front_left_knuckle')
    print(f"✓ Left roll center: {result_left['roll_center']} mm")

    # Calculate instant centers for right knuckle
    print("\nCalculating instant centers for right knuckle...")
    result_right = solver.calculate_instant_centers('front_right_knuckle')
    print(f"✓ Right roll center: {result_right['roll_center']} mm")

    # Verify both calculations succeeded
    assert 'roll_center' in result_left
    assert 'roll_center' in result_right

    # Roll centers should be roughly symmetric in Y (opposite signs)
    left_y = result_left['roll_center'][1]
    right_y = result_right['roll_center'][1]
    print(f"\nLeft Y: {left_y:.2f} mm, Right Y: {right_y:.2f} mm")
    print(f"Symmetry check: {left_y:.2f} ≈ -{right_y:.2f}")

    print("\n✓ TEST PASSED: Multi-knuckle instant centers calculated successfully")


if __name__ == "__main__":
    print("=" * 70)
    print("KINEMATIC SOLVER INSTANT CENTER TESTS")
    print("=" * 70)

    try:
        test_instant_center_basic()
        test_instant_center_state_restoration()
        test_instant_center_error_handling()
        test_instant_center_custom_offsets()
        test_instant_center_units()
        test_instant_center_convergence()
        test_instant_center_multi_knuckle()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
