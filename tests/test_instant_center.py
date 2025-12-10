"""
Integration test for instant center analysis.

Tests the calculate_instant_centers method with constraint-based kinematics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.corner_solver import CornerSolver
from pysuspension.suspension_link import SuspensionLink
from pysuspension.attachment_point import AttachmentPoint


def create_simple_suspension():
    """
    Create a simple double wishbone suspension for testing.

    Returns:
        CornerSolver with configured suspension geometry
    """
    solver = CornerSolver("test_corner")

    # Upper control arm links
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

    # Lower control arm links
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

    # Wheel center (on knuckle, between ball joints)
    wheel_center = AttachmentPoint("wheel_center", [1400, 750, 390], unit='mm')

    # Add links to solver
    solver.add_link(upper_front_link, end1_is_chassis=True, end2_is_chassis=False)
    solver.add_link(upper_rear_link, end1_is_chassis=True, end2_is_chassis=False)
    solver.add_link(lower_front_link, end1_is_chassis=True, end2_is_chassis=False)
    solver.add_link(lower_rear_link, end1_is_chassis=True, end2_is_chassis=False)

    # Connect wheel center to ball joints (simulates knuckle)
    upper_ball_joint = upper_front_link.endpoint2
    lower_ball_joint = lower_front_link.endpoint2

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

    solver.add_link(upper_to_wheel, end1_is_chassis=False, end2_is_chassis=False)
    solver.add_link(lower_to_wheel, end1_is_chassis=False, end2_is_chassis=False)

    # Set wheel center
    solver.set_wheel_center(wheel_center)

    # Solve initial configuration
    solver.save_initial_state()
    result = solver.solve()

    return solver


def test_instant_center_basic():
    """Test instant center calculation with double wishbone suspension."""
    print("=" * 70)
    print("TEST: Basic Instant Center Calculation")
    print("=" * 70)

    # Create suspension with constraint-based linkage
    solver = create_simple_suspension()

    print(f"\nSuspension configuration:")
    print(f"  Solver: {solver}")
    print(f"  Wheel center: {solver.wheel_center.position} mm")
    print(f"  Number of constraints: {len(solver.constraints)}")

    # Calculate instant centers with default z offsets
    print("\n--- Calculating Instant Centers ---")
    print("Using default z_offsets: [0, 5, 10, -5, -10] mm")

    result = solver.calculate_instant_centers(unit='mm')

    print(f"\n--- Results ---")
    print(f"Roll center:  {result['roll_center']} mm")
    print(f"Pitch center: {result['pitch_center']} mm")
    print(f"Roll radius:  {result['roll_radius']:.2f} mm")
    print(f"Pitch radius: {result['pitch_radius']:.2f} mm")
    print(f"Roll fit quality:  {result['roll_fit_quality']:.10f}")
    print(f"Pitch fit quality: {result['pitch_fit_quality']:.10f}")

    # Verify contact points were captured
    assert result['contact_points'].shape == (5, 3), "Should have 5 contact points"
    print(f"\nCaptured {len(result['contact_points'])} contact points")

    # Verify wheel centers were captured
    assert result['wheel_centers'].shape == (5, 3), "Should have 5 wheel center positions"
    print(f"Captured {len(result['wheel_centers'])} wheel center positions")

    # Check solver errors
    print(f"\nSolver RMS errors at each position:")
    for i, err in enumerate(result['solve_errors']):
        print(f"  Position {i}: {err:.6f} mm")

    max_solve_error = max(result['solve_errors'])
    assert max_solve_error < 0.01, f"Solver error too large: {max_solve_error}"
    print(f"✓ All solver errors < 0.01mm")

    # Verify suspension returns to original configuration
    final_wheel_pos = solver.wheel_center.position.copy()
    initial_wheel_pos = np.array([1400.0, 750.0, 390.0])
    assert np.allclose(final_wheel_pos, initial_wheel_pos, atol=1e-6), \
        "Suspension should return to original position"
    print("\n✓ Suspension correctly returned to original position")

    print("\n✓ Basic instant center test passed!\n")


def test_instant_center_custom_offsets():
    """Test instant center calculation with custom z offsets."""
    print("=" * 70)
    print("TEST: Instant Center with Custom Z Offsets")
    print("=" * 70)

    # Create suspension
    solver = create_simple_suspension()

    # Use larger range of motion
    z_offsets = [-20, -10, 0, 10, 20, 30]
    print(f"\nUsing custom z_offsets: {z_offsets} mm")

    result = solver.calculate_instant_centers(
        z_offsets=z_offsets,
        unit='mm'
    )

    print(f"\n--- Results ---")
    print(f"Roll center:  {result['roll_center']} mm")
    print(f"Pitch center: {result['pitch_center']} mm")
    print(f"Number of contact points: {len(result['contact_points'])}")

    # Verify correct number of points
    assert len(result['contact_points']) == len(z_offsets), \
        f"Should have {len(z_offsets)} contact points"

    print("\n✓ Custom offset test passed!\n")


def test_instant_center_units():
    """Test instant center calculation with different units."""
    print("=" * 70)
    print("TEST: Instant Center with Different Units")
    print("=" * 70)

    # Create suspension
    solver = create_simple_suspension()

    # Test with millimeter offsets
    print("\n--- Test 1: mm offsets, mm output ---")
    result_mm = solver.calculate_instant_centers(
        z_offsets=[0, 5, 10, -5, -10],
        unit='mm'
    )
    print(f"Roll center (mm): {result_mm['roll_center']}")
    print(f"Roll radius (mm): {result_mm['roll_radius']:.2f}")

    # Reset suspension for second test
    solver.reset_to_initial_state()

    # Test with meter offsets
    print("\n--- Test 2: m offsets, m output ---")
    result_m = solver.calculate_instant_centers(
        z_offsets=[0, 0.005, 0.010, -0.005, -0.010],
        unit='m'
    )
    print(f"Roll center (m): {result_m['roll_center']}")
    print(f"Roll radius (m): {result_m['roll_radius']:.6f}")

    # Results should be equivalent (allowing for floating point precision)
    roll_center_mm_to_m = result_mm['roll_center'] / 1000.0
    assert np.allclose(roll_center_mm_to_m, result_m['roll_center'], rtol=1e-3), \
        "Results should be equivalent across units"

    print("\n✓ Units test passed!\n")


def test_instant_center_error_handling():
    """Test error handling for invalid inputs."""
    print("=" * 70)
    print("TEST: Instant Center Error Handling")
    print("=" * 70)

    # Test without wheel_center set
    print("\n--- Testing missing wheel_center ---")
    solver_no_wheel = CornerSolver("error_test")
    try:
        solver_no_wheel.calculate_instant_centers()
        assert False, "Should raise ValueError when wheel_center not set"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Create properly configured suspension
    solver = create_simple_suspension()

    # Test insufficient z_offsets
    print("\n--- Testing insufficient z_offsets ---")
    try:
        solver.calculate_instant_centers(z_offsets=[0, 5])
        assert False, "Should raise ValueError for insufficient offsets"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    print("\n✓ Error handling test passed!\n")


def test_instant_center_realistic_suspension():
    """Test with realistic suspension geometry using constraint-based kinematics."""
    print("=" * 70)
    print("TEST: Instant Center with Realistic Suspension Arc")
    print("=" * 70)

    # Create suspension with constraint-based linkage
    # This will produce realistic kinematic motion through the control arms
    solver = create_simple_suspension()

    # Use a finer set of measurements
    z_offsets = np.linspace(-15, 15, 7).tolist()
    print(f"\nUsing {len(z_offsets)} measurement points")
    print(f"Z offset range: {z_offsets[0]:.1f} to {z_offsets[-1]:.1f} mm")

    result = solver.calculate_instant_centers(
        z_offsets=z_offsets,
        unit='mm'
    )

    print(f"\n--- Results ---")
    print(f"Roll center:  [{result['roll_center'][0]:.2f}, "
          f"{result['roll_center'][1]:.2f}, {result['roll_center'][2]:.2f}] mm")
    print(f"Pitch center: [{result['pitch_center'][0]:.2f}, "
          f"{result['pitch_center'][1]:.2f}, {result['pitch_center'][2]:.2f}] mm")
    print(f"Roll radius:  {result['roll_radius']:.2f} mm")
    print(f"Pitch radius: {result['pitch_radius']:.2f} mm")

    # Verify all required keys are present
    required_keys = [
        'roll_center', 'pitch_center', 'roll_radius', 'pitch_radius',
        'contact_points', 'wheel_centers', 'roll_fit_quality', 'pitch_fit_quality',
        'roll_residuals', 'pitch_residuals', 'solve_errors'
    ]

    for key in required_keys:
        assert key in result, f"Result missing required key: {key}"

    print(f"\n✓ All required result keys present")
    print(f"✓ Fit quality metrics:")
    print(f"  Roll:  {result['roll_fit_quality']:.10f}")
    print(f"  Pitch: {result['pitch_fit_quality']:.10f}")

    # Verify solver converged for all positions
    max_solve_error = max(result['solve_errors'])
    print(f"✓ Max solver error: {max_solve_error:.6f} mm")
    assert max_solve_error < 0.01, f"Solver error too large: {max_solve_error}"

    print("\n✓ Realistic suspension test passed!\n")


def run_all_tests():
    """Run all instant center integration tests."""
    print("\n" + "=" * 70)
    print("INSTANT CENTER INTEGRATION TEST SUITE")
    print("=" * 70 + "\n")

    test_instant_center_basic()
    test_instant_center_custom_offsets()
    test_instant_center_units()
    test_instant_center_error_handling()
    test_instant_center_realistic_suspension()

    print("=" * 70)
    print("ALL INSTANT CENTER INTEGRATION TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
