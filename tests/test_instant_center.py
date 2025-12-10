"""
Integration test for instant center analysis.

Tests the calculate_instant_centers method with SuspensionKnuckle.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.corner_solver import CornerSolver
from pysuspension.suspension_knuckle import SuspensionKnuckle


def test_instant_center_basic():
    """Test instant center calculation with a simple knuckle."""
    print("=" * 70)
    print("TEST: Basic Instant Center Calculation")
    print("=" * 70)

    # Create a suspension knuckle
    # Position at typical location: 1.5m forward, 0.75m lateral, 0.35m up
    knuckle = SuspensionKnuckle(
        tire_center_x=1500,  # mm
        tire_center_y=750,   # mm
        rolling_radius=350,  # mm
        toe_angle=0.0,       # degrees
        camber_angle=0.0,    # degrees
        unit='mm',
        name='front_left_knuckle'
    )

    print(f"\nKnuckle tire center: {knuckle.tire_center} mm")
    print(f"Initial contact patch: {knuckle.get_tire_contact_patch()} mm")

    # Create corner solver
    solver = CornerSolver("test_corner")

    # Calculate instant centers with default z offsets
    print("\n--- Calculating Instant Centers ---")
    print("Using default z_offsets: [0, 5, 10, -5, -10] mm")

    result = solver.calculate_instant_centers(knuckle, unit='mm')

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

    # For a knuckle that only moves vertically (no arc motion),
    # the instant centers should be very far away (large radius)
    # because the contact patch moves nearly straight up/down
    print(f"\nNote: Large radii indicate nearly linear vertical motion")
    print(f"(Expected for a knuckle moving purely in Z-axis)")

    # Verify knuckle returned to original position
    final_contact = knuckle.get_tire_contact_patch(unit='mm')
    initial_contact = np.array([1500.0, 750.0, 0.0])
    assert np.allclose(final_contact, initial_contact, atol=1e-10), \
        "Knuckle should return to original position"
    print("\n✓ Knuckle correctly returned to original position")

    print("\n✓ Basic instant center test passed!\n")


def test_instant_center_custom_offsets():
    """Test instant center calculation with custom z offsets."""
    print("=" * 70)
    print("TEST: Instant Center with Custom Z Offsets")
    print("=" * 70)

    # Create knuckle
    knuckle = SuspensionKnuckle(
        tire_center_x=1400,
        tire_center_y=700,
        rolling_radius=330,
        unit='mm'
    )

    solver = CornerSolver("custom_test")

    # Use larger range of motion
    z_offsets = [-20, -10, 0, 10, 20, 30]
    print(f"\nUsing custom z_offsets: {z_offsets} mm")

    result = solver.calculate_instant_centers(
        knuckle,
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

    # Create knuckle using meters
    knuckle = SuspensionKnuckle(
        tire_center_x=1.5,
        tire_center_y=0.75,
        rolling_radius=0.35,
        unit='m'
    )

    solver = CornerSolver("units_test")

    # Test with millimeter offsets
    print("\n--- Test 1: mm offsets, mm output ---")
    result_mm = solver.calculate_instant_centers(
        knuckle,
        z_offsets=[0, 5, 10, -5, -10],
        unit='mm'
    )
    print(f"Roll center (mm): {result_mm['roll_center']}")
    print(f"Roll radius (mm): {result_mm['roll_radius']:.2f}")

    # Test with meter offsets
    print("\n--- Test 2: m offsets, m output ---")
    result_m = solver.calculate_instant_centers(
        knuckle,
        z_offsets=[0, 0.005, 0.010, -0.005, -0.010],
        unit='m'
    )
    print(f"Roll center (m): {result_m['roll_center']}")
    print(f"Roll radius (m): {result_m['roll_radius']:.6f}")

    # Results should be equivalent (allowing for floating point precision)
    roll_center_mm_to_m = result_mm['roll_center'] / 1000.0
    assert np.allclose(roll_center_mm_to_m, result_m['roll_center'], rtol=1e-5), \
        "Results should be equivalent across units"

    print("\n✓ Units test passed!\n")


def test_instant_center_error_handling():
    """Test error handling for invalid inputs."""
    print("=" * 70)
    print("TEST: Instant Center Error Handling")
    print("=" * 70)

    knuckle = SuspensionKnuckle(
        tire_center_x=1500,
        tire_center_y=750,
        rolling_radius=350,
        unit='mm'
    )

    solver = CornerSolver("error_test")

    # Test insufficient z_offsets
    print("\n--- Testing insufficient z_offsets ---")
    try:
        solver.calculate_instant_centers(knuckle, z_offsets=[0, 5])
        assert False, "Should raise ValueError for insufficient offsets"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test invalid knuckle type
    print("\n--- Testing invalid knuckle type ---")
    try:
        solver.calculate_instant_centers("not a knuckle")
        assert False, "Should raise ValueError for invalid knuckle"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    print("\n✓ Error handling test passed!\n")


def test_instant_center_realistic_suspension():
    """Test with more realistic suspension geometry that produces an arc."""
    print("=" * 70)
    print("TEST: Instant Center with Realistic Suspension Arc")
    print("=" * 70)

    # For a more realistic test, we would need to simulate actual suspension
    # motion where the tire contact patch traces an arc. This would require
    # a full suspension linkage simulation.

    # For now, we can verify that the method works and returns reasonable values
    knuckle = SuspensionKnuckle(
        tire_center_x=1400,
        tire_center_y=700,
        rolling_radius=330,
        unit='mm'
    )

    solver = CornerSolver("realistic_test")

    # Use a finer set of measurements
    z_offsets = np.linspace(-15, 15, 7).tolist()
    print(f"\nUsing {len(z_offsets)} measurement points")
    print(f"Z offset range: {z_offsets[0]:.1f} to {z_offsets[-1]:.1f} mm")

    result = solver.calculate_instant_centers(
        knuckle,
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
        'contact_points', 'roll_fit_quality', 'pitch_fit_quality',
        'roll_residuals', 'pitch_residuals'
    ]

    for key in required_keys:
        assert key in result, f"Result missing required key: {key}"

    print(f"\n✓ All required result keys present")
    print(f"✓ Fit quality metrics:")
    print(f"  Roll:  {result['roll_fit_quality']:.10f}")
    print(f"  Pitch: {result['pitch_fit_quality']:.10f}")

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
