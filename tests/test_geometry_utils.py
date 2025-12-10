"""
Test geometric utility functions for suspension analysis.

Tests circle fitting, point projection, and instant center calculation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.geometry_utils import (
    project_points_to_plane,
    fit_circle_2d,
    calculate_instant_center_from_points
)


def test_project_points_to_plane():
    """Test projection of 3D points onto coordinate planes."""
    print("=" * 70)
    print("TEST: Point Projection")
    print("=" * 70)

    # Test data: 3D points
    points_3d = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])

    # Test YZ projection
    print("\n--- Testing YZ Plane Projection ---")
    projected_yz = project_points_to_plane(points_3d, 'YZ')
    expected_yz = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 9.0]])

    print(f"Input 3D points:\n{points_3d}")
    print(f"Projected to YZ:\n{projected_yz}")
    print(f"Expected:\n{expected_yz}")

    assert np.allclose(projected_yz, expected_yz), "YZ projection failed"
    print("✓ YZ projection correct")

    # Test XZ projection
    print("\n--- Testing XZ Plane Projection ---")
    projected_xz = project_points_to_plane(points_3d, 'XZ')
    expected_xz = np.array([[1.0, 3.0], [4.0, 6.0], [7.0, 9.0]])

    print(f"Projected to XZ:\n{projected_xz}")
    print(f"Expected:\n{expected_xz}")

    assert np.allclose(projected_xz, expected_xz), "XZ projection failed"
    print("✓ XZ projection correct")

    # Test case insensitivity
    print("\n--- Testing Case Insensitivity ---")
    projected_yz_lower = project_points_to_plane(points_3d, 'yz')
    assert np.allclose(projected_yz_lower, expected_yz), "Case insensitive projection failed"
    print("✓ Case insensitive projection works")

    # Test error handling
    print("\n--- Testing Error Handling ---")
    try:
        project_points_to_plane(points_3d, 'XY')
        assert False, "Should have raised ValueError for invalid plane"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for invalid plane: {e}")

    try:
        project_points_to_plane(np.array([[1, 2]]), 'YZ')
        assert False, "Should have raised ValueError for insufficient points"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError for insufficient points: {e}")

    print("\n✓ All projection tests passed!\n")


def test_fit_circle_2d_perfect():
    """Test circle fitting with perfect circular data."""
    print("=" * 70)
    print("TEST: Circle Fitting - Perfect Circle")
    print("=" * 70)

    # Generate perfect circle: center at (2, 3), radius 5
    center_true = np.array([2.0, 3.0])
    radius_true = 5.0

    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    points = np.column_stack([
        center_true[0] + radius_true * np.cos(theta),
        center_true[1] + radius_true * np.sin(theta)
    ])

    print(f"\nTrue center: {center_true}")
    print(f"True radius: {radius_true}")
    print(f"Number of points: {len(points)}")

    # Fit circle
    result = fit_circle_2d(points)

    print(f"\nFitted center: {result['center']}")
    print(f"Fitted radius: {result['radius']:.6f}")
    print(f"RMS residual: {result['residuals']:.10f} (should be near zero)")
    print(f"Fit quality: {result['fit_quality']:.10f} (normalized residual)")

    # Check accuracy
    assert np.allclose(result['center'], center_true, atol=1e-10), "Center fit failed"
    assert np.isclose(result['radius'], radius_true, atol=1e-10), "Radius fit failed"
    assert result['residuals'] < 1e-8, "Residuals too large for perfect circle"
    assert result['fit_quality'] < 1e-8, "Fit quality too poor for perfect circle"

    print("\n✓ Perfect circle fitting test passed!\n")


def test_fit_circle_2d_with_noise():
    """Test circle fitting with noisy data."""
    print("=" * 70)
    print("TEST: Circle Fitting - Noisy Circle")
    print("=" * 70)

    # Generate circle with noise
    center_true = np.array([10.0, -5.0])
    radius_true = 15.0

    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    points_perfect = np.column_stack([
        center_true[0] + radius_true * np.cos(theta),
        center_true[1] + radius_true * np.sin(theta)
    ])

    # Add small Gaussian noise
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 0.1, points_perfect.shape)
    points = points_perfect + noise

    print(f"\nTrue center: {center_true}")
    print(f"True radius: {radius_true}")
    print(f"Noise std dev: 0.1")

    # Fit circle
    result = fit_circle_2d(points)

    print(f"\nFitted center: {result['center']}")
    print(f"Fitted radius: {result['radius']:.6f}")
    print(f"RMS residual: {result['residuals']:.6f}")
    print(f"Fit quality: {result['fit_quality']:.6f}")

    # Check accuracy (should be within noise level)
    center_error = np.linalg.norm(result['center'] - center_true)
    radius_error = abs(result['radius'] - radius_true)

    print(f"\nCenter error: {center_error:.6f}")
    print(f"Radius error: {radius_error:.6f}")

    assert center_error < 0.2, f"Center error too large: {center_error}"
    assert radius_error < 0.2, f"Radius error too large: {radius_error}"
    assert result['residuals'] < 0.3, "Residuals unexpectedly large"

    print("\n✓ Noisy circle fitting test passed!\n")


def test_fit_circle_2d_arc():
    """Test circle fitting with partial arc (not full circle)."""
    print("=" * 70)
    print("TEST: Circle Fitting - Partial Arc")
    print("=" * 70)

    # Generate 90-degree arc
    center_true = np.array([0.0, 0.0])
    radius_true = 10.0

    # Arc from 0 to 90 degrees
    theta = np.linspace(0, np.pi / 2, 10)
    points = np.column_stack([
        center_true[0] + radius_true * np.cos(theta),
        center_true[1] + radius_true * np.sin(theta)
    ])

    print(f"\nTrue center: {center_true}")
    print(f"True radius: {radius_true}")
    print(f"Arc span: 0° to 90°")
    print(f"Number of points: {len(points)}")

    # Fit circle
    result = fit_circle_2d(points)

    print(f"\nFitted center: {result['center']}")
    print(f"Fitted radius: {result['radius']:.6f}")
    print(f"RMS residual: {result['residuals']:.10f}")
    print(f"Fit quality: {result['fit_quality']:.10f}")

    # Check accuracy
    assert np.allclose(result['center'], center_true, atol=1e-9), "Center fit failed for arc"
    assert np.isclose(result['radius'], radius_true, atol=1e-9), "Radius fit failed for arc"

    print("\n✓ Partial arc fitting test passed!\n")


def test_fit_circle_2d_small_arc():
    """Test circle fitting with small arc typical of suspension motion."""
    print("=" * 70)
    print("TEST: Circle Fitting - Small Suspension Arc")
    print("=" * 70)

    # Simulate suspension motion: small vertical arc
    # This is more representative of actual suspension instant center analysis
    center_true = np.array([0.0, 500.0])  # Instant center 500mm above ground
    radius_true = 600.0  # 600mm arc radius

    # Small arc: ±5 degrees around horizontal
    theta_center = np.pi / 2  # 90 degrees (pointing down from center)
    theta_span = np.radians(10)  # ±5 degrees
    theta = np.linspace(theta_center - theta_span / 2,
                       theta_center + theta_span / 2, 5)

    points = np.column_stack([
        center_true[0] + radius_true * np.cos(theta),
        center_true[1] + radius_true * np.sin(theta)
    ])

    print(f"\nTrue center: {center_true} mm")
    print(f"True radius: {radius_true} mm")
    print(f"Arc span: ±5° around horizontal")
    print(f"Number of points: {len(points)}")
    print(f"Point range Y: {points[:, 1].min():.2f} to {points[:, 1].max():.2f} mm")

    # Fit circle
    result = fit_circle_2d(points)

    print(f"\nFitted center: {result['center']}")
    print(f"Fitted radius: {result['radius']:.6f} mm")
    print(f"RMS residual: {result['residuals']:.6f} mm")
    print(f"Fit quality: {result['fit_quality']:.10f}")

    # For small arcs, accuracy may be reduced but should still be reasonable
    center_error = np.linalg.norm(result['center'] - center_true)
    radius_error = abs(result['radius'] - radius_true)

    print(f"\nCenter error: {center_error:.6f} mm")
    print(f"Radius error: {radius_error:.6f} mm")

    # Relaxed tolerances for small arc
    assert center_error < 50, f"Center error too large for small arc: {center_error}"
    assert radius_error < 50, f"Radius error too large for small arc: {radius_error}"

    print("\n✓ Small suspension arc fitting test passed!\n")


def test_calculate_instant_center_from_points():
    """Test instant center calculation from 3D contact points."""
    print("=" * 70)
    print("TEST: Instant Center Calculation")
    print("=" * 70)

    # Simulate suspension motion creating circular arc in YZ plane
    # Center at (1500, 700, 400) mm, radius 600 mm
    center_yz_true = np.array([700.0, 400.0])  # Y, Z in YZ plane
    radius_true = 600.0

    # Generate points along arc
    theta = np.linspace(np.pi / 2 - 0.1, np.pi / 2 + 0.1, 5)
    points_3d = np.column_stack([
        np.full(len(theta), 1500.0),  # X constant
        center_yz_true[0] + radius_true * np.cos(theta),  # Y
        center_yz_true[1] + radius_true * np.sin(theta)   # Z
    ])

    print(f"\nSimulated contact points (3D):")
    for i, pt in enumerate(points_3d):
        print(f"  Point {i}: [{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}] mm")

    # Calculate roll instant center (YZ plane)
    print("\n--- Roll Instant Center (YZ Plane) ---")
    result = calculate_instant_center_from_points(points_3d, 'YZ')

    print(f"Center 2D (YZ): {result['center_2d']}")
    print(f"Center 3D: {result['center_3d']}")
    print(f"Radius: {result['radius']:.6f} mm")
    print(f"RMS residual: {result['residuals']:.6f} mm")
    print(f"Fit quality: {result['fit_quality']:.10f}")

    # Verify results
    assert result['center_3d'][0] == np.mean(points_3d[:, 0]), "X coordinate should be mean"
    assert np.allclose(result['center_2d'], center_yz_true, atol=1), "YZ center incorrect"
    assert np.isclose(result['radius'], radius_true, atol=1), "Radius incorrect"

    print("✓ Roll instant center calculation correct")

    # Test pitch instant center (XZ plane)
    print("\n--- Pitch Instant Center (XZ Plane) ---")
    # Create points with arc in XZ plane
    center_xz_true = np.array([1500.0, 400.0])  # X, Z in XZ plane
    points_3d_xz = np.column_stack([
        center_xz_true[0] + radius_true * np.cos(theta),  # X
        np.full(len(theta), 700.0),  # Y constant
        center_xz_true[1] + radius_true * np.sin(theta)   # Z
    ])

    result_xz = calculate_instant_center_from_points(points_3d_xz, 'XZ')

    print(f"Center 2D (XZ): {result_xz['center_2d']}")
    print(f"Center 3D: {result_xz['center_3d']}")
    print(f"Radius: {result_xz['radius']:.6f} mm")

    assert result_xz['center_3d'][1] == np.mean(points_3d_xz[:, 1]), "Y coordinate should be mean"
    assert np.allclose(result_xz['center_2d'], center_xz_true, atol=1), "XZ center incorrect"

    print("✓ Pitch instant center calculation correct")
    print("\n✓ All instant center tests passed!\n")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("=" * 70)
    print("TEST: Error Handling")
    print("=" * 70)

    # Test collinear points
    print("\n--- Testing Collinear Points Detection ---")
    collinear_points = np.array([[0, 0], [1, 1], [2, 2]])
    try:
        result = fit_circle_2d(collinear_points)
        print("Warning: Collinear points did not raise error (may be acceptable)")
    except ValueError as e:
        print(f"✓ Correctly detected collinear points: {e}")

    # Test insufficient points
    print("\n--- Testing Insufficient Points ---")
    try:
        fit_circle_2d(np.array([[0, 0], [1, 1]]))
        assert False, "Should have raised ValueError for insufficient points"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test invalid plane
    print("\n--- Testing Invalid Plane ---")
    points_3d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    try:
        calculate_instant_center_from_points(points_3d, 'AB')
        assert False, "Should have raised ValueError for invalid plane"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    print("\n✓ All error handling tests passed!\n")


def run_all_tests():
    """Run all geometry utility tests."""
    print("\n" + "=" * 70)
    print("GEOMETRY UTILS TEST SUITE")
    print("=" * 70 + "\n")

    test_project_points_to_plane()
    test_fit_circle_2d_perfect()
    test_fit_circle_2d_with_noise()
    test_fit_circle_2d_arc()
    test_fit_circle_2d_small_arc()
    test_calculate_instant_center_from_points()
    test_error_handling()

    print("=" * 70)
    print("ALL GEOMETRY UTILS TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
