#!/usr/bin/env python3
"""
Test reset_to_origin() functionality for SuspensionLink and CoilSpring classes.
"""
import sys
import os
import numpy as np

# Add parent directory to path for pysuspension imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pysuspension.suspension_link import SuspensionLink
from pysuspension.coil_spring import CoilSpring


def test_suspension_link_reset():
    """Test SuspensionLink reset_to_origin()"""
    print("\n" + "="*60)
    print("TEST: SuspensionLink reset_to_origin()")
    print("="*60)

    # Create a suspension link
    link = SuspensionLink(
        endpoint1=[1400, 750, 600],
        endpoint2=[1500, 800, 650],
        name="test_link",
        unit='mm'
    )

    # Store original values
    original_endpoint1 = link.get_endpoint1(unit='mm').copy()
    original_endpoint2 = link.get_endpoint2(unit='mm').copy()
    original_center = link.get_center(unit='mm').copy()
    original_length = link.get_length(unit='mm')

    print(f"Original endpoint1 (mm): {original_endpoint1}")
    print(f"Original endpoint2 (mm): {original_endpoint2}")
    print(f"Original center (mm): {original_center}")
    print(f"Original length (mm): {original_length:.3f}")

    # Apply transformations
    link.translate([50, -25, 10], unit='mm')
    angle_rad = np.radians(15.0)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    link.rotate_about_center(R)

    print(f"\nAfter transformations:")
    print(f"  Endpoint1 (mm): {link.get_endpoint1(unit='mm')}")
    print(f"  Endpoint2 (mm): {link.get_endpoint2(unit='mm')}")
    print(f"  Center (mm): {link.get_center(unit='mm')}")
    print(f"  Length (mm): {link.get_length(unit='mm'):.3f} (should be unchanged)")

    # Verify transformations were applied
    assert not np.allclose(link.get_endpoint1(unit='mm'), original_endpoint1, atol=1e-6), \
        "Endpoint1 should have been transformed!"
    assert not np.allclose(link.get_center(unit='mm'), original_center, atol=1e-6), \
        "Center should have been transformed!"
    # Length should remain constant
    assert np.isclose(link.get_length(unit='mm'), original_length, atol=1e-6), \
        "Length should remain constant!"

    # Reset to origin
    link.reset_to_origin()

    print(f"\nAfter reset_to_origin():")
    print(f"  Endpoint1 (mm): {link.get_endpoint1(unit='mm')}")
    print(f"  Endpoint2 (mm): {link.get_endpoint2(unit='mm')}")
    print(f"  Center (mm): {link.get_center(unit='mm')}")
    print(f"  Length (mm): {link.get_length(unit='mm'):.3f}")

    # Verify reset
    reset_endpoint1 = link.get_endpoint1(unit='mm')
    reset_endpoint2 = link.get_endpoint2(unit='mm')
    reset_center = link.get_center(unit='mm')
    reset_length = link.get_length(unit='mm')

    assert np.allclose(reset_endpoint1, original_endpoint1, atol=1e-6), \
        f"Endpoint1 not reset! Expected {original_endpoint1}, got {reset_endpoint1}"
    assert np.allclose(reset_endpoint2, original_endpoint2, atol=1e-6), \
        f"Endpoint2 not reset! Expected {original_endpoint2}, got {reset_endpoint2}"
    assert np.allclose(reset_center, original_center, atol=1e-6), \
        f"Center not reset! Expected {original_center}, got {reset_center}"
    assert np.isclose(reset_length, original_length, atol=1e-6), \
        f"Length not maintained! Expected {original_length:.3f}, got {reset_length:.3f}"

    print("\n✓ SuspensionLink reset test PASSED")
    return True


def test_coil_spring_reset():
    """Test CoilSpring reset_to_origin()"""
    print("\n" + "="*60)
    print("TEST: CoilSpring reset_to_origin()")
    print("="*60)

    # Create a coil spring
    spring = CoilSpring(
        endpoint1=[1000, 500, 300],
        endpoint2=[1000, 500, 500],
        spring_rate=10.0,
        preload_force=500.0,
        mass=2.0,
        name="test_spring",
        unit='mm',
        spring_rate_unit='kg/mm',
        force_unit='N',
        mass_unit='kg'
    )

    # Store original values
    original_endpoint1 = spring.get_endpoint1(unit='mm').copy()
    original_endpoint2 = spring.get_endpoint2(unit='mm').copy()
    original_center = spring.get_center(unit='mm').copy()
    original_initial_length = spring.get_initial_length(unit='mm')
    original_current_length = spring.get_current_length(unit='mm')
    original_force = spring.get_reaction_force_magnitude(unit='N')

    print(f"Original endpoint1 (mm): {original_endpoint1}")
    print(f"Original endpoint2 (mm): {original_endpoint2}")
    print(f"Original center (mm): {original_center}")
    print(f"Original initial length (mm): {original_initial_length:.3f}")
    print(f"Original current length (mm): {original_current_length:.3f}")
    print(f"Original reaction force (N): {original_force:.3f}")

    # Apply transformations - translate the spring
    spring.translate([30, -20, 15], unit='mm')

    # Compress the spring by fitting to new targets
    compressed_targets = [
        [1030, 480, 315],  # endpoint1 after translation
        [1030, 480, 465]   # endpoint2 compressed by 50mm from translated position
    ]
    spring.fit_to_attachment_targets(compressed_targets, unit='mm')

    print(f"\nAfter transformations (translation + compression):")
    print(f"  Endpoint1 (mm): {spring.get_endpoint1(unit='mm')}")
    print(f"  Endpoint2 (mm): {spring.get_endpoint2(unit='mm')}")
    print(f"  Center (mm): {spring.get_center(unit='mm')}")
    print(f"  Current length (mm): {spring.get_current_length(unit='mm'):.3f}")
    print(f"  Length change (mm): {spring.get_length_change(unit='mm'):.3f}")
    print(f"  Reaction force (N): {spring.get_reaction_force_magnitude(unit='N'):.3f}")

    # Verify transformations were applied
    assert not np.allclose(spring.get_endpoint1(unit='mm'), original_endpoint1, atol=1e-6), \
        "Endpoint1 should have been transformed!"
    assert not np.allclose(spring.get_center(unit='mm'), original_center, atol=1e-6), \
        "Center should have been transformed!"
    assert not np.isclose(spring.get_current_length(unit='mm'), original_current_length, atol=1e-6), \
        "Current length should have changed (compression)!"
    assert not np.isclose(spring.get_reaction_force_magnitude(unit='N'), original_force, atol=1e-3), \
        "Reaction force should have changed!"

    # Reset to origin
    spring.reset_to_origin()

    print(f"\nAfter reset_to_origin():")
    print(f"  Endpoint1 (mm): {spring.get_endpoint1(unit='mm')}")
    print(f"  Endpoint2 (mm): {spring.get_endpoint2(unit='mm')}")
    print(f"  Center (mm): {spring.get_center(unit='mm')}")
    print(f"  Current length (mm): {spring.get_current_length(unit='mm'):.3f}")
    print(f"  Length change (mm): {spring.get_length_change(unit='mm'):.3f}")
    print(f"  Reaction force (N): {spring.get_reaction_force_magnitude(unit='N'):.3f}")

    # Verify reset
    reset_endpoint1 = spring.get_endpoint1(unit='mm')
    reset_endpoint2 = spring.get_endpoint2(unit='mm')
    reset_center = spring.get_center(unit='mm')
    reset_current_length = spring.get_current_length(unit='mm')
    reset_force = spring.get_reaction_force_magnitude(unit='N')

    assert np.allclose(reset_endpoint1, original_endpoint1, atol=1e-6), \
        f"Endpoint1 not reset! Expected {original_endpoint1}, got {reset_endpoint1}"
    assert np.allclose(reset_endpoint2, original_endpoint2, atol=1e-6), \
        f"Endpoint2 not reset! Expected {original_endpoint2}, got {reset_endpoint2}"
    assert np.allclose(reset_center, original_center, atol=1e-6), \
        f"Center not reset! Expected {original_center}, got {reset_center}"
    assert np.isclose(reset_current_length, original_current_length, atol=1e-6), \
        f"Current length not reset! Expected {original_current_length:.3f}, got {reset_current_length:.3f}"
    assert np.isclose(reset_force, original_force, atol=1e-3), \
        f"Reaction force not reset! Expected {original_force:.3f}, got {reset_force:.3f}"

    print("\n✓ CoilSpring reset test PASSED")
    return True


def main():
    """Run all reset tests for SuspensionLink and CoilSpring"""
    print("\n" + "="*60)
    print("SUSPENSION LINK & COIL SPRING RESET TEST SUITE")
    print("="*60)

    all_passed = True
    tests = [
        ("SuspensionLink", test_suspension_link_reset),
        ("CoilSpring", test_coil_spring_reset),
    ]

    for test_name, test_func in tests:
        try:
            all_passed &= test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
