"""
Test CoilSpring behavior before and after refactoring to extend SuspensionLink.

This test ensures backward compatibility and verifies that CoilSpring
properly inherits from SuspensionLink.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.coil_spring import CoilSpring
from pysuspension.suspension_link import SuspensionLink


def test_basic_spring_creation():
    """Test basic CoilSpring creation and properties."""
    print("\n--- Test: Basic Spring Creation ---")

    spring = CoilSpring(
        endpoint1=[0, 0, 0],
        endpoint2=[0, 0, 300],  # 300mm vertical spring
        spring_rate=5.0,  # 5 kg/mm
        preload_force=500.0,  # 500 N preload
        name="test_spring",
        unit='mm',
        spring_rate_unit='kg/mm',
        force_unit='N'
    )

    # Check basic properties
    assert spring.name == "test_spring"
    assert abs(spring.get_initial_length() - 300.0) < 0.001
    assert abs(spring.get_current_length() - 300.0) < 0.001
    assert abs(spring.get_spring_rate('kg/mm') - 5.0) < 0.001
    assert abs(spring.get_preload_force('N') - 500.0) < 0.001

    print(f"✓ Initial length: {spring.get_initial_length():.3f} mm")
    print(f"✓ Current length: {spring.get_current_length():.3f} mm")
    print(f"✓ Spring rate: {spring.get_spring_rate('kg/mm'):.3f} kg/mm")
    print(f"✓ Preload force: {spring.get_preload_force('N'):.3f} N")


def test_spring_compression():
    """Test spring compression and reaction force."""
    print("\n--- Test: Spring Compression ---")

    spring = CoilSpring(
        endpoint1=[0, 0, 0],
        endpoint2=[0, 0, 300],
        spring_rate=5.0,  # 5 kg/mm
        preload_force=500.0,  # 500 N preload
        unit='mm',
        spring_rate_unit='kg/mm',
        force_unit='N'
    )

    # Compress spring by 50mm
    spring.endpoint2.set_position([0, 0, 250], unit='mm')
    spring._update_local_frame()

    # Check compression
    assert abs(spring.get_current_length() - 250.0) < 0.001
    assert abs(spring.get_length_change() - (-50.0)) < 0.001

    # Check reaction force
    # Force = preload + spring_rate * compression
    # Force = 500 N + (5 kg/mm * 9.80665 N/kg) * 50mm
    expected_force = 500.0 + (5.0 * 9.80665) * 50.0
    actual_force = spring.get_reaction_force_magnitude('N')

    print(f"✓ Compressed length: {spring.get_current_length():.3f} mm")
    print(f"✓ Length change: {spring.get_length_change():.3f} mm")
    print(f"✓ Expected force: {expected_force:.3f} N")
    print(f"✓ Actual force: {actual_force:.3f} N")
    assert abs(actual_force - expected_force) < 1.0


def test_fit_to_targets():
    """Test fit_to_attachment_targets allows length change."""
    print("\n--- Test: Fit to Targets (Variable Length) ---")

    spring = CoilSpring(
        endpoint1=[0, 0, 0],
        endpoint2=[0, 0, 300],
        spring_rate=5.0,
        preload_force=500.0,
        unit='mm',
        spring_rate_unit='kg/mm',
        force_unit='N'
    )

    initial_length = spring.get_current_length()

    # Fit to targets with different length (compressed)
    target1 = np.array([10, 0, 0])
    target2 = np.array([10, 0, 240])  # 240mm length (60mm compression)

    rms_error = spring.fit_to_attachment_targets([target1, target2], unit='mm')

    # Should fit exactly (near-zero error)
    assert rms_error < 0.001

    # Length should have changed
    new_length = spring.get_current_length()
    assert abs(new_length - 240.0) < 0.001
    assert abs(new_length - initial_length) > 50.0  # Significant change

    print(f"✓ Initial length: {initial_length:.3f} mm")
    print(f"✓ New length: {new_length:.3f} mm")
    print(f"✓ RMS error: {rms_error:.6f} mm")
    print(f"✓ Length change allowed: {new_length != initial_length}")


def test_inheritance():
    """Test that CoilSpring properly inherits from SuspensionLink."""
    print("\n--- Test: Inheritance from SuspensionLink ---")

    spring = CoilSpring(
        endpoint1=[0, 0, 0],
        endpoint2=[0, 0, 300],
        spring_rate=5.0,
        preload_force=500.0,
        unit='mm',
        spring_rate_unit='kg/mm',
        force_unit='N'
    )

    # Check inheritance
    is_suspension_link = isinstance(spring, SuspensionLink)
    print(f"✓ isinstance(spring, SuspensionLink): {is_suspension_link}")
    assert is_suspension_link, "CoilSpring should inherit from SuspensionLink"

    # Check inherited methods exist
    assert hasattr(spring, 'get_endpoint1')
    assert hasattr(spring, 'get_endpoint2')
    assert hasattr(spring, 'get_endpoints')
    assert hasattr(spring, 'get_axis')
    assert hasattr(spring, 'get_center')
    assert hasattr(spring, 'translate')

    print(f"✓ All expected inherited methods present")

    # Check spring-specific methods exist
    assert hasattr(spring, 'get_current_length')
    assert hasattr(spring, 'get_spring_rate')
    assert hasattr(spring, 'get_reaction_force_magnitude')
    assert hasattr(spring, 'get_preload_force')

    print(f"✓ All spring-specific methods present")


def test_inherited_methods():
    """Test that inherited methods work correctly."""
    print("\n--- Test: Inherited Methods ---")

    spring = CoilSpring(
        endpoint1=[100, 200, 300],
        endpoint2=[100, 200, 600],
        spring_rate=5.0,
        preload_force=500.0,
        unit='mm',
        spring_rate_unit='kg/mm',
        force_unit='N'
    )

    # Test get_axis
    axis = spring.get_axis()
    expected_axis = np.array([0, 0, 1])
    assert np.allclose(axis, expected_axis), f"Expected axis {expected_axis}, got {axis}"
    print(f"✓ get_axis(): {axis}")

    # Test get_center
    center = spring.get_center()
    expected_center = np.array([100, 200, 450])
    assert np.allclose(center, expected_center), f"Expected center {expected_center}, got {center}"
    print(f"✓ get_center(): {center}")

    # Test translate
    spring.translate([10, 20, 30], unit='mm')
    new_center = spring.get_center()
    expected_new_center = expected_center + np.array([10, 20, 30])
    assert np.allclose(new_center, expected_new_center), f"Expected {expected_new_center}, got {new_center}"
    print(f"✓ translate([10, 20, 30]): new center = {new_center}")


def test_length_semantics():
    """Test that length semantics are correct for spring."""
    print("\n--- Test: Length Semantics ---")

    spring = CoilSpring(
        endpoint1=[0, 0, 0],
        endpoint2=[0, 0, 300],
        spring_rate=5.0,
        preload_force=500.0,
        unit='mm',
        spring_rate_unit='kg/mm',
        force_unit='N'
    )

    # For CoilSpring, self.length should be the free length (initial length)
    # This is inherited from SuspensionLink
    assert hasattr(spring, 'length'), "Should have length attribute from parent"

    # Initial length and free length should be the same
    initial_length = spring.get_initial_length()
    free_length = spring.length  # From parent
    assert abs(initial_length - free_length) < 0.001

    print(f"✓ Free length (self.length): {free_length:.3f} mm")
    print(f"✓ Initial length: {initial_length:.3f} mm")
    print(f"✓ Current length: {spring.get_current_length():.3f} mm")

    # Compress the spring
    spring.endpoint2.set_position([0, 0, 250], unit='mm')
    spring._update_local_frame()

    # Free length should not change, but current length should
    assert abs(spring.length - 300.0) < 0.001, "Free length should not change"
    assert abs(spring.get_current_length() - 250.0) < 0.001, "Current length should change"

    print(f"✓ After compression:")
    print(f"  - Free length: {spring.length:.3f} mm (unchanged)")
    print(f"  - Current length: {spring.get_current_length():.3f} mm (changed)")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("COIL SPRING REFACTORING TEST SUITE")
    print("=" * 70)

    try:
        test_basic_spring_creation()
        test_spring_compression()
        test_fit_to_targets()
        test_inheritance()
        test_inherited_methods()
        test_length_semantics()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
