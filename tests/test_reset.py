#!/usr/bin/env python3
"""
Comprehensive test for reset_to_origin() functionality across all suspension classes.
"""
import sys
import numpy as np
sys.path.insert(0, 'pysuspension')

from suspension_knuckle import SuspensionKnuckle
from control_arm import ControlArm
from suspension_link import SuspensionLink
from chassis import Chassis
from chassis_corner import ChassisCorner
from chassis_axle import ChassisAxle

def test_suspension_knuckle_reset():
    """Test SuspensionKnuckle reset_to_origin()"""
    print("\n" + "="*60)
    print("TEST: SuspensionKnuckle reset_to_origin()")
    print("="*60)

    # Create knuckle with initial position
    knuckle = SuspensionKnuckle(
        tire_center_x=1.5,
        tire_center_y=0.8,
        rolling_radius=0.6,
        toe_angle=2.0,
        camber_angle=-1.5,
        wheel_offset=0.0,
        mass=5.0,
        unit='m',
        mass_unit='kg'
    )

    # Store original values (tire_center is in mm, angles in radians)
    original_tire_center = knuckle.tire_center.copy()
    original_toe = knuckle.toe_angle
    original_camber = knuckle.camber_angle
    original_com = knuckle.get_center_of_mass(unit='mm').copy()

    print(f"Original tire center (mm): {original_tire_center}")
    print(f"Original toe angle (rad): {original_toe}")
    print(f"Original camber angle (rad): {original_camber}")
    print(f"Original COM (mm): {original_com}")

    # Apply transformations by directly modifying the knuckle's state
    knuckle.tire_center += np.array([50, -20, -30])  # Translate in mm
    knuckle.toe_angle += np.radians(5.0)  # Rotate (adjust toe)

    print(f"\nAfter transformations:")
    print(f"  Tire center (mm): {knuckle.tire_center}")
    print(f"  Toe angle (rad): {knuckle.toe_angle}")
    print(f"  COM (mm): {knuckle.get_center_of_mass(unit='mm')}")

    # Reset to origin
    knuckle.reset_to_origin()

    print(f"\nAfter reset_to_origin():")
    print(f"  Tire center (mm): {knuckle.tire_center}")
    print(f"  Toe angle (rad): {knuckle.toe_angle}")
    print(f"  Camber angle (rad): {knuckle.camber_angle}")
    print(f"  COM (mm): {knuckle.get_center_of_mass(unit='mm')}")

    # Verify reset
    assert np.allclose(knuckle.tire_center, original_tire_center), "Tire center not reset!"
    assert np.isclose(knuckle.toe_angle, original_toe), "Toe angle not reset!"
    assert np.isclose(knuckle.camber_angle, original_camber), "Camber angle not reset!"
    assert np.allclose(knuckle.get_center_of_mass(unit='mm'), original_com), "COM not reset!"

    print("\n✓ SuspensionKnuckle reset test PASSED")
    return True

def test_control_arm_reset():
    """Test ControlArm reset_to_origin()"""
    print("\n" + "="*60)
    print("TEST: ControlArm reset_to_origin()")
    print("="*60)

    # Create control arm with links
    control_arm = ControlArm(name="test_arm", mass=2.5, mass_unit='kg')

    link1 = SuspensionLink(
        endpoint1=[1300, 400, 550],
        endpoint2=[1500, 750, 600],
        name="front_link",
        unit='mm'
    )

    link2 = SuspensionLink(
        endpoint1=[1200, 400, 550],
        endpoint2=[1500, 750, 600],
        name="rear_link",
        unit='mm'
    )

    control_arm.add_link(link1)
    control_arm.add_link(link2)
    control_arm.add_attachment_point("sway_bar", [1250, 500, 500], unit='mm')

    # Store original values AFTER all attachments are added
    original_positions = [pos.copy() for pos in control_arm.get_all_attachment_positions(unit='mm')]
    original_centroid = control_arm.centroid.copy()
    original_com = control_arm.get_center_of_mass(unit='mm').copy()

    print(f"Original centroid (mm): {original_centroid}")
    print(f"Original COM (mm): {original_com}")
    print(f"Original attachment count: {len(original_positions)}")
    print(f"Original positions:")
    for i, pos in enumerate(original_positions):
        print(f"  {i}: {pos}")

    # Apply transformation via fit
    target_positions = [pos + np.array([20, -10, -30]) for pos in original_positions]
    rms_error = control_arm.fit_to_attachment_targets(target_positions, unit='mm')

    print(f"\nAfter fit transformation:")
    print(f"  Centroid (mm): {control_arm.centroid}")
    print(f"  COM (mm): {control_arm.get_center_of_mass(unit='mm')}")
    print(f"  RMS error: {rms_error:.3f} mm")

    # Reset to origin
    control_arm.reset_to_origin()

    print(f"\nAfter reset_to_origin():")
    print(f"  Centroid (mm): {control_arm.centroid}")
    print(f"  COM (mm): {control_arm.get_center_of_mass(unit='mm')}")

    reset_positions = control_arm.get_all_attachment_positions(unit='mm')
    print(f"Reset positions:")
    for i, pos in enumerate(reset_positions):
        print(f"  {i}: {pos}")

    # Verify reset
    assert np.allclose(control_arm.centroid, original_centroid, atol=1e-6), \
        f"Centroid not reset! Expected {original_centroid}, got {control_arm.centroid}"
    assert np.allclose(control_arm.get_center_of_mass(unit='mm'), original_com, atol=1e-6), \
        f"COM not reset! Expected {original_com}, got {control_arm.get_center_of_mass(unit='mm')}"
    assert len(reset_positions) == len(original_positions), "Position count mismatch!"

    for i, (orig, reset) in enumerate(zip(original_positions, reset_positions)):
        assert np.allclose(reset, orig, atol=1e-6), f"Position {i} not reset correctly! Expected {orig}, got {reset}"

    print("\n✓ ControlArm reset test PASSED")
    return True

def test_chassis_corner_reset():
    """Test ChassisCorner reset_to_origin()"""
    print("\n" + "="*60)
    print("TEST: ChassisCorner reset_to_origin()")
    print("="*60)

    corner = ChassisCorner("test_corner")
    corner.add_attachment_point("upper", [1400, 750, 650], unit='mm')
    corner.add_attachment_point("lower", [1400, 750, 450], unit='mm')

    # Store original
    original_positions = [pos.copy() for pos in corner.get_attachment_positions(unit='mm')]

    print(f"Original positions: {original_positions}")

    # Modify positions manually (simulate transformation)
    corner.attachment_points[0].set_position(
        corner.attachment_points[0].position + np.array([100, 50, -20]), unit='mm')
    corner.attachment_points[1].set_position(
        corner.attachment_points[1].position + np.array([100, 50, -20]), unit='mm')

    print(f"After modification: {[pos for pos in corner.get_attachment_positions(unit='mm')]}")

    # Reset
    corner.reset_to_origin()

    reset_positions = corner.get_attachment_positions(unit='mm')
    print(f"After reset: {reset_positions}")

    # Verify
    for i, (orig, reset) in enumerate(zip(original_positions, reset_positions)):
        assert np.allclose(reset, orig, atol=1e-6), f"Position {i} not reset!"

    print("\n✓ ChassisCorner reset test PASSED")
    return True

def test_chassis_axle_reset():
    """Test ChassisAxle reset_to_origin()"""
    print("\n" + "="*60)
    print("TEST: ChassisAxle reset_to_origin()")
    print("="*60)

    # Create a simple chassis with corners
    chassis = Chassis(mass=500, mass_unit='kg')
    chassis.create_corner("front_left")
    chassis.create_corner("front_right")

    # Create axle
    axle = chassis.create_axle("front_axle", ["front_left", "front_right"])
    axle.add_attachment_point("steering_rack", [1400, 0, 500], unit='mm')

    # Store original
    original_positions = [pos.copy() for pos in axle.get_all_attachment_positions(unit='mm')]

    print(f"Original axle attachment positions: {original_positions}")

    # Modify manually
    axle.attachment_points[0].set_position(
        axle.attachment_points[0].position + np.array([50, 20, -30]), unit='mm')

    print(f"After modification: {axle.get_all_attachment_positions(unit='mm')}")

    # Reset
    axle.reset_to_origin()

    reset_positions = axle.get_all_attachment_positions(unit='mm')
    print(f"After reset: {reset_positions}")

    # Verify
    for i, (orig, reset) in enumerate(zip(original_positions, reset_positions)):
        assert np.allclose(reset, orig, atol=1e-6), f"Position {i} not reset!"

    print("\n✓ ChassisAxle reset test PASSED")
    return True

def test_chassis_hierarchical_reset():
    """Test Chassis hierarchical reset_to_origin()"""
    print("\n" + "="*60)
    print("TEST: Chassis hierarchical reset_to_origin()")
    print("="*60)

    # Create chassis with corners
    chassis = Chassis(mass=500, mass_unit='kg')

    # Create corners and add attachments
    fl_corner = chassis.create_corner("front_left")
    fr_corner = chassis.create_corner("front_right")

    fl_corner.add_attachment_point("upper", [1400, 750, 650], unit='mm')
    fl_corner.add_attachment_point("lower", [1400, 750, 450], unit='mm')
    fr_corner.add_attachment_point("upper", [1400, -750, 650], unit='mm')
    fr_corner.add_attachment_point("lower", [1400, -750, 450], unit='mm')

    # Create axle
    axle = chassis.create_axle("front_axle", ["front_left", "front_right"])
    axle.add_attachment_point("steering_rack", [1400, 0, 500], unit='mm')

    # Store original values
    original_centroid = chassis.centroid.copy()
    original_com = chassis.get_center_of_mass(unit='mm').copy()
    original_fl_positions = [pos.copy() for pos in fl_corner.get_attachment_positions(unit='mm')]
    original_axle_positions = [pos.copy() for pos in axle.get_all_attachment_positions(unit='mm')]

    print(f"Original chassis centroid (mm): {original_centroid}")
    print(f"Original chassis COM (mm): {original_com}")
    print(f"Original FL corner positions: {original_fl_positions}")
    print(f"Original axle positions: {original_axle_positions}")

    # Apply transformations
    chassis.translate([50, 20, -30], unit='mm')
    # Create a simple rotation matrix (10 degrees about z-axis)
    angle_rad = np.radians(10.0)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    chassis.rotate_about_centroid(R)

    print(f"\nAfter transformations:")
    print(f"  Chassis centroid (mm): {chassis.centroid}")
    print(f"  Chassis COM (mm): {chassis.get_center_of_mass(unit='mm')}")
    print(f"  FL corner positions: {fl_corner.get_attachment_positions(unit='mm')}")

    # Reset (hierarchical)
    chassis.reset_to_origin()

    print(f"\nAfter chassis.reset_to_origin():")
    print(f"  Chassis centroid (mm): {chassis.centroid}")
    print(f"  Chassis COM (mm): {chassis.get_center_of_mass(unit='mm')}")
    print(f"  FL corner positions: {fl_corner.get_attachment_positions(unit='mm')}")
    print(f"  Axle positions: {axle.get_all_attachment_positions(unit='mm')}")

    # Verify chassis reset
    assert np.allclose(chassis.centroid, original_centroid, atol=1e-6), \
        f"Chassis centroid not reset! Expected {original_centroid}, got {chassis.centroid}"
    assert np.allclose(chassis.get_center_of_mass(unit='mm'), original_com, atol=1e-6), \
        f"Chassis COM not reset!"

    # Verify corner reset
    reset_fl_positions = fl_corner.get_attachment_positions(unit='mm')
    for i, (orig, reset) in enumerate(zip(original_fl_positions, reset_fl_positions)):
        assert np.allclose(reset, orig, atol=1e-6), \
            f"Corner position {i} not reset! Expected {orig}, got {reset}"

    # Verify axle reset
    reset_axle_positions = axle.get_all_attachment_positions(unit='mm')
    for i, (orig, reset) in enumerate(zip(original_axle_positions, reset_axle_positions)):
        assert np.allclose(reset, orig, atol=1e-6), \
            f"Axle position {i} not reset! Expected {orig}, got {reset}"

    print("\n✓ Chassis hierarchical reset test PASSED")
    return True

def main():
    """Run all reset tests"""
    print("\n" + "="*60)
    print("RESET FUNCTIONALITY TEST SUITE")
    print("="*60)

    all_passed = True
    tests = [
        ("SuspensionKnuckle", test_suspension_knuckle_reset),
        ("ControlArm", test_control_arm_reset),
        ("ChassisCorner", test_chassis_corner_reset),
        ("ChassisAxle", test_chassis_axle_reset),
        ("Chassis hierarchical", test_chassis_hierarchical_reset),
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
        print("✓ ALL RESET TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
