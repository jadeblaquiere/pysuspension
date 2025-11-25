#!/usr/bin/env python3
"""
Test the refactored AttachmentPoint system with connection tracking.
"""
import sys
import numpy as np
sys.path.insert(0, 'pysuspension')

from attachment_point import AttachmentPoint
from control_arm import ControlArm
from suspension_link import SuspensionLink
from chassis import Chassis
from suspension_knuckle import SuspensionKnuckle

def test_attachment_point_connections():
    """Test AttachmentPoint connection tracking"""
    print("\n" + "="*60)
    print("TEST: AttachmentPoint Connection Tracking")
    print("="*60)

    # Create two attachment points
    ap1 = AttachmentPoint("point1", [100, 200, 300], unit='mm')
    ap2 = AttachmentPoint("point2", [150, 250, 350], unit='mm')

    print(f"Created {ap1.name} at {ap1.position}")
    print(f"Created {ap2.name} at {ap2.position}")

    # Connect them
    ap1.connect_to(ap2)

    print(f"\n{ap1.name} connected to: {[p.name for p in ap1.get_connected_points()]}")
    print(f"{ap2.name} connected to: {[p.name for p in ap2.get_connected_points()]}")

    assert ap1.is_connected_to(ap2), "ap1 should be connected to ap2"
    assert ap2.is_connected_to(ap1), "ap2 should be connected to ap1 (bidirectional)"

    # Disconnect
    ap1.disconnect_from(ap2)

    print(f"\nAfter disconnection:")
    print(f"{ap1.name} connected to: {[p.name for p in ap1.get_connected_points()]}")
    print(f"{ap2.name} connected to: {[p.name for p in ap2.get_connected_points()]}")

    assert not ap1.is_connected_to(ap2), "ap1 should not be connected to ap2"
    assert not ap2.is_connected_to(ap1), "ap2 should not be connected to ap1"

    print("\n✓ AttachmentPoint connection tracking PASSED")
    return True

def test_control_arm_connections():
    """Test AttachmentPoint usage in ControlArm with connections"""
    print("\n" + "="*60)
    print("TEST: ControlArm AttachmentPoint Connections")
    print("="*60)

    # Create control arm
    control_arm = ControlArm(name="upper_arm", mass=2.5, mass_unit='kg')

    link1 = SuspensionLink(
        endpoint1=[1300, 400, 550],
        endpoint2=[1500, 750, 600],
        name="front_link",
        unit='mm'
    )

    control_arm.add_link(link1)
    ball_joint = control_arm.add_attachment_point("ball_joint", [1400, 575, 575], unit='mm')
    sway_bar = control_arm.add_attachment_point("sway_bar", [1250, 500, 500], unit='mm')

    print(f"Control arm has {len(control_arm.attachment_points)} attachment points:")
    for ap in control_arm.attachment_points:
        print(f"  - {ap.name} at {ap.position}")

    # Get attachment points by name
    retrieved_ball_joint = control_arm.get_attachment_point("ball_joint")
    assert retrieved_ball_joint is not None, "Should find ball_joint"
    assert retrieved_ball_joint.name == "ball_joint", "Name should match"
    print(f"\nRetrieved '{retrieved_ball_joint.name}' by name")

    # Test parent component reference
    assert ball_joint.parent_component == control_arm, "Parent component should be set"
    print(f"Parent component correctly set: {ball_joint.parent_component.name}")

    print("\n✓ ControlArm AttachmentPoint usage PASSED")
    return True

def test_chassis_corner_connections():
    """Test connecting chassis corner attachments to knuckle attachments"""
    print("\n" + "="*60)
    print("TEST: Chassis Corner to Knuckle Connections")
    print("="*60)

    # Create chassis with corner
    chassis = Chassis(mass=500, mass_unit='kg')
    fl_corner = chassis.create_corner("front_left")

    # Add attachment points to corner
    upper_mount = fl_corner.add_attachment_point("upper", [1400, 750, 650], unit='mm')
    lower_mount = fl_corner.add_attachment_point("lower", [1400, 750, 450], unit='mm')

    print(f"Chassis corner '{fl_corner.name}' has {len(fl_corner.attachment_points)} attachment points")

    # Create knuckle with attachment points
    knuckle = SuspensionKnuckle(
        tire_center_x=1.5,
        tire_center_y=0.8,
        rolling_radius=0.6,
        unit='m'
    )

    knuckle_upper = knuckle.add_attachment_point(
        "upper_ball_joint",
        [0, 0, 0.15],
        relative=True
    )
    knuckle_lower = knuckle.add_attachment_point(
        "lower_ball_joint",
        [0, 0, -0.15],
        relative=True
    )

    print(f"Knuckle has {len(knuckle.attachment_points)} attachment points")

    # Connect chassis mounts to knuckle ball joints
    upper_mount.connect_to(knuckle_upper)
    lower_mount.connect_to(knuckle_lower)

    print(f"\nConnections established:")
    print(f"  {upper_mount.name} <-> {knuckle_upper.name}")
    print(f"  {lower_mount.name} <-> {knuckle_lower.name}")

    # Verify connections
    assert upper_mount.is_connected_to(knuckle_upper), "Upper mount should be connected"
    assert knuckle_upper.is_connected_to(upper_mount), "Connection should be bidirectional"

    # Get all connected points
    upper_connections = upper_mount.get_connected_points()
    print(f"\nUpper mount connected to: {[p.name for p in upper_connections]}")
    assert len(upper_connections) == 1, "Should have one connection"

    print("\n✓ Chassis-Knuckle connections PASSED")
    return True

def test_connections_persist_through_transforms():
    """Test that connections persist through transformations"""
    print("\n" + "="*60)
    print("TEST: Connections Persist Through Transformations")
    print("="*60)

    # Create control arm with attachments
    control_arm = ControlArm(name="test_arm", mass=2.5, mass_unit='kg')

    link1 = SuspensionLink(
        endpoint1=[1300, 400, 550],
        endpoint2=[1500, 750, 600],
        name="link1",
        unit='mm'
    )

    control_arm.add_link(link1)
    ap1 = control_arm.add_attachment_point("ap1", [1400, 575, 575], unit='mm')
    ap2 = control_arm.add_attachment_point("ap2", [1350, 500, 500], unit='mm')

    # Create external attachment point and connect it
    external_ap = AttachmentPoint("external", [1450, 600, 600], unit='mm')
    ap1.connect_to(external_ap)

    print(f"Initial connection: {ap1.name} <-> {external_ap.name}")
    print(f"  {ap1.name} position: {ap1.position}")
    print(f"  {external_ap.name} position: {external_ap.position}")

    # Apply transformation
    original_positions = control_arm.get_all_attachment_positions(unit='mm')
    target_positions = [pos + np.array([20, -10, -30]) for pos in original_positions]
    rms_error = control_arm.fit_to_attachment_targets(target_positions, unit='mm')

    print(f"\nAfter transformation (RMS error: {rms_error:.3f} mm):")
    print(f"  {ap1.name} position: {ap1.position}")
    print(f"  {external_ap.name} position: {external_ap.position}")

    # Verify connection still exists
    assert ap1.is_connected_to(external_ap), "Connection should persist after transformation"
    print(f"Connection persisted: {ap1.name} still connected to {external_ap.name}")

    # Reset and verify connection still exists
    control_arm.reset_to_origin()

    print(f"\nAfter reset to origin:")
    print(f"  {ap1.name} position: {ap1.position}")

    assert ap1.is_connected_to(external_ap), "Connection should persist after reset"
    print(f"Connection persisted: {ap1.name} still connected to {external_ap.name}")

    print("\n✓ Connections persist through transformations PASSED")
    return True

def test_multi_component_connections():
    """Test connecting attachment points across multiple components"""
    print("\n" + "="*60)
    print("TEST: Multi-Component Attachment Connections")
    print("="*60)

    # Create chassis, control arm, and knuckle
    chassis = Chassis(mass=500, mass_unit='kg')
    fl_corner = chassis.create_corner("front_left")

    # Chassis attachment
    chassis_upper = fl_corner.add_attachment_point("chassis_upper", [1400, 750, 650], unit='mm')
    chassis_lower = fl_corner.add_attachment_point("chassis_lower", [1400, 750, 450], unit='mm')

    # Control arm
    upper_arm = ControlArm(name="upper_arm", mass=1.5, mass_unit='kg')
    link1 = SuspensionLink([1300, 400, 550], [1500, 750, 600], "link1", unit='mm')
    upper_arm.add_link(link1)
    arm_chassis = upper_arm.add_attachment_point("chassis_mount", [1400, 575, 650], unit='mm')
    arm_knuckle = upper_arm.add_attachment_point("knuckle_mount", [1500, 750, 600], unit='mm')

    # Knuckle
    knuckle = SuspensionKnuckle(1.5, 0.8, 0.6, unit='m')
    knuckle_upper = knuckle.add_attachment_point("upper_ball", [0, 0, 0.05], relative=True)

    # Create connection chain: chassis <-> control_arm <-> knuckle
    chassis_upper.connect_to(arm_chassis)
    arm_knuckle.connect_to(knuckle_upper)

    print("Connection chain established:")
    print(f"  Chassis '{chassis_upper.name}' <-> ControlArm '{arm_chassis.name}'")
    print(f"  ControlArm '{arm_knuckle.name}' <-> Knuckle '{knuckle_upper.name}'")

    # Verify all connections
    assert chassis_upper.is_connected_to(arm_chassis), "Chassis-Arm connection"
    assert arm_chassis.is_connected_to(chassis_upper), "Arm-Chassis connection (bidirectional)"
    assert arm_knuckle.is_connected_to(knuckle_upper), "Arm-Knuckle connection"
    assert knuckle_upper.is_connected_to(arm_knuckle), "Knuckle-Arm connection (bidirectional)"

    # Test connection traversal
    print(f"\nConnection traversal from chassis:")
    print(f"  {chassis_upper.name} connects to: {[p.name for p in chassis_upper.get_connected_points()]}")
    for connected in chassis_upper.get_connected_points():
        print(f"    {connected.name} (on {connected.parent_component.name if connected.parent_component else 'unknown'})")
        print(f"      connects to: {[p.name for p in connected.get_connected_points()]}")

    print("\n✓ Multi-component connections PASSED")
    return True

def main():
    """Run all attachment point tests"""
    print("\n" + "="*60)
    print("ATTACHMENT POINT REFACTORING TEST SUITE")
    print("="*60)

    all_passed = True
    tests = [
        ("AttachmentPoint connections", test_attachment_point_connections),
        ("ControlArm attachments", test_control_arm_connections),
        ("Chassis-Knuckle connections", test_chassis_corner_connections),
        ("Transform persistence", test_connections_persist_through_transforms),
        ("Multi-component connections", test_multi_component_connections),
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
        print("✓ ALL ATTACHMENT POINT TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
