#!/usr/bin/env python3
"""
Test SuspensionLink with AttachmentPoint endpoints.
"""
import sys
import numpy as np
sys.path.insert(0, 'pysuspension')

from suspension_link import SuspensionLink
from attachment_point import AttachmentPoint
from control_arm import ControlArm

def test_link_with_attachment_points():
    """Test SuspensionLink with AttachmentPoint endpoints"""
    print("\n" + "="*60)
    print("TEST: SuspensionLink with AttachmentPoint Endpoints")
    print("="*60)

    # Create a link using numpy arrays (should create AttachmentPoints internally)
    link1 = SuspensionLink([1300, 400, 550], [1500, 750, 600], "link1", unit='mm')

    print(f"Link created from arrays:")
    print(f"  Endpoint1: {link1.endpoint1.name} at {link1.endpoint1.position}")
    print(f"  Endpoint2: {link1.endpoint2.name} at {link1.endpoint2.position}")
    print(f"  Length: {link1.length:.3f} mm")

    # Verify endpoints are AttachmentPoint objects
    assert isinstance(link1.endpoint1, AttachmentPoint), "Endpoint1 should be AttachmentPoint"
    assert isinstance(link1.endpoint2, AttachmentPoint), "Endpoint2 should be AttachmentPoint"
    print("✓ Endpoints are AttachmentPoint objects")

    # Get endpoint AttachmentPoint objects
    ep1 = link1.get_endpoint1_attachment()
    ep2 = link1.get_endpoint2_attachment()

    print(f"\nRetrieved endpoint AttachmentPoint objects:")
    print(f"  EP1 name: {ep1.name}")
    print(f"  EP2 name: {ep2.name}")

    # Test connection tracking through link endpoints
    external_point = AttachmentPoint("external", [1500, 750, 700], unit='mm')
    ep2.connect_to(external_point)

    print(f"\nConnected {ep2.name} to {external_point.name}")
    print(f"  {ep2.name} connected to: {[p.name for p in ep2.get_connected_points()]}")
    assert ep2.is_connected_to(external_point), "Connection should exist"
    print("✓ Connection tracking works on link endpoints")

    # Test that transformations preserve connections
    original_ep2_pos = ep2.position.copy()
    link1.translate([100, 50, -25], unit='mm')

    print(f"\nAfter link translation:")
    print(f"  EP2 position changed from {original_ep2_pos} to {ep2.position}")
    assert ep2.is_connected_to(external_point), "Connection should persist"
    print("✓ Connection persisted through transformation")

    print("\n✓ SuspensionLink AttachmentPoint test PASSED")
    return True

def test_link_created_from_attachment_points():
    """Test creating SuspensionLink directly from AttachmentPoints"""
    print("\n" + "="*60)
    print("TEST: Create SuspensionLink from AttachmentPoint Objects")
    print("="*60)

    # Create AttachmentPoints first
    ap1 = AttachmentPoint("chassis_mount", [1400, 500, 600], unit='mm')
    ap2 = AttachmentPoint("knuckle_mount", [1550, 750, 650], unit='mm')

    print(f"Created AttachmentPoints:")
    print(f"  {ap1.name} at {ap1.position}")
    print(f"  {ap2.name} at {ap2.position}")

    # Create link using these AttachmentPoints
    link = SuspensionLink(ap1, ap2, name="control_arm_link")

    print(f"\nCreated link from AttachmentPoints:")
    print(f"  Link name: {link.name}")
    print(f"  Length: {link.length:.3f} mm")
    print(f"  Endpoint1 name: {link.endpoint1.name}")
    print(f"  Endpoint2 name: {link.endpoint2.name}")

    # Verify the link is using the same AttachmentPoint objects (not copies)
    assert link.endpoint1 is ap1, "Link should use the same AttachmentPoint object"
    assert link.endpoint2 is ap2, "Link should use the same AttachmentPoint object"
    print("✓ Link uses the same AttachmentPoint objects (not copies)")

    # Test that modifying the link moves the AttachmentPoints
    original_ap1_pos = ap1.position.copy()
    link.translate([50, 0, -20], unit='mm')

    print(f"\nAfter link translation:")
    print(f"  AP1 position changed: {not np.allclose(ap1.position, original_ap1_pos)}")
    print(f"  New AP1 position: {ap1.position}")
    assert not np.allclose(ap1.position, original_ap1_pos), "AP1 should have moved"
    print("✓ Modifying link position updates the AttachmentPoints")

    print("\n✓ Create link from AttachmentPoints test PASSED")
    return True

def test_link_endpoints_in_control_arm():
    """Test that link endpoints can be connected to other components"""
    print("\n" + "="*60)
    print("TEST: Connect Link Endpoints to Other Components")
    print("="*60)

    # Create control arm with attachment points
    control_arm = ControlArm(name="upper_arm", mass=2.0, mass_unit='kg')

    # Create external AttachmentPoints (simulating chassis and knuckle mounts)
    chassis_mount = AttachmentPoint("chassis_mount", [1400, 500, 650], unit='mm')
    knuckle_mount = AttachmentPoint("knuckle_mount", [1550, 750, 600], unit='mm')

    # Add attachment points to control arm
    control_arm.add_attachment_point("chassis_mount", [1400, 500, 650], unit='mm')
    control_arm.add_attachment_point("knuckle_mount", [1550, 750, 600], unit='mm')

    # Create link from these attachment points
    link = SuspensionLink(chassis_mount, knuckle_mount, name="link1")

    print(f"Created control arm with link connecting:")
    print(f"  {chassis_mount.name} <-> {knuckle_mount.name}")

    # Now the link endpoints ARE the chassis and knuckle mounts
    # We can query connections through the link
    print(f"\nLink endpoint1 name: {link.endpoint1.name}")
    print(f"Link endpoint2 name: {link.endpoint2.name}")

    # Add another component and connect it
    external_point = AttachmentPoint("external", [1400, 500, 750], unit='mm')
    chassis_mount.connect_to(external_point)

    print(f"\nConnected {chassis_mount.name} to {external_point.name}")
    print(f"Connections via link.endpoint1: {[p.name for p in link.endpoint1.get_connected_points()]}")

    assert link.endpoint1.is_connected_to(external_point), "Connection should exist"
    print("✓ Can connect link endpoints to other components")

    # Test that link transformations preserve connections
    # (Control arm fit requires at least 3 points, so we just transform the link directly)
    original_pos = link.endpoint1.position.copy()
    link.translate([20, -10, -30], unit='mm')

    print(f"\nAfter link transformation:")
    print(f"  Endpoint1 moved from {original_pos} to {link.endpoint1.position}")
    assert link.endpoint1.is_connected_to(external_point), "Connection should persist"
    print("✓ Connections persist through link transformations")

    print("\n✓ Link endpoint connections test PASSED")
    return True

def main():
    """Run all SuspensionLink tests"""
    print("\n" + "="*60)
    print("SUSPENSION LINK WITH ATTACHMENTPOINT TEST SUITE")
    print("="*60)

    all_passed = True
    tests = [
        ("Link with AttachmentPoints", test_link_with_attachment_points),
        ("Create link from AttachmentPoints", test_link_created_from_attachment_points),
        ("Link endpoint connections", test_link_endpoints_in_control_arm),
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
        print("✓ ALL SUSPENSION LINK TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
