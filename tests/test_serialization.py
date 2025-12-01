import sys
import os
import json
import numpy as np

# Add parent directory to path for pysuspension imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pysuspension.attachment_point import AttachmentPoint
from pysuspension.suspension_link import SuspensionLink
from pysuspension.control_arm import ControlArm
from pysuspension.suspension_knuckle import SuspensionKnuckle
from pysuspension.steering_rack import SteeringRack
from pysuspension.chassis_corner import ChassisCorner
from pysuspension.chassis_axle import ChassisAxle
from pysuspension.chassis import Chassis


def test_attachment_point_serialization():
    """Test AttachmentPoint serialization and deserialization."""
    print("\n--- Testing AttachmentPoint Serialization ---")

    # Create an attachment point
    ap = AttachmentPoint("test_point", [100.0, 200.0, 300.0], is_relative=True, unit='mm')

    # Serialize
    data = ap.to_dict()
    print(f"Serialized: {data}")

    # Check JSON serializability
    json_str = json.dumps(data)
    print(f"JSON: {json_str[:80]}...")

    # Deserialize
    ap_restored = AttachmentPoint.from_dict(data)

    # Verify
    assert ap_restored.name == ap.name
    assert np.allclose(ap_restored.position, ap.position)
    assert ap_restored.is_relative == ap.is_relative
    print("✓ AttachmentPoint serialization test passed")


def test_suspension_link_serialization():
    """Test SuspensionLink serialization and deserialization."""
    print("\n--- Testing SuspensionLink Serialization ---")

    # Create a suspension link
    link = SuspensionLink(
        endpoint1=[1400, 500, 600],
        endpoint2=[1500, 750, 600],
        name="test_link",
        unit='mm'
    )

    # Serialize
    data = link.to_dict()
    print(f"Serialized: {json.dumps(data, indent=2)[:200]}...")

    # Check JSON serializability
    json_str = json.dumps(data)

    # Deserialize
    link_restored = SuspensionLink.from_dict(data)

    # Verify
    assert link_restored.name == link.name
    assert np.allclose(link_restored.endpoint1.position, link.endpoint1.position)
    assert np.allclose(link_restored.endpoint2.position, link.endpoint2.position)
    assert abs(link_restored.length - link.length) < 1e-3
    print("✓ SuspensionLink serialization test passed")


def test_control_arm_serialization():
    """Test ControlArm serialization and deserialization."""
    print("\n--- Testing ControlArm Serialization ---")

    # Create a control arm with links and attachment points
    control_arm = ControlArm(name="test_control_arm", mass=2.5, mass_unit='kg')

    link1 = SuspensionLink(
        endpoint1=[1300, 400, 550],
        endpoint2=[1500, 750, 600],
        name="front_link",
        unit='mm'
    )
    control_arm.add_link(link1)

    link2 = SuspensionLink(
        endpoint1=[1400, 350, 520],
        endpoint2=[1500, 750, 600],
        name="rear_link",
        unit='mm'
    )
    control_arm.add_link(link2)

    control_arm.add_attachment_point("mount1", [1300, 400, 550], unit='mm')
    control_arm.add_attachment_point("mount2", [1400, 350, 520], unit='mm')

    # Serialize
    data = control_arm.to_dict()
    print(f"Serialized control arm with {len(data['links'])} links and {len(data['attachment_points'])} attachment points")

    # Check JSON serializability
    json_str = json.dumps(data)

    # Deserialize
    ca_restored = ControlArm.from_dict(data)

    # Verify
    assert ca_restored.name == control_arm.name
    assert abs(ca_restored.mass - control_arm.mass) < 1e-6
    assert len(ca_restored.links) == len(control_arm.links)
    assert len(ca_restored.attachment_points) == len(control_arm.attachment_points)
    print("✓ ControlArm serialization test passed")


def test_suspension_knuckle_serialization():
    """Test SuspensionKnuckle serialization and deserialization."""
    print("\n--- Testing SuspensionKnuckle Serialization ---")

    # Create a suspension knuckle
    knuckle = SuspensionKnuckle(
        tire_center_x=1500,
        tire_center_y=800,
        rolling_radius=350,
        toe_angle=1.5,
        camber_angle=-2.0,
        wheel_offset=50,
        mass=15.0,
        unit='mm',
        mass_unit='kg'
    )

    # Use absolute positioning (tire_center + offset)
    # Note: Refactored API now uses absolute positioning only
    tire_center = np.array([1500, 800, 350])
    knuckle.add_attachment_point("upper_ball_joint", tire_center + np.array([0, 50, 100]), unit='mm')
    knuckle.add_attachment_point("lower_ball_joint", tire_center + np.array([0, 50, -100]), unit='mm')
    knuckle.add_attachment_point("tie_rod", tire_center + np.array([100, 0, 50]), unit='mm')
    knuckle.steering_attachment_name = "tie_rod"

    # Serialize
    data = knuckle.to_dict()
    print(f"Serialized knuckle: toe={data['toe_angle']}°, camber={data['camber_angle']}°")

    # Check JSON serializability
    json_str = json.dumps(data)

    # Deserialize
    knuckle_restored = SuspensionKnuckle.from_dict(data)

    # Verify
    assert np.allclose(knuckle_restored.tire_center, knuckle.tire_center)
    assert abs(knuckle_restored.toe_angle - knuckle.toe_angle) < 1e-6
    assert abs(knuckle_restored.camber_angle - knuckle.camber_angle) < 1e-6
    assert abs(knuckle_restored.wheel_offset - knuckle.wheel_offset) < 1e-3
    assert abs(knuckle_restored.mass - knuckle.mass) < 1e-6
    assert len(knuckle_restored.attachment_points) == len(knuckle.attachment_points)
    assert knuckle_restored.steering_attachment_name == knuckle.steering_attachment_name
    print("✓ SuspensionKnuckle serialization test passed")


def test_steering_rack_serialization():
    """Test SteeringRack serialization and deserialization."""
    print("\n--- Testing SteeringRack Serialization ---")

    # Create a steering rack
    housing_mounts = [
        [1400, 200, 350],
        [1400, -200, 350],
        [1500, 0, 350],
    ]

    rack = SteeringRack(
        housing_attachments=housing_mounts,
        left_inner_pivot=[1450, 300, 350],
        right_inner_pivot=[1450, -300, 350],
        left_outer_attachment=[1550, 650, 350],
        right_outer_attachment=[1550, -650, 350],
        travel_per_rotation=1.0,
        max_displacement=100.0,
        name="front_rack",
        unit='mm'
    )

    # Set a turn angle
    rack.set_turn_angle(15.0)

    # Serialize
    data = rack.to_dict()
    # Note: Refactored API now stores housing as RigidBody
    housing_points = len(data['housing']['attachment_points'])
    print(f"Serialized rack: {housing_points} housing points, angle={data['current_angle']}°")

    # Check JSON serializability
    json_str = json.dumps(data)

    # Deserialize
    rack_restored = SteeringRack.from_dict(data)

    # Verify
    assert rack_restored.name == rack.name
    assert len(rack_restored.housing.attachment_points) == len(rack.housing.attachment_points)
    assert abs(rack_restored.travel_per_rotation - rack.travel_per_rotation) < 1e-3
    assert abs(rack_restored.max_displacement - rack.max_displacement) < 1e-3
    assert abs(rack_restored.current_angle - rack.current_angle) < 1e-6
    print("✓ SteeringRack serialization test passed")


def test_chassis_corner_serialization():
    """Test ChassisCorner serialization and deserialization."""
    print("\n--- Testing ChassisCorner Serialization ---")

    # Create a chassis corner
    corner = ChassisCorner("front_left")
    corner.add_attachment_point("upper_front", [1300, 800, 600], unit='mm')
    corner.add_attachment_point("upper_rear", [1400, 800, 550], unit='mm')
    corner.add_attachment_point("lower_front", [1300, 800, 200], unit='mm')
    corner.add_attachment_point("lower_rear", [1400, 800, 150], unit='mm')

    # Serialize
    data = corner.to_dict()
    print(f"Serialized corner: {data['name']} with {len(data['attachment_points'])} attachment points")

    # Check JSON serializability
    json_str = json.dumps(data)

    # Deserialize
    corner_restored = ChassisCorner.from_dict(data)

    # Verify
    assert corner_restored.name == corner.name
    assert len(corner_restored.attachment_points) == len(corner.attachment_points)
    for i, ap in enumerate(corner_restored.attachment_points):
        assert ap.name == corner.attachment_points[i].name
        assert np.allclose(ap.position, corner.attachment_points[i].position)
    print("✓ ChassisCorner serialization test passed")


def test_chassis_serialization():
    """Test full Chassis serialization with corners and axles."""
    print("\n--- Testing Chassis Serialization ---")

    # Create a chassis
    chassis = Chassis(name="test_chassis", mass=150.0, mass_unit='kg')

    # Add corners
    fl = chassis.create_corner("front_left")
    fl.add_attachment_point("upper", [1300, 800, 600], unit='mm')
    fl.add_attachment_point("lower", [1300, 800, 200], unit='mm')

    fr = chassis.create_corner("front_right")
    fr.add_attachment_point("upper", [1300, -800, 600], unit='mm')
    fr.add_attachment_point("lower", [1300, -800, 200], unit='mm')

    # Add an axle
    axle = chassis.create_axle("front_axle", ["front_left", "front_right"])
    axle.add_attachment_point("rack_mount1", [1400, 200, 350], unit='mm')
    axle.add_attachment_point("rack_mount2", [1400, -200, 350], unit='mm')

    # Serialize
    data = chassis.to_dict()
    print(f"Serialized chassis: {len(data['corners'])} corners, {len(data['axles'])} axles")
    print(f"Corners: {list(data['corners'].keys())}")
    print(f"Axles: {list(data['axles'].keys())}")

    # Check JSON serializability
    json_str = json.dumps(data, indent=2)
    print(f"JSON length: {len(json_str)} characters")

    # Deserialize
    chassis_restored = Chassis.from_dict(data)

    # Verify
    assert chassis_restored.name == chassis.name
    assert abs(chassis_restored.mass - chassis.mass) < 1e-6
    assert len(chassis_restored.corners) == len(chassis.corners)
    assert len(chassis_restored.axles) == len(chassis.axles)

    # Verify corners
    for corner_name in chassis.corners:
        assert corner_name in chassis_restored.corners
        orig_corner = chassis.corners[corner_name]
        rest_corner = chassis_restored.corners[corner_name]
        assert len(rest_corner.attachment_points) == len(orig_corner.attachment_points)

    # Verify axles
    for axle_name in chassis.axles:
        assert axle_name in chassis_restored.axles
        orig_axle = chassis.axles[axle_name]
        rest_axle = chassis_restored.axles[axle_name]
        assert rest_axle.corner_names == orig_axle.corner_names
        assert len(rest_axle.attachment_points) == len(orig_axle.attachment_points)

    print("✓ Chassis serialization test passed")


def test_json_round_trip():
    """Test complete JSON round-trip for a complex chassis configuration."""
    print("\n--- Testing JSON Round-Trip ---")

    # Create a complex chassis
    chassis = Chassis(name="complete_suspension", mass=200.0, mass_unit='kg')

    # Add all four corners
    for side in ['left', 'right']:
        for end in ['front', 'rear']:
            corner_name = f"{end}_{side}"
            corner = chassis.create_corner(corner_name)
            y_sign = 1 if side == 'left' else -1
            x_pos = 1300 if end == 'front' else 2700
            corner.add_attachment_point("upper", [x_pos, y_sign * 800, 600], unit='mm')
            corner.add_attachment_point("lower", [x_pos, y_sign * 800, 200], unit='mm')

    # Add axles
    front_axle = chassis.create_axle("front_axle", ["front_left", "front_right"])
    front_axle.add_attachment_point("steering_rack_mount", [1400, 0, 350], unit='mm')

    rear_axle = chassis.create_axle("rear_axle", ["rear_left", "rear_right"])

    # Serialize to JSON string
    data = chassis.to_dict()
    json_str = json.dumps(data, indent=2)
    print(f"JSON serialization successful ({len(json_str)} chars)")

    # Parse JSON back to dict
    data_from_json = json.loads(json_str)

    # Deserialize from dict
    chassis_restored = Chassis.from_dict(data_from_json)

    # Verify structure
    assert chassis_restored.name == chassis.name
    assert len(chassis_restored.corners) == 4
    assert len(chassis_restored.axles) == 2
    assert "front_axle" in chassis_restored.axles
    assert "rear_axle" in chassis_restored.axles

    print("✓ JSON round-trip test passed")


def test_serialization_after_transformation():
    """Test serialization after transforming components."""
    print("\n--- Testing Serialization After Transformation ---")

    # Create a chassis and transform it
    chassis = Chassis(name="transformed_chassis", mass=100.0, mass_unit='kg')
    corner = chassis.create_corner("test_corner")
    corner.add_attachment_point("point1", [1000, 500, 300], unit='mm')
    corner.add_attachment_point("point2", [1100, 500, 300], unit='mm')

    # Transform the chassis
    chassis.translate([100, 0, 0], unit='mm')

    # Serialize
    data = chassis.to_dict()
    json_str = json.dumps(data)

    # Deserialize
    chassis_restored = Chassis.from_dict(data)

    # Verify transformed positions are preserved
    orig_positions = corner.get_attachment_positions(unit='mm')
    rest_positions = chassis_restored.corners["test_corner"].get_attachment_positions(unit='mm')

    for orig, rest in zip(orig_positions, rest_positions):
        assert np.allclose(orig, rest, atol=1e-6)

    print("✓ Serialization after transformation test passed")


if __name__ == "__main__":
    print("=" * 70)
    print("SERIALIZATION TESTS")
    print("=" * 70)

    test_attachment_point_serialization()
    test_suspension_link_serialization()
    test_control_arm_serialization()
    test_suspension_knuckle_serialization()
    test_steering_rack_serialization()
    test_chassis_corner_serialization()
    test_chassis_serialization()
    test_json_round_trip()
    test_serialization_after_transformation()

    print("\n" + "=" * 70)
    print("✓ ALL SERIALIZATION TESTS PASSED")
    print("=" * 70)
