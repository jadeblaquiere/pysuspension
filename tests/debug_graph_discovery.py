"""
Debug script to investigate graph discovery.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.chassis import Chassis
from pysuspension.chassis_corner import ChassisCorner
from pysuspension.suspension_knuckle import SuspensionKnuckle
from pysuspension.suspension_link import SuspensionLink
from pysuspension.suspension_joint import SuspensionJoint
from pysuspension.joint_types import JointType
from pysuspension.suspension_graph import discover_suspension_graph


def setup_simple_suspension():
    """Set up a double-wishbone suspension."""
    chassis = Chassis("test_chassis")

    # Create chassis corner with attachment points
    corner = ChassisCorner("front_left")
    uf_chassis = corner.add_attachment_point("upper_front", [1400, 0, 600], unit='mm')
    ur_chassis = corner.add_attachment_point("upper_rear", [1200, 0, 600], unit='mm')
    lf_chassis = corner.add_attachment_point("lower_front", [1500, 0, 300], unit='mm')
    lr_chassis = corner.add_attachment_point("lower_rear", [1100, 0, 300], unit='mm')
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

    # Create upper A-arm links
    upper_front_link = SuspensionLink(
        endpoint1=[1400, 0, 600],
        endpoint2=[1400, 650, 580],
        name="upper_front",
        unit='mm'
    )

    upper_rear_link = SuspensionLink(
        endpoint1=[1200, 0, 600],
        endpoint2=[1400, 650, 580],
        name="upper_rear",
        unit='mm'
    )

    # Create lower A-arm links
    lower_front_link = SuspensionLink(
        endpoint1=[1500, 0, 300],
        endpoint2=[1400, 700, 200],
        name="lower_front",
        unit='mm'
    )

    lower_rear_link = SuspensionLink(
        endpoint1=[1100, 0, 300],
        endpoint2=[1400, 700, 200],
        name="lower_rear",
        unit='mm'
    )

    print("\n=== CHECKING PARENT COMPONENTS ===")
    print(f"\nChassis points:")
    print(f"  uf_chassis.parent_component: {uf_chassis.parent_component} (type: {type(uf_chassis.parent_component).__name__})")
    print(f"  ur_chassis.parent_component: {ur_chassis.parent_component} (type: {type(ur_chassis.parent_component).__name__})")
    print(f"  lf_chassis.parent_component: {lf_chassis.parent_component} (type: {type(lf_chassis.parent_component).__name__})")
    print(f"  lr_chassis.parent_component: {lr_chassis.parent_component} (type: {type(lr_chassis.parent_component).__name__})")

    print(f"\nLink endpoints:")
    print(f"  upper_front_link.endpoint1.parent_component: {upper_front_link.endpoint1.parent_component} (type: {type(upper_front_link.endpoint1.parent_component).__name__})")
    print(f"  upper_front_link.endpoint2.parent_component: {upper_front_link.endpoint2.parent_component} (type: {type(upper_front_link.endpoint2.parent_component).__name__})")

    # Create joints connecting components
    uf_bushing = SuspensionJoint("uf_bushing", JointType.BUSHING_SOFT)
    uf_bushing.add_attachment_point(uf_chassis)
    uf_bushing.add_attachment_point(upper_front_link.endpoint1)

    print(f"\n=== CHECKING JOINT CONNECTIONS ===")
    print(f"uf_bushing has {len(uf_bushing.attachment_points)} attachment points:")
    for i, ap in enumerate(uf_bushing.attachment_points):
        print(f"  {i}: {ap.name} (parent: {type(ap.parent_component).__name__}, joint: {ap.joint.name if ap.joint else None})")

    ur_bushing = SuspensionJoint("ur_bushing", JointType.BUSHING_SOFT)
    ur_bushing.add_attachment_point(ur_chassis)
    ur_bushing.add_attachment_point(upper_rear_link.endpoint1)

    upper_ball_joint = SuspensionJoint("upper_ball", JointType.BALL_JOINT)
    upper_ball_joint.add_attachment_point(upper_front_link.endpoint2)
    upper_ball_joint.add_attachment_point(upper_rear_link.endpoint2)
    upper_ball_joint.add_attachment_point(upper_ball)

    lf_bushing = SuspensionJoint("lf_bushing", JointType.BUSHING_SOFT)
    lf_bushing.add_attachment_point(lf_chassis)
    lf_bushing.add_attachment_point(lower_front_link.endpoint1)

    lr_bushing = SuspensionJoint("lr_bushing", JointType.BUSHING_SOFT)
    lr_bushing.add_attachment_point(lr_chassis)
    lr_bushing.add_attachment_point(lower_rear_link.endpoint1)

    lower_ball_joint = SuspensionJoint("lower_ball", JointType.BALL_JOINT)
    lower_ball_joint.add_attachment_point(lower_front_link.endpoint2)
    lower_ball_joint.add_attachment_point(lower_rear_link.endpoint2)
    lower_ball_joint.add_attachment_point(lower_ball)

    # Register all components with chassis
    chassis.add_component(knuckle)
    chassis.add_joint(uf_bushing)
    chassis.add_joint(ur_bushing)
    chassis.add_joint(upper_ball_joint)
    chassis.add_joint(lf_bushing)
    chassis.add_joint(lr_bushing)
    chassis.add_joint(lower_ball_joint)

    return chassis, 'front_left_knuckle', knuckle


def debug_graph_discovery():
    """Debug the graph discovery."""
    print("=" * 70)
    print("DEBUG: Graph Discovery Investigation")
    print("=" * 70)

    chassis, knuckle_name, knuckle = setup_simple_suspension()

    print("\n=== DISCOVERING SUSPENSION GRAPH ===")
    graph = discover_suspension_graph(knuckle)

    print(f"\nDiscovery results:")
    print(f"  Knuckle: {graph.knuckle.name}")
    print(f"  Control arms: {len(graph.control_arms)}")
    print(f"  Links: {len(graph.links)}")
    print(f"  Joints: {len(graph.joints)}")
    print(f"  Chassis points: {len(graph.chassis_points)}")
    print(f"  All attachment points: {len(graph.all_attachment_points)}")

    print(f"\nChassispoints found:")
    for cp in graph.chassis_points:
        print(f"  {cp.name}: {cp.position} (parent: {type(cp.parent_component).__name__})")

    print(f"\nAll attachment points found:")
    for ap in graph.all_attachment_points:
        parent_name = type(ap.parent_component).__name__ if ap.parent_component else "None"
        print(f"  {ap.name}: parent={parent_name}, joint={ap.joint.name if ap.joint else 'None'}")


if __name__ == "__main__":
    debug_graph_discovery()
