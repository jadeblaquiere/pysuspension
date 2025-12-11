"""
Integration tests for the new joint-centric API.

Tests the Phase 2 implementation where joints are defined explicitly
and used to generate constraints automatically.
"""

import pytest
import numpy as np
from pysuspension.corner_solver import CornerSolver
from pysuspension.attachment_point import AttachmentPoint
from pysuspension.control_arm import ControlArm
from pysuspension.suspension_link import SuspensionLink
from pysuspension.joint_types import JointType
from pysuspension.suspension_knuckle import SuspensionKnuckle


class TestNewControlArmAPI:
    """Test the new add_control_arm() API with explicit joints."""

    def test_basic_control_arm_with_joints(self):
        """Test adding a control arm using the new joint-centric API."""
        solver = CornerSolver("test")

        # Create knuckle attachment point
        knuckle_upper = AttachmentPoint("knuckle_upper", [1400, 1400, 580], unit='mm')

        # Create chassis attachment points
        chassis_front = AttachmentPoint("chassis_front", [1300, 0, 600], unit='mm')
        chassis_rear = AttachmentPoint("chassis_rear", [1200, 0, 600], unit='mm')

        # Create control arm
        upper_arm = ControlArm("upper_control_arm")
        front_link = SuspensionLink([1300, 0, 600], [1400, 1400, 580], "front_link", unit='mm')
        rear_link = SuspensionLink([1200, 0, 600], [1400, 1400, 580], "rear_link", unit='mm')
        upper_arm.add_link(front_link)
        upper_arm.add_link(rear_link)

        # Get control arm link endpoints (these are the attachment points)
        arm_front = front_link.endpoint1   # Front chassis mount on arm
        arm_rear = rear_link.endpoint1     # Rear chassis mount on arm
        arm_knuckle = front_link.endpoint2  # Knuckle mount on arm (shared by both links)

        # Define joints explicitly (NEW API)
        solver.add_joint(
            name="upper_ball_joint",
            points=[arm_knuckle, knuckle_upper],
            joint_type=JointType.BALL_JOINT
        )

        solver.add_joint(
            name="front_bushing",
            points=[arm_front, chassis_front],
            joint_type=JointType.BUSHING_SOFT
        )

        solver.add_joint(
            name="rear_bushing",
            points=[arm_rear, chassis_rear],
            joint_type=JointType.BUSHING_SOFT
        )

        # Add control arm with new API (no joint_type parameter!)
        solver.add_control_arm(
            control_arm=upper_arm,
            chassis_mount_points=[arm_front, arm_rear],
            knuckle_mount_points=[arm_knuckle]
        )

        # Verify joints were created
        assert len(solver.joints) == 3
        assert "upper_ball_joint" in solver.list_joints()
        assert "front_bushing" in solver.list_joints()
        assert "rear_bushing" in solver.list_joints()

        # Verify control arm was added
        assert len(solver.control_arms) == 1
        assert solver.control_arms[0] is upper_arm

        # Verify chassis mounts were marked correctly
        assert arm_front in solver.chassis_mounts
        assert arm_rear in solver.chassis_mounts

        # Verify knuckle points were marked
        assert arm_knuckle in solver.knuckle_points

        # Verify constraints were generated
        # Should have: 2 fixed point constraints (chassis), 2 distance constraints (links),
        # and 3 coincident constraints (one for each joint)
        constraint_types = {}
        for constraint in solver.constraints:
            constraint_type = type(constraint).__name__
            constraint_types[constraint_type] = constraint_types.get(constraint_type, 0) + 1

        assert constraint_types.get('FixedPointConstraint', 0) == 2  # chassis mounts
        assert constraint_types.get('DistanceConstraint', 0) == 2   # rigid links
        assert constraint_types.get('CoincidentPointConstraint', 0) == 3  # joints

    def test_mixed_joint_types(self):
        """Test control arm with different joint types at different mounts."""
        solver = CornerSolver("test")

        # Create attachment points
        knuckle_pt = AttachmentPoint("knuckle", [1400, 1400, 580], unit='mm')
        chassis_f = AttachmentPoint("chassis_f", [1300, 0, 600], unit='mm')
        chassis_r = AttachmentPoint("chassis_r", [1200, 0, 600], unit='mm')

        # Create control arm
        arm = ControlArm("test_arm")
        link1 = SuspensionLink([1300, 0, 600], [1400, 1400, 580], "link1", unit='mm')
        link2 = SuspensionLink([1200, 0, 600], [1400, 1400, 580], "link2", unit='mm')
        arm.add_link(link1)
        arm.add_link(link2)

        # Define joints with DIFFERENT types - this is the key feature!
        solver.add_joint("ball", [link1.endpoint2, knuckle_pt], JointType.BALL_JOINT)  # Stiff
        solver.add_joint("bush1", [link1.endpoint1, chassis_f], JointType.BUSHING_SOFT)  # Compliant
        solver.add_joint("bush2", [link2.endpoint1, chassis_r], JointType.BUSHING_SOFT)  # Compliant

        # Add control arm
        solver.add_control_arm(
            control_arm=arm,
            chassis_mount_points=[link1.endpoint1, link2.endpoint1],
            knuckle_mount_points=[link1.endpoint2]
        )

        # Verify joint stiffnesses are different
        ball_stiffness = solver.get_joint_stiffness("ball")
        bush_stiffness = solver.get_joint_stiffness("bush1")

        assert ball_stiffness > bush_stiffness  # Ball joint is stiffer than bushing
        assert ball_stiffness == 100000.0  # BALL_JOINT stiffness
        assert bush_stiffness == 100.0     # BUSHING_SOFT stiffness

    def test_control_arm_without_chassis_mounts(self):
        """Test adding a control arm with no chassis mount points specified."""
        solver = CornerSolver("test")

        # Create a simple control arm (e.g., a floating link in space)
        arm = ControlArm("floating_arm")
        link = SuspensionLink([0, 0, 0], [100, 0, 0], "link", unit='mm')
        arm.add_link(link)

        # Add without specifying any chassis or knuckle mounts
        solver.add_control_arm(control_arm=arm)

        assert len(solver.control_arms) == 1
        assert len(solver.chassis_mounts) == 0


class TestNewLinkAPI:
    """Test the new add_link() API with explicit joints."""

    def test_basic_link_with_joints(self):
        """Test adding a link using the new joint-centric API."""
        solver = CornerSolver("test")

        # Create link
        damper = SuspensionLink([1400, 1400, 500], [1300, 200, 800], "damper", unit='mm')

        # Create mount points
        lower_mount = AttachmentPoint("lower_mount", [1400, 1400, 500], unit='mm')
        upper_mount = AttachmentPoint("upper_mount", [1300, 200, 800], unit='mm')

        # Mark upper mount as chassis
        solver.chassis_mounts.append(upper_mount)

        # Define joints
        solver.add_joint("damper_lower", [damper.endpoint1, lower_mount], JointType.BALL_JOINT)
        solver.add_joint("damper_upper", [damper.endpoint2, upper_mount], JointType.BUSHING_HARD)

        # Add link with new API
        solver.add_link(
            link=damper,
            end1_mount_point=lower_mount,
            end2_mount_point=upper_mount
        )

        # Verify link was added
        assert len(solver.links) == 1
        assert solver.links[0] is damper

        # Verify joints exist
        assert len(solver.joints) == 2

        # Verify different joint types
        lower_stiffness = solver.get_joint_stiffness("damper_lower")
        upper_stiffness = solver.get_joint_stiffness("damper_upper")

        assert lower_stiffness == 100000.0  # BALL_JOINT
        assert upper_stiffness == 1000.0    # BUSHING_HARD

    def test_link_without_mount_points(self):
        """Test adding a link without specifying mount points."""
        solver = CornerSolver("test")

        link = SuspensionLink([0, 0, 0], [100, 0, 0], "test_link", unit='mm')

        # Add link without mount points
        solver.add_link(link=link)

        assert len(solver.links) == 1
        assert solver.links[0] is link


class TestConstraintGeneration:
    """Test that constraints are generated correctly from joints."""

    def test_coincident_constraints_from_joints(self):
        """Test that coincident constraints are auto-generated from joints."""
        solver = CornerSolver("test")

        # Create a control arm with a link
        arm = ControlArm("test_arm")
        link = SuspensionLink([0, 0, 0], [100, 0, 0], "link", unit='mm')  # Non-zero length
        arm.add_link(link)

        # Create separate point to connect to
        point1 = link.endpoint1
        point2 = AttachmentPoint("p2", [0, 0, 0], unit='mm')  # Same position as endpoint1

        # Define joint connecting the points
        solver.add_joint("joint1", [point1, point2], JointType.BALL_JOINT)

        # Add control arm - should auto-generate coincident constraint
        solver.add_control_arm(control_arm=arm)

        # Find the coincident constraint
        coincident_constraints = [c for c in solver.constraints
                                 if type(c).__name__ == 'CoincidentPointConstraint']

        assert len(coincident_constraints) >= 1

        # Verify the constraint has correct joint type
        constraint = coincident_constraints[0]
        assert constraint.joint_type == JointType.BALL_JOINT

    def test_no_duplicate_constraints(self):
        """Test that duplicate coincident constraints are not created."""
        solver = CornerSolver("test")

        # Create control arm
        arm = ControlArm("arm")
        link = SuspensionLink([0, 0, 0], [100, 0, 0], "link", unit='mm')
        arm.add_link(link)

        # Use link endpoints
        p1 = link.endpoint1
        p2 = AttachmentPoint("p2", [0, 0, 0], unit='mm')

        # Define joint
        solver.add_joint("joint1", [p1, p2], JointType.BALL_JOINT)

        # Add control arm twice (shouldn't create duplicate constraints)
        solver.add_control_arm(control_arm=arm)

        # Count coincident constraints
        coincident_count = sum(1 for c in solver.constraints
                              if type(c).__name__ == 'CoincidentPointConstraint')

        # Should only have one coincident constraint for the joint
        assert coincident_count == 1


class TestAutoInferenceIntegration:
    """Test that joint type auto-inference works end-to-end."""

    def test_end_to_end_auto_inference(self):
        """Test complete workflow with auto-inference."""
        solver = CornerSolver("test")

        # Setup points
        knuckle = AttachmentPoint("knuckle", [1400, 1400, 580], unit='mm')
        chassis = AttachmentPoint("chassis", [1300, 0, 600], unit='mm')

        # Create control arm
        arm = ControlArm("arm")
        link = SuspensionLink([1300, 0, 600], [1400, 1400, 580], "link", unit='mm')
        arm.add_link(link)

        # Define joints using link endpoints
        solver.add_joint("soft_bush", [link.endpoint1, chassis], JointType.BUSHING_SOFT)
        solver.add_joint("ball", [link.endpoint2, knuckle], JointType.BALL_JOINT)

        # Add control arm
        solver.add_control_arm(
            control_arm=arm,
            chassis_mount_points=[link.endpoint1],
            knuckle_mount_points=[link.endpoint2]
        )

        # Find constraints and verify they have correct joint types
        for constraint in solver.constraints:
            if type(constraint).__name__ == 'CoincidentPointConstraint':
                if 'soft_bush' in constraint.name:
                    assert constraint.joint_type == JointType.BUSHING_SOFT
                    assert constraint.stiffness == 100.0
                elif 'ball' in constraint.name:
                    assert constraint.joint_type == JointType.BALL_JOINT
                    assert constraint.stiffness == 100000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
