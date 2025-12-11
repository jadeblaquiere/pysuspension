"""
Tests for joint API in CornerSolver.

This module tests the joint-centric design where joints are first-class objects:
- Joint registry and creation
- Joint inspection methods
- Auto-inference of joint types in constraints
- Validation and error handling
"""

import pytest
import numpy as np
from pysuspension.corner_solver import CornerSolver
from pysuspension.attachment_point import AttachmentPoint
from pysuspension.joint_types import JointType, JOINT_STIFFNESS
from pysuspension.constraints import CoincidentPointConstraint


class TestJointRegistry:
    """Test joint registry and basic operations."""

    def test_empty_registry(self):
        """Test that solver starts with empty joint registry."""
        solver = CornerSolver("test")
        assert len(solver.joints) == 0
        assert solver.list_joints() == []

    def test_add_joint_basic(self):
        """Test adding a basic joint."""
        solver = CornerSolver("test")

        # Create attachment points
        point1 = AttachmentPoint("point1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("point2", [100, 0, 0], unit='mm')

        # Add joint
        joint = solver.add_joint(
            name="test_joint",
            points=[point1, point2],
            joint_type=JointType.BALL_JOINT
        )

        # Verify joint was created and registered
        assert joint.name == "test_joint"
        assert joint.joint_type == JointType.BALL_JOINT
        assert len(joint.attachment_points) == 2
        assert "test_joint" in solver.joints
        assert len(solver.list_joints()) == 1

    def test_add_joint_multiple_points(self):
        """Test adding a joint with more than 2 points."""
        solver = CornerSolver("test")

        # Create 3 attachment points (e.g., for a multi-point bushing)
        points = [
            AttachmentPoint(f"point{i}", [i*100, 0, 0], unit='mm')
            for i in range(3)
        ]

        # Add joint connecting all 3 points
        joint = solver.add_joint(
            name="multi_joint",
            points=points,
            joint_type=JointType.BUSHING_SOFT
        )

        assert len(joint.attachment_points) == 3
        assert joint.joint_type == JointType.BUSHING_SOFT

    def test_add_joint_custom_stiffness(self):
        """Test adding a joint with custom stiffness."""
        solver = CornerSolver("test")

        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

        # Add joint with custom stiffness
        custom_stiffness = 750.0  # N/mm
        joint = solver.add_joint(
            name="custom_joint",
            points=[point1, point2],
            joint_type=JointType.CUSTOM,
            stiffness=custom_stiffness
        )

        # Verify custom stiffness is stored
        assert hasattr(joint, 'stiffness')
        assert joint.stiffness == custom_stiffness

        # Verify inspection methods return custom stiffness
        assert solver.get_joint_stiffness("custom_joint") == custom_stiffness
        assert solver.get_joint_compliance("custom_joint") == 1.0 / custom_stiffness

    def test_add_joint_duplicate_name(self):
        """Test that duplicate joint names raise an error."""
        solver = CornerSolver("test")

        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')
        point3 = AttachmentPoint("p3", [200, 0, 0], unit='mm')

        # Add first joint
        solver.add_joint("joint1", [point1, point2], JointType.BALL_JOINT)

        # Try to add another joint with same name
        with pytest.raises(ValueError, match="already exists"):
            solver.add_joint("joint1", [point2, point3], JointType.BALL_JOINT)

    def test_add_joint_insufficient_points(self):
        """Test that joints require at least 2 points."""
        solver = CornerSolver("test")

        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')

        # Try to create joint with only 1 point
        with pytest.raises(ValueError, match="at least 2 points"):
            solver.add_joint("bad_joint", [point1], JointType.BALL_JOINT)

        # Try to create joint with 0 points
        with pytest.raises(ValueError, match="at least 2 points"):
            solver.add_joint("bad_joint", [], JointType.BALL_JOINT)

    def test_joint_back_references(self):
        """Test that attachment points have back-references to joints."""
        solver = CornerSolver("test")

        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

        # Initially, points should have no joint
        assert point1.joint is None
        assert point2.joint is None

        # Add joint
        joint = solver.add_joint("joint1", [point1, point2], JointType.BALL_JOINT)

        # Now points should reference the joint
        assert point1.joint is joint
        assert point2.joint is joint


class TestJointInspection:
    """Test joint inspection and query methods."""

    def test_get_joint(self):
        """Test getting a joint by name."""
        solver = CornerSolver("test")

        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

        original_joint = solver.add_joint("joint1", [point1, point2], JointType.BALL_JOINT)
        retrieved_joint = solver.get_joint("joint1")

        assert retrieved_joint is original_joint
        assert retrieved_joint.name == "joint1"

    def test_get_joint_not_found(self):
        """Test that getting nonexistent joint raises KeyError."""
        solver = CornerSolver("test")

        with pytest.raises(KeyError, match="not found"):
            solver.get_joint("nonexistent")

    def test_get_joints_at_point(self):
        """Test finding all joints connected to a point."""
        solver = CornerSolver("test")

        # Create points
        chassis = AttachmentPoint("chassis", [0, 0, 0], unit='mm')
        arm_front = AttachmentPoint("arm_front", [100, 0, 0], unit='mm')
        arm_rear = AttachmentPoint("arm_rear", [200, 0, 0], unit='mm')
        knuckle = AttachmentPoint("knuckle", [150, 100, 0], unit='mm')

        # Add joints
        # arm_front connects to both chassis and knuckle
        joint1 = solver.add_joint("joint1", [chassis, arm_front], JointType.BUSHING_SOFT)
        joint2 = solver.add_joint("joint2", [arm_front, knuckle], JointType.BALL_JOINT)
        # arm_rear only connects to chassis
        joint3 = solver.add_joint("joint3", [chassis, arm_rear], JointType.BUSHING_SOFT)

        # Query joints at arm_front - should get joint1 and joint2
        joints_at_arm_front = solver.get_joints_at_point(arm_front)
        assert len(joints_at_arm_front) == 2
        assert joint1 in joints_at_arm_front
        assert joint2 in joints_at_arm_front

        # Query joints at chassis - should get joint1 and joint3
        joints_at_chassis = solver.get_joints_at_point(chassis)
        assert len(joints_at_chassis) == 2
        assert joint1 in joints_at_chassis
        assert joint3 in joints_at_chassis

        # Query joints at knuckle - should only get joint2
        joints_at_knuckle = solver.get_joints_at_point(knuckle)
        assert len(joints_at_knuckle) == 1
        assert joint2 in joints_at_knuckle

        # Query joints at arm_rear - should only get joint3
        joints_at_arm_rear = solver.get_joints_at_point(arm_rear)
        assert len(joints_at_arm_rear) == 1
        assert joint3 in joints_at_arm_rear

    def test_list_joints(self):
        """Test listing all joint names."""
        solver = CornerSolver("test")

        assert solver.list_joints() == []

        # Add some joints
        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')
        point3 = AttachmentPoint("p3", [200, 0, 0], unit='mm')

        solver.add_joint("joint_a", [point1, point2], JointType.BALL_JOINT)
        solver.add_joint("joint_b", [point2, point3], JointType.BUSHING_SOFT)

        joint_names = solver.list_joints()
        assert len(joint_names) == 2
        assert "joint_a" in joint_names
        assert "joint_b" in joint_names

    def test_get_joint_stiffness_standard(self):
        """Test getting stiffness for standard joint types."""
        solver = CornerSolver("test")

        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

        # Test ball joint (very stiff)
        solver.add_joint("ball", [point1, point2], JointType.BALL_JOINT)
        ball_stiffness = solver.get_joint_stiffness("ball")
        assert ball_stiffness == JOINT_STIFFNESS[JointType.BALL_JOINT]
        assert ball_stiffness == 100000.0  # N/mm

        # Test soft bushing (compliant)
        point3 = AttachmentPoint("p3", [200, 0, 0], unit='mm')
        solver.add_joint("bushing", [point2, point3], JointType.BUSHING_SOFT)
        bushing_stiffness = solver.get_joint_stiffness("bushing")
        assert bushing_stiffness == JOINT_STIFFNESS[JointType.BUSHING_SOFT]
        assert bushing_stiffness == 100.0  # N/mm

    def test_get_joint_compliance(self):
        """Test getting compliance (inverse of stiffness)."""
        solver = CornerSolver("test")

        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

        # Add a bushing with known stiffness
        solver.add_joint("bushing", [point1, point2], JointType.BUSHING_SOFT)

        stiffness = solver.get_joint_stiffness("bushing")
        compliance = solver.get_joint_compliance("bushing")

        # Compliance should be inverse of stiffness
        assert abs(compliance - 1.0 / stiffness) < 1e-9
        assert abs(compliance - 1.0 / 100.0) < 1e-9  # BUSHING_SOFT = 100 N/mm


class TestJointAutoInference:
    """Test automatic inference of joint types in constraints."""

    def test_auto_inference_from_joint(self):
        """Test that constraints automatically infer joint type from attachment points."""
        solver = CornerSolver("test")

        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

        # Add joint connecting the points
        joint = solver.add_joint(
            "test_joint",
            [point1, point2],
            JointType.BUSHING_SOFT
        )

        # Create constraint with joint_type=None (auto-infer)
        constraint = CoincidentPointConstraint(
            point1,
            point2,
            name="test_constraint",
            joint_type=None  # Should auto-infer from joint
        )

        # Constraint should have inferred the joint type
        assert constraint.joint_type == JointType.BUSHING_SOFT
        assert constraint.stiffness == JOINT_STIFFNESS[JointType.BUSHING_SOFT]

    def test_auto_inference_fallback(self):
        """Test that auto-inference falls back to default when no joint is found."""
        # Create points without connecting them to a joint
        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

        # Create constraint with joint_type=None
        constraint = CoincidentPointConstraint(
            point1,
            point2,
            name="test_constraint",
            joint_type=None  # No joint, should use default
        )

        # Should default to BALL_JOINT
        assert constraint.joint_type == JointType.BALL_JOINT
        assert constraint.stiffness == JOINT_STIFFNESS[JointType.BALL_JOINT]

    def test_auto_inference_with_different_joints(self):
        """Test auto-inference when points belong to different joints."""
        solver = CornerSolver("test")

        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')
        point3 = AttachmentPoint("p3", [200, 0, 0], unit='mm')

        # Connect point1 to its own joint
        solver.add_joint("joint1", [point1, point3], JointType.BALL_JOINT)

        # Connect point2 to a different joint
        solver.add_joint("joint2", [point2, point3], JointType.BUSHING_SOFT)

        # Create constraint between point1 and point2 (different joints)
        constraint = CoincidentPointConstraint(
            point1,
            point2,
            name="test_constraint",
            joint_type=None
        )

        # Should fall back to default since points have different joints
        assert constraint.joint_type == JointType.BALL_JOINT

    def test_explicit_override_auto_inference(self):
        """Test that explicit joint_type overrides auto-inference."""
        solver = CornerSolver("test")

        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

        # Add joint with BUSHING_SOFT
        solver.add_joint("joint", [point1, point2], JointType.BUSHING_SOFT)

        # Create constraint with explicit joint type (override)
        constraint = CoincidentPointConstraint(
            point1,
            point2,
            name="test_constraint",
            joint_type=JointType.BALL_JOINT  # Explicit override
        )

        # Should use the explicit type, not the inferred type
        assert constraint.joint_type == JointType.BALL_JOINT
        assert constraint.stiffness == JOINT_STIFFNESS[JointType.BALL_JOINT]


class TestJointTypes:
    """Test different joint types and their properties."""

    def test_all_joint_types(self):
        """Test that all joint types can be created."""
        solver = CornerSolver("test")

        point1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        point2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

        # Test all standard joint types
        joint_types = [
            JointType.RIGID,
            JointType.BALL_JOINT,
            JointType.BUSHING_HARD,
            JointType.BUSHING_SOFT,
            JointType.RUBBER_MOUNT,
        ]

        for i, jtype in enumerate(joint_types):
            # Create new points for each joint
            p1 = AttachmentPoint(f"p{2*i}", [i*100, 0, 0], unit='mm')
            p2 = AttachmentPoint(f"p{2*i+1}", [i*100+50, 0, 0], unit='mm')

            joint = solver.add_joint(f"joint_{jtype.value}", [p1, p2], jtype)

            assert joint.joint_type == jtype
            assert solver.get_joint_stiffness(f"joint_{jtype.value}") == JOINT_STIFFNESS[jtype]

    def test_stiffness_hierarchy(self):
        """Test that joint types have correct stiffness hierarchy."""
        # RIGID should be stiffest
        # BALL_JOINT should be very stiff
        # BUSHING_HARD should be moderate
        # BUSHING_SOFT should be compliant
        # RUBBER_MOUNT should be most compliant

        assert JOINT_STIFFNESS[JointType.RIGID] > JOINT_STIFFNESS[JointType.BALL_JOINT]
        assert JOINT_STIFFNESS[JointType.BALL_JOINT] > JOINT_STIFFNESS[JointType.BUSHING_HARD]
        assert JOINT_STIFFNESS[JointType.BUSHING_HARD] > JOINT_STIFFNESS[JointType.BUSHING_SOFT]
        assert JOINT_STIFFNESS[JointType.BUSHING_SOFT] > JOINT_STIFFNESS[JointType.RUBBER_MOUNT]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
