"""
Unit tests for constraint framework.

Tests the constraint base classes, geometric constraints, and compliance modeling.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from pysuspension.attachment_point import AttachmentPoint
from pysuspension.constraints import (
    Constraint,
    GeometricConstraint,
    DistanceConstraint,
    FixedPointConstraint,
    CoincidentPointConstraint,
    JointType,
    JOINT_STIFFNESS
)
from pysuspension.solver_state import SolverState, DOFSpecification


class TestJointTypes:
    """Test joint type definitions and stiffness values."""

    def test_joint_type_enum(self):
        """Test that all joint types are defined."""
        assert JointType.RIGID.value == "rigid"
        assert JointType.BALL_JOINT.value == "ball_joint"
        assert JointType.BUSHING_SOFT.value == "bushing_soft"

    def test_joint_stiffness_values(self):
        """Test that stiffness values are reasonable."""
        # Rigid should be much stiffer than compliant
        assert JOINT_STIFFNESS[JointType.RIGID] > JOINT_STIFFNESS[JointType.BALL_JOINT]
        assert JOINT_STIFFNESS[JointType.BALL_JOINT] > JOINT_STIFFNESS[JointType.BUSHING_HARD]
        assert JOINT_STIFFNESS[JointType.BUSHING_HARD] > JOINT_STIFFNESS[JointType.BUSHING_SOFT]
        assert JOINT_STIFFNESS[JointType.BUSHING_SOFT] > JOINT_STIFFNESS[JointType.RUBBER_MOUNT]

        # Check reasonable ranges
        assert JOINT_STIFFNESS[JointType.RIGID] >= 1e5
        assert JOINT_STIFFNESS[JointType.RUBBER_MOUNT] >= 1.0


class TestDistanceConstraint:
    """Test distance constraint functionality."""

    def test_basic_distance_constraint(self):
        """Test creating and evaluating a distance constraint."""
        p1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        p2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

        constraint = DistanceConstraint(p1, p2, target_distance=100.0)

        assert constraint.target_distance == 100.0
        assert constraint.get_current_distance() == pytest.approx(100.0)
        assert constraint.evaluate() == pytest.approx(0.0)

    def test_distance_constraint_with_error(self):
        """Test distance constraint with non-zero error."""
        p1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        p2 = AttachmentPoint("p2", [105, 0, 0], unit='mm')

        constraint = DistanceConstraint(p1, p2, target_distance=100.0)

        assert constraint.get_current_distance() == pytest.approx(105.0)
        assert constraint.get_physical_error() == pytest.approx(5.0)
        assert constraint.evaluate() == pytest.approx(25.0)  # 5^2

    def test_distance_constraint_axial_force(self):
        """Test axial force calculation in distance constraint."""
        p1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        p2 = AttachmentPoint("p2", [105, 0, 0], unit='mm')

        # Rigid link (very stiff)
        constraint = DistanceConstraint(
            p1, p2, target_distance=100.0,
            joint_type=JointType.RIGID
        )

        # Extension = 5mm, Force = stiffness × extension
        expected_force = JOINT_STIFFNESS[JointType.RIGID] * 5.0
        assert constraint.get_axial_force() == pytest.approx(expected_force)

        # Force vector should point from p1 to p2
        force_vec = constraint.get_force_vector()
        assert force_vec[0] > 0  # Positive X direction
        assert np.linalg.norm(force_vec) == pytest.approx(expected_force)

    def test_distance_constraint_compliance(self):
        """Test distance constraint with different stiffness values."""
        p1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        p2 = AttachmentPoint("p2", [105, 0, 0], unit='mm')

        rigid = DistanceConstraint(p1, p2, 100.0, joint_type=JointType.RIGID)
        soft = DistanceConstraint(p1, p2, 100.0, stiffness=10.0)

        # Same error, different weights
        assert rigid.evaluate() == soft.evaluate()
        assert rigid.weight > soft.weight
        assert rigid.get_weighted_error() > soft.get_weighted_error()

    def test_distance_constraint_involved_points(self):
        """Test getting involved points."""
        p1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        p2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

        constraint = DistanceConstraint(p1, p2, 100.0)
        points = constraint.get_involved_points()

        assert len(points) == 2
        assert p1 in points
        assert p2 in points


class TestFixedPointConstraint:
    """Test fixed point constraint functionality."""

    def test_basic_fixed_constraint(self):
        """Test creating and evaluating a fixed point constraint."""
        p = AttachmentPoint("p", [100, 200, 300], unit='mm')
        target = np.array([100, 200, 300])

        constraint = FixedPointConstraint(p, target)

        assert constraint.evaluate() == pytest.approx(0.0)
        assert np.allclose(constraint.get_displacement(), [0, 0, 0])

    def test_fixed_constraint_with_error(self):
        """Test fixed constraint with non-zero displacement."""
        p = AttachmentPoint("p", [105, 200, 300], unit='mm')
        target = np.array([100, 200, 300])

        constraint = FixedPointConstraint(p, target)

        displacement = constraint.get_displacement()
        assert displacement[0] == pytest.approx(5.0)
        assert displacement[1] == pytest.approx(0.0)
        assert displacement[2] == pytest.approx(0.0)

        # Error = sum of squared displacements
        assert constraint.evaluate() == pytest.approx(25.0)  # 5^2
        assert constraint.get_physical_error() == pytest.approx(5.0)

    def test_fixed_constraint_force(self):
        """Test force calculation in fixed constraint."""
        p = AttachmentPoint("p", [105, 200, 300], unit='mm')
        target = np.array([100, 200, 300])

        constraint = FixedPointConstraint(
            p, target,
            joint_type=JointType.RIGID
        )

        force = constraint.get_force()
        # Force = stiffness × displacement
        expected_force = JOINT_STIFFNESS[JointType.RIGID] * 5.0

        assert force[0] == pytest.approx(expected_force)
        assert force[1] == pytest.approx(0.0)
        assert force[2] == pytest.approx(0.0)

    def test_fixed_constraint_3d_displacement(self):
        """Test fixed constraint with 3D displacement."""
        p = AttachmentPoint("p", [103, 204, 300], unit='mm')
        target = np.array([100, 200, 300])

        constraint = FixedPointConstraint(p, target)

        # Displacement = [3, 4, 0]
        # Error = 3^2 + 4^2 + 0^2 = 25
        assert constraint.evaluate() == pytest.approx(25.0)
        assert constraint.get_physical_error() == pytest.approx(5.0)  # sqrt(25)

    def test_fixed_constraint_involved_points(self):
        """Test getting involved points."""
        p = AttachmentPoint("p", [100, 200, 300], unit='mm')
        target = np.array([100, 200, 300])

        constraint = FixedPointConstraint(p, target)
        points = constraint.get_involved_points()

        assert len(points) == 1
        assert p in points


class TestCoincidentPointConstraint:
    """Test coincident point constraint functionality."""

    def test_basic_coincident_constraint(self):
        """Test creating and evaluating a coincident constraint."""
        p1 = AttachmentPoint("p1", [100, 200, 300], unit='mm')
        p2 = AttachmentPoint("p2", [100, 200, 300], unit='mm')

        constraint = CoincidentPointConstraint(p1, p2)

        assert constraint.evaluate() == pytest.approx(0.0)
        assert constraint.get_separation() == pytest.approx(0.0)

    def test_coincident_constraint_with_separation(self):
        """Test coincident constraint with non-zero separation."""
        p1 = AttachmentPoint("p1", [100, 200, 300], unit='mm')
        p2 = AttachmentPoint("p2", [103, 204, 300], unit='mm')

        constraint = CoincidentPointConstraint(p1, p2)

        # Separation = sqrt(3^2 + 4^2) = 5
        assert constraint.get_separation() == pytest.approx(5.0)
        # Error = 3^2 + 4^2 = 25
        assert constraint.evaluate() == pytest.approx(25.0)
        assert constraint.get_physical_error() == pytest.approx(5.0)

    def test_coincident_constraint_force(self):
        """Test force calculation in coincident constraint."""
        p1 = AttachmentPoint("p1", [100, 200, 300], unit='mm')
        p2 = AttachmentPoint("p2", [102, 200, 300], unit='mm')

        constraint = CoincidentPointConstraint(
            p1, p2,
            joint_type=JointType.BALL_JOINT
        )

        force = constraint.get_force()
        # Force = stiffness × displacement
        # Displacement = [2, 0, 0]
        expected_force = JOINT_STIFFNESS[JointType.BALL_JOINT] * 2.0

        assert force[0] == pytest.approx(expected_force)
        assert force[1] == pytest.approx(0.0)
        assert force[2] == pytest.approx(0.0)

    def test_coincident_constraint_compliance_comparison(self):
        """Test compliance effects on coincident constraints."""
        p1 = AttachmentPoint("p1", [100, 200, 300], unit='mm')
        p2 = AttachmentPoint("p2", [102, 200, 300], unit='mm')

        ball_joint = CoincidentPointConstraint(
            p1, p2, joint_type=JointType.BALL_JOINT
        )
        soft_bushing = CoincidentPointConstraint(
            p1, p2, joint_type=JointType.BUSHING_SOFT
        )

        # Same error, different stiffness and weights
        assert ball_joint.evaluate() == soft_bushing.evaluate()
        assert ball_joint.stiffness > soft_bushing.stiffness
        assert ball_joint.weight > soft_bushing.weight

        # Force proportional to stiffness
        force_ball = np.linalg.norm(ball_joint.get_force())
        force_bushing = np.linalg.norm(soft_bushing.get_force())
        assert force_ball > force_bushing

        # Compliance is inverse of stiffness
        assert ball_joint.compliance < soft_bushing.compliance

    def test_coincident_constraint_custom_stiffness(self):
        """Test custom stiffness value."""
        p1 = AttachmentPoint("p1", [100, 200, 300], unit='mm')
        p2 = AttachmentPoint("p2", [100, 200, 300], unit='mm')

        custom_stiffness = 500.0  # N/mm
        constraint = CoincidentPointConstraint(
            p1, p2, stiffness=custom_stiffness
        )

        assert constraint.stiffness == pytest.approx(custom_stiffness)
        assert constraint.compliance == pytest.approx(1.0 / custom_stiffness)

    def test_coincident_constraint_involved_points(self):
        """Test getting involved points."""
        p1 = AttachmentPoint("p1", [100, 200, 300], unit='mm')
        p2 = AttachmentPoint("p2", [100, 200, 300], unit='mm')

        constraint = CoincidentPointConstraint(p1, p2)
        points = constraint.get_involved_points()

        assert len(points) == 2
        assert p1 in points
        assert p2 in points


class TestSolverState:
    """Test solver state management."""

    def test_create_empty_state(self):
        """Test creating an empty solver state."""
        state = SolverState("test")
        assert state.name == "test"
        assert state.get_dof() == 0
        assert len(state.points) == 0

    def test_add_points(self):
        """Test adding points to state."""
        state = SolverState()
        p1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
        p2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

        state.add_free_point(p1)
        state.add_fixed_point(p2)

        assert len(state.points) == 2
        assert len(state.free_points) == 1
        assert len(state.fixed_points) == 1
        assert state.get_dof() == 3

    def test_vector_conversion(self):
        """Test converting state to/from vector."""
        state = SolverState()
        p1 = AttachmentPoint("p1", [100, 200, 300], unit='mm')
        p2 = AttachmentPoint("p2", [400, 500, 600], unit='mm')

        state.add_free_point(p1)
        state.add_free_point(p2)

        vec = state.to_vector()
        assert vec.shape == (6,)
        assert np.allclose(vec, [100, 200, 300, 400, 500, 600])

        # Modify vector and restore
        new_vec = vec + np.array([10, 20, 30, 40, 50, 60])
        state.from_vector(new_vec)

        assert np.allclose(p1.position, [110, 220, 330])
        assert np.allclose(p2.position, [440, 550, 660])

    def test_snapshots(self):
        """Test saving and restoring snapshots."""
        state = SolverState()
        p = AttachmentPoint("p", [100, 200, 300], unit='mm')
        state.add_free_point(p)

        # Save original
        snapshot = state.save_snapshot()

        # Modify
        p.set_position([150, 250, 350], unit='mm')
        assert np.allclose(p.position, [150, 250, 350])

        # Restore
        state.restore_snapshot(snapshot)
        assert np.allclose(p.position, [100, 200, 300])

    def test_change_point_freedom(self):
        """Test changing whether point is free or fixed."""
        state = SolverState()
        p = AttachmentPoint("p", [0, 0, 0], unit='mm')

        state.add_free_point(p)
        assert state.get_dof() == 3
        assert "p" in state.free_points

        state.set_point_free("p", False)
        assert state.get_dof() == 0
        assert "p" in state.fixed_points


class TestDOFSpecification:
    """Test DOF specification functionality."""

    def test_create_empty_dof(self):
        """Test creating empty DOF specification."""
        dof = DOFSpecification("test")
        assert dof.name == "test"
        assert len(dof.variables) == 0

    def test_add_variables(self):
        """Test adding DOF variables."""
        dof = DOFSpecification()
        dof.add_variable('heave', initial=0.0, min_val=-100, max_val=100, unit='mm')
        dof.add_variable('roll', initial=0.0, min_val=-0.1, max_val=0.1, unit='rad')

        assert len(dof.variables) == 2
        assert 'heave' in dof.variables
        assert 'roll' in dof.variables

    def test_get_initial_values(self):
        """Test getting initial values."""
        dof = DOFSpecification()
        dof.add_variable('heave', initial=10.0)
        dof.add_variable('roll', initial=0.05)

        initial = dof.get_initial_values()
        assert initial['heave'] == pytest.approx(10.0)
        assert initial['roll'] == pytest.approx(0.05)

    def test_get_bounds(self):
        """Test getting variable bounds."""
        dof = DOFSpecification()
        dof.add_variable('heave', initial=0.0, min_val=-50, max_val=50)
        dof.add_variable('roll', initial=0.0, min_val=-0.1, max_val=0.1)

        bounds = dof.get_bounds()
        assert len(bounds) == 2
        assert bounds[0] == (-50, 50)
        assert bounds[1] == (-0.1, 0.1)

    def test_validate_values(self):
        """Test value validation."""
        dof = DOFSpecification()
        dof.add_variable('heave', initial=0.0, min_val=-50, max_val=50)

        assert dof.validate_values({'heave': 25.0})
        assert not dof.validate_values({'heave': 75.0})
        assert not dof.validate_values({'heave': -75.0})

    def test_clamp_values(self):
        """Test value clamping."""
        dof = DOFSpecification()
        dof.add_variable('heave', initial=0.0, min_val=-50, max_val=50)

        clamped = dof.clamp_values({'heave': 75.0})
        assert clamped['heave'] == pytest.approx(50.0)

        clamped = dof.clamp_values({'heave': -75.0})
        assert clamped['heave'] == pytest.approx(-50.0)

        clamped = dof.clamp_values({'heave': 25.0})
        assert clamped['heave'] == pytest.approx(25.0)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
