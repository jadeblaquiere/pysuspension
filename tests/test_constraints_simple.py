"""
Simple unit tests for constraint framework (no pytest required).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.attachment_point import AttachmentPoint
from pysuspension.constraints import (
    DistanceConstraint,
    FixedPointConstraint,
    CoincidentPointConstraint,
    JointType,
    JOINT_STIFFNESS
)
from pysuspension.solver_state import SolverState, DOFSpecification


def assert_approx(a, b, tol=1e-6):
    """Simple approximate equality check."""
    if abs(a - b) > tol:
        raise AssertionError(f"{a} != {b} (tolerance {tol})")


def test_distance_constraint():
    """Test distance constraint functionality."""
    print("Testing DistanceConstraint...")

    p1 = AttachmentPoint("p1", [0, 0, 0], unit='mm')
    p2 = AttachmentPoint("p2", [100, 0, 0], unit='mm')

    constraint = DistanceConstraint(p1, p2, target_distance=100.0)

    assert constraint.target_distance == 100.0
    assert_approx(constraint.get_current_distance(), 100.0)
    assert_approx(constraint.evaluate(), 0.0)

    # Test with error
    p2.set_position([105, 0, 0], unit='mm')
    assert_approx(constraint.get_current_distance(), 105.0)
    assert_approx(constraint.get_physical_error(), 5.0)
    assert_approx(constraint.evaluate(), 25.0)

    # Test force calculation
    force = constraint.get_axial_force()
    assert force > 0  # Tension

    print("✓ DistanceConstraint tests passed")


def test_fixed_point_constraint():
    """Test fixed point constraint functionality."""
    print("Testing FixedPointConstraint...")

    p = AttachmentPoint("p", [100, 200, 300], unit='mm')
    target = np.array([100, 200, 300])

    constraint = FixedPointConstraint(p, target)

    assert_approx(constraint.evaluate(), 0.0)
    assert np.allclose(constraint.get_displacement(), [0, 0, 0])

    # Test with displacement
    p.set_position([105, 200, 300], unit='mm')
    displacement = constraint.get_displacement()
    assert_approx(displacement[0], 5.0)
    assert_approx(constraint.evaluate(), 25.0)
    assert_approx(constraint.get_physical_error(), 5.0)

    print("✓ FixedPointConstraint tests passed")


def test_coincident_point_constraint():
    """Test coincident point constraint functionality."""
    print("Testing CoincidentPointConstraint...")

    p1 = AttachmentPoint("p1", [100, 200, 300], unit='mm')
    p2 = AttachmentPoint("p2", [100, 200, 300], unit='mm')

    constraint = CoincidentPointConstraint(p1, p2)

    assert_approx(constraint.evaluate(), 0.0)
    assert_approx(constraint.get_separation(), 0.0)

    # Test with separation
    p2.set_position([103, 204, 300], unit='mm')
    assert_approx(constraint.get_separation(), 5.0)
    assert_approx(constraint.evaluate(), 25.0)

    # Test compliance
    ball_joint = CoincidentPointConstraint(
        p1, p2, joint_type=JointType.BALL_JOINT
    )
    soft_bushing = CoincidentPointConstraint(
        p1, p2, joint_type=JointType.BUSHING_SOFT
    )

    assert ball_joint.stiffness > soft_bushing.stiffness
    assert ball_joint.weight > soft_bushing.weight

    print("✓ CoincidentPointConstraint tests passed")


def test_joint_types():
    """Test joint type stiffness values."""
    print("Testing JointTypes...")

    # Check stiffness ordering
    assert JOINT_STIFFNESS[JointType.RIGID] > JOINT_STIFFNESS[JointType.BALL_JOINT]
    assert JOINT_STIFFNESS[JointType.BALL_JOINT] > JOINT_STIFFNESS[JointType.BUSHING_HARD]
    assert JOINT_STIFFNESS[JointType.BUSHING_HARD] > JOINT_STIFFNESS[JointType.BUSHING_SOFT]
    assert JOINT_STIFFNESS[JointType.BUSHING_SOFT] > JOINT_STIFFNESS[JointType.RUBBER_MOUNT]

    print("✓ JointType tests passed")


def test_solver_state():
    """Test solver state management."""
    print("Testing SolverState...")

    state = SolverState("test")
    p1 = AttachmentPoint("p1", [100, 200, 300], unit='mm')
    p2 = AttachmentPoint("p2", [400, 500, 600], unit='mm')

    state.add_free_point(p1)
    state.add_fixed_point(p2)

    assert state.get_dof() == 3
    assert len(state.free_points) == 1
    assert len(state.fixed_points) == 1

    # Test vector conversion
    vec = state.to_vector()
    assert vec.shape == (3,)
    assert np.allclose(vec, [100, 200, 300])

    # Modify and restore
    new_vec = vec + np.array([10, 20, 30])
    state.from_vector(new_vec)
    assert np.allclose(p1.position, [110, 220, 330])

    # Test snapshots
    snapshot = state.save_snapshot()
    p1.set_position([0, 0, 0], unit='mm')
    state.restore_snapshot(snapshot)
    assert np.allclose(p1.position, [110, 220, 330])

    print("✓ SolverState tests passed")


def test_dof_specification():
    """Test DOF specification functionality."""
    print("Testing DOFSpecification...")

    dof = DOFSpecification("test")
    dof.add_variable('heave', initial=0.0, min_val=-50, max_val=50, unit='mm')
    dof.add_variable('roll', initial=0.0, min_val=-0.1, max_val=0.1, unit='rad')

    assert len(dof.variables) == 2

    # Test initial values
    initial = dof.get_initial_values()
    assert_approx(initial['heave'], 0.0)
    assert_approx(initial['roll'], 0.0)

    # Test bounds
    bounds = dof.get_bounds()
    assert len(bounds) == 2
    assert bounds[0] == (-50, 50)
    assert bounds[1] == (-0.1, 0.1)

    # Test validation
    assert dof.validate_values({'heave': 25.0, 'roll': 0.05})
    assert not dof.validate_values({'heave': 75.0, 'roll': 0.05})

    # Test clamping
    clamped = dof.clamp_values({'heave': 75.0, 'roll': 0.05})
    assert_approx(clamped['heave'], 50.0)

    print("✓ DOFSpecification tests passed")


if __name__ == "__main__":
    print("=" * 70)
    print("CONSTRAINT FRAMEWORK UNIT TESTS")
    print("=" * 70)
    print()

    try:
        test_joint_types()
        test_distance_constraint()
        test_fixed_point_constraint()
        test_coincident_point_constraint()
        test_solver_state()
        test_dof_specification()

        print()
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
