"""
Debug script to check scipy solver behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import optimize
from pysuspension.kinematic_solver import KinematicSolver
from tests.test_kinematic_solver import setup_simple_suspension


def debug_scipy_solver():
    """Debug the scipy solver behavior."""
    print("=" * 70)
    print("DEBUG: Scipy Solver Behavior")
    print("=" * 70)

    chassis, knuckle_name = setup_simple_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    print(f"\nSolver state: {solver.state}")
    print(f"DOF: {solver.state.get_dof()}")

    # Generate constraints
    constraints = solver._generate_constraints_from_components()
    print(f"\nConstraints: {len(constraints)}")

    # Get knuckle
    knuckle = solver.get_knuckle(knuckle_name)
    reference_point = knuckle.attachment_points[0]

    # Create heave constraint
    from pysuspension.constraints import PartialPositionConstraint
    from pysuspension.joint_types import JointType

    current_pos = reference_point.position.copy()
    target_pos = current_pos.copy()
    target_pos[2] += -30.0  # 30mm down

    heave_constraint = PartialPositionConstraint(
        reference_point,
        target_pos,
        constrain_axes=['z'],
        name=f"{knuckle_name}_heave",
        joint_type=JointType.RIGID
    )

    constraints.append(heave_constraint)

    print(f"Total constraints (with heave): {len(constraints)}")

    # Build residual function
    def residual_function(x: np.ndarray) -> np.ndarray:
        solver.state.from_vector(x)
        residuals = []
        for constraint in constraints:
            weighted_residual = np.sqrt(constraint.weight * constraint.evaluate())
            residuals.append(weighted_residual)
        return np.array(residuals)

    # Get initial state
    x0 = solver.state.to_vector()

    print(f"\nInitial guess shape: {x0.shape}")
    print(f"Initial guess (first 9 values): {x0[:9]}")

    # Evaluate initial residuals
    r0 = residual_function(x0)
    print(f"\nInitial residuals shape: {r0.shape}")
    print(f"Initial residual sum: {np.sum(r0**2):.6e}")
    print(f"Initial residual max: {np.max(np.abs(r0)):.6f}")

    # Check which constraints have largest initial residuals
    print(f"\nTop 5 largest initial residuals:")
    residual_dict = [(constraints[i].name, r0[i]) for i in range(len(r0))]
    residual_dict.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, val in residual_dict[:5]:
        print(f"  {name}: {val:.6f}")

    # Try solving
    print(f"\n{'='*70}")
    print("Running scipy.optimize.least_squares...")
    print(f"{'='*70}\n")

    result = optimize.least_squares(
        residual_function,
        x0,
        method='trf',
        max_nfev=100,  # Limit iterations for debugging
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        verbose=2  # Verbose output
    )

    print(f"\n{'='*70}")
    print("Optimization result:")
    print(f"{'='*70}")
    print(f"Success: {result.success}")
    print(f"Status: {result.status}")
    print(f"Message: {result.message}")
    print(f"Function evals: {result.nfev}")
    print(f"Final cost: {result.cost:.6e}")
    print(f"Optimality: {result.optimality:.6e}")

    # Check final residuals
    r_final = result.fun
    print(f"\nFinal residual sum: {np.sum(r_final**2):.6e}")
    print(f"Final residual max: {np.max(np.abs(r_final)):.6f}")

    print(f"\nTop 5 largest final residuals:")
    final_residual_dict = [(constraints[i].name, r_final[i]) for i in range(len(r_final))]
    final_residual_dict.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, val in final_residual_dict[:5]:
        print(f"  {name}: {val:.6f}")

    # Check how much the solution changed
    dx = result.x - x0
    print(f"\nSolution change:")
    print(f"  Max change: {np.max(np.abs(dx)):.6f} mm")
    print(f"  Mean change: {np.mean(np.abs(dx)):.6f} mm")
    print(f"  RMS change: {np.sqrt(np.mean(dx**2)):.6f} mm")


if __name__ == "__main__":
    debug_scipy_solver()
