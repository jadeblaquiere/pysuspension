"""
Debug script to investigate constraint errors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.kinematic_solver import KinematicSolver
from tests.test_kinematic_solver import setup_simple_suspension


def debug_constraint_errors():
    """Debug constraint error calculation."""
    print("=" * 70)
    print("DEBUG: Constraint Error Investigation")
    print("=" * 70)

    chassis, knuckle_name = setup_simple_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    print("\nBEFORE SOLVE:")
    print("Checking bushing constraint:")

    # Find a bushing joint
    uf_bushing = None
    for joint in solver.registry.joints.values():
        if joint.name == "uf_bushing":
            uf_bushing = joint
            break

    if uf_bushing:
        points = uf_bushing.get_all_attachment_points()
        print(f"uf_bushing has {len(points)} points:")
        for i, p in enumerate(points):
            fixed = "FIXED" if p in solver.registry.chassis_points else "FREE"
            print(f"  {i}: {p.name} at {p.position} ({fixed})")

        if len(points) >= 2:
            dist = np.linalg.norm(points[1].position - points[0].position)
            print(f"  Initial distance: {dist:.6f} mm")

    # Solve
    print(f"\n{'='*70}")
    print("SOLVING for heave displacement: -30 mm")
    print(f"{'='*70}")

    result = solver.solve_for_heave(knuckle_name, -30.0, unit='mm')

    print(f"\nSolver result: success={result.success}, iterations={result.iterations}")
    print(f"Final cost from scipy: {result.total_error:.6e}")

    print("\nAFTER SOLVE:")
    if uf_bushing:
        points = uf_bushing.get_all_attachment_points()
        print(f"uf_bushing has {len(points)} points:")
        for i, p in enumerate(points):
            fixed = "FIXED" if p in solver.registry.chassis_points else "FREE"
            print(f"  {i}: {p.name} at {p.position} ({fixed})")

        if len(points) >= 2:
            dist = np.linalg.norm(points[1].position - points[0].position)
            print(f"  Final distance: {dist:.6f} mm")

    # Check constraint errors
    print(f"\n{'='*70}")
    print("CONSTRAINT ERRORS from result:")
    print(f"{'='*70}")

    bushing_errors = [(name, err) for name, err in result.constraint_errors.items() if 'bushing' in name]
    bushing_errors.sort(key=lambda x: x[1], reverse=True)

    print("\nBushing constraint errors:")
    for name, error in bushing_errors:
        print(f"  {name}: {error:.3f} mm")

    # Manually re-evaluate constraints
    print(f"\n{'='*70}")
    print("MANUALLY RE-EVALUATING CONSTRAINTS:")
    print(f"{'='*70}")

    for constraint in solver._active_constraints:
        if 'uf_bushing' in constraint.name:
            error = constraint.get_physical_error()
            print(f"{constraint.name}:")
            print(f"  Physical error: {error:.6f} mm")
            print(f"  Weight: {constraint.weight}")
            print(f"  evaluate() returns: {constraint.evaluate():.6f}")

            # Check the points
            if hasattr(constraint, 'point1') and hasattr(constraint, 'point2'):
                p1 = constraint.point1
                p2 = constraint.point2
                print(f"  Point1: {p1.name} at {p1.position}")
                print(f"  Point2: {p2.name} at {p2.position}")
                actual_dist = np.linalg.norm(p2.position - p1.position)
                print(f"  Actual distance: {actual_dist:.6f} mm")


if __name__ == "__main__":
    debug_constraint_errors()
