"""
Debug script to check component position updates.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.kinematic_solver import KinematicSolver
from tests.test_kinematic_solver import setup_simple_suspension


def debug_component_update():
    """Debug component position updating."""
    print("=" * 70)
    print("DEBUG: Component Position Update")
    print("=" * 70)

    chassis, knuckle_name = setup_simple_suspension()
    solver = KinematicSolver.from_chassis(chassis, ['front_left'])

    knuckle = solver.get_knuckle(knuckle_name)

    print("\nBEFORE SOLVE:")
    print(f"Knuckle tire_center: {knuckle.tire_center}")
    print(f"Knuckle tire contact patch: {knuckle.get_tire_contact_patch(unit='mm')}")
    print(f"\nKnuckle attachment points:")
    initial_ap_positions = {}
    for ap in knuckle.attachment_points:
        print(f"  {ap.name}: {ap.position}")
        initial_ap_positions[ap.name] = ap.position.copy()

    # Solve for heave
    print(f"\n{'='*70}")
    print("SOLVING for heave displacement: -30 mm")
    print(f"{'='*70}")

    result = solver.solve_for_heave(knuckle_name, -30.0, unit='mm')

    print(f"\nSolver result: {result}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")

    print(f"\nAFTER SOLVE:")
    print(f"Knuckle tire_center: {knuckle.tire_center}")
    print(f"Knuckle tire contact patch: {knuckle.get_tire_contact_patch(unit='mm')}")
    print(f"\nKnuckle attachment points:")
    for ap in knuckle.attachment_points:
        initial_pos = initial_ap_positions[ap.name]
        delta = ap.position - initial_pos
        print(f"  {ap.name}: {ap.position}")
        print(f"    Delta: {delta} (magnitude: {np.linalg.norm(delta):.2f} mm)")

    # Check if attachment points moved in solver state
    print(f"\n{'='*70}")
    print("CHECKING SOLVER STATE:")
    print(f"{'='*70}")
    for ap in knuckle.attachment_points:
        state_pos = solver.state.points[ap.name].position
        print(f"{ap.name}:")
        print(f"  AttachmentPoint.position: {ap.position}")
        print(f"  SolverState position:     {state_pos}")
        print(f"  Same object? {ap is solver.state.points[ap.name]}")


if __name__ == "__main__":
    debug_component_update()
