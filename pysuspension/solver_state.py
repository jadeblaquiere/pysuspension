"""
Solver state management for constraint-based suspension solving.

This module provides classes for managing the state of the solver,
including tracking which points are free to move, storing degrees of
freedom specifications, and converting between point positions and
optimization vectors.
"""

import numpy as np
from typing import Dict, Set, List, Tuple, Optional
from attachment_point import AttachmentPoint


class SolverState:
    """
    Represents the state of all moving points in the suspension system.

    The solver state manages:
    - All attachment points in the system
    - Which points are free to move (variables) vs fixed (constants)
    - Conversion between point positions and flat optimization vectors
    - State snapshots for comparison and rollback
    """

    def __init__(self, name: str = "solver_state"):
        """
        Initialize an empty solver state.

        Args:
            name: Identifier for this state
        """
        self.name = name
        self.points: Dict[str, AttachmentPoint] = {}  # point_id -> AttachmentPoint
        self.free_points: Set[str] = set()  # Point IDs that can move
        self.fixed_points: Set[str] = set()  # Point IDs that are fixed

    def add_point(self, point: AttachmentPoint, is_free: bool = True):
        """
        Add an attachment point to the state.

        Args:
            point: AttachmentPoint to add
            is_free: True if point can move, False if fixed
        """
        point_id = point.name

        if point_id in self.points:
            raise ValueError(f"Point '{point_id}' already exists in state")

        self.points[point_id] = point

        if is_free:
            self.free_points.add(point_id)
        else:
            self.fixed_points.add(point_id)

    def add_free_point(self, point: AttachmentPoint):
        """Add a point that can move during solving."""
        self.add_point(point, is_free=True)

    def add_fixed_point(self, point: AttachmentPoint):
        """Add a point that remains fixed during solving."""
        self.add_point(point, is_free=False)

    def set_point_free(self, point_id: str, is_free: bool = True):
        """
        Change whether a point is free or fixed.

        Args:
            point_id: Name of the point
            is_free: True to make free, False to make fixed
        """
        if point_id not in self.points:
            raise ValueError(f"Point '{point_id}' not found in state")

        if is_free:
            self.free_points.add(point_id)
            self.fixed_points.discard(point_id)
        else:
            self.fixed_points.add(point_id)
            self.free_points.discard(point_id)

    def get_dof(self) -> int:
        """
        Get number of degrees of freedom.

        Returns:
            Number of free variables (3 × number of free points)
        """
        return len(self.free_points) * 3

    def to_vector(self) -> np.ndarray:
        """
        Convert free point positions to a flat vector for optimization.

        The vector is organized as:
        [point1_x, point1_y, point1_z, point2_x, point2_y, point2_z, ...]

        Returns:
            Flat numpy array of free point positions in mm
        """
        positions = []
        for point_id in sorted(self.free_points):  # Sort for consistency
            point = self.points[point_id]
            positions.append(point.position)

        if not positions:
            return np.array([])

        return np.concatenate(positions)

    def from_vector(self, vec: np.ndarray):
        """
        Update free point positions from a flat vector.

        Args:
            vec: Flat array of positions (must have length 3 × num_free_points)
        """
        expected_size = self.get_dof()
        if len(vec) != expected_size:
            raise ValueError(
                f"Vector size {len(vec)} doesn't match DOF {expected_size}"
            )

        idx = 0
        for point_id in sorted(self.free_points):
            point = self.points[point_id]
            point.set_position(vec[idx:idx+3], unit='mm')
            idx += 3

    def get_point(self, point_id: str) -> AttachmentPoint:
        """
        Get an attachment point by ID.

        Args:
            point_id: Name of the point

        Returns:
            AttachmentPoint object
        """
        if point_id not in self.points:
            raise ValueError(f"Point '{point_id}' not found in state")
        return self.points[point_id]

    def save_snapshot(self) -> Dict[str, np.ndarray]:
        """
        Save current positions of all points.

        Returns:
            Dictionary mapping point_id -> position (copy)
        """
        return {
            point_id: point.position.copy()
            for point_id, point in self.points.items()
        }

    def restore_snapshot(self, snapshot: Dict[str, np.ndarray]):
        """
        Restore point positions from a snapshot.

        Args:
            snapshot: Dictionary from save_snapshot()
        """
        for point_id, position in snapshot.items():
            if point_id in self.points:
                self.points[point_id].set_position(position, unit='mm')

    def __repr__(self) -> str:
        return (f"SolverState('{self.name}', "
                f"points={len(self.points)}, "
                f"free={len(self.free_points)}, "
                f"fixed={len(self.fixed_points)}, "
                f"dof={self.get_dof()})")


class DOFSpecification:
    """
    Defines degrees of freedom (input variables) for solving.

    DOF specifications define which high-level parameters can vary
    during solving (e.g., heave, roll, steering angle) along with
    their valid ranges and initial values.

    Example DOF variables:
    - heave: Vertical chassis displacement
    - roll: Chassis roll angle
    - steering_angle: Steering input angle
    - spring_compression: Spring displacement
    """

    def __init__(self, name: str = "dof_spec"):
        """
        Initialize an empty DOF specification.

        Args:
            name: Identifier for this specification
        """
        self.name = name
        self.variables: Dict[str, Dict] = {}  # name -> {initial, min, max, unit}

    def add_variable(self,
                     name: str,
                     initial: float,
                     min_val: float = -np.inf,
                     max_val: float = np.inf,
                     unit: str = 'mm'):
        """
        Add a degree of freedom variable.

        Args:
            name: Variable name (e.g., 'heave', 'steering_angle')
            initial: Initial/default value
            min_val: Minimum allowed value (default: -inf)
            max_val: Maximum allowed value (default: +inf)
            unit: Unit of this variable (default: 'mm')
        """
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists")

        if min_val > max_val:
            raise ValueError(f"min_val ({min_val}) must be <= max_val ({max_val})")

        if not (min_val <= initial <= max_val):
            raise ValueError(
                f"Initial value {initial} outside range [{min_val}, {max_val}]"
            )

        self.variables[name] = {
            'initial': float(initial),
            'min': float(min_val),
            'max': float(max_val),
            'unit': unit
        }

    def get_initial_values(self) -> Dict[str, float]:
        """
        Get initial values for all variables.

        Returns:
            Dictionary mapping variable name -> initial value
        """
        return {
            name: var['initial']
            for name, var in self.variables.items()
        }

    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get bounds for all variables (for bounded optimization).

        Returns:
            List of (min, max) tuples in same order as variables
        """
        return [
            (var['min'], var['max'])
            for var in self.variables.values()
        ]

    def get_variable_names(self) -> List[str]:
        """Get list of variable names."""
        return list(self.variables.keys())

    def validate_values(self, values: Dict[str, float]) -> bool:
        """
        Check if values are within bounds.

        Args:
            values: Dictionary of variable name -> value

        Returns:
            True if all values are within bounds
        """
        for name, value in values.items():
            if name not in self.variables:
                return False
            var = self.variables[name]
            if not (var['min'] <= value <= var['max']):
                return False
        return True

    def clamp_values(self, values: Dict[str, float]) -> Dict[str, float]:
        """
        Clamp values to be within bounds.

        Args:
            values: Dictionary of variable name -> value

        Returns:
            Dictionary with clamped values
        """
        clamped = {}
        for name, value in values.items():
            if name in self.variables:
                var = self.variables[name]
                clamped[name] = np.clip(value, var['min'], var['max'])
            else:
                clamped[name] = value
        return clamped

    def __repr__(self) -> str:
        var_summary = ", ".join(
            f"{name}={var['initial']}{var['unit']}"
            for name, var in self.variables.items()
        )
        return f"DOFSpecification('{self.name}', variables=[{var_summary}])"


if __name__ == "__main__":
    print("=" * 70)
    print("SOLVER STATE TEST")
    print("=" * 70)

    # Create test attachment points
    p1 = AttachmentPoint("chassis_mount_1", [1000, 400, 500], unit='mm')
    p2 = AttachmentPoint("chassis_mount_2", [1200, 400, 500], unit='mm')
    p3 = AttachmentPoint("arm_ball_joint", [1100, 600, 400], unit='mm')
    p4 = AttachmentPoint("knuckle_mount", [1100, 700, 300], unit='mm')

    print("\n--- Testing SolverState ---")
    state = SolverState("test_state")

    # Chassis mounts are fixed
    state.add_fixed_point(p1)
    state.add_fixed_point(p2)

    # Arm and knuckle points can move
    state.add_free_point(p3)
    state.add_free_point(p4)

    print(state)
    print(f"DOF: {state.get_dof()}")

    print("\n--- Testing vector conversion ---")
    vec = state.to_vector()
    print(f"State vector shape: {vec.shape}")
    print(f"State vector: {vec}")

    # Modify vector and restore
    vec_modified = vec + np.array([10, 0, 0, 0, 10, 0])  # Move points in X and Y
    print(f"\nModified vector: {vec_modified}")

    state.from_vector(vec_modified)
    print(f"Ball joint new position: {p3.position}")
    print(f"Knuckle new position: {p4.position}")

    print("\n--- Testing snapshots ---")
    snapshot = state.save_snapshot()
    print("Saved snapshot")

    # Move points again
    state.from_vector(vec)  # Restore original
    print(f"After restore - ball joint position: {p3.position}")

    state.restore_snapshot(snapshot)
    print(f"After snapshot restore - ball joint position: {p3.position}")

    print("\n" + "=" * 70)
    print("DOF SPECIFICATION TEST")
    print("=" * 70)

    dof = DOFSpecification("suspension_dof")

    # Add common suspension DOF variables
    dof.add_variable('heave', initial=0.0, min_val=-100, max_val=100, unit='mm')
    dof.add_variable('roll', initial=0.0, min_val=-0.1, max_val=0.1, unit='rad')
    dof.add_variable('steering_angle', initial=0.0, min_val=-30, max_val=30, unit='deg')

    print(f"\n{dof}")
    print(f"Variables: {dof.get_variable_names()}")
    print(f"Initial values: {dof.get_initial_values()}")
    print(f"Bounds: {dof.get_bounds()}")

    print("\n--- Testing value validation ---")
    test_values = {'heave': 50.0, 'roll': 0.05, 'steering_angle': 15.0}
    print(f"Values {test_values}")
    print(f"Valid: {dof.validate_values(test_values)}")

    invalid_values = {'heave': 150.0, 'roll': 0.05, 'steering_angle': 15.0}
    print(f"\nValues {invalid_values}")
    print(f"Valid: {dof.validate_values(invalid_values)}")

    clamped = dof.clamp_values(invalid_values)
    print(f"Clamped: {clamped}")
    print(f"Valid after clamp: {dof.validate_values(clamped)}")

    print("\n✓ All solver state tests completed successfully!")
