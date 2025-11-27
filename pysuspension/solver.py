"""
Constraint-based solver for suspension kinematics.

This module provides solvers that use geometric constraints and compliance
modeling to solve suspension positions. The solver minimizes weighted least-
squares error across all constraints, effectively minimizing elastic energy
in the system.

Requirements:
    - scipy: For numerical optimization (scipy.optimize)
    Install with: pip install scipy
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Solver will not function.")
    print("Install with: pip install scipy")
from constraints import Constraint, GeometricConstraint
from solver_state import SolverState, DOFSpecification
from attachment_point import AttachmentPoint


class SolverResult:
    """
    Results from a solver run.

    Contains final positions, errors, convergence information, and
    computed geometry (camber, toe, etc.).
    """

    def __init__(self,
                 success: bool,
                 message: str,
                 iterations: int,
                 total_error: float,
                 positions: Dict[str, np.ndarray],
                 constraint_errors: Dict[str, float]):
        """
        Initialize solver result.

        Args:
            success: Whether solver converged successfully
            message: Solver status message
            iterations: Number of iterations performed
            total_error: Total weighted error (objective function value)
            positions: Dictionary of point_name -> final position
            constraint_errors: Dictionary of constraint_name -> error value
        """
        self.success = success
        self.message = message
        self.iterations = iterations
        self.total_error = total_error
        self.positions = positions
        self.constraint_errors = constraint_errors

        # Additional computed quantities (filled in by specialized solvers)
        self.geometry = {}  # Camber, toe, etc.
        self.forces = {}    # Forces at each joint
        self.compliance_energy = 0.0  # Elastic energy in compliant joints

    def get_position(self, point_name: str, unit: str = 'mm') -> np.ndarray:
        """Get final position of a point."""
        from units import from_mm
        if point_name not in self.positions:
            raise ValueError(f"Point '{point_name}' not found in results")
        return from_mm(self.positions[point_name].copy(), unit)

    def get_rms_error(self) -> float:
        """Get RMS error across all constraints."""
        if not self.constraint_errors:
            return 0.0
        errors = np.array(list(self.constraint_errors.values()))
        return np.sqrt(np.mean(errors))

    def get_max_error(self) -> Tuple[str, float]:
        """
        Get maximum constraint error.

        Returns:
            Tuple of (constraint_name, error_value)
        """
        if not self.constraint_errors:
            return ("none", 0.0)
        max_name = max(self.constraint_errors, key=self.constraint_errors.get)
        return (max_name, self.constraint_errors[max_name])

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return (f"SolverResult({status}, "
                f"iterations={self.iterations}, "
                f"total_error={self.total_error:.6e}, "
                f"rms_error={self.get_rms_error():.6f} mm)")


class SuspensionSolver:
    """
    Base class for constraint-based suspension solving.

    The solver minimizes weighted least-squares error:
        E = Σ weight_i × error_i²

    This is equivalent to minimizing elastic energy in compliant joints.
    """

    def __init__(self, name: str = "solver"):
        """
        Initialize solver.

        Args:
            name: Identifier for this solver
        """
        self.name = name
        self.constraints: List[Constraint] = []
        self.state = SolverState(name=f"{name}_state")
        self.dof_spec = DOFSpecification(name=f"{name}_dof")

        # Solver settings
        self.max_iterations = 1000
        self.tolerance = 1e-9  # Convergence tolerance
        self.method = 'least-squares'  # or 'minimize'

    def add_constraint(self, constraint: Constraint):
        """
        Add a constraint to the system.

        Args:
            constraint: Constraint to add
        """
        self.constraints.append(constraint)

        # Add involved points to state if not already present
        for point in constraint.get_involved_points():
            if point.name not in self.state.points:
                # Default to free (can be changed later)
                self.state.add_free_point(point)

    def set_point_fixed(self, point_name: str):
        """Mark a point as fixed (not a variable in optimization)."""
        self.state.set_point_free(point_name, False)

    def set_point_free(self, point_name: str):
        """Mark a point as free (variable in optimization)."""
        self.state.set_point_free(point_name, True)

    def compute_total_error(self) -> float:
        """
        Compute total weighted error across all constraints.

        Returns:
            Sum of weighted errors
        """
        total = 0.0
        for constraint in self.constraints:
            total += constraint.get_weighted_error()
        return total

    def compute_constraint_errors(self) -> Dict[str, float]:
        """
        Compute error for each constraint.

        Returns:
            Dictionary mapping constraint name to physical error
        """
        errors = {}
        for constraint in self.constraints:
            errors[constraint.name] = constraint.get_physical_error()
        return errors

    def _objective_function(self, x: np.ndarray) -> float:
        """
        Objective function for optimization (total weighted error).

        Args:
            x: Flat array of free point positions

        Returns:
            Total weighted error
        """
        # Update state from optimization vector
        self.state.from_vector(x)

        # Compute total error
        return self.compute_total_error()

    def _residual_vector(self, x: np.ndarray) -> np.ndarray:
        """
        Residual vector for least-squares (weighted square root of errors).

        Args:
            x: Flat array of free point positions

        Returns:
            Array of weighted residuals
        """
        # Update state from optimization vector
        self.state.from_vector(x)

        # Build residual vector
        residuals = []
        for constraint in self.constraints:
            # Residual = sqrt(weight) × sqrt(error)
            # This makes least-squares minimize Σ weight × error
            weighted_residual = np.sqrt(constraint.weight * constraint.evaluate())
            residuals.append(weighted_residual)

        return np.array(residuals)

    def solve(self,
              initial_guess: Optional[np.ndarray] = None,
              method: Optional[str] = None) -> SolverResult:
        """
        Solve the constraint system.

        Args:
            initial_guess: Initial positions (uses current state if None)
            method: Solving method ('least-squares' or 'minimize')

        Returns:
            SolverResult with final positions and diagnostics
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for solving. Install with: pip install scipy")

        if not self.constraints:
            raise ValueError("No constraints added to solver")

        # Use specified method or default
        solve_method = method if method is not None else self.method

        # Initial guess
        if initial_guess is None:
            x0 = self.state.to_vector()
        else:
            x0 = initial_guess

        if len(x0) == 0:
            # No free variables - system is fully constrained
            return SolverResult(
                success=True,
                message="No free variables (fully constrained)",
                iterations=0,
                total_error=self.compute_total_error(),
                positions={name: point.position.copy()
                          for name, point in self.state.points.items()},
                constraint_errors=self.compute_constraint_errors()
            )

        # Solve using specified method
        if solve_method == 'least-squares':
            # Use 'trf' (Trust Region Reflective) - handles both over and under-constrained
            # 'lm' (Levenberg-Marquardt) only works for over-constrained systems
            result = optimize.least_squares(
                self._residual_vector,
                x0,
                method='trf',  # Trust Region Reflective - handles any constraint level
                max_nfev=self.max_iterations,
                ftol=self.tolerance,
                xtol=self.tolerance,
                gtol=self.tolerance
            )
            success = result.success
            message = result.message
            iterations = result.nfev
            total_error = np.sum(result.fun ** 2)  # Sum of squared residuals

        elif solve_method == 'minimize':
            result = optimize.minimize(
                self._objective_function,
                x0,
                method='SLSQP',
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance
                }
            )
            success = result.success
            message = result.message
            iterations = result.nit
            total_error = result.fun

        else:
            raise ValueError(f"Unknown method '{solve_method}'. Use 'least-squares' or 'minimize'")

        # Update state with final solution
        self.state.from_vector(result.x)

        # Build result
        solver_result = SolverResult(
            success=success,
            message=message,
            iterations=iterations,
            total_error=total_error,
            positions={name: point.position.copy()
                      for name, point in self.state.points.items()},
            constraint_errors=self.compute_constraint_errors()
        )

        # Compute forces and compliance energy
        self._compute_forces_and_energy(solver_result)

        return solver_result

    def _compute_forces_and_energy(self, result: SolverResult):
        """
        Compute forces at each constraint and total compliance energy.

        Args:
            result: SolverResult to populate with force data
        """
        total_energy = 0.0

        for constraint in self.constraints:
            if hasattr(constraint, 'get_force'):
                force = constraint.get_force()
                force_magnitude = np.linalg.norm(force)
                result.forces[constraint.name] = force_magnitude

                # Energy = 0.5 × k × x²
                error = constraint.evaluate()  # Already squared
                energy = 0.5 * constraint.stiffness * error
                total_energy += energy

        result.compliance_energy = total_energy

    def get_state_snapshot(self) -> Dict[str, np.ndarray]:
        """Save current state."""
        return self.state.save_snapshot()

    def restore_state_snapshot(self, snapshot: Dict[str, np.ndarray]):
        """Restore saved state."""
        self.state.restore_snapshot(snapshot)

    def __repr__(self) -> str:
        return (f"SuspensionSolver('{self.name}', "
                f"constraints={len(self.constraints)}, "
                f"free_points={len(self.state.free_points)}, "
                f"dof={self.state.get_dof()})")


if __name__ == "__main__":
    print("=" * 70)
    print("SUSPENSION SOLVER TEST")
    print("=" * 70)

    from constraints import DistanceConstraint, FixedPointConstraint, CoincidentPointConstraint

    # Create a simple 2D linkage to test the solver
    # Four-bar linkage: ground, crank, coupler, rocker

    # Ground points (fixed)
    ground_left = AttachmentPoint("ground_left", [0, 0, 0], unit='mm')
    ground_right = AttachmentPoint("ground_right", [400, 0, 0], unit='mm')

    # Moving points (free)
    crank_end = AttachmentPoint("crank_end", [100, 100, 0], unit='mm')
    coupler_end = AttachmentPoint("coupler_end", [300, 100, 0], unit='mm')

    print("\n--- Creating Four-Bar Linkage Solver ---")
    solver = SuspensionSolver("four_bar")

    # Ground pivots are fixed
    solver.add_constraint(FixedPointConstraint(ground_left, [0, 0, 0]))
    solver.add_constraint(FixedPointConstraint(ground_right, [400, 0, 0]))

    # Link lengths (rigid)
    crank_length = 100.0  # Ground to crank_end
    coupler_length = 200.0  # Crank_end to coupler_end
    rocker_length = 100.0  # Coupler_end to ground_right

    solver.add_constraint(DistanceConstraint(ground_left, crank_end, crank_length, name="crank"))
    solver.add_constraint(DistanceConstraint(crank_end, coupler_end, coupler_length, name="coupler"))
    solver.add_constraint(DistanceConstraint(coupler_end, ground_right, rocker_length, name="rocker"))

    # Ground points are fixed, others are free
    solver.set_point_fixed("ground_left")
    solver.set_point_fixed("ground_right")

    print(solver)
    print(f"DOF: {solver.state.get_dof()}")

    print("\n--- Solving Initial Configuration ---")
    result = solver.solve()
    print(result)
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"RMS error: {result.get_rms_error():.6f} mm")

    print("\n--- Final Positions ---")
    for name in ["crank_end", "coupler_end"]:
        pos = result.get_position(name)
        print(f"  {name}: {pos}")

    print("\n--- Constraint Errors ---")
    for name, error in result.constraint_errors.items():
        print(f"  {name}: {error:.6f} mm")

    print("\n--- Link Length Verification ---")
    crank_pos = result.get_position("crank_end")
    coupler_pos = result.get_position("coupler_end")
    ground_r_pos = result.get_position("ground_right")

    actual_crank = np.linalg.norm(crank_pos - [0, 0, 0])
    actual_coupler = np.linalg.norm(coupler_pos - crank_pos)
    actual_rocker = np.linalg.norm(ground_r_pos - coupler_pos)

    print(f"  Crank: {actual_crank:.3f} mm (target: {crank_length:.3f} mm)")
    print(f"  Coupler: {actual_coupler:.3f} mm (target: {coupler_length:.3f} mm)")
    print(f"  Rocker: {actual_rocker:.3f} mm (target: {rocker_length:.3f} mm)")

    print("\n✓ Solver test completed successfully!")
