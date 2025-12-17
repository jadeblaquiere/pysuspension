"""
Kinematic solver with direct component-level integration.

This module provides the KinematicSolver class that works directly with suspension
components (RigidBody, SuspensionLink, SuspensionJoint) rather than abstract
constraints. It generates constraints dynamically from component geometry and
automatically updates component positions after solving.

Key differences from CornerSolver:
1. Works directly with component objects (not just constraints)
2. Automatically handles variable-length components (springs)
3. Updates component state after solving
4. Provides component-level force/energy queries

Requirements:
    - scipy: For numerical optimization (scipy.optimize)
    Install with: pip install scipy
"""

import numpy as np
from typing import List, Dict, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass, field

try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. KinematicSolver will not function.")
    print("Install with: pip install scipy")

from .solver import SuspensionSolver, SolverResult
from .solver_state import SolverState
from .constraints import (
    Constraint,
    DistanceConstraint,
    FixedPointConstraint,
    CoincidentPointConstraint,
    PartialPositionConstraint
)
from .joint_types import JointType, JOINT_STIFFNESS
from .attachment_point import AttachmentPoint
from .suspension_graph import discover_suspension_graph, SuspensionGraph

if TYPE_CHECKING:
    from .chassis import Chassis
    from .rigid_body import RigidBody
    from .suspension_link import SuspensionLink
    from .coil_spring import CoilSpring
    from .suspension_joint import SuspensionJoint
    from .suspension_knuckle import SuspensionKnuckle


@dataclass
class ComponentRegistry:
    """
    Registry of suspension components organized by type.

    This structure maintains references to all components in the solver's
    working domain, organized for efficient access and constraint generation.
    """
    # Components by type
    rigid_bodies: Dict[str, 'RigidBody'] = field(default_factory=dict)
    links: Dict[str, 'SuspensionLink'] = field(default_factory=dict)
    springs: Dict[str, 'CoilSpring'] = field(default_factory=dict)
    joints: Dict[str, 'SuspensionJoint'] = field(default_factory=dict)

    # Special components
    knuckles: Dict[str, 'SuspensionKnuckle'] = field(default_factory=dict)

    # Attachment points
    chassis_points: List[AttachmentPoint] = field(default_factory=list)
    all_attachment_points: List[AttachmentPoint] = field(default_factory=list)

    # Mapping from attachment point id to parent component
    component_map: Dict[int, object] = field(default_factory=dict)

    def get_component_count(self) -> int:
        """Get total number of components."""
        return (len(self.rigid_bodies) + len(self.links) +
                len(self.springs) + len(self.joints))

    def get_free_point_count(self) -> int:
        """Get number of non-chassis attachment points."""
        return len(self.all_attachment_points) - len(self.chassis_points)


class KinematicSolver:
    """
    Direct component-level kinematic solver for suspension systems.

    This solver maintains an internal representation using actual suspension
    components (RigidBody, SuspensionLink, SuspensionJoint) and generates
    constraints dynamically from their current geometry. After solving, it
    automatically updates all component positions.

    Key features:
    - Direct construction from Chassis object
    - Automatic constraint generation from component geometry
    - Special handling for variable-length springs (no length constraint)
    - Automatic component state updates after solving
    - Component-level force and energy analysis

    The solver minimizes:
        E = Σ (JOINT_STIFFNESS[joint_type] × error²)

    This is equivalent to minimizing elastic energy in the compliant joints.

    Example:
        >>> from pysuspension import Chassis, KinematicSolver
        >>> chassis = Chassis(...)  # Load or create chassis
        >>> solver = KinematicSolver.from_chassis(chassis, ['front_left'])
        >>> result = solver.solve_for_heave('front_left_knuckle', -50, unit='mm')
        >>> print(f"Solved in {result.iterations} iterations")
        >>> print(f"Spring force: {solver.get_spring_force('front_left_spring'):.1f} N")
    """

    def __init__(self, name: str = "kinematic_solver"):
        """
        Initialize a kinematic solver.

        Typically, you should use the from_chassis() class method instead
        of calling this constructor directly.

        Args:
            name: Identifier for this solver instance
        """
        self.name = name

        # Component registry
        self.registry = ComponentRegistry()

        # Solver infrastructure (reuse from SuspensionSolver)
        self.state = SolverState(name=f"{name}_state")

        # Generated constraints (rebuilt each solve)
        self._active_constraints: List[Constraint] = []

        # Solver settings
        self.max_iterations = 1000
        self.tolerance = 1e-9
        self.method = 'least-squares'

        # Last solve result
        self._last_result: Optional[SolverResult] = None

    @classmethod
    def from_chassis(cls,
                     chassis: 'Chassis',
                     corner_names: Optional[List[str]] = None) -> 'KinematicSolver':
        """
        Create a KinematicSolver from a Chassis object.

        Discovers all suspension components connected to the specified corners
        using graph traversal from the knuckle through joints and attachment points.

        Args:
            chassis: Chassis object containing suspension corners
            corner_names: List of corner names to include (e.g., ['front_left', 'front_right'])
                         If None, includes all corners

        Returns:
            New KinematicSolver instance with all components registered

        Raises:
            ValueError: If chassis has no corners or specified corner names don't exist

        Example:
            >>> solver = KinematicSolver.from_chassis(chassis, ['front_left'])
            >>> # Solver now contains all components from front left suspension
        """
        # Implementation will be in Step 2
        raise NotImplementedError("from_chassis() will be implemented in Step 2")

    def add_component(self, component: object, component_type: str = "auto"):
        """
        Add a component to the registry.

        Automatically categorizes the component by type and registers all
        its attachment points.

        Args:
            component: Component object to add (RigidBody, SuspensionLink, etc.)
            component_type: Type hint ('rigid_body', 'link', 'spring', 'joint', 'auto')
                           If 'auto', type is inferred from component class
        """
        from .rigid_body import RigidBody
        from .suspension_link import SuspensionLink
        from .coil_spring import CoilSpring
        from .suspension_joint import SuspensionJoint
        from .suspension_knuckle import SuspensionKnuckle

        # Infer type if auto
        if component_type == "auto":
            if isinstance(component, CoilSpring):
                component_type = "spring"
            elif isinstance(component, SuspensionLink):
                component_type = "link"
            elif isinstance(component, SuspensionJoint):
                component_type = "joint"
            elif isinstance(component, SuspensionKnuckle):
                component_type = "knuckle"
            elif isinstance(component, RigidBody):
                component_type = "rigid_body"
            else:
                raise ValueError(f"Cannot infer type for component: {component}")

        # Add to appropriate registry
        if component_type == "rigid_body":
            self.registry.rigid_bodies[component.name] = component
            # Register attachment points
            for ap in component.attachment_points:
                if ap not in self.registry.all_attachment_points:
                    self.registry.all_attachment_points.append(ap)
                    self.registry.component_map[id(ap)] = component

        elif component_type == "knuckle":
            self.registry.knuckles[component.name] = component
            self.registry.rigid_bodies[component.name] = component
            # Register attachment points
            for ap in component.attachment_points:
                if ap not in self.registry.all_attachment_points:
                    self.registry.all_attachment_points.append(ap)
                    self.registry.component_map[id(ap)] = component

        elif component_type == "link":
            self.registry.links[component.name] = component
            # Register endpoints
            for ap in [component.endpoint1, component.endpoint2]:
                if ap not in self.registry.all_attachment_points:
                    self.registry.all_attachment_points.append(ap)
                    self.registry.component_map[id(ap)] = component

        elif component_type == "spring":
            self.registry.springs[component.name] = component
            # Register endpoints
            for ap in [component.endpoint1, component.endpoint2]:
                if ap not in self.registry.all_attachment_points:
                    self.registry.all_attachment_points.append(ap)
                    self.registry.component_map[id(ap)] = component

        elif component_type == "joint":
            self.registry.joints[component.name] = component
            # Joints don't have attachment points directly
            # Their attachment points are registered with their parent components

        else:
            raise ValueError(f"Unknown component type: {component_type}")

    def get_knuckle(self, knuckle_name: str) -> 'SuspensionKnuckle':
        """
        Get a knuckle by name.

        Args:
            knuckle_name: Name of the knuckle

        Returns:
            SuspensionKnuckle object

        Raises:
            KeyError: If knuckle not found
        """
        if knuckle_name not in self.registry.knuckles:
            raise KeyError(f"Knuckle '{knuckle_name}' not found in solver. "
                          f"Available knuckles: {list(self.registry.knuckles.keys())}")
        return self.registry.knuckles[knuckle_name]

    def get_spring_force(self, spring_name: str) -> float:
        """
        Get current reaction force of a spring.

        Args:
            spring_name: Name of the spring

        Returns:
            Reaction force in Newtons (positive = compression)

        Raises:
            KeyError: If spring not found
        """
        if spring_name not in self.registry.springs:
            raise KeyError(f"Spring '{spring_name}' not found in solver. "
                          f"Available springs: {list(self.registry.springs.keys())}")
        spring = self.registry.springs[spring_name]
        return spring.get_reaction_force_magnitude()

    def _generate_constraints_from_components(self) -> List[Constraint]:
        """
        Generate constraints dynamically from current component geometry.

        This is called at the start of each solve to ensure constraints
        match the current component configuration.

        Constraint generation rules:
        1. RigidBody: Distance constraints between all attachment point pairs
        2. SuspensionLink: Single distance constraint for link length
        3. CoilSpring: NO distance constraint (allows compression/extension)
        4. SuspensionJoint: Coincident point constraints weighted by stiffness
        5. Chassis mounts: Fixed point constraints

        Returns:
            List of Constraint objects
        """
        # Implementation will be in Step 3
        raise NotImplementedError("_generate_constraints_from_components() will be implemented in Step 3")

    def _update_component_positions(self, solution_vector: np.ndarray):
        """
        Update all component positions from the solved state.

        This propagates the optimized attachment point positions back to
        the component objects using their fit_to_attachment_targets() methods.

        Args:
            solution_vector: Optimized position vector from solver
        """
        # Implementation will be in Step 5
        raise NotImplementedError("_update_component_positions() will be implemented in Step 5")

    def solve_for_target(self,
                        knuckle_name: str,
                        target_constraint: Constraint) -> SolverResult:
        """
        Solve for a target constraint on a knuckle.

        Args:
            knuckle_name: Name of the knuckle to constrain
            target_constraint: Constraint defining the target (e.g., heave position)

        Returns:
            SolverResult with convergence info and final positions
        """
        # Implementation will be in Step 4
        raise NotImplementedError("solve_for_target() will be implemented in Step 4")

    def solve_for_heave(self,
                       knuckle_name: str,
                       displacement: float,
                       unit: str = 'mm') -> SolverResult:
        """
        Solve for suspension heave (vertical wheel motion).

        Convenience method that creates a vertical position constraint
        on the tire contact patch.

        Args:
            knuckle_name: Name of the knuckle to move
            displacement: Vertical displacement (positive = up, negative = down)
            unit: Unit of displacement (default: 'mm')

        Returns:
            SolverResult with convergence info and final positions

        Example:
            >>> # Compress suspension by 50mm
            >>> result = solver.solve_for_heave('front_left_knuckle', -50, unit='mm')
            >>> print(f"Spring force: {solver.get_spring_force('spring'):.1f} N")
        """
        # Implementation will be in Step 6
        raise NotImplementedError("solve_for_heave() will be implemented in Step 6")

    def compute_total_error(self) -> float:
        """
        Compute total weighted error across all active constraints.

        Returns:
            Sum of weighted errors
        """
        total = 0.0
        for constraint in self._active_constraints:
            total += constraint.get_weighted_error()
        return total

    def compute_constraint_errors(self) -> Dict[str, float]:
        """
        Compute error for each active constraint.

        Returns:
            Dictionary mapping constraint name to physical error (mm)
        """
        errors = {}
        for constraint in self._active_constraints:
            errors[constraint.name] = constraint.get_physical_error()
        return errors

    def __repr__(self) -> str:
        return (f"KinematicSolver('{self.name}', "
                f"rigid_bodies={len(self.registry.rigid_bodies)}, "
                f"links={len(self.registry.links)}, "
                f"springs={len(self.registry.springs)}, "
                f"joints={len(self.registry.joints)})")


if __name__ == "__main__":
    print("=" * 70)
    print("KINEMATIC SOLVER")
    print("=" * 70)
    print("\nKinematicSolver: Direct component-level suspension kinematics solver")
    print("\nKey features:")
    print("  - Works directly with suspension components")
    print("  - Automatic constraint generation from geometry")
    print("  - Special handling for variable-length springs")
    print("  - Automatic component state updates after solving")
    print("\nUsage:")
    print("  solver = KinematicSolver.from_chassis(chassis, ['front_left'])")
    print("  result = solver.solve_for_heave('front_left_knuckle', -50)")
    print("\nSee tests/test_kinematic_solver.py for examples.")
