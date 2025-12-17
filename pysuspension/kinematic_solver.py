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
        from .suspension_knuckle import SuspensionKnuckle

        # Validate chassis has corners
        if not chassis.corners:
            raise ValueError("Chassis has no corners")

        # Determine which corners to include
        if corner_names is None:
            corner_names = list(chassis.corners.keys())
        else:
            # Validate corner names exist
            for corner_name in corner_names:
                if corner_name not in chassis.corners:
                    available = list(chassis.corners.keys())
                    raise ValueError(f"Corner '{corner_name}' not found in chassis. "
                                   f"Available corners: {available}")

        # Create solver instance
        solver_name = f"kinematic_solver_{chassis.name}"
        solver = cls(name=solver_name)

        # Track all discovered components to avoid duplicates
        discovered_component_ids: Set[int] = set()
        discovered_joint_ids: Set[int] = set()

        # Process each corner
        for corner_name in corner_names:
            corner = chassis.get_corner(corner_name)

            # Find the knuckle associated with this corner
            # Strategy: Look through chassis components for SuspensionKnuckle objects
            # and match by corner name pattern or by checking joint connections
            knuckle = solver._find_knuckle_for_corner(chassis, corner_name)

            if knuckle is None:
                # Try alternative: search through corner's joint connections
                knuckle = solver._find_knuckle_through_joints(corner)

            if knuckle is None:
                raise ValueError(f"Could not find SuspensionKnuckle for corner '{corner_name}'. "
                               f"Ensure the knuckle is registered with the chassis and connected "
                               f"to the corner's attachment points through joints.")

            # Discover suspension graph from this knuckle
            graph = discover_suspension_graph(knuckle)

            # Register knuckle
            if id(knuckle) not in discovered_component_ids:
                solver.add_component(knuckle, component_type="knuckle")
                discovered_component_ids.add(id(knuckle))

            # Register control arms (RigidBody components)
            for control_arm in graph.control_arms:
                if id(control_arm) not in discovered_component_ids:
                    solver.add_component(control_arm, component_type="rigid_body")
                    discovered_component_ids.add(id(control_arm))

            # Register standalone links (fixed length)
            for link in graph.links:
                if id(link) not in discovered_component_ids:
                    solver.add_component(link, component_type="link")
                    discovered_component_ids.add(id(link))

            # Register coil springs (variable length)
            for spring in graph.coil_springs:
                if id(spring) not in discovered_component_ids:
                    solver.add_component(spring, component_type="spring")
                    discovered_component_ids.add(id(spring))

            # Register steering racks
            for rack in graph.steering_racks:
                if id(rack) not in discovered_component_ids:
                    solver.add_component(rack, component_type="rigid_body")
                    discovered_component_ids.add(id(rack))

            # Register joints
            for joint_name, joint in graph.joints.items():
                if id(joint) not in discovered_joint_ids:
                    solver.add_component(joint, component_type="joint")
                    discovered_joint_ids.add(id(joint))

            # Register chassis points (these are fixed)
            for chassis_point in graph.chassis_points:
                if chassis_point not in solver.registry.chassis_points:
                    solver.registry.chassis_points.append(chassis_point)

        # Initialize solver state with all attachment points
        for ap in solver.registry.all_attachment_points:
            # Check if this point is a chassis mount (fixed)
            if ap in solver.registry.chassis_points:
                solver.state.add_fixed_point(ap)
            else:
                solver.state.add_free_point(ap)

        return solver

    @staticmethod
    def _find_knuckle_for_corner(chassis: 'Chassis', corner_name: str) -> Optional['SuspensionKnuckle']:
        """
        Find the SuspensionKnuckle associated with a corner by name matching.

        Searches through chassis components for SuspensionKnuckle objects
        whose name contains the corner name pattern.

        Args:
            chassis: Chassis object to search
            corner_name: Name of the corner (e.g., 'front_left')

        Returns:
            SuspensionKnuckle if found, None otherwise
        """
        from .suspension_knuckle import SuspensionKnuckle

        # Search through all chassis components
        for component_name, component in chassis.components.items():
            if isinstance(component, SuspensionKnuckle):
                # Check if corner name is in component name
                # e.g., 'front_left' in 'front_left_knuckle'
                if corner_name.lower() in component_name.lower():
                    return component

        return None

    @staticmethod
    def _find_knuckle_through_joints(corner: 'ChassisCorner') -> Optional['SuspensionKnuckle']:
        """
        Find the SuspensionKnuckle connected to a corner through joint connections.

        Traverses from the corner's attachment points through their joints
        to find connected SuspensionKnuckle components.

        Args:
            corner: ChassisCorner to search from

        Returns:
            SuspensionKnuckle if found, None otherwise
        """
        from .suspension_knuckle import SuspensionKnuckle

        # Check each attachment point in the corner
        for corner_ap in corner.attachment_points:
            # If this point has a joint
            if corner_ap.joint is not None:
                # Get all points connected through this joint
                connected_points = corner_ap.joint.get_all_attachment_points()

                # Check each connected point's parent component
                for connected_ap in connected_points:
                    if connected_ap.parent_component is not None:
                        parent = connected_ap.parent_component
                        if isinstance(parent, SuspensionKnuckle):
                            return parent

        return None

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
        constraints = []

        # 1. RigidBody constraints: Maintain all pairwise distances
        for name, rigid_body in self.registry.rigid_bodies.items():
            points = rigid_body.attachment_points

            # Generate all pairwise distance constraints
            for i, p1 in enumerate(points):
                for j, p2 in enumerate(points[i+1:], start=i+1):
                    # Calculate current distance
                    distance = np.linalg.norm(p2.position - p1.position)

                    # Create constraint with RIGID joint type (very stiff)
                    constraint = DistanceConstraint(
                        p1, p2, distance,
                        name=f"{name}_rigid_{p1.name}_{p2.name}",
                        joint_type=JointType.RIGID
                    )
                    constraints.append(constraint)

        # 2. SuspensionLink constraints: Fixed length
        for name, link in self.registry.links.items():
            # Get current length (should be constant)
            distance = link.length

            constraint = DistanceConstraint(
                link.endpoint1, link.endpoint2, distance,
                name=f"{name}_link_length",
                joint_type=JointType.RIGID
            )
            constraints.append(constraint)

        # 3. CoilSpring: NO distance constraint (allows length to vary)
        # Springs are handled specially - no constraint added here
        # Spring forces are computed post-solve based on length change

        # 4. SuspensionJoint constraints: Points should coincide
        for name, joint in self.registry.joints.items():
            points = joint.get_all_attachment_points()

            if len(points) < 2:
                continue  # Need at least 2 points to constrain

            # Create coincident constraints between all pairs
            # This ensures all points at this joint converge to the same location
            for i, p1 in enumerate(points):
                for j, p2 in enumerate(points[i+1:], start=i+1):
                    constraint = CoincidentPointConstraint(
                        p1, p2,
                        name=f"{name}_coincident_{i}_{j}",
                        joint_type=joint.joint_type
                    )
                    constraints.append(constraint)

        # 5. Chassis mount points: Fixed in space
        for i, chassis_point in enumerate(self.registry.chassis_points):
            # Create fixed point constraint at current position
            constraint = FixedPointConstraint(
                chassis_point,
                chassis_point.position.copy(),
                name=f"chassis_mount_{chassis_point.name}",
                joint_type=JointType.RIGID
            )
            constraints.append(constraint)

        return constraints

    def _update_component_positions(self, solution_vector: np.ndarray):
        """
        Update all component positions from the solved state.

        This propagates the optimized attachment point positions back to
        the component objects using their fit_to_attachment_targets() methods.

        Args:
            solution_vector: Optimized position vector from solver
        """
        # Ensure state is updated
        self.state.from_vector(solution_vector)

        # Update RigidBody components (knuckles, control arms, steering racks)
        for name, rigid_body in self.registry.rigid_bodies.items():
            # Get target positions for all attachment points
            target_positions = [
                ap.position.copy() for ap in rigid_body.attachment_points
            ]

            # Update rigid body to best fit target positions
            # This uses SVD-based Kabsch algorithm for optimal rigid transformation
            rigid_body.fit_to_attachment_targets(target_positions, unit='mm')

        # Update SuspensionLink components (fixed length)
        for name, link in self.registry.links.items():
            # Update link endpoints
            target_positions = [
                link.endpoint1.position.copy(),
                link.endpoint2.position.copy()
            ]
            link.fit_to_attachment_targets(target_positions, unit='mm')

        # Update CoilSpring components (variable length)
        for name, spring in self.registry.springs.items():
            # Update spring endpoints - this allows compression/extension
            target_positions = [
                spring.endpoint1.position.copy(),
                spring.endpoint2.position.copy()
            ]
            spring.fit_to_attachment_targets(target_positions, unit='mm')
            # Spring force is automatically recalculated by fit_to_attachment_targets

    def _compute_forces_and_energy(self, result: SolverResult):
        """
        Compute forces at constraints and total compliance energy.

        Populates the SolverResult with:
        - forces: Dictionary of constraint forces
        - compliance_energy: Total elastic energy in compliant joints
        - spring_forces: Dictionary of spring reaction forces (added to result.geometry)
        - spring_energy: Total elastic energy in springs

        Args:
            result: SolverResult to populate with force/energy data
        """
        total_energy = 0.0
        spring_forces = {}
        spring_energy = 0.0

        # Compute forces and energy from constraints
        for constraint in self._active_constraints:
            if hasattr(constraint, 'get_force'):
                try:
                    force = constraint.get_force()
                    force_magnitude = np.linalg.norm(force) if hasattr(force, '__len__') else abs(force)
                    result.forces[constraint.name] = force_magnitude
                except:
                    pass  # Some constraints may not support force calculation

            # Energy = 0.5 × k × x²
            error = constraint.evaluate()  # Already squared
            energy = 0.5 * constraint.stiffness * error
            total_energy += energy

        # Compute spring forces and energy separately
        for name, spring in self.registry.springs.items():
            # Get current force
            force = spring.get_reaction_force_magnitude()
            spring_forces[name] = force

            # Calculate spring energy
            length_change = spring.get_length_change()

            # Spring rate is in kg/mm, convert to N/mm
            k_n_per_mm = spring.spring_rate * 9.80665

            # Energy = 0.5 * k * delta_x²
            energy = 0.5 * k_n_per_mm * (length_change ** 2)
            spring_energy += energy

        # Store results
        result.compliance_energy = total_energy
        result.geometry['spring_forces'] = spring_forces
        result.geometry['spring_energy'] = spring_energy
        result.geometry['total_elastic_energy'] = total_energy + spring_energy

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

        Raises:
            ImportError: If scipy is not available
            ValueError: If knuckle not found or no constraints
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for solving. Install with: pip install scipy")

        # Validate knuckle exists
        if knuckle_name not in self.registry.knuckles:
            raise ValueError(f"Knuckle '{knuckle_name}' not found. "
                           f"Available: {list(self.registry.knuckles.keys())}")

        # Generate constraints from current component geometry
        self._active_constraints = self._generate_constraints_from_components()

        # Add target constraint
        self._active_constraints.append(target_constraint)

        if not self._active_constraints:
            raise ValueError("No constraints generated. Cannot solve.")

        # Get initial guess from current state
        x0 = self.state.to_vector()

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

        # Build residual function for least-squares
        def residual_function(x: np.ndarray) -> np.ndarray:
            # Update state from optimization vector
            self.state.from_vector(x)

            # Build residual vector
            residuals = []
            for constraint in self._active_constraints:
                # Residual = sqrt(weight) × sqrt(error)
                # This makes least-squares minimize Σ weight × error
                weighted_residual = np.sqrt(constraint.weight * constraint.evaluate())
                residuals.append(weighted_residual)

            return np.array(residuals)

        # Solve using Trust Region Reflective (handles over/under-constrained)
        result = optimize.least_squares(
            residual_function,
            x0,
            method='trf',  # Trust Region Reflective
            max_nfev=self.max_iterations,
            ftol=self.tolerance,
            xtol=self.tolerance,
            gtol=self.tolerance
        )

        # Update state with final solution
        self.state.from_vector(result.x)

        # Update component positions from solved state
        self._update_component_positions(result.x)

        # Build solver result
        solver_result = SolverResult(
            success=result.success,
            message=result.message,
            iterations=result.nfev,
            total_error=np.sum(result.fun ** 2),  # Sum of squared residuals
            positions={name: point.position.copy()
                      for name, point in self.state.points.items()},
            constraint_errors=self.compute_constraint_errors()
        )

        # Compute forces and compliance energy
        self._compute_forces_and_energy(solver_result)

        # Store result for later queries
        self._last_result = solver_result

        return solver_result

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
