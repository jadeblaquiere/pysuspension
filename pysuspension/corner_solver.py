"""
Corner-level solver for single suspension corners.

This module provides a specialized solver that automatically builds constraints
from suspension component geometry (control arms, knuckle, links, etc.) and
provides convenient methods for common suspension analysis tasks.

Requirements:
    - scipy: For numerical optimization
    Install with: pip install scipy
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from .solver import SuspensionSolver, SolverResult

if TYPE_CHECKING:
    from .suspension_knuckle import SuspensionKnuckle
from .constraints import (
    DistanceConstraint,
    FixedPointConstraint,
    CoincidentPointConstraint,
    PartialPositionConstraint
)
from .joint_types import JointType
from .attachment_point import AttachmentPoint
from .control_arm import ControlArm
from .suspension_link import SuspensionLink
from .suspension_joint import SuspensionJoint
from .units import to_mm, from_mm
from .geometry_utils import calculate_instant_center_from_points


class CornerSolver(SuspensionSolver):
    """
    Specialized solver for single suspension corners.

    Automatically builds constraints from suspension geometry:
    - Fixed constraints for chassis mount points
    - Distance constraints for links (control arms, tie rods, etc.)
    - Coincident constraints for ball joints and bushings

    Provides convenience methods:
    - solve_for_heave(displacement): Solve for vertical wheel travel
    - solve_for_wheel_position(position): Solve for specific wheel position
    - get_camber(): Calculate camber angle from solved position
    - get_toe(): Calculate toe angle from solved position
    """

    def __init__(self, name: str = "corner_solver"):
        """
        Initialize corner solver.

        Args:
            name: Identifier for this solver
        """
        super().__init__(name=name)

        # Joint registry - joints are first-class objects
        self.joints: Dict[str, SuspensionJoint] = {}

        # Component tracking
        self.control_arms: List[ControlArm] = []
        self.links: List[SuspensionLink] = []
        self.chassis_mounts: List[AttachmentPoint] = []

        # Reference points for geometry calculations
        self.wheel_center: Optional[AttachmentPoint] = None
        self.knuckle_points: List[AttachmentPoint] = []

        # Initial state for resetting
        self._initial_snapshot = None

        # Model integration (for from_suspension_knuckle)
        self.original_knuckle: Optional['SuspensionKnuckle'] = None
        self.copied_knuckle: Optional['SuspensionKnuckle'] = None
        self.component_mapping: Dict = {}  # original_id -> copy
        self.reverse_mapping: Dict = {}    # copy_id -> original

    def add_joint(self,
                 name: str,
                 points: List[AttachmentPoint],
                 joint_type: JointType,
                 stiffness: Optional[float] = None) -> SuspensionJoint:
        """
        Add a joint connecting multiple attachment points.

        This is the primary method for defining joints in the solver. Joints are
        first-class objects that define compliance between connected points.
        All components (links, control arms) are rigid - only joints have compliance.

        Args:
            name: Unique identifier for the joint
            points: List of AttachmentPoint objects to connect (must be 2+)
            joint_type: Type of joint (BALL_JOINT, BUSHING_SOFT, etc.)
            stiffness: Custom stiffness in N/mm (overrides joint_type default)

        Returns:
            SuspensionJoint object that was created and registered

        Raises:
            ValueError: If name already exists or insufficient points provided

        Examples:
            # Ball joint connecting control arm to knuckle
            ball_joint = solver.add_joint(
                name="upper_ball_joint",
                points=[control_arm.endpoint, knuckle.upper_mount],
                joint_type=JointType.BALL_JOINT
            )

            # Soft bushing at chassis mount
            bushing = solver.add_joint(
                name="front_bushing",
                points=[control_arm.front_mount, chassis.mount_point],
                joint_type=JointType.BUSHING_SOFT
            )

            # Custom stiffness bushing
            custom_joint = solver.add_joint(
                name="custom_bushing",
                points=[link.end1, chassis.mount],
                joint_type=JointType.CUSTOM,
                stiffness=500.0  # 500 N/mm
            )
        """
        # Validate inputs
        if name in self.joints:
            raise ValueError(f"Joint '{name}' already exists in solver")

        if len(points) < 2:
            raise ValueError(
                f"Joint '{name}' must connect at least 2 points, got {len(points)}"
            )

        # Create the joint
        joint = SuspensionJoint(name=name, joint_type=joint_type)

        # Store custom stiffness if provided
        if stiffness is not None:
            joint.stiffness = stiffness

        # Connect all attachment points to this joint
        for point in points:
            joint.add_attachment_point(point)

        # Register the joint
        self.joints[name] = joint

        return joint

    def get_joint(self, name: str) -> SuspensionJoint:
        """
        Get a joint by name.

        Args:
            name: Name of the joint

        Returns:
            SuspensionJoint object

        Raises:
            KeyError: If joint name doesn't exist
        """
        if name not in self.joints:
            raise KeyError(f"Joint '{name}' not found in solver")
        return self.joints[name]

    def get_joints_at_point(self, point: AttachmentPoint) -> List[SuspensionJoint]:
        """
        Get all joints connected to a specific attachment point.

        Args:
            point: AttachmentPoint to query

        Returns:
            List of SuspensionJoint objects connected to this point
        """
        connected_joints = []
        for joint in self.joints.values():
            if any(p is point for p in joint.attachment_points):
                connected_joints.append(joint)
        return connected_joints

    def get_joint_compliance(self, joint_name: str) -> float:
        """
        Get compliance (mm/N) of a joint.

        Args:
            joint_name: Name of the joint

        Returns:
            Compliance in mm/N

        Raises:
            KeyError: If joint doesn't exist
        """
        from .joint_types import JOINT_STIFFNESS

        joint = self.get_joint(joint_name)

        # Use custom stiffness if available
        if hasattr(joint, 'stiffness') and joint.stiffness is not None:
            return 1.0 / joint.stiffness

        # Otherwise use standard stiffness for the joint type
        stiffness = JOINT_STIFFNESS[joint.joint_type]
        return 1.0 / stiffness

    def get_joint_stiffness(self, joint_name: str) -> float:
        """
        Get stiffness (N/mm) of a joint.

        Args:
            joint_name: Name of the joint

        Returns:
            Stiffness in N/mm

        Raises:
            KeyError: If joint doesn't exist
        """
        from .joint_types import JOINT_STIFFNESS

        joint = self.get_joint(joint_name)

        # Use custom stiffness if available
        if hasattr(joint, 'stiffness') and joint.stiffness is not None:
            return joint.stiffness

        # Otherwise use standard stiffness for the joint type
        return JOINT_STIFFNESS[joint.joint_type]

    def list_joints(self) -> List[str]:
        """
        Get list of all joint names in the solver.

        Returns:
            List of joint names
        """
        return list(self.joints.keys())

    @classmethod
    def from_suspension_knuckle(cls,
                                knuckle: 'SuspensionKnuckle',
                                wheel_center_point: Optional[AttachmentPoint] = None,
                                name: str = "corner_solver",
                                copy_components: bool = True) -> 'CornerSolver':
        """
        Create a CornerSolver from a suspension knuckle.

        This method automatically discovers all connected components by traversing
        from the knuckle through joints and attachment points to find control arms,
        links, and chassis mount points. It then configures the solver with the
        discovered geometry.

        The discovery process:
        1. Discovers all connected components via graph traversal
        2. Creates working copies of all components (if copy_components=True)
        3. Configures solver with discovered geometry
        4. Fixes chassis attachment points
        5. Sets up all joints and constraints

        Args:
            knuckle: SuspensionKnuckle to build solver from
            wheel_center_point: Optional specific point to use as wheel center.
                              If None, creates point from knuckle.tire_center
            name: Name for the solver (default: "corner_solver")
            copy_components: If True, creates copies of all components; if False,
                           uses originals (default: True). Using copies is safer
                           as it leaves original components unchanged.

        Returns:
            Configured CornerSolver ready for solving

        Raises:
            ValueError: If knuckle has no attachment points or invalid configuration

        Example:
            >>> from pysuspension import SuspensionKnuckle, CornerSolver
            >>> knuckle = SuspensionKnuckle(...)
            >>> # ... set up control arms, links, joints ...
            >>> solver = CornerSolver.from_suspension_knuckle(knuckle)
            >>> result = solver.solve_for_heave(25, unit='mm')
            >>> # Original knuckle unchanged, solver has copies
            >>> solved_knuckle = solver.get_solved_knuckle()
        """
        from .suspension_graph import discover_suspension_graph, create_working_copies

        # Phase 1: Discover the suspension graph
        print(f"Discovering suspension graph from knuckle '{knuckle.name}'...")
        graph = discover_suspension_graph(knuckle)

        print(f"  Found {len(graph.control_arms)} control arms")
        print(f"  Found {len(graph.links)} standalone links")
        print(f"  Found {len(graph.joints)} joints")
        print(f"  Found {len(graph.chassis_points)} chassis mount points")

        # Phase 2: Create working copies if requested
        if copy_components:
            print("Creating working copies of components...")
            copied_graph, mapping = create_working_copies(graph)
            work_graph = copied_graph
            original_knuckle = knuckle
            work_knuckle = copied_graph.knuckle
        else:
            print("Using original components (no copying)...")
            work_graph = graph
            mapping = {}
            original_knuckle = knuckle
            work_knuckle = knuckle

        # Phase 3: Create the solver instance
        print(f"Configuring CornerSolver '{name}'...")
        solver = cls(name=name)

        # Store references
        solver.original_knuckle = original_knuckle
        solver.copied_knuckle = work_knuckle
        solver.component_mapping = mapping
        solver.reverse_mapping = {id(v): k for k, v in mapping.items()}

        # Phase 4: Register all joints
        print("Registering joints...")
        for joint_name, joint in work_graph.joints.items():
            solver.joints[joint_name] = joint

        # Phase 5: Add chassis mounts as fixed points
        print("Setting up chassis mount constraints...")
        for chassis_point in work_graph.chassis_points:
            solver.chassis_mounts.append(chassis_point)
            # Add to solver state
            if chassis_point.name not in solver.state.points:
                solver.state.add_point(chassis_point)
            # Add fixed constraint
            solver.add_constraint(
                FixedPointConstraint(
                    chassis_point,
                    chassis_point.position.copy(),
                    name=f"chassis_{chassis_point.name}",
                    joint_type=JointType.RIGID
                )
            )
            solver.set_point_fixed(chassis_point.name)

        # Phase 6: Add all control arms
        print("Adding control arms...")
        for control_arm in work_graph.control_arms:
            # Determine which points are chassis mounts
            chassis_mount_points = []
            knuckle_mount_points = []

            # Check each attachment point in the control arm
            all_arm_points = []
            for link in control_arm.links:
                # Use identity comparison to avoid numpy array comparison issues
                if not any(link.endpoint1 is p for p in all_arm_points):
                    all_arm_points.append(link.endpoint1)
                if not any(link.endpoint2 is p for p in all_arm_points):
                    all_arm_points.append(link.endpoint2)
            for ap in control_arm.attachment_points:
                if not any(ap is p for p in all_arm_points):
                    all_arm_points.append(ap)

            for point in all_arm_points:
                # Use identity comparison to avoid numpy array comparison issues
                if any(point is cp for cp in work_graph.chassis_points):
                    chassis_mount_points.append(point)
                elif any(point is kp for kp in work_graph.knuckle_points):
                    knuckle_mount_points.append(point)

            solver.add_control_arm(
                control_arm,
                chassis_mount_points=chassis_mount_points,
                knuckle_mount_points=knuckle_mount_points
            )

        # Phase 7: Add standalone links
        print("Adding standalone links...")
        for link in work_graph.links:
            # Determine mount points for this link
            end1_mount = None
            end2_mount = None

            # Use identity comparison to avoid numpy array comparison issues
            if any(link.endpoint1 is cp for cp in work_graph.chassis_points):
                end1_mount = link.endpoint1
            if any(link.endpoint2 is cp for cp in work_graph.chassis_points):
                end2_mount = link.endpoint2

            solver.add_link(
                link,
                end1_mount_point=end1_mount,
                end2_mount_point=end2_mount
            )

        # Phase 7b: Add coil springs
        print("Adding coil springs...")
        for spring in work_graph.coil_springs:
            # Determine mount points for this spring
            end1_mount = None
            end2_mount = None

            # Use identity comparison to avoid numpy array comparison issues
            if any(spring.endpoint1 is cp for cp in work_graph.chassis_points):
                end1_mount = spring.endpoint1
            if any(spring.endpoint2 is cp for cp in work_graph.chassis_points):
                end2_mount = spring.endpoint2

            # Add coil spring as a link with distance constraint
            # Springs can compress/extend, so we use the current length as target
            solver.add_link(
                spring,
                end1_mount_point=end1_mount,
                end2_mount_point=end2_mount
            )

        # Phase 7c: Add steering racks
        print("Adding steering racks...")
        for rack in work_graph.steering_racks:
            # Add steering rack housing attachment points as fixed (chassis mounts)
            for housing_point in rack.housing.get_all_attachment_points():
                solver.chassis_mounts.append(housing_point)
                # Add to solver state
                if housing_point.name not in solver.state.points:
                    solver.state.add_point(housing_point)
                # Add fixed constraint
                solver.add_constraint(
                    FixedPointConstraint(
                        housing_point,
                        housing_point.position.copy(),
                        name=f"rack_housing_{housing_point.name}",
                        joint_type=JointType.RIGID
                    )
                )
                solver.set_point_fixed(housing_point.name)

            # Add inner pivot points to solver state as free points
            # These will be connected to tie rods via joints
            for pivot in [rack.left_inner_pivot, rack.right_inner_pivot]:
                if pivot.name not in solver.state.points:
                    solver.state.add_point(pivot)
                solver.set_point_free(pivot.name)

        # Phase 8: Set wheel center
        if wheel_center_point is not None:
            # Use the provided wheel center point
            # If it was copied, find the copy
            if copy_components and id(wheel_center_point) in mapping:
                solver.wheel_center = mapping[id(wheel_center_point)]
            else:
                solver.wheel_center = wheel_center_point
        else:
            # Create wheel center point from knuckle tire center
            wheel_center = AttachmentPoint(
                name=f"{work_knuckle.name}_wheel_center",
                position=work_knuckle.tire_center.copy(),
                unit='mm',
                parent_component=work_knuckle
            )
            # Add to solver state
            solver.state.add_point(wheel_center)
            solver.set_point_free(wheel_center.name)
            solver.wheel_center = wheel_center

        # Store knuckle points and add them to solver state
        solver.knuckle_points = list(work_graph.knuckle_points)
        for knuckle_point in work_graph.knuckle_points:
            if knuckle_point.name not in solver.state.points:
                solver.state.add_point(knuckle_point)
            solver.set_point_free(knuckle_point.name)

        # Add distance constraints between knuckle points to maintain knuckle as rigid body
        # This allows knuckle points to move with control arms while maintaining relative positions
        if len(solver.knuckle_points) >= 2:
            for i, kp1 in enumerate(solver.knuckle_points):
                for kp2 in solver.knuckle_points[i+1:]:
                    distance = np.linalg.norm(kp1.position - kp2.position)
                    solver.add_constraint(
                        DistanceConstraint(
                            kp1,
                            kp2,
                            target_distance=distance,
                            name=f"knuckle_rigid_{kp1.name}_{kp2.name}",
                            joint_type=JointType.RIGID
                        )
                    )

        # Also add distance constraint from wheel_center to one knuckle point
        # This links the wheel_center movement to the knuckle
        if solver.wheel_center is not None and solver.knuckle_points:
            ref_point = solver.knuckle_points[0]  # Use first knuckle point as reference
            distance = np.linalg.norm(solver.wheel_center.position - ref_point.position)
            solver.add_constraint(
                DistanceConstraint(
                    solver.wheel_center,
                    ref_point,
                    target_distance=distance,
                    name=f"wheel_to_knuckle",
                    joint_type=JointType.RIGID
                )
            )

        print(f"✓ CornerSolver configured successfully")
        print(f"  Total DOF: {solver.state.get_dof()}")
        print(f"  Total constraints: {len(solver.constraints)}")

        return solver

    def _generate_joint_constraints_for_component(self,
                                                   component,
                                                   component_points: List[AttachmentPoint]):
        """
        Generate coincident constraints for all joints involving the component.

        This method creates CoincidentPointConstraint objects for any registered
        joints that connect points in the component to other points. The joint
        type is automatically inferred from the SuspensionJoint object.

        Args:
            component: The component (ControlArm or SuspensionLink) being added
            component_points: List of attachment points belonging to the component
        """
        # Track which point pairs we've already created constraints for
        constrained_pairs = set()

        # Iterate through all registered joints
        for joint_name, joint in self.joints.items():
            joint_points = joint.attachment_points

            # Find which points in this joint belong to the component
            component_joint_points = [p for p in joint_points if any(p is cp for cp in component_points)]

            if not component_joint_points:
                # This joint doesn't involve the component
                continue

            # Add all points in this joint to solver state (including external points)
            for point in joint_points:
                if point.name not in self.state.points:
                    self.state.add_point(point)

            # For each point in the joint that belongs to the component,
            # create coincident constraints to all other points in the joint
            for comp_point in component_joint_points:
                for other_point in joint_points:
                    if other_point is comp_point:
                        continue

                    # Create a canonical pair key (order doesn't matter)
                    pair_key = tuple(sorted([id(comp_point), id(other_point)]))

                    if pair_key in constrained_pairs:
                        # Already created a constraint for this pair
                        continue

                    # Create coincident constraint with auto-inference of joint type
                    constraint_name = f"{component.name}_{joint_name}_{comp_point.name}_{other_point.name}"
                    self.add_constraint(
                        CoincidentPointConstraint(
                            comp_point,
                            other_point,
                            name=constraint_name,
                            joint_type=None  # Auto-infer from joint
                        )
                    )

                    constrained_pairs.add(pair_key)

    def _validate_joint_topology(self):
        """
        Validate that joint topology is physically consistent.

        Checks:
        - All non-chassis, non-fixed points have at least one joint connection
        - No redundant constraints

        Raises:
            ValueError: If topology is invalid
        """
        # Get all free points (not chassis mounts)
        free_points = []
        for control_arm in self.control_arms:
            for point in control_arm.get_all_attachment_points():
                if point not in self.chassis_mounts:
                    free_points.append(point)

        for link in self.links:
            if link.endpoint1 not in self.chassis_mounts:
                free_points.append(link.endpoint1)
            if link.endpoint2 not in self.chassis_mounts:
                free_points.append(link.endpoint2)

        # Check each free point has at least one joint
        for point in free_points:
            joints_at_point = self.get_joints_at_point(point)
            if not joints_at_point:
                # Check if this point is connected to other points through component structure
                # (e.g., endpoints of a link are connected by the link itself)
                # For now, just warn
                pass  # Could add warning here

    def add_control_arm(self,
                       control_arm: ControlArm,
                       chassis_mount_points: Optional[List[AttachmentPoint]] = None,
                       knuckle_mount_points: Optional[List[AttachmentPoint]] = None):
        """
        Add a control arm with automatic constraint generation.

        Joint types are automatically inferred from joints defined via add_joint().
        You must call add_joint() to define all joints before adding the control arm.

        Args:
            control_arm: ControlArm object
            chassis_mount_points: List of attachment points that mount to chassis (optional)
            knuckle_mount_points: List of attachment points that mount to knuckle (optional)

        Examples:
            # Define joints first
            solver.add_joint("upper_ball", [upper_arm.knuckle_point, knuckle.upper], JointType.BALL_JOINT)
            solver.add_joint("front_bush", [upper_arm.front_chassis, chassis.uf], JointType.BUSHING_SOFT)
            solver.add_joint("rear_bush", [upper_arm.rear_chassis, chassis.ur], JointType.BUSHING_SOFT)

            # Add control arm
            solver.add_control_arm(
                control_arm=upper_arm,
                chassis_mount_points=[upper_arm.front_chassis, upper_arm.rear_chassis],
                knuckle_mount_points=[upper_arm.knuckle_point]
            )
        """
        self.control_arms.append(control_arm)

        # Default to empty lists if not provided
        if chassis_mount_points is None:
            chassis_mount_points = []
        if knuckle_mount_points is None:
            knuckle_mount_points = []

        # Collect all attachment points from the control arm
        # This includes both link endpoints and any additional attachment points
        attachments = []
        for link in control_arm.links:
            if link.endpoint1 not in attachments:
                attachments.append(link.endpoint1)
            if link.endpoint2 not in attachments:
                attachments.append(link.endpoint2)
        # Also include any extra attachment points
        for ap in control_arm.attachment_points:
            if ap not in attachments:
                attachments.append(ap)

        # Add all attachment points to solver state first
        for attachment in attachments:
            if attachment.name not in self.state.points:
                self.state.add_point(attachment)

        # Mark chassis mount points and add fixed constraints
        for mount in chassis_mount_points:
            self.chassis_mounts.append(mount)
            self.add_constraint(
                FixedPointConstraint(
                    mount,
                    mount.position.copy(),
                    name=f"{control_arm.name}_chassis_{mount.name}",
                    joint_type=JointType.RIGID
                )
            )
            self.set_point_fixed(mount.name)

        # Mark knuckle mount points
        for mount in knuckle_mount_points:
            self.knuckle_points.append(mount)
            self.set_point_free(mount.name)

        # Mark all other attachment points as free to move
        for attachment in attachments:
            # Check by identity to handle attachment point objects
            is_chassis = any(attachment is m for m in chassis_mount_points)
            is_knuckle = any(attachment is m for m in knuckle_mount_points)
            if not is_chassis and not is_knuckle:
                self.set_point_free(attachment.name)

        # Add distance constraints for control arm links (always rigid)
        for link in control_arm.links:
            self.add_constraint(
                DistanceConstraint(
                    link.endpoint1,
                    link.endpoint2,
                    target_distance=link.length,
                    name=f"{control_arm.name}_{link.name}",
                    joint_type=JointType.RIGID  # Links are rigid
                )
            )

        # Add coincident constraints for all registered joints involving this control arm
        self._generate_joint_constraints_for_component(control_arm, attachments)

    def add_link(self,
                link: SuspensionLink,
                end1_mount_point: Optional[AttachmentPoint] = None,
                end2_mount_point: Optional[AttachmentPoint] = None):
        """
        Add a suspension link with automatic constraint generation.

        Joint types are automatically inferred from joints defined via add_joint().
        You must call add_joint() to define joints at the link endpoints before adding the link.

        Args:
            link: SuspensionLink object
            end1_mount_point: Chassis/knuckle point that end1 connects to (None if free-floating)
            end2_mount_point: Chassis/knuckle point that end2 connects to (None if free-floating)

        Examples:
            # Define joints first
            solver.add_joint("damper_lower", [damper.end1, knuckle.damper_mount], JointType.BALL_JOINT)
            solver.add_joint("damper_upper", [damper.end2, chassis.damper_mount], JointType.BUSHING_HARD)

            # Add link
            solver.add_link(
                link=damper,
                end1_mount_point=knuckle.damper_mount,
                end2_mount_point=chassis.damper_mount
            )
        """
        self.links.append(link)

        # Add link endpoints to solver state
        if link.endpoint1.name not in self.state.points:
            self.state.add_point(link.endpoint1)
        if link.endpoint2.name not in self.state.points:
            self.state.add_point(link.endpoint2)

        # Add distance constraint for link rigidity (links are always rigid)
        self.add_constraint(
            DistanceConstraint(
                link.endpoint1,
                link.endpoint2,
                target_distance=link.length,
                name=f"{link.name}_length",
                joint_type=JointType.RIGID
            )
        )

        # If end1 connects to a chassis mount, fix it
        if end1_mount_point is not None and end1_mount_point in self.chassis_mounts:
            self.add_constraint(
                FixedPointConstraint(
                    link.endpoint1,
                    link.endpoint1.position.copy(),
                    name=f"{link.name}_end1_chassis",
                    joint_type=JointType.RIGID
                )
            )
            self.set_point_fixed(link.endpoint1.name)
        else:
            self.set_point_free(link.endpoint1.name)

        # If end2 connects to a chassis mount, fix it
        if end2_mount_point is not None and end2_mount_point in self.chassis_mounts:
            self.add_constraint(
                FixedPointConstraint(
                    link.endpoint2,
                    link.endpoint2.position.copy(),
                    name=f"{link.name}_end2_chassis",
                    joint_type=JointType.RIGID
                )
            )
            self.set_point_fixed(link.endpoint2.name)
        else:
            self.set_point_free(link.endpoint2.name)

        # Add coincident constraints for registered joints involving this link
        link_points = [link.endpoint1, link.endpoint2]
        self._generate_joint_constraints_for_component(link, link_points)

    def add_ball_joint_coincident(self,
                                  point1: AttachmentPoint,
                                  point2: AttachmentPoint,
                                  joint_type: JointType = JointType.BALL_JOINT):
        """
        Add a coincident constraint for a ball joint.

        Use this to connect points that should move together
        (e.g., upper and lower control arms meeting at knuckle).

        Args:
            point1: First attachment point
            point2: Second attachment point
            joint_type: Type of joint (default: BALL_JOINT)
        """
        self.add_constraint(
            CoincidentPointConstraint(
                point1,
                point2,
                joint_type=joint_type,
                name=f"ball_joint_{point1.name}_{point2.name}"
            )
        )

    def set_wheel_center(self, point: AttachmentPoint):
        """
        Designate a point as the wheel center for geometry calculations.

        Args:
            point: AttachmentPoint representing wheel center
        """
        self.wheel_center = point

    def solve_for_heave(self,
                       displacement: float,
                       unit: str = 'mm',
                       initial_guess: Optional[np.ndarray] = None) -> SolverResult:
        """
        Solve for suspension position at a given heave displacement.

        Args:
            displacement: Vertical (Z) displacement from current position
            unit: Unit of displacement (default: 'mm')
            initial_guess: Initial guess for solver (uses current state if None)

        Returns:
            SolverResult with solved positions
        """
        if self.wheel_center is None:
            raise ValueError("Wheel center not set. Call set_wheel_center() first.")

        # Save initial state
        if self._initial_snapshot is None:
            self._initial_snapshot = self.get_state_snapshot()

        # Create constraint for Z position
        target_z = self.wheel_center.position[2] + to_mm(displacement, unit)
        target_position = self.wheel_center.position.copy()
        target_position[2] = target_z

        # Add temporary constraint for wheel Z position
        heave_constraint = PartialPositionConstraint(
            self.wheel_center,
            target_position,
            constrain_axes=['z'],
            name="heave_constraint"
        )
        self.add_constraint(heave_constraint)

        try:
            # Solve
            result = self.solve(initial_guess=initial_guess)

            # Update copied knuckle tire_center if it exists
            if self.copied_knuckle is not None and self.wheel_center is not None:
                self.copied_knuckle.tire_center = self.wheel_center.position.copy()

            return result
        finally:
            # Remove temporary constraint
            self.constraints.remove(heave_constraint)

    def solve_for_wheel_position(self,
                                position: np.ndarray,
                                unit: str = 'mm',
                                constrain_axes: Optional[List[str]] = None,
                                initial_guess: Optional[np.ndarray] = None) -> SolverResult:
        """
        Solve for suspension position with wheel at a specific location.

        Args:
            position: Target wheel center position [x, y, z]
            unit: Unit of position (default: 'mm')
            constrain_axes: Which axes to constrain (default: ['x', 'y', 'z'])
            initial_guess: Initial guess for solver

        Returns:
            SolverResult with solved positions
        """
        if self.wheel_center is None:
            raise ValueError("Wheel center not set. Call set_wheel_center() first.")

        if constrain_axes is None:
            constrain_axes = ['x', 'y', 'z']

        target_position = to_mm(np.array(position, dtype=float), unit)

        # Add temporary constraint
        wheel_constraint = PartialPositionConstraint(
            self.wheel_center,
            target_position,
            constrain_axes=constrain_axes,
            name="wheel_position_constraint"
        )
        self.add_constraint(wheel_constraint)

        try:
            result = self.solve(initial_guess=initial_guess)
            return result
        finally:
            self.constraints.remove(wheel_constraint)

    def get_camber(self,
                  knuckle_top: AttachmentPoint,
                  knuckle_bottom: AttachmentPoint,
                  unit: str = 'deg') -> float:
        """
        Calculate camber angle from knuckle orientation.

        Camber is the angle of the wheel from vertical in the Y-Z plane.
        Positive camber = top of wheel tilted outward.

        Args:
            knuckle_top: Upper knuckle attachment point
            knuckle_bottom: Lower knuckle attachment point
            unit: Unit for output angle ('deg' or 'rad')

        Returns:
            Camber angle in specified unit
        """
        # Vector from bottom to top
        knuckle_axis = knuckle_top.position - knuckle_bottom.position

        # Project onto Y-Z plane
        y_component = knuckle_axis[1]
        z_component = knuckle_axis[2]

        # Camber angle (from vertical)
        camber_rad = np.arctan2(y_component, z_component)

        if unit == 'deg':
            return np.degrees(camber_rad)
        elif unit == 'rad':
            return camber_rad
        else:
            raise ValueError(f"Unknown angle unit '{unit}'. Use 'deg' or 'rad'")

    def get_toe(self,
               knuckle_front: AttachmentPoint,
               knuckle_rear: AttachmentPoint,
               unit: str = 'deg') -> float:
        """
        Calculate toe angle from knuckle orientation.

        Toe is the angle of the wheel from straight ahead in the X-Y plane.
        Positive toe = front of wheel turned inward (toe-in).

        Args:
            knuckle_front: Forward-most knuckle attachment point
            knuckle_rear: Rearward-most knuckle attachment point
            unit: Unit for output angle ('deg' or 'rad')

        Returns:
            Toe angle in specified unit
        """
        # Vector from rear to front
        wheel_direction = knuckle_front.position - knuckle_rear.position

        # Project onto X-Y plane
        x_component = wheel_direction[0]
        y_component = wheel_direction[1]

        # Toe angle (from longitudinal axis - X direction)
        toe_rad = np.arctan2(y_component, x_component)

        if unit == 'deg':
            return np.degrees(toe_rad)
        elif unit == 'rad':
            return toe_rad
        else:
            raise ValueError(f"Unknown angle unit '{unit}'. Use 'deg' or 'rad'")

    def get_caster(self,
                  steering_axis_top: AttachmentPoint,
                  steering_axis_bottom: AttachmentPoint,
                  unit: str = 'deg') -> float:
        """
        Calculate caster angle from steering axis orientation.

        Caster is the angle of the steering axis from vertical in the X-Z plane.
        Positive caster = top of axis tilted rearward.

        Args:
            steering_axis_top: Upper steering axis point
            steering_axis_bottom: Lower steering axis point
            unit: Unit for output angle ('deg' or 'rad')

        Returns:
            Caster angle in specified unit
        """
        # Vector from bottom to top
        steering_axis = steering_axis_top.position - steering_axis_bottom.position

        # Project onto X-Z plane
        x_component = steering_axis[0]
        z_component = steering_axis[2]

        # Caster angle (from vertical)
        caster_rad = np.arctan2(-x_component, z_component)

        if unit == 'deg':
            return np.degrees(caster_rad)
        elif unit == 'rad':
            return caster_rad
        else:
            raise ValueError(f"Unknown angle unit '{unit}'. Use 'deg' or 'rad'")

    def calculate_instant_centers(self,
                                  knuckle: 'SuspensionKnuckle',
                                  z_offsets: Optional[List[float]] = None,
                                  unit: str = 'mm',
                                  initial_guess: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Calculate roll and pitch instant centers for the suspension corner.

        This method uses constraint-based kinematics to simulate vertical motion
        (heave) of the suspension through the control arms and linkage. It solves
        for the wheel center position at each displacement, updates the knuckle
        position, and uses the knuckle's tire geometry to calculate the proper
        contact patch position (accounting for camber). The instant centers are
        found by fitting circles to the projected contact point trajectories.

        The method:
        1. Verifies the suspension is properly configured with wheel_center set
        2. Saves the initial suspension and knuckle states
        3. For each z-offset, solves suspension kinematics using constraints
        4. Updates knuckle position to match solved wheel center
        5. Calculates contact patch using knuckle's tire geometry
        6. Restores initial suspension and knuckle states
        7. Projects contact points onto YZ plane (for roll center)
        8. Projects contact points onto XZ plane (for pitch center)
        9. Fits circles to the projected points
        10. Returns the circle centers as instant centers

        Prerequisites:
            - Suspension linkage must be configured (control arms, links added)
            - Wheel center must be set via set_wheel_center()
            - Chassis mount points must be properly constrained
            - SuspensionKnuckle must be provided with tire geometry

        Args:
            knuckle: SuspensionKnuckle object with tire geometry (rolling radius, camber, etc.)
            z_offsets: List of z-axis displacements from current position (heave)
                      Default: [0, 5, 10, -5, -10] mm
            unit: Unit for z_offsets and output positions (default: 'mm')
            initial_guess: Initial guess for solver (uses current state if None)

        Returns:
            Dictionary containing:
                - 'roll_center': [x, y, z] - Roll instant center position
                - 'pitch_center': [x, y, z] - Pitch instant center position
                - 'roll_radius': float - Fitted circle radius for roll motion
                - 'pitch_radius': float - Fitted circle radius for pitch motion
                - 'contact_points': Nx3 array - Captured tire contact positions
                - 'wheel_centers': Nx3 array - Wheel center positions at each offset
                - 'roll_fit_quality': float - Normalized RMS residual for roll fit
                - 'pitch_fit_quality': float - Normalized RMS residual for pitch fit
                - 'roll_residuals': float - Absolute RMS residual for roll fit
                - 'pitch_residuals': float - Absolute RMS residual for pitch fit
                - 'solve_errors': List[float] - RMS error from solver at each offset

        Raises:
            ValueError: If wheel_center not set, insufficient z_offsets, or invalid knuckle

        Example:
            >>> from pysuspension import CornerSolver, SuspensionLink, AttachmentPoint, SuspensionKnuckle
            >>> # Set up suspension geometry
            >>> solver = CornerSolver("front_left")
            >>> # ... add control arms, links, etc ...
            >>> wheel_center = AttachmentPoint("wheel_center", [1400, 750, 390], unit='mm')
            >>> solver.set_wheel_center(wheel_center)
            >>> # Create knuckle with tire geometry
            >>> knuckle = SuspensionKnuckle(tire_center_x=1400, tire_center_y=750,
            ...                             rolling_radius=390, camber_angle=-1.0, unit='mm')
            >>> # Calculate instant centers
            >>> result = solver.calculate_instant_centers(knuckle)
            >>> print(f"Roll center: {result['roll_center']} mm")
            >>> print(f"Pitch center: {result['pitch_center']} mm")
        """
        # Import here to avoid circular dependency
        from .suspension_knuckle import SuspensionKnuckle

        # Validate prerequisites
        if self.wheel_center is None:
            raise ValueError("Wheel center not set. Call set_wheel_center() before calculating instant centers.")

        if not isinstance(knuckle, SuspensionKnuckle):
            raise ValueError("knuckle must be a SuspensionKnuckle object")

        if z_offsets is None:
            z_offsets = [0, 5, 10, -5, -10]  # Default in mm

        if len(z_offsets) < 3:
            raise ValueError(f"Need at least 3 z_offsets for circle fitting, got {len(z_offsets)}")

        # Convert z_offsets to mm for internal calculations
        z_offsets_mm = [to_mm(z, unit) for z in z_offsets]

        # Save initial suspension state
        if self._initial_snapshot is None:
            self.save_initial_state()
        initial_snapshot = self.get_state_snapshot()

        # Save initial knuckle state
        knuckle.save_state()
        initial_tire_center = knuckle.tire_center.copy()

        # Collect wheel center and contact patch positions at each z offset
        wheel_centers = []
        contact_points = []
        solve_errors = []

        try:
            for z_offset in z_offsets_mm:
                # Reset to initial state
                self.restore_state_snapshot(initial_snapshot)
                knuckle.tire_center = initial_tire_center.copy()

                # Solve for this heave position using constraint-based kinematics
                result = self.solve_for_heave(z_offset, unit='mm', initial_guess=initial_guess)

                # Check if solve was successful
                rms_error = result.get_rms_error()
                solve_errors.append(rms_error)

                # Get wheel center position from solved state
                wheel_center_pos = self.wheel_center.position.copy()  # Already in mm
                wheel_centers.append(wheel_center_pos)

                # Update knuckle position to match solved wheel center
                knuckle.tire_center = wheel_center_pos.copy()

                # Calculate tire contact patch using knuckle's tire geometry
                # This accounts for camber, rolling radius, and tire axis orientation
                contact_point = knuckle.get_tire_contact_patch(unit='mm')
                contact_points.append(contact_point)

        finally:
            # Always restore original state
            self.restore_state_snapshot(initial_snapshot)
            knuckle.reset_to_origin()

        # Convert to numpy arrays for analysis
        contact_points = np.array(contact_points)
        wheel_centers = np.array(wheel_centers)

        # Calculate roll instant center (YZ plane projection)
        roll_result = calculate_instant_center_from_points(contact_points, 'YZ')

        # Calculate pitch instant center (XZ plane projection)
        pitch_result = calculate_instant_center_from_points(contact_points, 'XZ')

        # Convert results to requested unit
        roll_center = from_mm(roll_result['center_3d'], unit)
        pitch_center = from_mm(pitch_result['center_3d'], unit)
        roll_radius = from_mm(roll_result['radius'], unit)
        pitch_radius = from_mm(pitch_result['radius'], unit)
        roll_residuals = from_mm(roll_result['residuals'], unit)
        pitch_residuals = from_mm(pitch_result['residuals'], unit)
        contact_points_output = from_mm(contact_points, unit)
        wheel_centers_output = from_mm(wheel_centers, unit)
        solve_errors_output = [from_mm(err, unit) for err in solve_errors]

        return {
            'roll_center': roll_center,
            'pitch_center': pitch_center,
            'roll_radius': roll_radius,
            'pitch_radius': pitch_radius,
            'contact_points': contact_points_output,
            'wheel_centers': wheel_centers_output,
            'roll_fit_quality': roll_result['fit_quality'],
            'pitch_fit_quality': pitch_result['fit_quality'],
            'roll_residuals': roll_residuals,
            'pitch_residuals': pitch_residuals,
            'solve_errors': solve_errors_output
        }

    def reset_to_initial_state(self):
        """Reset suspension to initial configuration."""
        if self._initial_snapshot is not None:
            self.restore_state_snapshot(self._initial_snapshot)

    def save_initial_state(self):
        """Save current state as initial configuration."""
        self._initial_snapshot = self.get_state_snapshot()

    def solve_for_knuckle_heave(self,
                               displacement: float,
                               unit: str = 'mm',
                               initial_guess: Optional[np.ndarray] = None) -> SolverResult:
        """
        Solve for suspension position with knuckle heaved vertically.

        This is similar to solve_for_heave() but specifically works with the
        knuckle's tire center position when the solver was created from a
        SuspensionKnuckle using from_suspension_knuckle().

        Args:
            displacement: Vertical displacement from current knuckle position
            unit: Unit of displacement (default: 'mm')
            initial_guess: Initial guess for solver

        Returns:
            SolverResult with new positions in copied components

        Raises:
            ValueError: If solver was not created from a knuckle

        Example:
            >>> solver = CornerSolver.from_suspension_knuckle(knuckle)
            >>> result = solver.solve_for_knuckle_heave(25, unit='mm')
            >>> solved_knuckle = solver.get_solved_knuckle()
        """
        if self.copied_knuckle is None:
            raise ValueError(
                "solve_for_knuckle_heave() requires solver to be created from a knuckle. "
                "Use from_suspension_knuckle() or use solve_for_heave() instead."
            )

        # Use the standard solve_for_heave with the wheel center
        # Note: solve_for_heave will update wheel_center and copied_knuckle.tire_center
        result = self.solve_for_heave(displacement, unit=unit, initial_guess=initial_guess)

        return result

    def get_solved_knuckle(self) -> 'SuspensionKnuckle':
        """
        Get the solved knuckle with updated position/orientation.

        Returns the copied knuckle (if solver was created with copy_components=True)
        with positions updated from solving. The original knuckle remains unchanged.

        Returns:
            SuspensionKnuckle with solved positions

        Raises:
            ValueError: If solver was not created from a knuckle

        Example:
            >>> solver = CornerSolver.from_suspension_knuckle(knuckle)
            >>> result = solver.solve_for_heave(25, unit='mm')
            >>> solved_knuckle = solver.get_solved_knuckle()
            >>> print(f"New tire center: {solved_knuckle.tire_center}")
        """
        if self.copied_knuckle is None:
            raise ValueError(
                "get_solved_knuckle() requires solver to be created from a knuckle. "
                "Use from_suspension_knuckle() first."
            )

        return self.copied_knuckle

    def update_original_from_solved(self) -> None:
        """
        Update original components with solved positions from copies.

        WARNING: This modifies the original suspension model!
        Use with caution. This is only available if the solver was created
        with copy_components=True.

        Updates:
        - Original knuckle position and orientation
        - Original control arm positions
        - Original link endpoint positions

        Raises:
            ValueError: If solver was not created from a knuckle or no copies exist

        Example:
            >>> solver = CornerSolver.from_suspension_knuckle(knuckle, copy_components=True)
            >>> result = solver.solve_for_heave(25, unit='mm')
            >>> # Original still unchanged
            >>> solver.update_original_from_solved()
            >>> # Now original knuckle has new positions
        """
        if self.original_knuckle is None or self.copied_knuckle is None:
            raise ValueError(
                "update_original_from_solved() requires solver to be created from a knuckle. "
                "Use from_suspension_knuckle() first."
            )

        if not self.component_mapping:
            raise ValueError(
                "No component mapping available. "
                "Solver must be created with copy_components=True."
            )

        # Update original knuckle from solved state
        # Get the solved wheel center position and update the knuckle's tire_center
        if self.wheel_center is not None:
            self.original_knuckle.tire_center = self.wheel_center.position.copy()
            self.copied_knuckle.tire_center = self.wheel_center.position.copy()

        self.original_knuckle.toe_angle = self.copied_knuckle.toe_angle
        self.original_knuckle.camber_angle = self.copied_knuckle.camber_angle
        self.original_knuckle.rotation_matrix = self.copied_knuckle.rotation_matrix.copy()

        # Update attachment point positions
        for orig_ap, copy_ap in zip(self.original_knuckle.attachment_points,
                                    self.copied_knuckle.attachment_points):
            orig_ap.set_position(copy_ap.position.copy(), unit='mm')

        # Update control arms
        for control_arm_copy in self.control_arms:
            # Find original control arm using reverse mapping
            orig_id = self.reverse_mapping.get(id(control_arm_copy))
            if orig_id is not None:
                # Find original in component_mapping
                orig_arm = self.component_mapping.get(orig_id)
                if orig_arm is not None:
                    # Update link endpoints
                    for orig_link, copy_link in zip(orig_arm.links, control_arm_copy.links):
                        orig_link.endpoint1.set_position(copy_link.endpoint1.position.copy(), unit='mm')
                        orig_link.endpoint2.set_position(copy_link.endpoint2.position.copy(), unit='mm')
                        orig_link._update_local_frame()

        # Update standalone links
        for link_copy in self.links:
            orig_id = self.reverse_mapping.get(id(link_copy))
            if orig_id is not None:
                orig_link = self.component_mapping.get(orig_id)
                if orig_link is not None:
                    orig_link.endpoint1.set_position(link_copy.endpoint1.position.copy(), unit='mm')
                    orig_link.endpoint2.set_position(link_copy.endpoint2.position.copy(), unit='mm')
                    orig_link._update_local_frame()

    def __repr__(self) -> str:
        return (f"CornerSolver('{self.name}', "
                f"control_arms={len(self.control_arms)}, "
                f"links={len(self.links)}, "
                f"chassis_mounts={len(self.chassis_mounts)}, "
                f"constraints={len(self.constraints)}, "
                f"dof={self.state.get_dof()})")


if __name__ == "__main__":
    print("=" * 70)
    print("CORNER SOLVER TEST - Simple Double Wishbone")
    print("=" * 70)

    # Create a simple double wishbone suspension
    # Coordinate system: X = forward, Y = outboard, Z = up

    print("\n--- Creating Suspension Components ---")

    # Upper control arm (shorter, at top)
    upper_arm = ControlArm("upper_control_arm")
    upper_front_link = SuspensionLink(
        endpoint1=[1400, 0, 600],     # Chassis mount (front)
        endpoint2=[1400, 650, 580],   # Ball joint
        name="upper_front",
        unit='mm'
    )
    upper_rear_link = SuspensionLink(
        endpoint1=[1200, 0, 600],     # Chassis mount (rear)
        endpoint2=[1400, 650, 580],   # Ball joint (shared)
        name="upper_rear",
        unit='mm'
    )
    upper_arm.add_link(upper_front_link)
    upper_arm.add_link(upper_rear_link)

    # Lower control arm (longer, at bottom)
    lower_arm = ControlArm("lower_control_arm")
    lower_front_link = SuspensionLink(
        endpoint1=[1500, 0, 300],     # Chassis mount (front)
        endpoint2=[1400, 700, 200],   # Ball joint
        name="lower_front",
        unit='mm'
    )
    lower_rear_link = SuspensionLink(
        endpoint1=[1100, 0, 300],     # Chassis mount (rear)
        endpoint2=[1400, 700, 200],   # Ball joint (shared)
        name="lower_rear",
        unit='mm'
    )
    lower_arm.add_link(lower_front_link)
    lower_arm.add_link(lower_rear_link)

    # Wheel center (on knuckle, between ball joints)
    wheel_center = AttachmentPoint("wheel_center", [1400, 750, 390], unit='mm')

    print(f"Upper arm: {len(upper_arm.links)} links")
    print(f"Lower arm: {len(lower_arm.links)} links")
    print(f"Wheel center: {wheel_center.position} mm")

    print("\n--- Building CornerSolver ---")
    solver = CornerSolver("double_wishbone")

    # Add upper control arm
    # Endpoints 0 and 2 are chassis mounts, endpoint 1 is ball joint
    upper_attachments = upper_arm.get_all_attachment_positions()
    solver.add_link(upper_front_link, end1_is_chassis=True, end2_is_chassis=False)
    solver.add_link(upper_rear_link, end1_is_chassis=True, end2_is_chassis=False)

    # Add lower control arm
    solver.add_link(lower_front_link, end1_is_chassis=True, end2_is_chassis=False)
    solver.add_link(lower_rear_link, end1_is_chassis=True, end2_is_chassis=False)

    # Ball joints connect control arms to knuckle
    # Upper and lower ball joints should be coincident with their respective endpoints
    upper_ball_joint = upper_front_link.endpoint2
    lower_ball_joint = lower_front_link.endpoint2

    # Connect wheel center to ball joints with rigid links
    # (This simulates the knuckle connecting the ball joints to the wheel)
    upper_to_wheel = SuspensionLink(
        upper_ball_joint,
        wheel_center,
        name="upper_knuckle_link",
        unit='mm'
    )
    lower_to_wheel = SuspensionLink(
        lower_ball_joint,
        wheel_center,
        name="lower_knuckle_link",
        unit='mm'
    )
    solver.add_link(upper_to_wheel, end1_is_chassis=False, end2_is_chassis=False)
    solver.add_link(lower_to_wheel, end1_is_chassis=False, end2_is_chassis=False)

    # Set wheel center for heave calculations
    solver.set_wheel_center(wheel_center)

    print(solver)
    solver.save_initial_state()

    print("\n--- Solving Initial Configuration ---")
    result = solver.solve()
    print(result)
    print(f"RMS error: {result.get_rms_error():.6f} mm")

    print("\n--- Testing Heave Travel ---")
    heave_values = [-50, -25, 0, 25, 50]  # mm

    print(f"\n{'Heave (mm)':<12} {'Wheel Z (mm)':<15} {'Camber (deg)':<15} {'RMS Error (mm)':<15}")
    print("-" * 60)

    for heave in heave_values:
        # Reset to initial
        solver.reset_to_initial_state()

        # Solve for this heave position
        result = solver.solve_for_heave(heave, unit='mm')

        wheel_z = result.get_position("wheel_center")[2]
        camber = solver.get_camber(upper_ball_joint, lower_ball_joint, unit='deg')
        rms_error = result.get_rms_error()

        print(f"{heave:<12.1f} {wheel_z:<15.3f} {camber:<15.3f} {rms_error:<15.6f}")

    print("\n--- Final Positions at +50mm Heave ---")
    print(f"Upper ball joint: {result.get_position('upper_front_endpoint2')}")
    print(f"Lower ball joint: {result.get_position('lower_front_endpoint2')}")
    print(f"Wheel center: {result.get_position('wheel_center')}")

    print("\n✓ CornerSolver test completed successfully!")
