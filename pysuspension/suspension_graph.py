"""
Suspension graph discovery and component relationship mapping.

This module provides tools for discovering suspension component relationships
by traversing from a SuspensionKnuckle through joints and attachment points
to find all connected components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .suspension_knuckle import SuspensionKnuckle
    from .control_arm import ControlArm
    from .suspension_link import SuspensionLink
    from .coil_spring import CoilSpring
    from .steering_rack import SteeringRack
    from .suspension_joint import SuspensionJoint
    from .attachment_point import AttachmentPoint
    from .chassis_corner import ChassisCorner


@dataclass
class SuspensionGraph:
    """
    Represents the discovered suspension component graph.

    Contains all components and joints connected to a SuspensionKnuckle,
    organized by type for easy access.

    Attributes:
        knuckle: The starting SuspensionKnuckle
        control_arms: List of discovered ControlArm components (RigidBody with attachment points)
        links: List of discovered standalone SuspensionLink components (excluding CoilSpring)
        coil_springs: List of discovered CoilSpring components
        steering_racks: List of discovered SteeringRack components
        joints: Dict of discovered SuspensionJoint objects (name -> joint)
        chassis_points: List of ChassisCorner attachment points (to be fixed)
        knuckle_points: List of attachment points on the knuckle
        all_attachment_points: List of all discovered attachment points
        component_map: Dict mapping attachment point id to their parent components
    """
    knuckle: 'SuspensionKnuckle'
    control_arms: List['ControlArm'] = field(default_factory=list)
    links: List['SuspensionLink'] = field(default_factory=list)
    coil_springs: List['CoilSpring'] = field(default_factory=list)
    steering_racks: List['SteeringRack'] = field(default_factory=list)
    joints: Dict[str, 'SuspensionJoint'] = field(default_factory=dict)
    chassis_points: List['AttachmentPoint'] = field(default_factory=list)
    knuckle_points: List['AttachmentPoint'] = field(default_factory=list)
    all_attachment_points: List['AttachmentPoint'] = field(default_factory=list)
    component_map: Dict[int, object] = field(default_factory=dict)  # id(point) -> component


def discover_suspension_graph(knuckle: 'SuspensionKnuckle') -> SuspensionGraph:
    """
    Traverse from knuckle attachment points to discover all connected components.

    This function performs a breadth-first traversal starting from the knuckle's
    attachment points, following joint connections to discover all related
    suspension components (control arms, links) until reaching chassis attachment
    points.

    The traversal algorithm:
    1. Start with knuckle's attachment points
    2. For each point with a joint:
       - Get all connected points from the joint
       - Check each connected point's parent_component:
         * If ChassisCorner: add to chassis_points (stop here)
         * If ControlArm: add to control_arms, continue with its points
         * If SuspensionLink: add to links, continue with other endpoint
    3. Track visited points to avoid cycles
    4. Collect all discovered joints

    Args:
        knuckle: SuspensionKnuckle to start traversal from

    Returns:
        SuspensionGraph containing all discovered components and relationships

    Raises:
        ValueError: If knuckle has no attachment points

    Example:
        >>> knuckle = SuspensionKnuckle(...)
        >>> # ... set up suspension with control arms, links, joints ...
        >>> graph = discover_suspension_graph(knuckle)
        >>> print(f"Found {len(graph.control_arms)} control arms")
        >>> print(f"Found {len(graph.chassis_points)} chassis mount points")
    """
    from .chassis_corner import ChassisCorner
    from .control_arm import ControlArm
    from .suspension_link import SuspensionLink
    from .coil_spring import CoilSpring
    from .steering_rack import SteeringRack

    if not knuckle.attachment_points:
        raise ValueError(f"Knuckle '{knuckle.name}' has no attachment points")

    graph = SuspensionGraph(knuckle=knuckle)

    # Track components we've already processed to avoid duplicates
    visited_points: Set[int] = set()  # Use ids instead of objects
    visited_components: Set[int] = set()  # Use ids

    # Queue for breadth-first traversal (point, parent_component)
    queue: List['AttachmentPoint'] = []

    # Start with knuckle attachment points
    for point in knuckle.attachment_points:
        graph.knuckle_points.append(point)
        graph.all_attachment_points.append(point)
        graph.component_map[id(point)] = knuckle
        queue.append(point)
        visited_points.add(id(point))

    # Add knuckle to visited components
    visited_components.add(id(knuckle))

    # Breadth-first traversal
    while queue:
        current_point = queue.pop(0)

        # If this point has a joint, explore connected points
        if current_point.joint is not None:
            joint = current_point.joint

            # Add joint to graph if not already there
            if joint.name not in graph.joints:
                graph.joints[joint.name] = joint

            # Get all points connected through this joint
            connected_points = joint.get_all_attachment_points()

            for connected_point in connected_points:
                # Skip if we've already visited this point
                if id(connected_point) in visited_points:
                    continue

                visited_points.add(id(connected_point))
                graph.all_attachment_points.append(connected_point)

                # Get the parent component of this connected point
                parent = connected_point.parent_component

                if parent is None:
                    # Point has no parent - just track it
                    continue

                # Map this point to its component
                graph.component_map[id(connected_point)] = parent

                # Check parent component type and handle accordingly
                if isinstance(parent, ChassisCorner):
                    # Reached a chassis attachment point - this is a boundary
                    graph.chassis_points.append(connected_point)
                    # Don't continue traversal from chassis points

                elif isinstance(parent, ControlArm):
                    # Found a control arm
                    component_id = id(parent)
                    if component_id not in visited_components:
                        graph.control_arms.append(parent)
                        visited_components.add(component_id)

                        # Add all attachment points from this control arm to queue
                        for ap in parent.attachment_points:
                            if id(ap) not in visited_points:
                                queue.append(ap)
                                graph.all_attachment_points.append(ap)
                                graph.component_map[id(ap)] = parent
                                visited_points.add(id(ap))

                elif isinstance(parent, CoilSpring):
                    # Found a coil spring (must check before SuspensionLink since CoilSpring extends it)
                    component_id = id(parent)
                    if component_id not in visited_components:
                        graph.coil_springs.append(parent)
                        visited_components.add(component_id)

                        # Add both endpoints to queue
                        if id(parent.endpoint1) not in visited_points:
                            queue.append(parent.endpoint1)
                            graph.all_attachment_points.append(parent.endpoint1)
                            graph.component_map[id(parent.endpoint1)] = parent
                            visited_points.add(id(parent.endpoint1))
                        if id(parent.endpoint2) not in visited_points:
                            queue.append(parent.endpoint2)
                            graph.all_attachment_points.append(parent.endpoint2)
                            graph.component_map[id(parent.endpoint2)] = parent
                            visited_points.add(id(parent.endpoint2))

                elif isinstance(parent, SuspensionLink):
                    # Found a standalone suspension link
                    component_id = id(parent)
                    if component_id not in visited_components:
                        graph.links.append(parent)
                        visited_components.add(component_id)

                        # Add both endpoints to queue
                        if id(parent.endpoint1) not in visited_points:
                            queue.append(parent.endpoint1)
                            graph.all_attachment_points.append(parent.endpoint1)
                            graph.component_map[id(parent.endpoint1)] = parent
                            visited_points.add(id(parent.endpoint1))
                        if id(parent.endpoint2) not in visited_points:
                            queue.append(parent.endpoint2)
                            graph.all_attachment_points.append(parent.endpoint2)
                            graph.component_map[id(parent.endpoint2)] = parent
                            visited_points.add(id(parent.endpoint2))

                elif isinstance(parent, SteeringRack):
                    # Found a steering rack
                    component_id = id(parent)
                    if component_id not in visited_components:
                        graph.steering_racks.append(parent)
                        visited_components.add(component_id)

                        # Add all attachment points (housing + inner pivots) to queue
                        # SteeringRack now inherits from RigidBody with all attachment points
                        # in self.attachment_points (housing mounts + inner pivots)
                        for ap in parent.attachment_points:
                            if id(ap) not in visited_points:
                                queue.append(ap)
                                graph.all_attachment_points.append(ap)
                                graph.component_map[id(ap)] = parent
                                visited_points.add(id(ap))

                else:
                    # Unknown component type - just track the point
                    # Could be a different rigid body type
                    pass

    return graph


def validate_suspension_graph(graph: SuspensionGraph) -> Dict[str, any]:
    """
    Validate that a suspension graph is properly configured.

    Checks for common issues:
    - Disconnected components
    - Missing joints
    - Points without parent components
    - Incomplete suspension (no chassis points found)

    Args:
        graph: SuspensionGraph to validate

    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'warnings': List[str],
            'errors': List[str],
            'stats': Dict with component counts
        }
    """
    warnings = []
    errors = []

    # Check that we found chassis points
    if not graph.chassis_points:
        errors.append("No chassis attachment points found - suspension not grounded")

    # Check that we found at least one control arm or link
    if not graph.control_arms and not graph.links:
        errors.append("No control arms or links found - invalid suspension")

    # Check that all points have parent components
    orphan_points = []
    for point in graph.all_attachment_points:
        # Use identity comparison to avoid numpy array comparison issues
        if point.parent_component is None and not any(point is kp for kp in graph.knuckle_points):
            orphan_points.append(point.name)

    if orphan_points:
        warnings.append(f"Found {len(orphan_points)} attachment points without parent components: {orphan_points}")

    # Check that all non-chassis points have joints
    unjoint_points = []
    for point in graph.all_attachment_points:
        # Use identity comparison to avoid numpy array comparison issues
        if not any(point is cp for cp in graph.chassis_points) and point.joint is None:
            # Some points might not have joints if they're internal to a component
            # This is just a warning, not an error
            unjoint_points.append(point.name)

    if unjoint_points:
        warnings.append(f"Found {len(unjoint_points)} non-chassis points without joints: {unjoint_points}")

    # Calculate statistics
    stats = {
        'control_arms': len(graph.control_arms),
        'links': len(graph.links),
        'joints': len(graph.joints),
        'chassis_points': len(graph.chassis_points),
        'knuckle_points': len(graph.knuckle_points),
        'total_points': len(graph.all_attachment_points)
    }

    return {
        'valid': len(errors) == 0,
        'warnings': warnings,
        'errors': errors,
        'stats': stats
    }


def create_working_copies(graph: SuspensionGraph) -> tuple[SuspensionGraph, Dict]:
    """
    Create deep copies of all components with maintained relationships.

    This function uses a two-pass approach:
    1. Pass 1: Copy all component objects and attachment points
    2. Pass 2: Reconstruct joint connections between copied points

    Args:
        graph: Original SuspensionGraph to copy

    Returns:
        Tuple of:
        - New SuspensionGraph with copied components
        - Mapping dict: {id(original): copied_object}

    Example:
        >>> graph = discover_suspension_graph(knuckle)
        >>> copied_graph, mapping = create_working_copies(graph)
        >>> # Original components unchanged
        >>> # copied_graph contains independent copies with same relationships
    """
    from .suspension_joint import SuspensionJoint

    # Mapping from original objects to copies
    mapping = {}

    # === PASS 1: Copy all components and attachment points ===

    # Copy the knuckle
    knuckle_copy = graph.knuckle.copy(copy_joints=False)
    mapping[id(graph.knuckle)] = knuckle_copy

    # Map knuckle attachment points
    for i, original_ap in enumerate(graph.knuckle.attachment_points):
        copied_ap = knuckle_copy.attachment_points[i]
        mapping[id(original_ap)] = copied_ap

    # Copy all control arms
    control_arms_copy = []
    for control_arm in graph.control_arms:
        arm_copy = control_arm.copy(copy_joints=False)
        control_arms_copy.append(arm_copy)
        mapping[id(control_arm)] = arm_copy

        # Map all attachment points in this control arm
        # The copy() method creates new attachment points, need to map them
        for orig_ap, copy_ap in zip(control_arm.attachment_points, arm_copy.attachment_points):
            mapping[id(orig_ap)] = copy_ap

    # Copy all standalone links
    links_copy = []
    for link in graph.links:
        link_copy = link.copy(copy_joints=False)
        links_copy.append(link_copy)
        mapping[id(link)] = link_copy
        mapping[id(link.endpoint1)] = link_copy.endpoint1
        mapping[id(link.endpoint2)] = link_copy.endpoint2

    # Copy all coil springs
    coil_springs_copy = []
    for spring in graph.coil_springs:
        spring_copy = spring.copy(copy_joints=False)
        coil_springs_copy.append(spring_copy)
        mapping[id(spring)] = spring_copy
        mapping[id(spring.endpoint1)] = spring_copy.endpoint1
        mapping[id(spring.endpoint2)] = spring_copy.endpoint2

    # Copy all steering racks
    steering_racks_copy = []
    for rack in graph.steering_racks:
        rack_copy = rack.copy(copy_joints=False)
        steering_racks_copy.append(rack_copy)
        mapping[id(rack)] = rack_copy

        # Map all attachment points (housing + inner pivots)
        # SteeringRack now inherits from RigidBody with all attachment points in self.attachment_points
        for orig_ap, copy_ap in zip(rack.attachment_points, rack_copy.attachment_points):
            mapping[id(orig_ap)] = copy_ap

    # Copy chassis attachment points (these are just references, not new objects)
    # Chassis points should remain as references to the original chassis
    # OR we can create simple copies
    chassis_points_copy = []
    for chassis_point in graph.chassis_points:
        # Create a simple copy of the chassis point
        # These will be fixed in the solver, so they don't need complex copying
        chassis_point_copy = chassis_point.copy(copy_joint=False, copy_parent=True)
        chassis_points_copy.append(chassis_point_copy)
        mapping[id(chassis_point)] = chassis_point_copy

    # === PASS 2: Reconstruct joint connections ===

    joints_copy = {}

    for joint_name, original_joint in graph.joints.items():
        # Create new joint with same name and type
        joint_copy = SuspensionJoint(
            name=original_joint.name,
            joint_type=original_joint.joint_type
        )

        # Copy custom stiffness if present
        if hasattr(original_joint, 'stiffness') and original_joint.stiffness is not None:
            joint_copy.stiffness = original_joint.stiffness

        # Reconnect all attachment points using the mapping
        for original_point in original_joint.attachment_points:
            # Find the copied point
            point_id = id(original_point)
            if point_id in mapping:
                copied_point = mapping[point_id]
                # Add the copied point to the copied joint
                joint_copy.add_attachment_point(copied_point)
            else:
                # This shouldn't happen if graph discovery worked correctly
                print(f"Warning: Could not find copied point for {original_point.name}")

        joints_copy[joint_name] = joint_copy

    # Build the copied graph
    # component_map uses id(point) -> component, so we need to map:
    # id(copied_point) -> copied_component
    copied_component_map = {}
    for point_id, component in graph.component_map.items():
        # point_id is already an id from original
        if point_id in mapping:
            copied_point = mapping[point_id]
            copied_point_id = id(copied_point)
            # Map the component
            component_id = id(component)
            if component_id in mapping:
                copied_component = mapping[component_id]
            else:
                copied_component = component  # Keep original if not copied
            copied_component_map[copied_point_id] = copied_component

    copied_graph = SuspensionGraph(
        knuckle=knuckle_copy,
        control_arms=control_arms_copy,
        links=links_copy,
        coil_springs=coil_springs_copy,
        steering_racks=steering_racks_copy,
        joints=joints_copy,
        chassis_points=chassis_points_copy,
        knuckle_points=[mapping[id(p)] for p in graph.knuckle_points if id(p) in mapping],
        all_attachment_points=[mapping[id(p)] for p in graph.all_attachment_points if id(p) in mapping],
        component_map=copied_component_map
    )

    return copied_graph, mapping


if __name__ == "__main__":
    print("=" * 70)
    print("SUSPENSION GRAPH DISCOVERY TEST")
    print("=" * 70)
    print("\nThis module provides graph discovery for suspension systems.")
    print("See tests/test_suspension_graph.py for usage examples.")
    print("\nKey functions:")
    print("  - discover_suspension_graph(knuckle)")
    print("  - create_working_copies(graph)")
    print("  - validate_suspension_graph(graph)")
