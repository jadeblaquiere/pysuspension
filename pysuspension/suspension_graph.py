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
        control_arms: List of discovered ControlArm components
        links: List of discovered SuspensionLink components
        joints: Dict of discovered SuspensionJoint objects (name -> joint)
        chassis_points: List of ChassisCorner attachment points (to be fixed)
        knuckle_points: List of attachment points on the knuckle
        all_attachment_points: Set of all discovered attachment points
        component_map: Dict mapping attachment points to their parent components
    """
    knuckle: 'SuspensionKnuckle'
    control_arms: List['ControlArm'] = field(default_factory=list)
    links: List['SuspensionLink'] = field(default_factory=list)
    joints: Dict[str, 'SuspensionJoint'] = field(default_factory=dict)
    chassis_points: List['AttachmentPoint'] = field(default_factory=list)
    knuckle_points: List['AttachmentPoint'] = field(default_factory=list)
    all_attachment_points: Set['AttachmentPoint'] = field(default_factory=set)
    component_map: Dict['AttachmentPoint', object] = field(default_factory=dict)


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

    if not knuckle.attachment_points:
        raise ValueError(f"Knuckle '{knuckle.name}' has no attachment points")

    graph = SuspensionGraph(knuckle=knuckle)

    # Track components we've already processed to avoid duplicates
    visited_points: Set['AttachmentPoint'] = set()
    visited_components: Set[object] = set()

    # Queue for breadth-first traversal (point, parent_component)
    queue: List['AttachmentPoint'] = []

    # Start with knuckle attachment points
    for point in knuckle.attachment_points:
        graph.knuckle_points.append(point)
        graph.all_attachment_points.add(point)
        graph.component_map[point] = knuckle
        queue.append(point)
        visited_points.add(point)

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
                if connected_point in visited_points:
                    continue

                visited_points.add(connected_point)
                graph.all_attachment_points.add(connected_point)

                # Get the parent component of this connected point
                parent = connected_point.parent_component

                if parent is None:
                    # Point has no parent - just track it
                    continue

                # Map this point to its component
                graph.component_map[connected_point] = parent

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
                        for arm_point in parent.get_all_attachment_points(unit='mm'):
                            # Find the actual AttachmentPoint object
                            # (get_all_attachment_points returns positions, we need objects)
                            # Need to get attachment points from links
                            for link in parent.links:
                                if link.endpoint1 not in visited_points:
                                    queue.append(link.endpoint1)
                                    graph.all_attachment_points.add(link.endpoint1)
                                    graph.component_map[link.endpoint1] = parent
                                    visited_points.add(link.endpoint1)
                                if link.endpoint2 not in visited_points:
                                    queue.append(link.endpoint2)
                                    graph.all_attachment_points.add(link.endpoint2)
                                    graph.component_map[link.endpoint2] = parent
                                    visited_points.add(link.endpoint2)

                            # Also check additional attachment points
                            for ap in parent.attachment_points:
                                if ap not in visited_points:
                                    queue.append(ap)
                                    graph.all_attachment_points.add(ap)
                                    graph.component_map[ap] = parent
                                    visited_points.add(ap)

                elif isinstance(parent, SuspensionLink):
                    # Found a standalone suspension link
                    component_id = id(parent)
                    if component_id not in visited_components:
                        graph.links.append(parent)
                        visited_components.add(component_id)

                        # Add both endpoints to queue
                        if parent.endpoint1 not in visited_points:
                            queue.append(parent.endpoint1)
                            graph.all_attachment_points.add(parent.endpoint1)
                            graph.component_map[parent.endpoint1] = parent
                            visited_points.add(parent.endpoint1)
                        if parent.endpoint2 not in visited_points:
                            queue.append(parent.endpoint2)
                            graph.all_attachment_points.add(parent.endpoint2)
                            graph.component_map[parent.endpoint2] = parent
                            visited_points.add(parent.endpoint2)

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
        if point.parent_component is None and point not in graph.knuckle_points:
            orphan_points.append(point.name)

    if orphan_points:
        warnings.append(f"Found {len(orphan_points)} attachment points without parent components: {orphan_points}")

    # Check that all non-chassis points have joints
    unjoint_points = []
    for point in graph.all_attachment_points:
        if point not in graph.chassis_points and point.joint is None:
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
        # Need to match original points to copied points
        # The copy() method creates new attachment points, need to map them

        # Map link endpoints
        for orig_link, copy_link in zip(control_arm.links, arm_copy.links):
            mapping[id(orig_link)] = copy_link
            mapping[id(orig_link.endpoint1)] = copy_link.endpoint1
            mapping[id(orig_link.endpoint2)] = copy_link.endpoint2

        # Map additional attachment points
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
    copied_graph = SuspensionGraph(
        knuckle=knuckle_copy,
        control_arms=control_arms_copy,
        links=links_copy,
        joints=joints_copy,
        chassis_points=chassis_points_copy,
        knuckle_points=[mapping[id(p)] for p in graph.knuckle_points if id(p) in mapping],
        all_attachment_points={mapping[id(p)] for p in graph.all_attachment_points if id(p) in mapping},
        component_map={mapping[id(p)]: mapping[id(c)] if id(c) in mapping else c
                      for p, c in graph.component_map.items() if id(p) in mapping}
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
