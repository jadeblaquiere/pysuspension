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
from typing import List, Dict, Optional, Tuple
from .solver import SuspensionSolver, SolverResult
from .constraints import (
    DistanceConstraint,
    FixedPointConstraint,
    CoincidentPointConstraint,
    PartialPositionConstraint,
    JointType
)
from .attachment_point import AttachmentPoint
from .control_arm import ControlArm
from .suspension_link import SuspensionLink
from .units import to_mm, from_mm


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

        # Component tracking
        self.control_arms: List[ControlArm] = []
        self.links: List[SuspensionLink] = []
        self.chassis_mounts: List[AttachmentPoint] = []

        # Reference points for geometry calculations
        self.wheel_center: Optional[AttachmentPoint] = None
        self.knuckle_points: List[AttachmentPoint] = []

        # Initial state for resetting
        self._initial_snapshot = None

    def add_control_arm(self,
                       control_arm: ControlArm,
                       chassis_mount_indices: List[int],
                       knuckle_mount_index: int,
                       joint_type: JointType = JointType.BALL_JOINT):
        """
        Add a control arm with automatic constraint generation.

        Args:
            control_arm: ControlArm object
            chassis_mount_indices: Indices of attachment points that mount to chassis
            knuckle_mount_index: Index of attachment point that mounts to knuckle
            joint_type: Type of joint at knuckle mount (default: BALL_JOINT)
        """
        self.control_arms.append(control_arm)

        attachments = control_arm.get_all_attachment_points()

        # Add all attachment points to state
        for i, attachment in enumerate(attachments):
            if i in chassis_mount_indices:
                # Chassis mount - fixed
                self.chassis_mounts.append(attachment)
                self.add_constraint(
                    FixedPointConstraint(
                        attachment,
                        attachment.position.copy(),
                        name=f"{control_arm.name}_chassis_mount_{i}"
                    )
                )
                self.set_point_fixed(attachment.name)
            elif i == knuckle_mount_index:
                # Knuckle mount - free to move
                self.knuckle_points.append(attachment)
                self.set_point_free(attachment.name)
            else:
                # Other attachment - free to move with control arm
                self.set_point_free(attachment.name)

        # Add distance constraints for control arm links
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

        # Add coincident constraints for points that move together
        # (e.g., multiple links meeting at ball joint)
        for i, attachment in enumerate(attachments):
            if i == knuckle_mount_index:
                # This is the ball joint - may need to connect to other control arms
                pass

    def add_link(self,
                link: SuspensionLink,
                end1_is_chassis: bool,
                end2_is_chassis: bool,
                joint_type: JointType = JointType.BALL_JOINT):
        """
        Add a suspension link with automatic constraint generation.

        Args:
            link: SuspensionLink object
            end1_is_chassis: True if endpoint1 is chassis-mounted
            end2_is_chassis: True if endpoint2 is chassis-mounted
            joint_type: Type of joint at non-chassis end
        """
        self.links.append(link)

        # Add distance constraint for link rigidity
        self.add_constraint(
            DistanceConstraint(
                link.endpoint1,
                link.endpoint2,
                target_distance=link.length,
                name=f"{link.name}_length",
                joint_type=JointType.RIGID
            )
        )

        # Fix chassis end(s)
        if end1_is_chassis:
            self.chassis_mounts.append(link.endpoint1)
            self.add_constraint(
                FixedPointConstraint(
                    link.endpoint1,
                    link.endpoint1.position.copy(),
                    name=f"{link.name}_end1_chassis"
                )
            )
            self.set_point_fixed(link.endpoint1.name)
        else:
            self.set_point_free(link.endpoint1.name)

        if end2_is_chassis:
            self.chassis_mounts.append(link.endpoint2)
            self.add_constraint(
                FixedPointConstraint(
                    link.endpoint2,
                    link.endpoint2.position.copy(),
                    name=f"{link.name}_end2_chassis"
                )
            )
            self.set_point_fixed(link.endpoint2.name)
        else:
            self.set_point_free(link.endpoint2.name)

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

    def reset_to_initial_state(self):
        """Reset suspension to initial configuration."""
        if self._initial_snapshot is not None:
            self.restore_state_snapshot(self._initial_snapshot)

    def save_initial_state(self):
        """Save current state as initial configuration."""
        self._initial_snapshot = self.get_state_snapshot()

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
