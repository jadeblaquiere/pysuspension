"""
Constraint classes for suspension geometry solving.

This module provides a constraint-based framework for solving suspension kinematics.
Constraints can represent geometric relationships (distances, coincident points) with
optional compliance modeling for realistic joint behavior.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List
from .attachment_point import AttachmentPoint
from .joint_types import JointType, JOINT_STIFFNESS


class Constraint(ABC):
    """
    Base class for all constraints with compliance modeling.

    Constraints define relationships between attachment points that must be
    satisfied (or minimized) during solving. Each constraint has an associated
    stiffness/compliance that affects how strictly it must be satisfied.

    In weighted least-squares solving, the weight is proportional to stiffness:
    - High stiffness → High weight → Must be satisfied closely
    - Low stiffness → Low weight → Can deviate more (compliant)

    The total error being minimized is:
        E = Σ weight_i × error_i²

    Which is equivalent to minimizing elastic energy in compliant joints.
    """

    def __init__(self,
                 name: str,
                 weight: float = 1.0,
                 joint_type: JointType = JointType.RIGID,
                 stiffness: Optional[float] = None,
                 compliance: Optional[float] = None):
        """
        Initialize a constraint.

        Args:
            name: Constraint identifier
            weight: Direct weight (if not using stiffness/compliance)
            joint_type: Predefined joint type
            stiffness: Joint stiffness in N/mm (overrides joint_type)
            compliance: Joint compliance in mm/N (inverse of stiffness)

        Priority order: stiffness > compliance > joint_type > weight
        """
        self.name = name
        self.joint_type = joint_type

        # Determine effective stiffness
        if stiffness is not None:
            self._stiffness = float(stiffness)
        elif compliance is not None:
            if compliance <= 0:
                raise ValueError(f"Compliance must be positive, got {compliance}")
            self._stiffness = 1.0 / compliance
        elif joint_type != JointType.CUSTOM:
            self._stiffness = JOINT_STIFFNESS[joint_type]
        else:
            # Use weight directly, interpret as normalized stiffness
            self._stiffness = float(weight)

        # Reference stiffness for normalization (1000 N/mm)
        # This keeps weights in a reasonable range for numerical stability
        self._stiffness_ref = 1e3

        # Compute normalized weight for least-squares solving
        self.weight = self._stiffness / self._stiffness_ref

    @property
    def stiffness(self) -> float:
        """Get joint stiffness in N/mm."""
        return self._stiffness

    @property
    def compliance(self) -> float:
        """Get joint compliance in mm/N (inverse of stiffness)."""
        return 1.0 / self._stiffness

    @abstractmethod
    def evaluate(self) -> float:
        """
        Evaluate constraint error (squared error).

        Returns:
            Squared error value (0 = perfectly satisfied)
        """
        pass

    @abstractmethod
    def get_involved_points(self) -> List[AttachmentPoint]:
        """
        Get all attachment points involved in this constraint.

        Returns:
            List of AttachmentPoint objects
        """
        pass

    def get_physical_error(self) -> float:
        """
        Get the physical error magnitude (in mm for distance/position errors).

        Returns:
            RMS error in physical units
        """
        return np.sqrt(self.evaluate())

    def get_weighted_error(self) -> float:
        """
        Get the weighted error used in optimization.

        Returns:
            weight × error²
        """
        return self.weight * self.evaluate()

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}('{self.name}', "
                f"stiffness={self.stiffness:.1f} N/mm, "
                f"weight={self.weight:.3f})")


class GeometricConstraint(Constraint):
    """
    Base class for geometric constraints (positions, distances, angles).

    Geometric constraints define spatial relationships between points
    that must be maintained during kinematic solving.
    """
    pass


class ForceConstraint(Constraint):
    """
    Base class for force equilibrium constraints (future implementation).

    Force constraints ensure that forces on components are in equilibrium.
    These will be used for static force-based solving.
    """
    pass


class DistanceConstraint(GeometricConstraint):
    """
    Constraint to maintain a fixed distance between two points.

    This is used for rigid suspension links, or flexible elements with
    specified axial stiffness (like soft links or cables).

    For rigid links: use joint_type=JointType.RIGID (default)
    For flexible links: specify custom stiffness
    """

    def __init__(self,
                 point1: AttachmentPoint,
                 point2: AttachmentPoint,
                 target_distance: float,
                 name: Optional[str] = None,
                 joint_type: JointType = JointType.RIGID,
                 stiffness: Optional[float] = None):
        """
        Initialize a distance constraint.

        Args:
            point1: First endpoint
            point2: Second endpoint
            target_distance: Desired distance in mm
            name: Constraint name (auto-generated if None)
            joint_type: RIGID for solid links, CUSTOM for flexible
            stiffness: For flexible links, axial stiffness in N/mm
        """
        if name is None:
            name = f"distance_{point1.name}_{point2.name}"

        super().__init__(name, joint_type=joint_type, stiffness=stiffness)
        self.point1 = point1
        self.point2 = point2
        self.target_distance = float(target_distance)

        if self.target_distance < 0:
            raise ValueError(f"Target distance must be non-negative, got {target_distance}")

    def evaluate(self) -> float:
        """
        Evaluate squared distance error.

        Returns:
            (current_distance - target_distance)²
        """
        current_distance = np.linalg.norm(
            self.point2.position - self.point1.position
        )
        error = current_distance - self.target_distance
        return error ** 2

    def get_current_distance(self) -> float:
        """Get current distance between points in mm."""
        return np.linalg.norm(self.point2.position - self.point1.position)

    def get_axial_force(self) -> float:
        """
        Compute axial force in link.

        Returns:
            Force in N (positive = tension, negative = compression)
        """
        current_distance = self.get_current_distance()
        extension = current_distance - self.target_distance
        return self.stiffness * extension

    def get_force_vector(self) -> np.ndarray:
        """
        Compute force vector on point2.

        Returns:
            3D force vector in N
        """
        direction = self.point2.position - self.point1.position
        current_distance = np.linalg.norm(direction)

        if current_distance < 1e-9:
            return np.zeros(3)

        unit_direction = direction / current_distance
        force_magnitude = self.get_axial_force()
        return force_magnitude * unit_direction

    def get_involved_points(self) -> List[AttachmentPoint]:
        """Return both endpoints."""
        return [self.point1, self.point2]

    def __repr__(self) -> str:
        return (f"DistanceConstraint('{self.name}', "
                f"target={self.target_distance:.1f} mm, "
                f"current={self.get_current_distance():.1f} mm, "
                f"stiffness={self.stiffness:.1f} N/mm)")


class FixedPointConstraint(GeometricConstraint):
    """
    Constraint to pin a point to fixed coordinates.

    This is used for chassis mount points or other fixed attachment locations.
    These are typically very stiff (effectively infinite stiffness).
    """

    def __init__(self,
                 point: AttachmentPoint,
                 target_position: np.ndarray,
                 name: Optional[str] = None,
                 joint_type: JointType = JointType.RIGID,
                 stiffness: Optional[float] = None):
        """
        Initialize a fixed point constraint.

        Args:
            point: Point to be fixed
            target_position: Target position in mm (3D array)
            name: Constraint name (auto-generated if None)
            joint_type: Usually RIGID for chassis mounts
            stiffness: Custom stiffness in N/mm
        """
        if name is None:
            name = f"fixed_{point.name}"

        super().__init__(name, joint_type=joint_type, stiffness=stiffness)
        self.point = point
        self.target_position = np.array(target_position, dtype=float)

        if self.target_position.shape != (3,):
            raise ValueError(f"Target position must be 3D, got shape {self.target_position.shape}")

    def evaluate(self) -> float:
        """
        Evaluate squared position error.

        Returns:
            Sum of squared position errors in all 3 dimensions
        """
        error = self.point.position - self.target_position
        return np.sum(error ** 2)

    def get_displacement(self) -> np.ndarray:
        """
        Get displacement vector from target position.

        Returns:
            3D displacement vector in mm
        """
        return self.point.position - self.target_position

    def get_force(self) -> np.ndarray:
        """
        Compute reaction force at fixed point.

        Returns:
            3D force vector in N
        """
        displacement = self.get_displacement()
        return self.stiffness * displacement

    def get_involved_points(self) -> List[AttachmentPoint]:
        """Return the fixed point."""
        return [self.point]

    def __repr__(self) -> str:
        displacement = np.linalg.norm(self.get_displacement())
        return (f"FixedPointConstraint('{self.name}', "
                f"target={self.target_position}, "
                f"displacement={displacement:.3f} mm, "
                f"stiffness={self.stiffness:.1f} N/mm)")


class CoincidentPointConstraint(GeometricConstraint):
    """
    Constraint requiring two points to be at the same location.

    This is used for:
    - Ball joints (very stiff)
    - Bushings (moderate compliance)
    - Rubber mounts (high compliance)

    The joint_type or stiffness parameter determines how strictly
    the coincidence must be maintained.
    """

    def __init__(self,
                 point1: AttachmentPoint,
                 point2: AttachmentPoint,
                 name: Optional[str] = None,
                 joint_type: JointType = JointType.BALL_JOINT,
                 stiffness: Optional[float] = None):
        """
        Initialize a coincident point constraint.

        Args:
            point1: First point
            point2: Second point (should coincide with first)
            name: Constraint name (auto-generated if None)
            joint_type: Type of joint (ball joint, bushing, etc.)
            stiffness: Custom stiffness in N/mm (overrides joint_type)
        """
        if name is None:
            name = f"coincident_{point1.name}_{point2.name}"

        super().__init__(name, joint_type=joint_type, stiffness=stiffness)
        self.point1 = point1
        self.point2 = point2

    def evaluate(self) -> float:
        """
        Evaluate squared distance between points.

        Returns:
            Sum of squared position errors in all 3 dimensions
        """
        error = self.point1.position - self.point2.position
        return np.sum(error ** 2)

    def get_separation(self) -> float:
        """
        Get distance between points in mm.

        Returns:
            Separation distance in mm
        """
        return np.linalg.norm(self.point2.position - self.point1.position)

    def get_force(self) -> np.ndarray:
        """
        Compute force vector exerted by this joint.
        Force = stiffness × displacement

        Returns:
            3D force vector in N (force on point2)
        """
        displacement = self.point2.position - self.point1.position
        return self.stiffness * displacement

    def get_involved_points(self) -> List[AttachmentPoint]:
        """Return both points."""
        return [self.point1, self.point2]

    def __repr__(self) -> str:
        separation = self.get_separation()
        return (f"CoincidentPointConstraint('{self.name}', "
                f"separation={separation:.3f} mm, "
                f"joint_type={self.joint_type.value}, "
                f"stiffness={self.stiffness:.1f} N/mm)")


class PartialPositionConstraint(GeometricConstraint):
    """
    Constraint that fixes one or more dimensions (axes) of a point position.

    This allows constraining specific axes while leaving others free to move.
    Very useful for suspension analysis:
    - Fix Z (vertical) to simulate specific wheel height
    - Fix X, Y to constrain lateral position
    - Any combination of axes

    Example use cases:
    - Solve suspension with wheel raised 25mm (fix Z, free X/Y)
    - Constrain motion to a plane (fix one axis)
    - Multi-axis positioning (fix any combination)
    """

    # Axis name to index mapping
    AXIS_MAP = {'x': 0, 'y': 1, 'z': 2}

    def __init__(self,
                 point: AttachmentPoint,
                 target_position: np.ndarray,
                 constrain_axes: List[str],
                 name: Optional[str] = None,
                 joint_type: JointType = JointType.RIGID,
                 stiffness: Optional[float] = None):
        """
        Initialize a partial position constraint.

        Args:
            point: Point to be constrained
            target_position: Target position in mm (3D array)
            constrain_axes: List of axes to constrain: ['x'], ['y'], ['z'],
                           ['x', 'y'], ['x', 'z'], ['y', 'z'], or ['x', 'y', 'z']
            name: Constraint name (auto-generated if None)
            joint_type: Usually RIGID for hard constraints
            stiffness: Custom stiffness in N/mm

        Examples:
            # Fix only vertical position (wheel height)
            PartialPositionConstraint(wheel_center, [0, 0, 350], ['z'])

            # Fix X and Y, allow Z to vary
            PartialPositionConstraint(point, [100, 200, 0], ['x', 'y'])

            # Fix all (equivalent to FixedPointConstraint)
            PartialPositionConstraint(point, [100, 200, 300], ['x', 'y', 'z'])
        """
        if name is None:
            axes_str = '_'.join(sorted(constrain_axes))
            name = f"partial_{point.name}_{axes_str}"

        super().__init__(name, joint_type=joint_type, stiffness=stiffness)
        self.point = point
        self.target_position = np.array(target_position, dtype=float)

        if self.target_position.shape != (3,):
            raise ValueError(f"Target position must be 3D, got shape {self.target_position.shape}")

        # Validate and store constrained axes
        self.constrain_axes = []
        for axis in constrain_axes:
            axis_lower = axis.lower()
            if axis_lower not in self.AXIS_MAP:
                raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'")
            self.constrain_axes.append(axis_lower)

        if not self.constrain_axes:
            raise ValueError("Must constrain at least one axis")

        # Remove duplicates and sort for consistency
        self.constrain_axes = sorted(list(set(self.constrain_axes)))

        # Create mask for constrained axes
        self.axis_mask = np.zeros(3, dtype=bool)
        for axis in self.constrain_axes:
            self.axis_mask[self.AXIS_MAP[axis]] = True

        # Create index array for constrained axes
        self.constrained_indices = [self.AXIS_MAP[axis] for axis in self.constrain_axes]

    def evaluate(self) -> float:
        """
        Evaluate squared position error for constrained axes only.

        Returns:
            Sum of squared position errors in constrained dimensions
        """
        error = self.point.position - self.target_position
        # Only include error from constrained axes
        constrained_error = error[self.axis_mask]
        return np.sum(constrained_error ** 2)

    def get_displacement(self) -> np.ndarray:
        """
        Get displacement vector from target position (all axes).

        Returns:
            3D displacement vector in mm
        """
        return self.point.position - self.target_position

    def get_constrained_displacement(self) -> np.ndarray:
        """
        Get displacement only for constrained axes.

        Returns:
            Array of displacements for constrained axes only
        """
        displacement = self.get_displacement()
        return displacement[self.axis_mask]

    def get_force(self) -> np.ndarray:
        """
        Compute reaction force at point (only on constrained axes).

        Returns:
            3D force vector in N (zero on unconstrained axes)
        """
        force = np.zeros(3)
        displacement = self.get_displacement()
        # Apply force only on constrained axes
        force[self.axis_mask] = self.stiffness * displacement[self.axis_mask]
        return force

    def get_involved_points(self) -> List[AttachmentPoint]:
        """Return the constrained point."""
        return [self.point]

    def is_axis_constrained(self, axis: str) -> bool:
        """
        Check if a specific axis is constrained.

        Args:
            axis: 'x', 'y', or 'z'

        Returns:
            True if axis is constrained
        """
        return axis.lower() in self.constrain_axes

    def get_free_axes(self) -> List[str]:
        """
        Get list of axes that are NOT constrained.

        Returns:
            List of free axis names
        """
        all_axes = ['x', 'y', 'z']
        return [axis for axis in all_axes if axis not in self.constrain_axes]

    def __repr__(self) -> str:
        displacement = self.get_constrained_displacement()
        axes_str = ', '.join(self.constrain_axes)
        rms_error = np.sqrt(np.mean(displacement ** 2)) if len(displacement) > 0 else 0.0
        return (f"PartialPositionConstraint('{self.name}', "
                f"axes=[{axes_str}], "
                f"target={self.target_position[self.axis_mask]}, "
                f"error={rms_error:.3f} mm, "
                f"stiffness={self.stiffness:.1f} N/mm)")


if __name__ == "__main__":
    print("=" * 70)
    print("CONSTRAINT FRAMEWORK TEST")
    print("=" * 70)

    # Create test attachment points
    point1 = AttachmentPoint("point1", [0, 0, 0], unit='mm')
    point2 = AttachmentPoint("point2", [100, 0, 0], unit='mm')
    point3 = AttachmentPoint("point3", [100, 1, 0], unit='mm')

    print("\n--- Testing DistanceConstraint (Rigid Link) ---")
    dist_constraint = DistanceConstraint(
        point1, point2,
        target_distance=100.0,
        joint_type=JointType.RIGID
    )
    print(dist_constraint)
    print(f"Error: {dist_constraint.evaluate():.6f} mm²")
    print(f"Physical error: {dist_constraint.get_physical_error():.6f} mm")
    print(f"Weight: {dist_constraint.weight:.1f}")

    print("\n--- Testing CoincidentPointConstraint (Ball Joint) ---")
    coincident_constraint = CoincidentPointConstraint(
        point2, point3,
        joint_type=JointType.BALL_JOINT
    )
    print(coincident_constraint)
    print(f"Error: {coincident_constraint.evaluate():.6f} mm²")
    print(f"Separation: {coincident_constraint.get_separation():.6f} mm")
    print(f"Force: {coincident_constraint.get_force()} N")

    print("\n--- Testing CoincidentPointConstraint (Soft Bushing) ---")
    bushing_constraint = CoincidentPointConstraint(
        point2, point3,
        joint_type=JointType.BUSHING_SOFT
    )
    print(bushing_constraint)
    print(f"Stiffness: {bushing_constraint.stiffness:.1f} N/mm")
    print(f"Compliance: {bushing_constraint.compliance:.6f} mm/N")
    print(f"Weight (relative to ball joint): {bushing_constraint.weight / coincident_constraint.weight:.3f}")

    print("\n--- Testing FixedPointConstraint ---")
    fixed_constraint = FixedPointConstraint(
        point1,
        target_position=[0, 0, 0],
        joint_type=JointType.RIGID
    )
    print(fixed_constraint)
    print(f"Error: {fixed_constraint.evaluate():.6f} mm²")

    print("\n--- Testing Compliance Effects ---")
    print("Moving point3 by 2mm in Y direction...")
    point3.set_position([100, 2, 0], unit='mm')

    print(f"\nBall joint (stiff):")
    print(f"  Separation: {CoincidentPointConstraint(point2, point3, joint_type=JointType.BALL_JOINT).get_separation():.3f} mm")
    print(f"  Force: {np.linalg.norm(CoincidentPointConstraint(point2, point3, joint_type=JointType.BALL_JOINT).get_force()):.1f} N")

    print(f"\nSoft bushing (compliant):")
    print(f"  Separation: {CoincidentPointConstraint(point2, point3, joint_type=JointType.BUSHING_SOFT).get_separation():.3f} mm")
    print(f"  Force: {np.linalg.norm(CoincidentPointConstraint(point2, point3, joint_type=JointType.BUSHING_SOFT).get_force()):.1f} N")

    print("\n--- Testing PartialPositionConstraint ---")
    # Create a wheel center point
    wheel_center = AttachmentPoint("wheel_center", [1500, 750, 350], unit='mm')

    # Constrain only Z (vertical) - useful for wheel travel analysis
    z_constraint = PartialPositionConstraint(
        wheel_center,
        target_position=[0, 0, 375],  # Only Z=375 matters
        constrain_axes=['z'],
        joint_type=JointType.RIGID
    )
    print(f"\nConstrain only Z (wheel height):")
    print(z_constraint)
    print(f"  Constrained axes: {z_constraint.constrain_axes}")
    print(f"  Free axes: {z_constraint.get_free_axes()}")
    print(f"  Error (Z only): {z_constraint.evaluate():.6f} mm²")
    print(f"  Physical error (Z): {z_constraint.get_physical_error():.6f} mm")
    print(f"  Force: {z_constraint.get_force()} N (only Z component)")

    # Constrain X and Y, leave Z free
    xy_constraint = PartialPositionConstraint(
        wheel_center,
        target_position=[1500, 750, 0],  # Only X, Y matter
        constrain_axes=['x', 'y'],
        joint_type=JointType.RIGID
    )
    print(f"\nConstrain X and Y (lateral position):")
    print(xy_constraint)
    print(f"  Constrained axes: {xy_constraint.constrain_axes}")
    print(f"  Free axes: {xy_constraint.get_free_axes()}")
    print(f"  Error (X,Y only): {xy_constraint.evaluate():.6f} mm²")

    # Move wheel and check
    wheel_center.set_position([1500, 750, 400], unit='mm')  # Changed Z from 350 to 400
    print(f"\nAfter moving wheel to Z=400mm:")
    print(f"  Z constraint error: {z_constraint.get_physical_error():.3f} mm (should be 25mm)")
    print(f"  XY constraint error: {xy_constraint.get_physical_error():.3f} mm (should be 0mm)")

    print("\n✓ All constraint tests completed successfully!")
