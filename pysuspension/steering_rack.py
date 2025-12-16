import numpy as np
from typing import List, Tuple, Union, Optional
from .rigid_body import RigidBody
from .suspension_link import SuspensionLink
from .attachment_point import AttachmentPoint
from .units import to_mm, from_mm, to_kg
from .joint_types import JointType, JOINT_STIFFNESS


class SteeringRack(RigidBody):
    """
    Represents a steering rack unit with housing and inner tie rod pivots.

    Inherits from RigidBody to provide rigid body transformation behavior.
    All attachment points (housing + inner pivots) are managed by the RigidBody base class.

    All positions are stored internally in millimeters (mm).

    The steering rack unit consists of:
    - Housing with attachment points that connect to the chassis (≥3 points)
    - A steering rack that moves linearly within the housing
    - Two inner tie rod pivots (on the rack) where tie rods attach
    - Travel per rotation (determines steering ratio)
    - Maximum displacement from center

    All attachment points (housing + inner pivots) are in self.attachment_points.
    Housing points (first N) are used for rigid body fitting.
    Inner pivots (last 2) follow transformations and maintain steering displacement.

    Tie rods are modeled separately as SuspensionLink objects and connected
    to the rack's inner pivots via SuspensionJoints.

    A positive travel_per_rotation indicates front-steer (rack forward of axle).
    """

    @staticmethod
    def _ensure_attachment_point(obj: Union[AttachmentPoint, np.ndarray, Tuple[float, float, float]],
                                   name: str,
                                   unit: str) -> AttachmentPoint:
        """
        Convert raw position to AttachmentPoint if needed.

        Args:
            obj: Either an AttachmentPoint or a raw position
            name: Name to use if creating a new AttachmentPoint
            unit: Unit of the position if creating a new AttachmentPoint

        Returns:
            AttachmentPoint object
        """
        if isinstance(obj, AttachmentPoint):
            return obj
        return AttachmentPoint(name=name, position=obj, unit=unit)

    def __init__(self,
                 housing_attachments: List[Union[AttachmentPoint, np.ndarray, Tuple[float, float, float]]],
                 left_inner_pivot: Union[AttachmentPoint, np.ndarray, Tuple[float, float, float]],
                 right_inner_pivot: Union[AttachmentPoint, np.ndarray, Tuple[float, float, float]],
                 travel_per_rotation: float,  # mm of rack travel per degree of steering
                 max_displacement: float,  # mm maximum displacement from center
                 name: str = "steering_rack",
                 unit: str = 'mm',
                 mass: float = 0.0,
                 mass_unit: str = 'kg'):
        """
        Initialize a steering rack unit.

        Args:
            housing_attachments: List of ≥3 AttachmentPoint objects (or raw positions) for chassis mounting
            left_inner_pivot: AttachmentPoint (or raw position) for left inner tie rod pivot (on the rack)
            right_inner_pivot: AttachmentPoint (or raw position) for right inner tie rod pivot (on the rack)
            travel_per_rotation: Rack travel per degree of steering wheel rotation (in specified unit)
            max_displacement: Maximum rack displacement from center (in specified unit)
            name: Identifier for the steering rack
            unit: Unit of input positions if raw positions are provided (default: 'mm')
            mass: Mass of the steering rack (default: 0.0)
            mass_unit: Unit of input mass (default: 'kg')

        Raises:
            ValueError: If fewer than 3 housing attachment points provided
        """
        if len(housing_attachments) < 3:
            raise ValueError("Housing requires at least 3 attachment points for rigid body mounting")

        # Initialize RigidBody base
        super().__init__(name=name, mass=mass, mass_unit=mass_unit)

        # Convert and add housing attachment points to self.attachment_points
        housing_points = [
            self._ensure_attachment_point(pos, f"{name}_mount_{i}", unit)
            for i, pos in enumerate(housing_attachments)
        ]
        for point in housing_points:
            self.add_attachment_point(point)

        # Track housing attachment count (for selective fitting)
        self._housing_attachment_count = len(housing_points)

        # Convert pivot inputs to AttachmentPoint objects and ADD to self.attachment_points
        self._left_inner_pivot = self._ensure_attachment_point(
            left_inner_pivot, f"{name}_left_inner", unit
        )
        self._left_inner_pivot.parent_component = self
        self.add_attachment_point(self._left_inner_pivot)

        self._right_inner_pivot = self._ensure_attachment_point(
            right_inner_pivot, f"{name}_right_inner", unit
        )
        self._right_inner_pivot.parent_component = self
        self.add_attachment_point(self._right_inner_pivot)

        # Calculate rack axis (direction of travel) from inner pivot points
        rack_vector = self._right_inner_pivot.position - self._left_inner_pivot.position
        self.rack_length = np.linalg.norm(rack_vector)
        if self.rack_length < 1e-6:
            raise ValueError("Inner pivot points are too close together")
        self.rack_axis = rack_vector / self.rack_length
        self.rack_axis_initial = self.rack_axis.copy()

        # Calculate rack center position
        self.rack_center_initial = (self._left_inner_pivot.position + self._right_inner_pivot.position) / 2.0
        self.rack_center = self.rack_center_initial.copy()

        # Store initial positions for reference (used by set_turn_angle as reference)
        self.left_inner_pivot_initial = self._left_inner_pivot.position.copy()
        self.right_inner_pivot_initial = self._right_inner_pivot.position.copy()

        # Store original positions for reset (never updated by transformations)
        self._original_left_inner_pivot = self._left_inner_pivot.position.copy()
        self._original_right_inner_pivot = self._right_inner_pivot.position.copy()
        self._original_rack_center = self.rack_center.copy()
        self._original_rack_axis = self.rack_axis.copy()

        # Travel parameters (in mm)
        self.travel_per_rotation = to_mm(travel_per_rotation, unit)
        self.max_displacement = to_mm(max_displacement, unit)

        # Current displacement from center (positive = right, negative = left)
        self.current_displacement = 0.0

        # Current steering angle
        self.current_angle = 0.0

    def get_left_inner_pivot(self) -> AttachmentPoint:
        """
        Get the left inner pivot attachment point.

        Returns:
            AttachmentPoint object for left inner pivot
        """
        return self._left_inner_pivot

    def get_right_inner_pivot(self) -> AttachmentPoint:
        """
        Get the right inner pivot attachment point.

        Returns:
            AttachmentPoint object for right inner pivot
        """
        return self._right_inner_pivot

    def get_housing_attachment_points(self) -> List[AttachmentPoint]:
        """
        Get only the housing attachment points (used for chassis mounting).

        Returns:
            List of housing AttachmentPoint objects (excludes inner pivots)
        """
        return self.attachment_points[:self._housing_attachment_count]

    def get_housing_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get only the housing attachment positions (used for chassis mounting).

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of housing attachment positions in specified unit
        """
        housing_points = self.get_housing_attachment_points()
        return [ap.get_position(unit) for ap in housing_points]

    # Backward compatibility properties
    @property
    def left_inner_pivot(self) -> AttachmentPoint:
        """Backward compatibility property for left inner pivot."""
        return self._left_inner_pivot

    @property
    def right_inner_pivot(self) -> AttachmentPoint:
        """Backward compatibility property for right inner pivot."""
        return self._right_inner_pivot

    def set_turn_angle(self, angle_degrees: float) -> None:
        """
        Set the steering angle, which displaces the rack along the rack axis.
        Updates the inner pivot positions based on the steering angle.

        Args:
            angle_degrees: Steering angle in degrees (positive = right turn)
        """
        self.current_angle = angle_degrees

        # Calculate rack displacement based on travel per rotation
        displacement = angle_degrees * self.travel_per_rotation

        # Clamp to maximum displacement
        displacement = np.clip(displacement, -self.max_displacement, self.max_displacement)
        self.current_displacement = displacement

        # Update inner pivot positions along rack axis
        displacement_vector = displacement * self.rack_axis

        # Calculate new positions
        new_left_inner = self.left_inner_pivot_initial + displacement_vector
        new_right_inner = self.right_inner_pivot_initial + displacement_vector
        self.rack_center = self.rack_center_initial + displacement_vector

        # Update AttachmentPoint positions
        self._left_inner_pivot.set_position(new_left_inner, unit='mm')
        self._right_inner_pivot.set_position(new_right_inner, unit='mm')

    def get_chassis_attachment_points(self) -> List[AttachmentPoint]:
        """
        Get the housing attachment point objects (connect to chassis).

        Deprecated: Use get_housing_attachment_points() instead.

        Returns:
            List of AttachmentPoint objects for the housing
        """
        return self.get_housing_attachment_points()

    def get_chassis_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get the housing attachment positions (connect to chassis).

        Deprecated: Use get_housing_attachment_positions() instead.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of housing attachment positions in specified unit
        """
        return self.get_housing_attachment_positions(unit=unit)

    def fit_chassis_to_attachment_targets(self, target_positions: List[np.ndarray],
                                          unit: str = 'mm') -> float:
        """
        Fit the steering rack housing to target chassis attachment positions.

        This method fits ONLY the housing attachment points (not inner pivots).
        The inner pivots follow the housing transformation while maintaining their
        steering displacement along the rack axis.

        Args:
            target_positions: List of target positions for HOUSING attachments only
                             (length must match housing attachment count, not total)
            unit: Unit of input positions (default: 'mm')

        Returns:
            RMS error of the fit (in mm)

        Raises:
            ValueError: If number of target positions doesn't match housing attachment count
        """
        if len(target_positions) != self._housing_attachment_count:
            raise ValueError(
                f"Expected {self._housing_attachment_count} target positions for housing, "
                f"got {len(target_positions)}"
            )

        # Freeze original state on first transformation
        self._original_state_frozen = True

        # Get current housing positions only
        housing_points = self.get_housing_attachment_points()
        current_positions = [ap.position.copy() for ap in housing_points]

        # Convert target positions to mm
        target_points = np.array([to_mm(np.array(p, dtype=float), unit) for p in target_positions])
        current_points = np.array(current_positions)

        # Get joint stiffness for housing points only
        stiffness_weights = []
        for ap in housing_points:
            if ap.joint is not None:
                stiffness = JOINT_STIFFNESS[ap.joint.joint_type]
            else:
                stiffness = JOINT_STIFFNESS[JointType.RIGID]
            stiffness_weights.append(stiffness)
        stiffness_weights = np.array(stiffness_weights)
        total_weight = np.sum(stiffness_weights)

        # Compute weighted centroids
        centroid_current = np.sum(current_points * stiffness_weights[:, np.newaxis], axis=0) / total_weight
        centroid_target = np.sum(target_points * stiffness_weights[:, np.newaxis], axis=0) / total_weight

        # Center the point sets
        current_centered = current_points - centroid_current
        target_centered = target_points - centroid_target

        # Compute weighted cross-covariance matrix H
        weighted_current = current_centered * np.sqrt(stiffness_weights[:, np.newaxis])
        weighted_target = target_centered * np.sqrt(stiffness_weights[:, np.newaxis])
        H = weighted_current.T @ weighted_target

        # SVD (Kabsch algorithm)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = centroid_target - R @ centroid_current

        # Apply transformation to ALL attachment points (housing + inner pivots) and rack state
        self._apply_transformation(R, t)

        # Calculate RMS error (housing points only)
        transformed_points = (R @ current_points.T).T + t
        errors = target_points - transformed_points
        squared_errors = np.sum(errors**2, axis=1)
        weighted_rms_error = np.sqrt(np.sum(stiffness_weights * squared_errors) / total_weight)

        # Reset steering angle and displacement after transformation
        # (transformation moves entire unit, not steering input)
        self.current_displacement = 0.0
        self.current_angle = 0.0

        return weighted_rms_error

    def get_rack_displacement(self, unit: str = 'mm') -> float:
        """
        Get the current rack displacement from center.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Current displacement in specified unit (positive = right)
        """
        return from_mm(self.current_displacement, unit)

    def get_steering_angle(self) -> float:
        """
        Get the current steering angle.

        Returns:
            Current steering angle in degrees
        """
        return self.current_angle

    def _apply_transformation(self, R: np.ndarray, t: np.ndarray) -> None:
        """
        Apply rigid body transformation to housing attachments and rack pivots.

        The base class handles transforming all attachment_points (housing + inner pivots).
        We only need to transform additional rack-specific state:
        - Rack center position
        - Rack axis direction
        - Initial positions for steering reference

        Args:
            R: 3x3 rotation matrix
            t: 3D translation vector (in mm)
        """
        # Call parent to handle ALL attachment points (housing + inner pivots) and center of mass
        super()._apply_transformation(R, t)

        # Transform rack axis (rotation only - it's a direction vector)
        self.rack_axis = R @ self.rack_axis
        self.rack_axis_initial = self.rack_axis.copy()

        # Transform rack center and its initial position
        self.rack_center = R @ self.rack_center + t
        self.rack_center_initial = R @ self.rack_center_initial + t

        # Transform initial positions (used as reference for set_turn_angle)
        self.left_inner_pivot_initial = R @ self.left_inner_pivot_initial + t
        self.right_inner_pivot_initial = R @ self.right_inner_pivot_initial + t

        # Note: Steering angle and displacement are NOT reset here
        # They represent user input, not transformation state

    def reset_to_origin(self) -> None:
        """
        Reset the steering rack to its originally defined position.

        Resets both housing (via parent) and rack-specific state:
        - Housing attachment points (via parent)
        - Inner pivot positions (via parent)
        - Rack center and axis
        - Steering angle and displacement
        """
        # Reset ALL attachment points (housing + inner pivots) via parent
        super().reset_to_origin()

        # Reset rack-specific state to original positions
        self.rack_center = self._original_rack_center.copy()
        self.rack_center_initial = self._original_rack_center.copy()
        self.rack_axis = self._original_rack_axis.copy()
        self.rack_axis_initial = self._original_rack_axis.copy()

        # Reset initial positions for steering reference
        self.left_inner_pivot_initial = self._original_left_inner_pivot.copy()
        self.right_inner_pivot_initial = self._original_right_inner_pivot.copy()

        # Reset steering angle and displacement
        self.current_displacement = 0.0
        self.current_angle = 0.0

    def copy(self, copy_joints: bool = False) -> 'SteeringRack':
        """
        Create a copy of the steering rack.

        Args:
            copy_joints: If True, copy joint references; if False, set joints to None

        Returns:
            New SteeringRack instance with copied state
        """
        # Copy housing attachment points only (first N attachment points)
        housing_attachments_copy = []
        for ap in self.get_housing_attachment_points():
            ap_copy = ap.copy(copy_joint=copy_joints, copy_parent=False)
            housing_attachments_copy.append(ap_copy)

        # Copy inner pivot points
        left_inner_copy = self._left_inner_pivot.copy(copy_joint=copy_joints, copy_parent=False)
        right_inner_copy = self._right_inner_pivot.copy(copy_joint=copy_joints, copy_parent=False)

        # Create new steering rack
        rack_copy = SteeringRack(
            housing_attachments=housing_attachments_copy,
            left_inner_pivot=left_inner_copy,
            right_inner_pivot=right_inner_copy,
            travel_per_rotation=self.travel_per_rotation,
            max_displacement=self.max_displacement,
            name=self.name,
            unit='mm',
            mass=self.mass,
            mass_unit='kg'
        )

        # Copy current state
        rack_copy.set_turn_angle(self.current_angle)

        return rack_copy

    def to_dict(self) -> dict:
        """
        Serialize the steering rack to a dictionary.

        Combines RigidBody data (housing + inner pivots in attachment_points)
        with SteeringRack-specific data.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        # Get base RigidBody serialization (includes ALL attachment points)
        data = super().to_dict()

        # Add SteeringRack-specific fields
        data.update({
            'type': 'SteeringRack',  # For type identification
            '_housing_attachment_count': self._housing_attachment_count,  # For partitioning
            'travel_per_rotation': float(self.travel_per_rotation),
            'max_displacement': float(self.max_displacement),
            'current_angle': float(self.current_angle),
            'current_displacement': float(self.current_displacement),
        })

        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'SteeringRack':
        """
        Deserialize a steering rack from a dictionary.

        Supports both new format (inherits RigidBody) and old format (housing member)
        for backward compatibility.

        Args:
            data: Dictionary containing steering rack data

        Returns:
            New SteeringRack instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # Handle backward compatibility with old format
        if 'housing' in data:
            # Old format: housing was a separate RigidBody member
            housing = RigidBody.from_dict(data['housing'])
            housing_positions = housing.get_all_attachment_points()

            # Extract inner pivot data from old format
            if 'left_inner_pivot' in data and 'right_inner_pivot' in data:
                left_inner = AttachmentPoint.from_dict(data['left_inner_pivot'])
                right_inner = AttachmentPoint.from_dict(data['right_inner_pivot'])
            else:
                raise KeyError("Missing inner pivot data in serialized SteeringRack")
        else:
            # New format: attachment_points includes housing + inner pivots
            all_attachment_points = [
                AttachmentPoint.from_dict(ap_data)
                for ap_data in data.get('attachment_points', [])
            ]

            # Partition based on _housing_attachment_count
            housing_count = data.get('_housing_attachment_count')
            if housing_count is None:
                raise KeyError("Missing _housing_attachment_count in new format")

            housing_positions = all_attachment_points[:housing_count]
            left_inner = all_attachment_points[housing_count]
            right_inner = all_attachment_points[housing_count + 1]

        # Create the steering rack
        rack = cls(
            housing_attachments=housing_positions,
            left_inner_pivot=left_inner,
            right_inner_pivot=right_inner,
            travel_per_rotation=data['travel_per_rotation'],
            max_displacement=data['max_displacement'],
            name=data['name'],
            unit='mm',
            mass=data.get('mass', 0.0),
            mass_unit=data.get('mass_unit', 'kg')
        )

        # Restore current state if present
        if 'current_angle' in data:
            rack.set_turn_angle(data['current_angle'])

        return rack

    def __repr__(self) -> str:
        """String representation showing housing and rack state."""
        centroid_str = f"{self.centroid} mm" if self.centroid is not None else "None"
        return (f"SteeringRack('{self.name}',\n"
                f"  centroid={centroid_str},\n"
                f"  housing_attachments={self._housing_attachment_count},\n"
                f"  total_attachments={len(self.attachment_points)},\n"
                f"  rack_center={self.rack_center} mm,\n"
                f"  rack_length={self.rack_length:.3f} mm,\n"
                f"  current_angle={self.current_angle:.2f}°,\n"
                f"  current_displacement={self.current_displacement:.3f} mm,\n"
                f"  travel_per_rotation={self.travel_per_rotation:.3f} mm/deg\n"
                f")")


if __name__ == "__main__":
    print("=" * 70)
    print("STEERING RACK TEST (rack-only, no tie rods)")
    print("=" * 70)

    # Test 1: Using AttachmentPoint objects (new API)
    print("\n--- Test 1: Creating with AttachmentPoint objects ---")

    housing_ap1 = AttachmentPoint("mount_0", [1.40, 0.2, 0.35], unit='m')
    housing_ap2 = AttachmentPoint("mount_1", [1.40, -0.2, 0.35], unit='m')
    housing_ap3 = AttachmentPoint("mount_2", [1.50, 0.0, 0.35], unit='m')

    left_inner_ap = AttachmentPoint("left_inner", [1.45, 0.3, 0.35], unit='m')
    right_inner_ap = AttachmentPoint("right_inner", [1.45, -0.3, 0.35], unit='m')

    rack = SteeringRack(
        housing_attachments=[housing_ap1, housing_ap2, housing_ap3],
        left_inner_pivot=left_inner_ap,
        right_inner_pivot=right_inner_ap,
        travel_per_rotation=0.001,  # 1mm per degree (in meters)
        max_displacement=0.1,  # 100mm max (in meters)
        name="front_rack",
        unit='m'
    )

    print("✓ Created SteeringRack with AttachmentPoint objects")

    # Test 2: Backward compatibility with raw positions
    print("\n--- Test 2: Backward compatibility test (raw positions) ---")

    rack_compat = SteeringRack(
        housing_attachments=[
            [1.40, 0.2, 0.35],   # Front right
            [1.40, -0.2, 0.35],  # Front left
            [1.50, 0.0, 0.35],   # Rear center
        ],
        left_inner_pivot=[1.45, 0.3, 0.35],
        right_inner_pivot=[1.45, -0.3, 0.35],
        travel_per_rotation=0.001,
        max_displacement=0.1,
        name="front_rack_compat",
        unit='m'
    )

    print("✓ Created SteeringRack with raw positions (backward compatibility)")

    # Continue with main tests using the AttachmentPoint-based rack
    print("\n--- Continuing tests with AttachmentPoint-based rack ---")

    print(f"\n{rack}")

    print("\n--- Initial positions ---")
    housing_pos = rack.get_chassis_attachment_positions(unit='m')
    print(f"Housing attachment points ({len(housing_pos)}):")
    for i, pos in enumerate(housing_pos):
        print(f"  {i}: {pos}")

    print(f"\nInner pivot positions:")
    print(f"  Left: {rack.left_inner_pivot.get_position('m')}")
    print(f"  Right: {rack.right_inner_pivot.get_position('m')}")

    # Test turning right (positive angle)
    print("\n--- Testing right turn (30 degrees) ---")
    rack.set_turn_angle(30.0)

    print(f"Steering angle: {rack.get_steering_angle():.2f}°")
    print(f"Rack displacement: {rack.get_rack_displacement('mm'):.3f} mm")

    print(f"\nInner pivot positions after turn:")
    print(f"  Left: {rack.left_inner_pivot.get_position('m')}")
    print(f"  Right: {rack.right_inner_pivot.get_position('m')}")

    # Test fitting chassis to new targets (rigid body transformation)
    print("\n--- Testing fit_chassis_to_attachment_targets (rigid body) ---")

    # Simulate chassis movement (translate and rotate housing)
    original_housing = rack.get_chassis_attachment_positions(unit='m')
    print(f"\nOriginal housing attachment 0 (m): {original_housing[0]}")

    # Create target positions: translate forward 50mm and rotate slightly
    angle_rad = np.radians(5.0)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    translation = np.array([0.05, 0.0, 0.0])  # 50mm forward

    new_housing_targets = []
    for pos in original_housing:
        new_pos = R @ pos + translation
        new_housing_targets.append(new_pos)

    print(f"Target housing attachment 0 (m): {new_housing_targets[0]}")

    housing_error = rack.fit_chassis_to_attachment_targets(new_housing_targets, unit='m')

    print(f"\nAfter fitting housing (rigid body):")
    print(f"  RMS error: {housing_error:.6f} mm = {from_mm(housing_error, 'm'):.9f} m")

    housing_pos = rack.get_chassis_attachment_positions(unit='m')
    print(f"  New housing attachment 0 (m): {housing_pos[0]}")

    print(f"\nInner pivot positions after fit:")
    print(f"  Left: {rack.left_inner_pivot.get_position('m')}")
    print(f"  Right: {rack.right_inner_pivot.get_position('m')}")

    # Test turning again after transformation
    print("\n--- Testing turn after transformation ---")
    rack.set_turn_angle(-20.0)
    print(f"Steering angle: {rack.get_steering_angle():.2f}°")
    print(f"Rack displacement: {rack.get_rack_displacement('mm'):.3f} mm")

    # Test reset
    print("\n--- Testing reset_to_origin ---")
    rack.reset_to_origin()

    print(f"After reset:")
    print(f"  Steering angle: {rack.get_steering_angle():.2f}°")
    print(f"  Rack displacement: {rack.get_rack_displacement('mm'):.3f} mm")

    housing_pos = rack.get_chassis_attachment_positions(unit='m')
    print(f"  Housing attachment 0 (m): {housing_pos[0]}")

    print(f"\nInner pivot positions after reset:")
    print(f"  Left: {rack.left_inner_pivot.get_position('m')}")
    print(f"  Right: {rack.right_inner_pivot.get_position('m')}")

    print("\n✓ All tests completed successfully!")
