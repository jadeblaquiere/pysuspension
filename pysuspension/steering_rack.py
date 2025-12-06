import numpy as np
from typing import List, Tuple, Union
from .rigid_body import RigidBody
from .suspension_link import SuspensionLink
from .attachment_point import AttachmentPoint
from .units import to_mm, from_mm


class SteeringRack:
    """
    Represents a steering rack unit with housing, movable rack, and tie rods.

    All positions are stored internally in millimeters (mm).

    The steering rack unit consists of:
    - Housing with attachment points that connect to the chassis (≥3 points, RigidBody)
    - A steering rack that moves linearly within the housing
    - Two inner tie rod pivots (on the rack)
    - Two outer tie rod attachment points (connect to knuckles)
    - Travel per rotation (determines steering ratio)
    - Maximum displacement from center

    The housing is implemented as a RigidBody member, providing rigid body
    transformation behavior for chassis mounting.

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
                 left_outer_attachment: Union[AttachmentPoint, np.ndarray, Tuple[float, float, float]],
                 right_outer_attachment: Union[AttachmentPoint, np.ndarray, Tuple[float, float, float]],
                 travel_per_rotation: float,  # mm of rack travel per degree of steering
                 max_displacement: float,  # mm maximum displacement from center
                 name: str = "steering_rack",
                 unit: str = 'mm'):
        """
        Initialize a steering rack unit.

        Args:
            housing_attachments: List of ≥3 AttachmentPoint objects (or raw positions) for chassis mounting
            left_inner_pivot: AttachmentPoint (or raw position) for left inner tie rod pivot (on the rack)
            right_inner_pivot: AttachmentPoint (or raw position) for right inner tie rod pivot (on the rack)
            left_outer_attachment: AttachmentPoint (or raw position) for left outer tie rod attachment (on knuckle)
            right_outer_attachment: AttachmentPoint (or raw position) for right outer tie rod attachment (on knuckle)
            travel_per_rotation: Rack travel per degree of steering wheel rotation (in specified unit)
            max_displacement: Maximum rack displacement from center (in specified unit)
            name: Identifier for the steering rack
            unit: Unit of input positions if raw positions are provided (default: 'mm')

        Raises:
            ValueError: If fewer than 3 housing attachment points provided
        """
        if len(housing_attachments) < 3:
            raise ValueError("Housing requires at least 3 attachment points for rigid body mounting")

        self.name = name

        # Convert housing attachments to AttachmentPoint objects if needed
        housing_points = [
            self._ensure_attachment_point(pos, f"{name}_mount_{i}", unit)
            for i, pos in enumerate(housing_attachments)
        ]

        # Create housing as a RigidBody (rigid body connected to chassis)
        self.housing = RigidBody(name=f"{name}_housing", mass=0.0, mass_unit='kg')
        for point in housing_points:
            self.housing.add_attachment_point(point)

        # Convert pivot/attachment inputs to AttachmentPoint objects if needed
        self.left_inner_pivot = self._ensure_attachment_point(
            left_inner_pivot, f"{name}_left_inner", unit
        )
        self.right_inner_pivot = self._ensure_attachment_point(
            right_inner_pivot, f"{name}_right_inner", unit
        )
        self.left_outer_attachment = self._ensure_attachment_point(
            left_outer_attachment, f"{name}_left_outer", unit
        )
        self.right_outer_attachment = self._ensure_attachment_point(
            right_outer_attachment, f"{name}_right_outer", unit
        )

        # Travel parameters (in mm)
        self.travel_per_rotation = to_mm(travel_per_rotation, unit)
        self.max_displacement = to_mm(max_displacement, unit)

        # Calculate rack axis (direction of travel) from inner pivot points
        rack_vector = self.right_inner_pivot.position - self.left_inner_pivot.position
        self.rack_length = np.linalg.norm(rack_vector)
        if self.rack_length < 1e-6:
            raise ValueError("Inner pivot points are too close together")
        self.rack_axis = rack_vector / self.rack_length
        self.rack_axis_initial = self.rack_axis.copy()

        # Calculate rack center position
        self.rack_center_initial = (self.left_inner_pivot.position + self.right_inner_pivot.position) / 2.0
        self.rack_center = self.rack_center_initial.copy()

        # Store initial positions for reference (used by set_turn_angle as reference)
        self.left_inner_pivot_initial = self.left_inner_pivot.position.copy()
        self.right_inner_pivot_initial = self.right_inner_pivot.position.copy()
        self.left_outer_attachment_initial = self.left_outer_attachment.position.copy()
        self.right_outer_attachment_initial = self.right_outer_attachment.position.copy()

        # Store original positions for reset (never updated by transformations)
        # Note: Housing original positions are now managed by self.housing (RigidBody)
        self._original_left_inner_pivot = self.left_inner_pivot.position.copy()
        self._original_right_inner_pivot = self.right_inner_pivot.position.copy()
        self._original_left_outer_attachment = self.left_outer_attachment.position.copy()
        self._original_right_outer_attachment = self.right_outer_attachment.position.copy()
        self._original_rack_center = self.rack_center.copy()
        self._original_rack_axis = self.rack_axis.copy()

        # Current displacement from center (positive = right, negative = left)
        self.current_displacement = 0.0

        # Create tie rod links (left and right)
        # SuspensionLink expects positions, so we pass the position from AttachmentPoint
        self.left_tie_rod = SuspensionLink(
            endpoint1=self.left_inner_pivot,
            endpoint2=self.left_outer_attachment,
            name=f"{name}_left_tie_rod",
            unit='mm'
        )

        self.right_tie_rod = SuspensionLink(
            endpoint1=self.right_inner_pivot,
            endpoint2=self.right_outer_attachment,
            name=f"{name}_right_tie_rod",
            unit='mm'
        )

        # Store initial tie rod lengths
        self.left_tie_rod_length = self.left_tie_rod.length
        self.right_tie_rod_length = self.right_tie_rod.length

        # Current steering angle
        self.current_angle = 0.0

    def set_turn_angle(self, angle_degrees: float) -> None:
        """
        Set the steering angle, which displaces the rack along the rack axis.
        Both inner and outer tie rod endpoints move together to maintain tie rod lengths.

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

        # Move outer tie rod endpoints the same distance to maintain tie rod length
        new_left_outer = self.left_outer_attachment_initial + displacement_vector
        new_right_outer = self.right_outer_attachment_initial + displacement_vector

        # Update AttachmentPoint positions
        self.left_inner_pivot.set_position(new_left_inner, unit='mm')
        self.right_inner_pivot.set_position(new_right_inner, unit='mm')
        self.left_outer_attachment.set_position(new_left_outer, unit='mm')
        self.right_outer_attachment.set_position(new_right_outer, unit='mm')

        # Update tie rod endpoints (use set_position on AttachmentPoint)
        self.left_tie_rod.endpoint1.set_position(new_left_inner, unit='mm')
        self.left_tie_rod.endpoint2.set_position(new_left_outer, unit='mm')
        self.right_tie_rod.endpoint1.set_position(new_right_inner, unit='mm')
        self.right_tie_rod.endpoint2.set_position(new_right_outer, unit='mm')

        # Update tie rod geometry
        self.left_tie_rod._update_local_frame()
        self.right_tie_rod._update_local_frame()

    def get_chassis_attachment_points(self) -> List[AttachmentPoint]:
        """
        Get the housing attachment point objects (connect to chassis).

        Returns:
            List of AttachmentPoint objects for the housing
        """
        return self.housing.get_all_attachment_points()

    def get_chassis_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get the housing attachment positions (connect to chassis).

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of housing attachment positions in specified unit
        """
        return self.housing.get_all_attachment_positions(unit=unit)

    def fit_chassis_to_attachment_targets(self, target_positions: List[np.ndarray],
                                          unit: str = 'mm') -> float:
        """
        Fit the steering rack housing to target chassis attachment positions.
        Performs rigid body fit on housing and applies the transformation to the entire unit
        (housing, rack, inner pivots, outer tie rod endpoints).

        Args:
            target_positions: List of target positions for housing attachments (≥3) in specified unit
            unit: Unit of input positions (default: 'mm')

        Returns:
            RMS error of the fit (in mm)

        Raises:
            ValueError: If number of target positions doesn't match housing attachment count
        """
        # Store original housing positions to calculate transformation
        original_housing_positions = self.housing.get_all_attachment_positions(unit='mm')

        # Use housing's fit_to_attachment_targets (which applies the transformation)
        rms_error = self.housing.fit_to_attachment_targets(target_positions, unit=unit)

        # Calculate the transformation that was applied by comparing before/after
        new_housing_positions = self.housing.get_all_attachment_positions(unit='mm')

        # Extract rotation and translation from the housing transformation
        # We can use the housing's rotation_matrix since it was updated by fit_to_attachment_targets
        R = self.housing.rotation_matrix

        # Calculate translation from centroid movement
        original_centroid = np.mean(original_housing_positions, axis=0)
        new_centroid = self.housing.centroid
        t = new_centroid - R @ original_centroid

        # Apply the same transformation to rack axis (rotation only, it's a direction vector)
        self.rack_axis = R @ self.rack_axis
        self.rack_axis_initial = self.rack_axis.copy()

        # Apply transformation to rack center and inner pivots
        self.rack_center = R @ self.rack_center + t
        self.rack_center_initial = R @ self.rack_center_initial + t

        # Transform inner pivot positions
        new_left_inner = R @ self.left_inner_pivot.position + t
        new_right_inner = R @ self.right_inner_pivot.position + t
        self.left_inner_pivot_initial = R @ self.left_inner_pivot_initial + t
        self.right_inner_pivot_initial = R @ self.right_inner_pivot_initial + t

        # Apply transformation to outer tie rod endpoints
        new_left_outer = R @ self.left_outer_attachment.position + t
        new_right_outer = R @ self.right_outer_attachment.position + t
        self.left_outer_attachment_initial = R @ self.left_outer_attachment_initial + t
        self.right_outer_attachment_initial = R @ self.right_outer_attachment_initial + t

        # Update AttachmentPoint positions
        self.left_inner_pivot.set_position(new_left_inner, unit='mm')
        self.right_inner_pivot.set_position(new_right_inner, unit='mm')
        self.left_outer_attachment.set_position(new_left_outer, unit='mm')
        self.right_outer_attachment.set_position(new_right_outer, unit='mm')

        # Update tie rod endpoints
        self.left_tie_rod.endpoint1.set_position(new_left_inner, unit='mm')
        self.left_tie_rod.endpoint2.set_position(new_left_outer, unit='mm')
        self.right_tie_rod.endpoint1.set_position(new_right_inner, unit='mm')
        self.right_tie_rod.endpoint2.set_position(new_right_outer, unit='mm')

        # Update tie rod geometry
        self.left_tie_rod._update_local_frame()
        self.right_tie_rod._update_local_frame()

        # Reset displacement and angle (transformation moves the entire unit, not steering input)
        self.current_displacement = 0.0
        self.current_angle = 0.0

        return rms_error

    def get_outer_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get the outer tie rod attachment positions.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of [left_outer, right_outer] in specified unit
        """
        return [
            from_mm(self.left_tie_rod.endpoint2.position.copy(), unit),
            from_mm(self.right_tie_rod.endpoint2.position.copy(), unit)
        ]

    def fit_outer_to_attachment_targets(self, target_positions: List[np.ndarray],
                                        unit: str = 'mm') -> float:
        """
        Fit the outer tie rod attachments to target positions.
        Maintains constant tie rod lengths while minimizing RMS error.

        Args:
            target_positions: List of [left_target, right_target] in specified unit
            unit: Unit of input positions (default: 'mm')

        Returns:
            RMS error of the fit (in mm)
        """
        if len(target_positions) != 2:
            raise ValueError("Expected 2 target positions (left and right)")

        # Convert targets to mm
        target_left = to_mm(np.array(target_positions[0], dtype=float), unit)
        target_right = to_mm(np.array(target_positions[1], dtype=float), unit)

        # Fit left tie rod to target
        left_error = self.left_tie_rod.fit_to_attachment_targets(
            [self.left_inner_pivot.position, target_left],
            unit='mm'
        )

        # Fit right tie rod to target
        right_error = self.right_tie_rod.fit_to_attachment_targets(
            [self.right_inner_pivot.position, target_right],
            unit='mm'
        )

        # Update outer attachment AttachmentPoint positions and initial positions
        new_left_outer = self.left_tie_rod.endpoint2.position.copy()
        new_right_outer = self.right_tie_rod.endpoint2.position.copy()

        self.left_outer_attachment.set_position(new_left_outer, unit='mm')
        self.right_outer_attachment.set_position(new_right_outer, unit='mm')
        self.left_outer_attachment_initial = new_left_outer.copy()
        self.right_outer_attachment_initial = new_right_outer.copy()

        # Calculate combined RMS error
        rms_error = np.sqrt((left_error**2 + right_error**2) / 2.0)

        return rms_error

    def get_tie_rod_lengths(self, unit: str = 'mm') -> Tuple[float, float]:
        """
        Get the lengths of the left and right tie rods.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Tuple of (left_length, right_length) in specified unit
        """
        return (
            from_mm(self.left_tie_rod.get_length(unit='mm'), unit),
            from_mm(self.right_tie_rod.get_length(unit='mm'), unit)
        )

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

    def reset_to_origin(self) -> None:
        """
        Reset the steering rack to its originally defined position.
        Resets housing, rack, inner pivots, outer attachments, and steering angle to
        the positions defined at construction.
        """
        # Reset housing to original positions (using RigidBody's reset)
        self.housing.reset_to_origin()

        # Reset rack center and axis to original
        self.rack_center = self._original_rack_center.copy()
        self.rack_center_initial = self._original_rack_center.copy()
        self.rack_axis = self._original_rack_axis.copy()
        self.rack_axis_initial = self._original_rack_axis.copy()

        # Reset inner pivots to original positions
        self.left_inner_pivot.set_position(self._original_left_inner_pivot, unit='mm')
        self.right_inner_pivot.set_position(self._original_right_inner_pivot, unit='mm')
        self.left_inner_pivot_initial = self._original_left_inner_pivot.copy()
        self.right_inner_pivot_initial = self._original_right_inner_pivot.copy()

        # Reset outer attachments to original positions
        self.left_outer_attachment.set_position(self._original_left_outer_attachment, unit='mm')
        self.right_outer_attachment.set_position(self._original_right_outer_attachment, unit='mm')
        self.left_outer_attachment_initial = self._original_left_outer_attachment.copy()
        self.right_outer_attachment_initial = self._original_right_outer_attachment.copy()

        # Reset steering angle and displacement
        self.current_displacement = 0.0
        self.current_angle = 0.0

        # Update tie rod endpoints
        self.left_tie_rod.endpoint1.set_position(self._original_left_inner_pivot, unit='mm')
        self.left_tie_rod.endpoint2.set_position(self._original_left_outer_attachment, unit='mm')
        self.right_tie_rod.endpoint1.set_position(self._original_right_inner_pivot, unit='mm')
        self.right_tie_rod.endpoint2.set_position(self._original_right_outer_attachment, unit='mm')

        # Update tie rod geometry
        self.left_tie_rod._update_local_frame()
        self.right_tie_rod._update_local_frame()

    def to_dict(self) -> dict:
        """
        Serialize the steering rack to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'name': self.name,
            'housing': self.housing.to_dict(),
            'left_tie_rod': self.left_tie_rod.to_dict(),
            'right_tie_rod': self.right_tie_rod.to_dict(),
            'travel_per_rotation': float(self.travel_per_rotation),  # Store in mm
            'max_displacement': float(self.max_displacement),  # Store in mm
            'current_angle': float(self.current_angle),  # Store current state
            'current_displacement': float(self.current_displacement),  # Store current state
            'unit': 'mm'
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SteeringRack':
        """
        Deserialize a steering rack from a dictionary.

        Args:
            data: Dictionary containing steering rack data

        Returns:
            New SteeringRack instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # Deserialize tie rods to get inner/outer positions
        left_tie_rod = SuspensionLink.from_dict(data['left_tie_rod'])
        right_tie_rod = SuspensionLink.from_dict(data['right_tie_rod'])

        # Extract housing attachment positions
        # Support both old format (housing_attachment_points) and new format (housing RigidBody)
        if 'housing' in data:
            # New format: housing is a RigidBody
            housing = RigidBody.from_dict(data['housing'])
            housing_positions = housing.get_all_attachment_positions(unit='mm')
        else:
            # Old format: housing_attachment_points list
            housing_positions = [ap_data['position'] for ap_data in data['housing_attachment_points']]

        # Create the steering rack
        rack = cls(
            housing_attachments=housing_positions,
            left_inner_pivot=left_tie_rod.endpoint1.position,
            right_inner_pivot=right_tie_rod.endpoint1.position,
            left_outer_attachment=left_tie_rod.endpoint2.position,
            right_outer_attachment=right_tie_rod.endpoint2.position,
            travel_per_rotation=data['travel_per_rotation'],
            max_displacement=data['max_displacement'],
            name=data['name'],
            unit=data.get('unit', 'mm')
        )

        # Restore current state if present
        if 'current_angle' in data:
            rack.set_turn_angle(data['current_angle'])

        return rack

    def __repr__(self) -> str:
        left_len, right_len = self.get_tie_rod_lengths(unit='mm')
        housing_centroid_str = f"{self.housing.centroid} mm" if self.housing.centroid is not None else "None"
        return (f"SteeringRack('{self.name}',\n"
                f"  housing_centroid={housing_centroid_str},\n"
                f"  housing_attachments={len(self.housing.attachment_points)},\n"
                f"  rack_center={self.rack_center} mm,\n"
                f"  rack_length={self.rack_length:.3f} mm,\n"
                f"  current_angle={self.current_angle:.2f}°,\n"
                f"  current_displacement={self.current_displacement:.3f} mm,\n"
                f"  left_tie_rod_length={left_len:.3f} mm,\n"
                f"  right_tie_rod_length={right_len:.3f} mm,\n"
                f"  travel_per_rotation={self.travel_per_rotation:.3f} mm/deg\n"
                f")")


if __name__ == "__main__":
    print("=" * 70)
    print("STEERING RACK TEST (with AttachmentPoint support)")
    print("=" * 70)

    # Test 1: Using AttachmentPoint objects (new API)
    print("\n--- Test 1: Creating with AttachmentPoint objects ---")

    housing_ap1 = AttachmentPoint("mount_0", [1.40, 0.2, 0.35], unit='m')
    housing_ap2 = AttachmentPoint("mount_1", [1.40, -0.2, 0.35], unit='m')
    housing_ap3 = AttachmentPoint("mount_2", [1.50, 0.0, 0.35], unit='m')

    left_inner_ap = AttachmentPoint("left_inner", [1.45, 0.3, 0.35], unit='m')
    right_inner_ap = AttachmentPoint("right_inner", [1.45, -0.3, 0.35], unit='m')
    left_outer_ap = AttachmentPoint("left_outer", [1.55, 0.65, 0.35], unit='m')
    right_outer_ap = AttachmentPoint("right_outer", [1.55, -0.65, 0.35], unit='m')

    rack = SteeringRack(
        housing_attachments=[housing_ap1, housing_ap2, housing_ap3],
        left_inner_pivot=left_inner_ap,
        right_inner_pivot=right_inner_ap,
        left_outer_attachment=left_outer_ap,
        right_outer_attachment=right_outer_ap,
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
        left_outer_attachment=[1.55, 0.65, 0.35],
        right_outer_attachment=[1.55, -0.65, 0.35],
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

    outer_pos = rack.get_outer_attachment_positions(unit='m')
    print(f"\nOuter tie rod attachments:")
    print(f"  Left: {outer_pos[0]}")
    print(f"  Right: {outer_pos[1]}")

    left_len, right_len = rack.get_tie_rod_lengths(unit='m')
    print(f"\nTie rod lengths:")
    print(f"  Left: {left_len:.6f} m")
    print(f"  Right: {right_len:.6f} m")

    # Test turning right (positive angle)
    print("\n--- Testing right turn (30 degrees) ---")
    rack.set_turn_angle(30.0)

    print(f"Steering angle: {rack.get_steering_angle():.2f}°")
    print(f"Rack displacement: {rack.get_rack_displacement('mm'):.3f} mm")

    outer_pos = rack.get_outer_attachment_positions(unit='m')
    print(f"\nOuter tie rod attachments after turn:")
    print(f"  Left: {outer_pos[0]}")
    print(f"  Right: {outer_pos[1]}")

    left_len, right_len = rack.get_tie_rod_lengths(unit='m')
    print(f"\nTie rod lengths maintained:")
    print(f"  Left: {left_len:.6f} m")
    print(f"  Right: {right_len:.6f} m")

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

    left_len, right_len = rack.get_tie_rod_lengths(unit='m')
    print(f"\nTie rod lengths maintained after fit:")
    print(f"  Left: {left_len:.6f} m")
    print(f"  Right: {right_len:.6f} m")

    # Test turning again after transformation
    print("\n--- Testing turn after transformation ---")
    rack.set_turn_angle(-20.0)
    print(f"Steering angle: {rack.get_steering_angle():.2f}°")

    left_len, right_len = rack.get_tie_rod_lengths(unit='m')
    print(f"Tie rod lengths still maintained:")
    print(f"  Left: {left_len:.6f} m")
    print(f"  Right: {right_len:.6f} m")

    # Test reset
    print("\n--- Testing reset_to_origin ---")
    rack.reset_to_origin()

    print(f"After reset:")
    print(f"  Steering angle: {rack.get_steering_angle():.2f}°")
    print(f"  Rack displacement: {rack.get_rack_displacement('mm'):.3f} mm")

    housing_pos = rack.get_chassis_attachment_positions(unit='m')
    print(f"  Housing attachment 0 (m): {housing_pos[0]}")

    print("\n✓ All tests completed successfully!")
