import numpy as np
from typing import List, Tuple, Union
from suspension_link import SuspensionLink
from units import to_mm, from_mm


class SteeringRack:
    """
    Represents a steering rack with left and right tie rods.

    All positions are stored internally in millimeters (mm).

    The steering rack consists of:
    - Two chassis attachment points (left and right inner tie rod pivots)
    - Two outer tie rod attachment points (connect to knuckles)
    - Travel per rotation (determines steering ratio)
    - Maximum displacement from center

    A positive travel_per_rotation indicates front-steer (rack forward of axle).
    """

    def __init__(self,
                 left_inner_pivot: Union[np.ndarray, Tuple[float, float, float]],
                 right_inner_pivot: Union[np.ndarray, Tuple[float, float, float]],
                 left_outer_attachment: Union[np.ndarray, Tuple[float, float, float]],
                 right_outer_attachment: Union[np.ndarray, Tuple[float, float, float]],
                 travel_per_rotation: float,  # mm of rack travel per degree of steering
                 max_displacement: float,  # mm maximum displacement from center
                 name: str = "steering_rack",
                 unit: str = 'mm'):
        """
        Initialize a steering rack.

        Args:
            left_inner_pivot: Position of left inner tie rod pivot (chassis attachment)
            right_inner_pivot: Position of right inner tie rod pivot (chassis attachment)
            left_outer_attachment: Position of left outer tie rod attachment
            right_outer_attachment: Position of right outer tie rod attachment
            travel_per_rotation: Rack travel per degree of steering wheel rotation (in specified unit)
            max_displacement: Maximum rack displacement from center (in specified unit)
            name: Identifier for the steering rack
            unit: Unit of input positions (default: 'mm')
        """
        self.name = name

        # Convert all inputs to mm for internal storage
        self.left_inner_pivot = to_mm(np.array(left_inner_pivot, dtype=float), unit)
        self.right_inner_pivot = to_mm(np.array(right_inner_pivot, dtype=float), unit)
        self.left_outer_attachment = to_mm(np.array(left_outer_attachment, dtype=float), unit)
        self.right_outer_attachment = to_mm(np.array(right_outer_attachment, dtype=float), unit)

        # Store initial positions for reference
        self.left_inner_pivot_initial = self.left_inner_pivot.copy()
        self.right_inner_pivot_initial = self.right_inner_pivot.copy()

        # Travel parameters (in mm)
        self.travel_per_rotation = to_mm(travel_per_rotation, unit)
        self.max_displacement = to_mm(max_displacement, unit)

        # Calculate rack axis (direction of travel) from inner pivot points
        rack_vector = self.right_inner_pivot - self.left_inner_pivot
        self.rack_length = np.linalg.norm(rack_vector)
        if self.rack_length < 1e-6:
            raise ValueError("Inner pivot points are too close together")
        self.rack_axis = rack_vector / self.rack_length

        # Calculate rack center position
        self.rack_center_initial = (self.left_inner_pivot + self.right_inner_pivot) / 2.0
        self.rack_center = self.rack_center_initial.copy()

        # Current displacement from center (positive = right, negative = left)
        self.current_displacement = 0.0

        # Create tie rod links (left and right)
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

        # Current steering angle
        self.current_angle = 0.0

    def set_turn_angle(self, angle_degrees: float) -> None:
        """
        Set the steering angle, which displaces the inner pivot points along the rack axis.

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

        self.left_inner_pivot = self.left_inner_pivot_initial + displacement_vector
        self.right_inner_pivot = self.right_inner_pivot_initial + displacement_vector
        self.rack_center = self.rack_center_initial + displacement_vector

        # Update tie rod inner endpoints (keeping outer endpoints fixed for now)
        self.left_tie_rod.endpoint1 = self.left_inner_pivot.copy()
        self.right_tie_rod.endpoint1 = self.right_inner_pivot.copy()

        # Update tie rod geometry
        self.left_tie_rod._update_local_frame()
        self.right_tie_rod._update_local_frame()

    def get_chassis_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get the chassis attachment positions (inner tie rod pivots).

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of [left_inner_pivot, right_inner_pivot] in specified unit
        """
        return [
            from_mm(self.left_inner_pivot.copy(), unit),
            from_mm(self.right_inner_pivot.copy(), unit)
        ]

    def fit_chassis_to_attachment_targets(self, target_positions: List[np.ndarray],
                                          unit: str = 'mm') -> float:
        """
        Fit the steering rack chassis attachments to target positions.
        Treats the rack (chassis attachments and inner pivots) as a rigid body.

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

        # Calculate new rack center and axis
        new_rack_center = (target_left + target_right) / 2.0
        new_rack_vector = target_right - target_left
        new_rack_length = np.linalg.norm(new_rack_vector)

        if new_rack_length < 1e-6:
            raise ValueError("Target positions are too close together")

        new_rack_axis = new_rack_vector / new_rack_length

        # Update rack properties
        self.rack_center = new_rack_center.copy()
        self.rack_center_initial = new_rack_center.copy()
        self.rack_axis = new_rack_axis.copy()
        self.rack_length = new_rack_length

        # Update inner pivot positions
        self.left_inner_pivot = target_left.copy()
        self.right_inner_pivot = target_right.copy()
        self.left_inner_pivot_initial = target_left.copy()
        self.right_inner_pivot_initial = target_right.copy()

        # Reset displacement
        self.current_displacement = 0.0
        self.current_angle = 0.0

        # Update tie rod inner endpoints
        self.left_tie_rod.endpoint1 = self.left_inner_pivot.copy()
        self.right_tie_rod.endpoint1 = self.right_inner_pivot.copy()
        self.left_tie_rod._update_local_frame()
        self.right_tie_rod._update_local_frame()

        # Calculate RMS error
        error_left = target_left - self.left_inner_pivot
        error_right = target_right - self.right_inner_pivot
        rms_error = np.sqrt((np.sum(error_left**2) + np.sum(error_right**2)) / 2.0)

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
            from_mm(self.left_tie_rod.endpoint2.copy(), unit),
            from_mm(self.right_tie_rod.endpoint2.copy(), unit)
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
            [self.left_inner_pivot, target_left],
            unit='mm'
        )

        # Fit right tie rod to target
        right_error = self.right_tie_rod.fit_to_attachment_targets(
            [self.right_inner_pivot, target_right],
            unit='mm'
        )

        # Update outer attachment positions
        self.left_outer_attachment = self.left_tie_rod.endpoint2.copy()
        self.right_outer_attachment = self.right_tie_rod.endpoint2.copy()

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

    def __repr__(self) -> str:
        left_len, right_len = self.get_tie_rod_lengths(unit='mm')
        return (f"SteeringRack('{self.name}',\n"
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
    print("STEERING RACK TEST (with unit support)")
    print("=" * 70)

    # Create a steering rack (using meters as input, stored as mm internally)
    rack = SteeringRack(
        left_inner_pivot=[1.45, 0.3, 0.35],      # Left chassis mount
        right_inner_pivot=[1.45, -0.3, 0.35],    # Right chassis mount
        left_outer_attachment=[1.55, 0.65, 0.35],  # Left knuckle attachment
        right_outer_attachment=[1.55, -0.65, 0.35],  # Right knuckle attachment
        travel_per_rotation=0.001,  # 1mm per degree (in meters)
        max_displacement=0.1,  # 100mm max (in meters)
        name="front_rack",
        unit='m'
    )

    print(f"\n{rack}")

    print("\n--- Initial positions ---")
    chassis_pos = rack.get_chassis_attachment_positions(unit='m')
    outer_pos = rack.get_outer_attachment_positions(unit='m')
    print(f"Left inner pivot (m): {chassis_pos[0]}")
    print(f"Right inner pivot (m): {chassis_pos[1]}")
    print(f"Left outer attachment (m): {outer_pos[0]}")
    print(f"Right outer attachment (m): {outer_pos[1]}")

    left_len, right_len = rack.get_tie_rod_lengths(unit='m')
    print(f"\nTie rod lengths:")
    print(f"  Left: {left_len:.6f} m")
    print(f"  Right: {right_len:.6f} m")

    # Test turning right (positive angle)
    print("\n--- Testing right turn (30 degrees) ---")
    rack.set_turn_angle(30.0)

    print(f"\n{rack}")
    print(f"Steering angle: {rack.get_steering_angle():.2f}°")
    print(f"Rack displacement: {rack.get_rack_displacement('mm'):.3f} mm = {rack.get_rack_displacement('m'):.6f} m")

    chassis_pos = rack.get_chassis_attachment_positions(unit='m')
    outer_pos = rack.get_outer_attachment_positions(unit='m')
    print(f"\nAfter turning right:")
    print(f"Left inner pivot (m): {chassis_pos[0]}")
    print(f"Right inner pivot (m): {chassis_pos[1]}")
    print(f"Left outer attachment (m): {outer_pos[0]}")
    print(f"Right outer attachment (m): {outer_pos[1]}")

    left_len, right_len = rack.get_tie_rod_lengths(unit='m')
    print(f"\nTie rod lengths maintained:")
    print(f"  Left: {left_len:.6f} m")
    print(f"  Right: {right_len:.6f} m")

    # Test turning left (negative angle)
    print("\n--- Testing left turn (-45 degrees) ---")
    rack.set_turn_angle(-45.0)

    print(f"Steering angle: {rack.get_steering_angle():.2f}°")
    print(f"Rack displacement: {rack.get_rack_displacement('mm'):.3f} mm = {rack.get_rack_displacement('m'):.6f} m")

    # Test fitting chassis to new targets
    print("\n--- Testing fit_chassis_to_attachment_targets ---")

    # Simulate chassis movement (translate by 50mm forward)
    original_chassis = rack.get_chassis_attachment_positions(unit='m')
    new_chassis_targets = [
        original_chassis[0] + np.array([0.05, 0.0, 0.0]),
        original_chassis[1] + np.array([0.05, 0.0, 0.0])
    ]

    print(f"Original left inner pivot (m): {original_chassis[0]}")
    print(f"Target left inner pivot (m): {new_chassis_targets[0]}")

    chassis_error = rack.fit_chassis_to_attachment_targets(new_chassis_targets, unit='m')

    print(f"\nAfter fitting chassis:")
    print(f"RMS error: {chassis_error:.6f} mm = {from_mm(chassis_error, 'm'):.9f} m")

    chassis_pos = rack.get_chassis_attachment_positions(unit='m')
    print(f"New left inner pivot (m): {chassis_pos[0]}")
    print(f"New right inner pivot (m): {chassis_pos[1]}")

    # Test fitting outer attachments to new targets
    print("\n--- Testing fit_outer_to_attachment_targets ---")

    # Simulate knuckle movement
    original_outer = rack.get_outer_attachment_positions(unit='m')
    new_outer_targets = [
        original_outer[0] + np.array([0.02, -0.01, -0.03]),
        original_outer[1] + np.array([0.02, 0.01, -0.03])
    ]

    print(f"Original left outer (m): {original_outer[0]}")
    print(f"Target left outer (m): {new_outer_targets[0]}")

    outer_error = rack.fit_outer_to_attachment_targets(new_outer_targets, unit='m')

    print(f"\nAfter fitting outer attachments:")
    print(f"RMS error: {outer_error:.6f} mm = {from_mm(outer_error, 'm'):.9f} m")

    outer_pos = rack.get_outer_attachment_positions(unit='m')
    print(f"New left outer (m): {outer_pos[0]}")
    print(f"New right outer (m): {outer_pos[1]}")

    left_len, right_len = rack.get_tie_rod_lengths(unit='m')
    print(f"\nTie rod lengths maintained:")
    print(f"  Left: {left_len:.6f} m")
    print(f"  Right: {right_len:.6f} m")

    print("\n✓ All tests completed successfully!")
