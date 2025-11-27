import numpy as np
from typing import List, Tuple, Optional, Union
from .attachment_point import AttachmentPoint
from .units import to_mm, from_mm, to_kg


class SuspensionKnuckle:
    """
    Represents a suspension knuckle with geometric properties.

    All positions are stored internally in millimeters (mm).

    Coordinate system:
    - x: longitudinal (+ front, - rear)
    - y: lateral (+ left, - right)
    - z: vertical (+ up, - down)

    Angular conventions:
    - toe: projected angle in xy plane (+ leading edge projects out)
    - camber: projected angle in yz plane (+ top leans out)
    """

    def __init__(self,
                 tire_center_x: float,
                 tire_center_y: float,
                 rolling_radius: float,
                 toe_angle: float = 0.0,
                 camber_angle: float = 0.0,
                 wheel_offset: float = 0.0,
                 mass: float = 0.0,
                 unit: str = 'mm',
                 mass_unit: str = 'kg'):
        """
        Initialize suspension knuckle.

        Args:
            tire_center_x: Longitudinal position (+ front, - rear)
            tire_center_y: Lateral position / track width (+ left, - right)
            rolling_radius: Vertical position (tire rolling radius)
            toe_angle: Toe setting in degrees (+ leading edge out)
            camber_angle: Camber angle in degrees (+ top leans out)
            wheel_offset: Offset of wheel mounting plane from tire center (+ outward)
            mass: Mass of the knuckle (default: 0.0)
            unit: Unit of input positions (default: 'mm')
            mass_unit: Unit of input mass (default: 'kg')
        """
        # Convert inputs to mm for internal storage
        tire_center_x_mm = to_mm(tire_center_x, unit)
        tire_center_y_mm = to_mm(tire_center_y, unit)
        rolling_radius_mm = to_mm(rolling_radius, unit)
        wheel_offset_mm = to_mm(wheel_offset, unit)

        # Tire center position (absolute, in mm)
        self.tire_center = np.array([tire_center_x_mm, tire_center_y_mm, rolling_radius_mm], dtype=float)

        # Angular orientation (store in radians internally)
        self.toe_angle = np.radians(toe_angle)
        self.camber_angle = np.radians(camber_angle)

        # Wheel mounting plane offset (in mm)
        self.wheel_offset = wheel_offset_mm

        # Mass (stored in kg)
        self.mass = to_kg(mass, mass_unit)

        # Center of mass (at tire center initially, in mm)
        self.center_of_mass = self.tire_center.copy()

        # Attachment points list
        self.attachment_points: List[AttachmentPoint] = []

        # Steering attachment designation (None if not set)
        self.steering_attachment_name: Optional[str] = None

        # Compute rotation matrix and tire axis
        self._update_geometry()

        # Store original state for reset
        self._original_state = {
            'tire_center': self.tire_center.copy(),
            'toe_angle': self.toe_angle,
            'camber_angle': self.camber_angle,
            'rotation_matrix': self.rotation_matrix.copy(),
            'center_of_mass': self.center_of_mass.copy(),
        }
    
    def _update_geometry(self):
        """Update rotation matrix and derived geometric properties."""
        # Rotation matrix from toe and camber angles
        # Apply camber first (rotation about x-axis), then toe (rotation about z-axis)
        
        # Camber rotation (about x-axis)
        R_camber = np.array([
            [1, 0, 0],
            [0, np.cos(self.camber_angle), -np.sin(self.camber_angle)],
            [0, np.sin(self.camber_angle), np.cos(self.camber_angle)]
        ])
        
        # Toe rotation (about z-axis)
        R_toe = np.array([
            [np.cos(self.toe_angle), -np.sin(self.toe_angle), 0],
            [np.sin(self.toe_angle), np.cos(self.toe_angle), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix (toe applied after camber)
        self.rotation_matrix = R_toe @ R_camber
        
        # Tire axis in relative coordinates (initially pointing in +y direction)
        self.tire_axis_relative = np.array([0, 1, 0], dtype=float)
        
        # Wheel mounting plane normal in relative coordinates (same as tire axis)
        self.mounting_plane_normal_relative = self.tire_axis_relative.copy()
        
        # Wheel mounting plane center point in relative coordinates
        self.mounting_plane_center_relative = self.wheel_offset * self.tire_axis_relative
    
    def add_attachment_point(self, name: str, position: Tuple[float, float, float],
                            relative: bool = True, unit: str = 'mm') -> AttachmentPoint:
        """
        Add an attachment point to the knuckle.

        Args:
            name: Identifier for the attachment point
            position: 3D position [x, y, z]
            relative: If True, position is relative to tire center; if False, absolute
            unit: Unit of input position (default: 'mm')

        Returns:
            The created AttachmentPoint object
        """
        pos_array = to_mm(np.array(position, dtype=float), unit)

        # Convert to relative coordinates if given in absolute
        if not relative:
            pos_array = pos_array - self.tire_center

        attachment = AttachmentPoint(name, pos_array, is_relative=True, unit='mm', parent_component=self)
        self.attachment_points.append(attachment)
        return attachment
    
    def get_attachment_position(self, name: str, absolute: bool = True, unit: str = 'mm') -> np.ndarray:
        """
        Get the position of an attachment point.

        Args:
            name: Name of the attachment point
            absolute: If True, return absolute position; if False, return relative to tire center
            unit: Unit for output (default: 'mm')

        Returns:
            3D position vector in specified unit
        """
        for attachment in self.attachment_points:
            if attachment.name == name:
                # All attachments are stored in relative coordinates (in mm)
                if absolute:
                    # Transform by rotation matrix and add tire center
                    local_pos = self.rotation_matrix @ attachment.position
                    return from_mm(self.tire_center + local_pos, unit)
                else:
                    # Return relative position as-is
                    return from_mm(attachment.position.copy(), unit)

        raise ValueError(f"Attachment point '{name}' not found")
    
    def get_all_attachment_positions(self, absolute: bool = True, unit: str = 'mm') -> dict:
        """
        Get all attachment point positions.

        Args:
            absolute: If True, return absolute positions; if False, return relative to tire center
            unit: Unit for output (default: 'mm')

        Returns:
            Dictionary mapping attachment point names to positions in specified unit
        """
        return {ap.name: self.get_attachment_position(ap.name, absolute, unit)
                for ap in self.attachment_points}
    
    def update_orientation(self, toe_angle: float, camber_angle: float) -> None:
        """
        Update the angular orientation of the knuckle.
        
        Args:
            toe_angle: New toe angle in degrees
            camber_angle: New camber angle in degrees
        """
        self.toe_angle = np.radians(toe_angle)
        self.camber_angle = np.radians(camber_angle)
        self._update_geometry()
    
    def get_tire_contact_patch(self, unit: str = 'mm') -> np.ndarray:
        """
        Calculate the position of the tire contact patch (ground contact point).

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            3D position of contact patch in specified unit
        """
        # Contact patch is at z=0, directly below the tire center considering camber
        # For small camber angles, this is approximately at ground level
        contact = self.tire_center.copy()
        contact[2] = 0.0
        return from_mm(contact, unit)
    
    def get_tire_axis(self, absolute: bool = True) -> np.ndarray:
        """
        Get the tire axis vector.
        
        Args:
            absolute: If True, return in global coordinates; if False, return in local coordinates
            
        Returns:
            3D tire axis vector
        """
        if absolute:
            return self.rotation_matrix @ self.tire_axis_relative
        else:
            return self.tire_axis_relative.copy()
    
    def get_wheel_mounting_plane_center(self, absolute: bool = True, unit: str = 'mm') -> np.ndarray:
        """
        Get the wheel mounting plane center position.

        Args:
            absolute: If True, return absolute position; if False, return relative to tire center
            unit: Unit for output (default: 'mm')

        Returns:
            3D position vector in specified unit
        """
        if absolute:
            local_pos = self.rotation_matrix @ self.mounting_plane_center_relative
            return from_mm(self.tire_center + local_pos, unit)
        else:
            return from_mm(self.mounting_plane_center_relative.copy(), unit)
    
    def get_wheel_mounting_plane_normal(self, absolute: bool = True) -> np.ndarray:
        """
        Get the wheel mounting plane normal vector.

        Args:
            absolute: If True, return in global coordinates; if False, return in local coordinates

        Returns:
            3D normal vector
        """
        if absolute:
            return self.rotation_matrix @ self.mounting_plane_normal_relative
        else:
            return self.mounting_plane_normal_relative.copy()

    def get_center_of_mass(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the center of mass position.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            3D position vector in specified unit
        """
        return from_mm(self.center_of_mass.copy(), unit)

    def set_center_of_mass(self, position: Union[np.ndarray, Tuple[float, float, float]], unit: str = 'mm') -> None:
        """
        Set the center of mass position.

        Args:
            position: New center of mass position [x, y, z]
            unit: Unit of input position (default: 'mm')
        """
        self.center_of_mass = to_mm(np.array(position, dtype=float), unit)

    def fit_to_attachment_targets(self, target_positions: List[np.ndarray],
                                  unit: str = 'mm') -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Fit the knuckle position and orientation to target attachment point positions.
        Uses SVD-based rigid body transformation to minimize RMS error.

        Args:
            target_positions: List of target positions in the same order as attachment points
            unit: Unit of input target positions (default: 'mm')

        Returns:
            Tuple of (rms_error in mm, new_tire_center in mm, new_rotation_matrix)
        """
        if len(target_positions) != len(self.attachment_points):
            raise ValueError(f"Expected {len(self.attachment_points)} target positions, got {len(target_positions)}")

        # Convert to numpy array and to mm
        target_points = np.array([to_mm(np.array(p, dtype=float), unit) for p in target_positions])
        
        # Get relative attachment positions (in knuckle local frame)
        relative_points = np.array([ap.position for ap in self.attachment_points])
        
        # Compute centroids
        centroid_target = np.mean(target_points, axis=0)
        centroid_relative = np.mean(relative_points, axis=0)
        
        # Center the point sets
        target_centered = target_points - centroid_target
        relative_centered = relative_points - centroid_relative
        
        # Compute the cross-covariance matrix H = sum(relative_i * target_i^T)
        H = relative_centered.T @ target_centered
        
        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1, not -1 for reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = centroid_target - R @ centroid_relative
        
        # Calculate RMS error
        transformed_points = (R @ relative_points.T).T + t
        errors = target_points - transformed_points
        rms_error = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        
        return rms_error, t, R
    
    def update_from_attachment_targets(self, target_positions: List[np.ndarray], unit: str = 'mm') -> float:
        """
        Update knuckle position and orientation to best fit target attachment positions.
        Updates tire_center, rotation_matrix (which determines toe and camber), and center_of_mass.
        The center of mass moves as a rigid body with the knuckle.

        Args:
            target_positions: List of target positions in the same order as attachment points
            unit: Unit of input target positions (default: 'mm')

        Returns:
            RMS error of the fit (in mm)
        """
        # Store old position for transformation
        old_tire_center = self.tire_center.copy()
        old_rotation = self.rotation_matrix.copy()

        rms_error, new_center, new_rotation = self.fit_to_attachment_targets(target_positions, unit)

        # Compute the transformation: new_pos = R @ old_pos + t
        # where R = new_rotation @ old_rotation.T and t = new_center - R @ old_tire_center
        R_transform = new_rotation @ old_rotation.T
        t_transform = new_center - R_transform @ old_tire_center

        # Transform center of mass
        self.center_of_mass = R_transform @ self.center_of_mass + t_transform

        # Update tire center
        self.tire_center = new_center

        # Update rotation matrix
        self.rotation_matrix = new_rotation

        # Extract toe and camber angles from rotation matrix
        # R = R_toe @ R_camber
        # For small angles, we can extract them, but for general case:
        # camber is arcsin(R[1,2])
        # toe is arctan2(R[1,0], R[1,1])
        self.camber_angle = np.arcsin(np.clip(self.rotation_matrix[1, 2], -1, 1))
        self.toe_angle = np.arctan2(self.rotation_matrix[1, 0],
                                     np.sqrt(self.rotation_matrix[1, 1]**2 + self.rotation_matrix[1, 2]**2))

        return rms_error

    def set_steering_attachment(self, attachment_name: str) -> None:
        """
        Designate which attachment point is the steering input.

        Args:
            attachment_name: Name of the attachment point to use as steering input

        Raises:
            ValueError: If attachment point doesn't exist
        """
        # Verify the attachment exists
        found = False
        for ap in self.attachment_points:
            if ap.name == attachment_name:
                found = True
                break

        if not found:
            raise ValueError(f"Attachment point '{attachment_name}' not found")

        self.steering_attachment_name = attachment_name

    def get_steering_attachment_position(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the position of the designated steering attachment point.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Position of steering attachment in specified unit

        Raises:
            ValueError: If no steering attachment has been designated
        """
        if self.steering_attachment_name is None:
            raise ValueError("No steering attachment has been designated. Use set_steering_attachment() first.")

        return self.get_attachment_position(self.steering_attachment_name, absolute=True, unit=unit)

    def fit_to_steering_attachment_target(self, target_position: Union[np.ndarray, Tuple[float, float, float]],
                                          unit: str = 'mm') -> float:
        """
        Fit the knuckle so the steering attachment exactly matches the target position.
        The knuckle rotates and translates to place the steering attachment at the target
        while minimizing RMS error for the remaining attachment points relative to their
        previous positions.

        This is useful for steering input where the tie rod end must be at a specific
        position dictated by the steering rack, and the knuckle rotates accordingly.

        Args:
            target_position: Target position for the steering attachment
            unit: Unit of input target position (default: 'mm')

        Returns:
            RMS error for non-steering attachments (in mm)

        Raises:
            ValueError: If no steering attachment has been designated
        """
        if self.steering_attachment_name is None:
            raise ValueError("No steering attachment has been designated. Use set_steering_attachment() first.")

        # Convert target to mm
        target = to_mm(np.array(target_position, dtype=float), unit)

        # Find the steering attachment and separate from others
        steering_attachment = None
        other_attachments = []
        current_positions = {}

        for ap in self.attachment_points:
            abs_pos = self.get_attachment_position(ap.name, absolute=True, unit='mm')
            current_positions[ap.name] = abs_pos

            if ap.name == self.steering_attachment_name:
                steering_attachment = ap
            else:
                other_attachments.append(ap)

        if len(other_attachments) == 0:
            # Only steering attachment exists - just move tire center
            # We want: tire_center + R @ steering_attachment.position = target
            # If we don't change R, then: tire_center = target - R @ steering_attachment.position
            local_pos = self.rotation_matrix @ steering_attachment.position
            self.tire_center = target - local_pos
            return 0.0

        # Get steering attachment position in knuckle local frame
        steering_rel = steering_attachment.position

        # For other attachments, get their positions in knuckle local frame and absolute
        other_rel = np.array([ap.position for ap in other_attachments])
        other_curr = np.array([current_positions[ap.name] for ap in other_attachments])

        # We need to find R and t such that:
        # 1. R @ steering_rel + t = target (exact constraint)
        # 2. Minimize sum ||R @ other_rel[i] + t - other_curr[i]||^2

        # From constraint 1: t = target - R @ steering_rel
        # Substituting into 2:
        # Minimize sum ||R @ other_rel[i] + target - R @ steering_rel - other_curr[i]||^2
        # = Minimize sum ||R @ (other_rel[i] - steering_rel) - (other_curr[i] - target)||^2

        # This is a Procrustes problem: find R to minimize ||R @ A - B||^2 where:
        # A = positions relative to steering attachment in local frame
        # B = positions relative to target in global frame

        A = other_rel - steering_rel  # Relative to steering attachment in local frame
        B = other_curr - target  # Relative to target in global frame

        # SVD solution for optimal rotation
        H = A.T @ B
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Calculate translation from constraint
        t = target - R @ steering_rel

        # Store old position for transformation of center of mass
        old_tire_center = self.tire_center.copy()
        old_rotation = self.rotation_matrix.copy()

        # Compute the transformation for center of mass
        R_transform = R @ old_rotation.T
        t_transform = t - R_transform @ old_tire_center

        # Transform center of mass
        self.center_of_mass = R_transform @ self.center_of_mass + t_transform

        # Update knuckle transformation
        self.tire_center = t
        self.rotation_matrix = R

        # Extract toe and camber angles from new rotation matrix
        self.camber_angle = np.arcsin(np.clip(R[1, 2], -1, 1))
        self.toe_angle = np.arctan2(R[1, 0], np.sqrt(R[1, 1]**2 + R[1, 2]**2))

        # Calculate RMS error for non-steering attachments
        transformed = (R @ other_rel.T).T + t
        errors = other_curr - transformed
        rms_error = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

        return rms_error

    def reset_to_origin(self) -> None:
        """
        Reset the knuckle to its originally defined position and orientation.

        This restores:
        - Tire center position
        - Toe and camber angles
        - Rotation matrix
        - Center of mass position

        Note: Attachment point relative positions are not changed as they define the knuckle geometry.
        """
        self.tire_center = self._original_state['tire_center'].copy()
        self.toe_angle = self._original_state['toe_angle']
        self.camber_angle = self._original_state['camber_angle']
        self.rotation_matrix = self._original_state['rotation_matrix'].copy()
        self.center_of_mass = self._original_state['center_of_mass'].copy()

    def to_dict(self) -> dict:
        """
        Serialize the suspension knuckle to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'tire_center': self.tire_center.tolist(),  # Convert numpy array to list
            'toe_angle': float(np.degrees(self.toe_angle)),  # Store in degrees for readability
            'camber_angle': float(np.degrees(self.camber_angle)),  # Store in degrees
            'wheel_offset': float(self.wheel_offset),  # Store in mm
            'mass': float(self.mass),  # Store in kg
            'mass_unit': 'kg',
            'unit': 'mm',
            'attachment_points': [ap.to_dict() for ap in self.attachment_points],
            'steering_attachment_name': self.steering_attachment_name
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SuspensionKnuckle':
        """
        Deserialize a suspension knuckle from a dictionary.

        Args:
            data: Dictionary containing suspension knuckle data

        Returns:
            New SuspensionKnuckle instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # Extract tire center components
        tire_center = data['tire_center']

        # Create the knuckle
        knuckle = cls(
            tire_center_x=tire_center[0],
            tire_center_y=tire_center[1],
            rolling_radius=tire_center[2],
            toe_angle=data.get('toe_angle', 0.0),
            camber_angle=data.get('camber_angle', 0.0),
            wheel_offset=data.get('wheel_offset', 0.0),
            mass=data.get('mass', 0.0),
            unit=data.get('unit', 'mm'),
            mass_unit=data.get('mass_unit', 'kg')
        )

        # Add attachment points
        for ap_data in data.get('attachment_points', []):
            position = ap_data['position']
            name = ap_data['name']
            is_relative = ap_data.get('is_relative', True)
            unit = ap_data.get('unit', 'mm')
            knuckle.add_attachment_point(name, position, relative=is_relative, unit=unit)

        # Set steering attachment if specified
        if 'steering_attachment_name' in data and data['steering_attachment_name'] is not None:
            knuckle.steering_attachment_name = data['steering_attachment_name']

        return knuckle

    def __repr__(self) -> str:
        return (f"SuspensionKnuckle(\n"
                f"  tire_center={self.tire_center} mm,\n"
                f"  toe={np.degrees(self.toe_angle):.2f}°,\n"
                f"  camber={np.degrees(self.camber_angle):.2f}°,\n"
                f"  wheel_offset={self.wheel_offset:.3f} mm,\n"
                f"  mass={self.mass:.3f} kg,\n"
                f"  center_of_mass={self.center_of_mass} mm,\n"
                f"  attachments={len(self.attachment_points)}\n"
                f")")


if __name__ == "__main__":
    print("=" * 60)
    print("SUSPENSION KNUCKLE TEST (with unit support)")
    print("=" * 60)

    # Create a front left knuckle (using meters as input, stored as mm internally)
    knuckle = SuspensionKnuckle(
        tire_center_x=1.5,
        tire_center_y=0.75,
        rolling_radius=0.35,
        toe_angle=0.5,
        camber_angle=-1.0,
        wheel_offset=0.05,
        unit='m'  # Input in meters
    )

    # Add attachment points (using meters as input)
    knuckle.add_attachment_point("upper_ball_joint", [0, 0, 0.25], relative=True, unit='m')
    knuckle.add_attachment_point("lower_ball_joint", [0, 0, -0.25], relative=True, unit='m')
    knuckle.add_attachment_point("tie_rod", [0.1, -0.1, 0.0], relative=True, unit='m')

    print(knuckle)
    print("\nAttachment positions (absolute, in mm):")
    for name, pos in knuckle.get_all_attachment_positions(absolute=True, unit='mm').items():
        print(f"  {name}: {pos}")

    print("\nAttachment positions (absolute, in m):")
    for name, pos in knuckle.get_all_attachment_positions(absolute=True, unit='m').items():
        print(f"  {name}: {pos}")

    print("\nAttachment positions (relative, in mm):")
    for name, pos in knuckle.get_all_attachment_positions(absolute=False, unit='mm').items():
        print(f"  {name}: {pos}")

    print(f"\nTire axis (absolute): {knuckle.get_tire_axis(absolute=True)}")
    print(f"Wheel mounting plane center (absolute, mm): {knuckle.get_wheel_mounting_plane_center(absolute=True)}")
    print(f"Wheel mounting plane center (absolute, m): {knuckle.get_wheel_mounting_plane_center(absolute=True, unit='m')}")
    print(f"Tire contact patch (mm): {knuckle.get_tire_contact_patch()}")
    print(f"Tire contact patch (m): {knuckle.get_tire_contact_patch('m')}")

    # Test fitting to new attachment positions
    print("\n--- Testing knuckle positioning from attachment targets ---")

    original_positions = list(knuckle.get_all_attachment_positions(absolute=True, unit='m').values())
    # Create target positions in meters: translate by 50mm x, 0 y, -30mm z
    target_positions = [pos + np.array([0.05, 0.0, -0.03]) for pos in original_positions]

    print(f"\nOriginal tire center (mm): {knuckle.tire_center}")
    print(f"Original tire center (m): {from_mm(knuckle.tire_center, 'm')}")
    print(f"Original toe: {np.degrees(knuckle.toe_angle):.3f}°, camber: {np.degrees(knuckle.camber_angle):.3f}°")

    rms_error = knuckle.update_from_attachment_targets(target_positions, unit='m')

    print(f"\nAfter fitting:")
    print(f"RMS error (mm): {rms_error:.3f}")
    print(f"RMS error (m): {from_mm(rms_error, 'm'):.6f}")
    print(f"New tire center (mm): {knuckle.tire_center}")
    print(f"New tire center (m): {from_mm(knuckle.tire_center, 'm')}")
    print(f"New toe: {np.degrees(knuckle.toe_angle):.3f}°, camber: {np.degrees(knuckle.camber_angle):.3f}°")

    # Test steering attachment functionality
    print("\n--- Testing steering attachment functionality ---")

    # Designate the tie_rod attachment as the steering input
    knuckle.set_steering_attachment("tie_rod")
    print(f"\nDesignated 'tie_rod' as steering attachment")

    # Get current steering attachment position
    steering_pos_initial = knuckle.get_steering_attachment_position(unit='m')
    print(f"Initial steering attachment position (m): {steering_pos_initial}")

    # Get positions of other attachments before steering input
    upper_ball_joint_before = knuckle.get_attachment_position("upper_ball_joint", absolute=True, unit='m')
    lower_ball_joint_before = knuckle.get_attachment_position("lower_ball_joint", absolute=True, unit='m')

    print(f"Upper ball joint before (m): {upper_ball_joint_before}")
    print(f"Lower ball joint before (m): {lower_ball_joint_before}")
    print(f"Toe before: {np.degrees(knuckle.toe_angle):.3f}°")

    # Simulate steering input - move tie rod inboard by 20mm (turning left)
    steering_target = steering_pos_initial + np.array([0.0, 0.02, 0.0])  # 20mm inboard
    print(f"\nApplying steering input - moving tie rod 20mm inboard")
    print(f"Target steering position (m): {steering_target}")

    # Fit knuckle to steering target
    rms_error_steering = knuckle.fit_to_steering_attachment_target(steering_target, unit='m')

    # Check results
    steering_pos_final = knuckle.get_steering_attachment_position(unit='m')
    upper_ball_joint_after = knuckle.get_attachment_position("upper_ball_joint", absolute=True, unit='m')
    lower_ball_joint_after = knuckle.get_attachment_position("lower_ball_joint", absolute=True, unit='m')

    print(f"\nAfter steering input:")
    print(f"Final steering attachment position (m): {steering_pos_final}")
    print(f"Steering position error: {np.linalg.norm(steering_pos_final - steering_target):.9f} m (should be ~0)")
    print(f"Upper ball joint after (m): {upper_ball_joint_after}")
    print(f"Lower ball joint after (m): {lower_ball_joint_after}")
    print(f"Upper ball joint movement (mm): {from_mm(np.linalg.norm(knuckle.tire_center + knuckle.rotation_matrix @ knuckle.attachment_points[0].position - to_mm(upper_ball_joint_before, 'm')), 'mm'):.3f}")
    print(f"Lower ball joint movement (mm): {from_mm(np.linalg.norm(knuckle.tire_center + knuckle.rotation_matrix @ knuckle.attachment_points[1].position - to_mm(lower_ball_joint_before, 'm')), 'mm'):.3f}")
    print(f"Toe after: {np.degrees(knuckle.toe_angle):.3f}°")
    print(f"Toe change: {np.degrees(knuckle.toe_angle - np.radians(0.5)):.3f}° (from initial 0.5°)")
    print(f"RMS error for other attachments (mm): {rms_error_steering:.3f}")

    print("\n✓ All tests completed successfully!")
