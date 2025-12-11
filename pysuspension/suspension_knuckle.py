import numpy as np
from typing import List, Tuple, Optional, Union
from .rigid_body import RigidBody
from .attachment_point import AttachmentPoint
from .units import to_mm, from_mm


class SuspensionKnuckle(RigidBody):
    """
    Represents a suspension knuckle with geometric properties.

    Extends RigidBody to provide rigid body transformation behavior.
    All positions are stored internally in millimeters (mm).
    All attachment points use absolute positioning.

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
                 mass_unit: str = 'kg',
                 name: str = 'knuckle'):
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
            name: Identifier for the knuckle (default: 'knuckle')
        """
        # Initialize parent RigidBody
        super().__init__(name=name, mass=mass, mass_unit=mass_unit)

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

        # Compute rotation matrix and tire axis
        self._update_geometry()

        # Set center of mass to tire center initially
        self.center_of_mass = self.tire_center.copy()

        # Store knuckle-specific original state (parent handles attachment points)
        self._knuckle_original_state = {
            'tire_center': self.tire_center.copy(),
            'toe_angle': self.toe_angle,
            'camber_angle': self.camber_angle,
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
    
    def add_attachment_point(self,
                            name_or_attachment: Union[str, AttachmentPoint],
                            position: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None,
                            unit: str = 'mm') -> AttachmentPoint:
        """
        Add an attachment point to the knuckle (absolute positioning only).

        Can be called in two ways:
        1. Pass an existing AttachmentPoint object:
           add_attachment_point(attachment_point)
        2. Create a new AttachmentPoint from parameters:
           add_attachment_point(name, position, unit='mm')

        Args:
            name_or_attachment: Either an AttachmentPoint object or a name string
            position: 3D position [x, y, z] in absolute coordinates (required if name_or_attachment is a string)
            unit: Unit of input position (default: 'mm')

        Returns:
            The AttachmentPoint object (either the one passed in or newly created)
        """
        # Use parent's add_attachment_point (which handles both signatures)
        return super().add_attachment_point(name_or_attachment, position, unit=unit)
    
    def get_attachment_position(self, name: str, unit: str = 'mm') -> np.ndarray:
        """
        Get the position of an attachment point (always absolute).

        Args:
            name: Name of the attachment point
            unit: Unit for output (default: 'mm')

        Returns:
            3D position vector in specified unit
        """
        attachment = self.get_attachment_point(name)
        if attachment is None:
            raise ValueError(f"Attachment point '{name}' not found")
        return attachment.get_position(unit)
    
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

        The contact patch is located at a distance of rolling_radius from the tire center,
        along a line perpendicular to the tire axis and constrained to the YZ plane
        (maintaining the same X coordinate as the tire center). This accounts for
        camber angle when determining the contact patch location.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            3D position of contact patch in specified unit
        """
        # Get the tire axis in global coordinates
        tire_axis = self.get_tire_axis()  # [a_x, a_y, a_z]

        # Find direction perpendicular to tire axis, constrained to YZ plane (X = 0)
        # This direction is [0, -a_z, a_y], which is perpendicular to tire axis
        # because: [0, -a_z, a_y] · [a_x, a_y, a_z] = 0*a_x + (-a_z)*a_y + a_y*a_z = 0
        direction = np.array([0.0, -tire_axis[2], tire_axis[1]])

        # Normalize the direction vector
        direction_magnitude = np.linalg.norm(direction)
        if direction_magnitude < 1e-10:
            # Tire axis is purely in X direction (shouldn't happen normally)
            # Fall back to straight down
            direction = np.array([0.0, 0.0, -1.0])
        else:
            direction = direction / direction_magnitude

        # Ensure direction points downward (negative Z component)
        # If it points upward, flip it
        if direction[2] > 0:
            direction = -direction

        # Calculate contact patch position
        # Start at tire center, move rolling_radius distance in the perpendicular direction
        rolling_radius_mm = self.tire_center[2]  # Stored as Z coordinate of tire center
        contact = self.tire_center + rolling_radius_mm * direction

        return from_mm(contact, unit)
    
    def get_tire_axis(self) -> np.ndarray:
        """
        Get the tire axis vector (in global coordinates).

        Returns:
            3D tire axis vector
        """
        return self.rotation_matrix @ self.tire_axis_relative

    def get_wheel_mounting_plane_center(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the wheel mounting plane center position (absolute).

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            3D position vector in specified unit
        """
        local_pos = self.rotation_matrix @ self.mounting_plane_center_relative
        return from_mm(self.tire_center + local_pos, unit)

    def get_wheel_mounting_plane_normal(self) -> np.ndarray:
        """
        Get the wheel mounting plane normal vector (in global coordinates).

        Returns:
            3D normal vector
        """
        return self.rotation_matrix @ self.mounting_plane_normal_relative

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
        # Store old tire center for transformation
        old_tire_center = self.tire_center.copy()

        # Use parent's fit_to_attachment_targets (which applies the transformation)
        # This updates all attachment points, center_of_mass, and rotation_matrix
        rms_error = super().fit_to_attachment_targets(target_positions, unit=unit)

        # Calculate how the centroid moved (tire center needs to move similarly)
        centroid_delta = self.centroid - old_tire_center

        # Update tire center to maintain its relationship with the knuckle
        self.tire_center = self.tire_center + centroid_delta

        # Extract toe and camber angles from rotation matrix
        # R = R_toe @ R_camber
        # For small angles, we can extract them, but for general case:
        # camber is arcsin(R[1,2])
        # toe is arctan2(R[1,0], sqrt(R[1,1]^2 + R[1,2]^2))
        self.camber_angle = np.arcsin(np.clip(self.rotation_matrix[1, 2], -1, 1))
        self.toe_angle = np.arctan2(self.rotation_matrix[1, 0],
                                     np.sqrt(self.rotation_matrix[1, 1]**2 + self.rotation_matrix[1, 2]**2))

        return rms_error

    def reset_to_origin(self) -> None:
        """
        Reset the knuckle to its originally defined position and orientation.

        This restores:
        - All attachment point positions (via parent)
        - Tire center position
        - Toe and camber angles
        - Rotation matrix
        - Center of mass position
        - Centroid
        """
        # Call parent's reset_to_origin to restore attachment points
        super().reset_to_origin()

        # Restore knuckle-specific state
        self.tire_center = self._knuckle_original_state['tire_center'].copy()
        self.toe_angle = self._knuckle_original_state['toe_angle']
        self.camber_angle = self._knuckle_original_state['camber_angle']

        # Recompute rotation matrix from toe and camber
        self._update_geometry()

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
            'attachment_points': [ap.to_dict() for ap in self.attachment_points]
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
            mass_unit=data.get('mass_unit', 'kg'),
            name=data.get('name', 'knuckle')
        )

        # Add attachment points (all absolute positioning now)
        for ap_data in data.get('attachment_points', []):
            position = ap_data['position']
            name = ap_data['name']
            unit = ap_data.get('unit', 'mm')
            # Note: Old data may have 'is_relative' flag, but we ignore it now
            # All attachments are absolute in the refactored version
            knuckle.add_attachment_point(name, position, unit=unit)

        return knuckle

    def __repr__(self) -> str:
        centroid_str = f"{self.centroid} mm" if self.centroid is not None else "None"
        com_str = f"{self.center_of_mass} mm" if self.center_of_mass is not None else "None"
        return (f"SuspensionKnuckle('{self.name}',\n"
                f"  tire_center={self.tire_center} mm,\n"
                f"  toe={np.degrees(self.toe_angle):.2f}°,\n"
                f"  camber={np.degrees(self.camber_angle):.2f}°,\n"
                f"  wheel_offset={self.wheel_offset:.3f} mm,\n"
                f"  mass={self.mass:.3f} kg,\n"
                f"  centroid={centroid_str},\n"
                f"  center_of_mass={com_str},\n"
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

    # Add attachment points (using absolute positioning, in meters)
    # Calculate absolute positions by adding tire center offset
    tire_center_m = np.array([1.5, 0.75, 0.35])
    knuckle.add_attachment_point("upper_ball_joint", tire_center_m + np.array([0, 0, 0.25]), unit='m')
    knuckle.add_attachment_point("lower_ball_joint", tire_center_m + np.array([0, 0, -0.25]), unit='m')
    knuckle.add_attachment_point("tie_rod", tire_center_m + np.array([0.1, -0.1, 0.0]), unit='m')

    print(knuckle)
    print("\nAttachment positions (in mm):")
    for ap in knuckle.attachment_points:
        print(f"  {ap.name}: {ap.get_position(unit='mm')}")

    print("\nAttachment positions (in m):")
    for ap in knuckle.attachment_points:
        print(f"  {ap.name}: {ap.get_position(unit='m')}")

    print(f"\nTire axis: {knuckle.get_tire_axis()}")
    print(f"Wheel mounting plane center (mm): {knuckle.get_wheel_mounting_plane_center()}")
    print(f"Wheel mounting plane center (m): {knuckle.get_wheel_mounting_plane_center(unit='m')}")
    print(f"Tire contact patch (mm): {knuckle.get_tire_contact_patch()}")
    print(f"Tire contact patch (m): {knuckle.get_tire_contact_patch('m')}")

    # Test fitting to new attachment positions
    print("\n--- Testing knuckle positioning from attachment targets ---")

    original_positions = knuckle.get_all_attachment_positions(unit='m')
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

    # Test inheritance
    print("\n--- Testing inheritance ---")
    print(f"isinstance(knuckle, RigidBody): {isinstance(knuckle, RigidBody)}")

    print("\n✓ All tests completed successfully!")
