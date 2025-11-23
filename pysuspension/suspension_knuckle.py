import numpy as np
from typing import List, Tuple, Optional
from attachment_point import AttachmentPoint
from units import to_mm, from_mm


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
                 unit: str = 'mm'):
        """
        Initialize suspension knuckle.

        Args:
            tire_center_x: Longitudinal position (+ front, - rear)
            tire_center_y: Lateral position / track width (+ left, - right)
            rolling_radius: Vertical position (tire rolling radius)
            toe_angle: Toe setting in degrees (+ leading edge out)
            camber_angle: Camber angle in degrees (+ top leans out)
            wheel_offset: Offset of wheel mounting plane from tire center (+ outward)
            unit: Unit of input positions (default: 'mm')
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
        
        # Attachment points list
        self.attachment_points: List[AttachmentPoint] = []
        
        # Compute rotation matrix and tire axis
        self._update_geometry()
    
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
                            relative: bool = True, unit: str = 'mm') -> None:
        """
        Add an attachment point to the knuckle.

        Args:
            name: Identifier for the attachment point
            position: 3D position [x, y, z]
            relative: If True, position is relative to tire center; if False, absolute
            unit: Unit of input position (default: 'mm')
        """
        pos_array = to_mm(np.array(position, dtype=float), unit)

        # Convert to relative coordinates if given in absolute
        if not relative:
            pos_array = pos_array - self.tire_center

        attachment = AttachmentPoint(name, pos_array, is_relative=True, unit='mm')
        self.attachment_points.append(attachment)
    
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
        Updates tire_center and rotation_matrix (which determines toe and camber).

        Args:
            target_positions: List of target positions in the same order as attachment points
            unit: Unit of input target positions (default: 'mm')

        Returns:
            RMS error of the fit (in mm)
        """
        rms_error, new_center, new_rotation = self.fit_to_attachment_targets(target_positions, unit)
        
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
    
    def __repr__(self) -> str:
        return (f"SuspensionKnuckle(\n"
                f"  tire_center={self.tire_center} mm,\n"
                f"  toe={np.degrees(self.toe_angle):.2f}°,\n"
                f"  camber={np.degrees(self.camber_angle):.2f}°,\n"
                f"  wheel_offset={self.wheel_offset:.3f} mm,\n"
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

    print("\n✓ All tests completed successfully!")
