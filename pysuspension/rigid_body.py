"""
Base class for rigid body components in suspension systems.

This module provides a RigidBody base class that encapsulates common behavior
for components that maintain a fixed set of attachment points and move as a
single rigid unit (preserving distances and angles between points).

Uses SVD-based Kabsch algorithm for optimal rigid body transformation fitting.
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from .attachment_point import AttachmentPoint
from .units import to_mm, from_mm, to_kg


class RigidBody:
    """
    Base class for components with rigid body behavior.

    A rigid body maintains a set of attachment points that move together,
    preserving distances and angles between them. All attachment points
    use absolute positioning (not relative).

    Provides automatic rigid body transformation (translation + rotation) to
    fit target positions using the SVD-based Kabsch algorithm for optimal fit.

    All positions are stored internally in millimeters (mm).
    Mass is stored internally in kilograms (kg).

    Subclasses:
    - ControlArm: Rigid body with multiple links
    - Chassis: Rigid body with corners and axles
    - SuspensionKnuckle: Rigid body with tire geometry
    - SteeringRack: Housing is a rigid body
    """

    def __init__(self, name: str = "rigid_body", mass: float = 0.0, mass_unit: str = 'kg'):
        """
        Initialize a rigid body.

        Args:
            name: Identifier for the rigid body
            mass: Mass of the rigid body (default: 0.0)
            mass_unit: Unit of input mass (default: 'kg')
        """
        self.name = name
        self.mass = to_kg(mass, mass_unit)

        # Attachment points (all use absolute positioning)
        self.attachment_points: List[AttachmentPoint] = []

        # Geometric properties
        self.centroid: Optional[np.ndarray] = None
        self.center_of_mass: Optional[np.ndarray] = None
        self.rotation_matrix = np.eye(3)

        # State management for reset functionality
        self._original_state = {
            'attachment_points': [],  # List of AttachmentPoint copies
            'centroid': None,
            'center_of_mass': None,
            'rotation_matrix': self.rotation_matrix.copy(),
        }

        # Flag to freeze original state after first transformation
        self._original_state_frozen = False

    def add_attachment_point(self,
                            name: str,
                            position: Union[np.ndarray, Tuple[float, float, float]],
                            unit: str = 'mm') -> AttachmentPoint:
        """
        Add an attachment point to this rigid body.

        Args:
            name: Identifier for the attachment point
            position: 3D position [x, y, z] in absolute coordinates
            unit: Unit of input position (default: 'mm')

        Returns:
            The created AttachmentPoint object
        """
        attachment = AttachmentPoint(
            name=name,
            position=position,
            is_relative=False,  # All rigid body attachments use absolute positioning
            unit=unit,
            parent_component=self
        )
        self.attachment_points.append(attachment)

        # Store original attachment point (copy without connections)
        if not self._original_state_frozen:
            self._original_state['attachment_points'].append(attachment.copy())

        self._update_centroid()
        return attachment

    def get_attachment_point(self, name: str) -> Optional[AttachmentPoint]:
        """
        Get an attachment point by name.

        Args:
            name: Name of the attachment point

        Returns:
            AttachmentPoint object if found, None otherwise
        """
        for attachment in self.attachment_points:
            if attachment.name == name:
                return attachment
        return None

    def get_all_attachment_points(self) -> List[AttachmentPoint]:
        """
        Get all attachment point objects.

        Returns:
            List of all AttachmentPoint objects
        """
        return self.attachment_points.copy()

    def get_all_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get all attachment point positions.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of attachment positions in specified unit
        """
        return [attachment.get_position(unit) for attachment in self.attachment_points]

    def _update_centroid(self) -> None:
        """
        Update the centroid and center of mass of all attachment points.

        The center of mass is located at the geometric center of all attachment points
        (can be overridden by subclasses for custom behavior).

        This method can be overridden by subclasses that need to gather attachment
        points from multiple sources (e.g., Chassis gathering from corners).
        """
        if not self.attachment_points:
            self.centroid = np.zeros(3)
            self.center_of_mass = np.zeros(3)
        else:
            positions = [ap.position for ap in self.attachment_points]
            self.centroid = np.mean(positions, axis=0)
            self.center_of_mass = self.centroid.copy()

        # Update original state only if not frozen (i.e., during setup, before transformations)
        if not self._original_state_frozen:
            self._original_state['centroid'] = self.centroid.copy() if self.centroid is not None else None
            self._original_state['center_of_mass'] = self.center_of_mass.copy() if self.center_of_mass is not None else None

    def get_centroid(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the geometric center (centroid) of all attachment points.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            3D position vector in specified unit
        """
        if self.centroid is None:
            return from_mm(np.zeros(3), unit)
        return from_mm(self.centroid.copy(), unit)

    def get_center_of_mass(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the center of mass position.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            3D position vector in specified unit
        """
        if self.center_of_mass is None:
            return from_mm(np.zeros(3), unit)
        return from_mm(self.center_of_mass.copy(), unit)

    def set_center_of_mass(self,
                          position: Union[np.ndarray, Tuple[float, float, float]],
                          unit: str = 'mm') -> None:
        """
        Set the center of mass position.

        Args:
            position: New center of mass position [x, y, z]
            unit: Unit of input position (default: 'mm')
        """
        self.center_of_mass = to_mm(np.array(position, dtype=float), unit)

    def fit_to_attachment_targets(self,
                                  target_positions: List[Union[np.ndarray, Tuple[float, float, float]]],
                                  unit: str = 'mm') -> float:
        """
        Fit the rigid body to target attachment positions using SVD-based transformation.

        Uses the Kabsch algorithm to find the optimal rotation and translation that
        minimizes RMS error between current and target positions. Updates all attachment
        point positions and center of mass based on the optimal rigid body fit.

        The center of mass moves as a rigid body with the attachment points.

        Args:
            target_positions: List of target positions in same order as get_all_attachment_positions()
            unit: Unit of input target positions (default: 'mm')

        Returns:
            RMS error of the fit (in mm)

        Raises:
            ValueError: If wrong number of target positions or fewer than 3 points
        """
        # Freeze original state on first transformation
        self._original_state_frozen = True

        current_positions = self.get_all_attachment_positions(unit='mm')

        if len(target_positions) != len(current_positions):
            raise ValueError(f"Expected {len(current_positions)} target positions, got {len(target_positions)}")

        if len(current_positions) < 3:
            raise ValueError("Need at least 3 attachment points for rigid body fit")

        # Convert to numpy arrays and to mm
        current_points = np.array(current_positions)
        target_points = np.array([to_mm(np.array(p, dtype=float), unit) for p in target_positions])

        # Compute centroids
        centroid_current = np.mean(current_points, axis=0)
        centroid_target = np.mean(target_points, axis=0)

        # Center the point sets
        current_centered = current_points - centroid_current
        target_centered = target_points - centroid_target

        # Compute the cross-covariance matrix H
        H = current_centered.T @ target_centered

        # Singular Value Decomposition (Kabsch algorithm)
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1, not -1 for reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = centroid_target - R @ centroid_current

        # Apply transformation to all attachment points and center of mass
        self._apply_transformation(R, t)

        # Calculate RMS error
        transformed_points = (R @ current_points.T).T + t
        errors = target_points - transformed_points
        rms_error = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

        return rms_error

    def _apply_transformation(self, R: np.ndarray, t: np.ndarray) -> None:
        """
        Apply rigid body transformation to all attachment points and center of mass.

        This is a protected method that can be overridden by subclasses to customize
        what gets transformed (e.g., Chassis needs to transform corners).

        Args:
            R: 3x3 rotation matrix
            t: 3D translation vector (in mm)
        """
        # Transform center of mass (if it exists)
        if self.center_of_mass is not None:
            self.center_of_mass = R @ self.center_of_mass + t

        # Update all attachment point positions
        for attachment in self.attachment_points:
            new_pos = R @ attachment.position + t
            attachment.set_position(new_pos, unit='mm')

        # Update rotation matrix
        self.rotation_matrix = R
        self._update_centroid()

    def transform(self, rotation_matrix: np.ndarray, translation: np.ndarray) -> None:
        """
        Apply an explicit rigid body transformation (rotation + translation).

        Args:
            rotation_matrix: 3x3 rotation matrix
            translation: 3D translation vector (in mm)
        """
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")

        if translation.shape != (3,):
            raise ValueError("Translation must be a 3D vector")

        self._apply_transformation(rotation_matrix, translation)

    def translate(self,
                 translation: Union[np.ndarray, Tuple[float, float, float]],
                 unit: str = 'mm') -> None:
        """
        Translate the rigid body by a given vector.

        Args:
            translation: Translation vector [dx, dy, dz]
            unit: Unit of input translation (default: 'mm')
        """
        t = to_mm(np.array(translation, dtype=float), unit)
        R = np.eye(3)  # Identity rotation
        self._apply_transformation(R, t)

    def rotate_about_center(self, rotation_matrix: np.ndarray) -> None:
        """
        Rotate the rigid body about its center of mass.

        Args:
            rotation_matrix: 3x3 rotation matrix
        """
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")

        if self.center_of_mass is None:
            raise ValueError("Center of mass not defined")

        # Translate to origin, rotate, translate back
        # This is equivalent to: new_pos = R @ (old_pos - com) + com
        for attachment in self.attachment_points:
            new_pos = self.center_of_mass + rotation_matrix @ (attachment.position - self.center_of_mass)
            attachment.set_position(new_pos, unit='mm')

        # Update rotation matrix
        self.rotation_matrix = rotation_matrix @ self.rotation_matrix
        self._update_centroid()

    def reset_to_origin(self) -> None:
        """
        Reset the rigid body to its originally defined position.

        This restores:
        - All attachment point positions
        - Centroid and center of mass (recalculated from restored positions)
        - Rotation matrix
        """
        # Restore attachment point positions from original state
        for i, attachment in enumerate(self.attachment_points):
            if i < len(self._original_state['attachment_points']):
                original = self._original_state['attachment_points'][i]
                attachment.set_position(original.position, unit='mm')

        # Restore transformation
        self.rotation_matrix = self._original_state['rotation_matrix'].copy()

        # Unfreeze original state to allow subsequent transformations
        self._original_state_frozen = False

        # Recalculate centroid and center of mass from restored attachment positions
        self._update_centroid()

    def save_state(self) -> None:
        """
        Save current state as the original state.

        Useful after building up the rigid body geometry.
        """
        self._original_state['attachment_points'] = [ap.copy() for ap in self.attachment_points]
        self._original_state['centroid'] = self.centroid.copy() if self.centroid is not None else None
        self._original_state['center_of_mass'] = self.center_of_mass.copy() if self.center_of_mass is not None else None
        self._original_state['rotation_matrix'] = self.rotation_matrix.copy()

    def to_dict(self) -> dict:
        """
        Serialize the rigid body to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'name': self.name,
            'mass': float(self.mass),  # Store in kg
            'mass_unit': 'kg',
            'attachment_points': [ap.to_dict() for ap in self.attachment_points]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RigidBody':
        """
        Deserialize a rigid body from a dictionary.

        Args:
            data: Dictionary containing rigid body data

        Returns:
            New RigidBody instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # Create the rigid body
        rigid_body = cls(
            name=data['name'],
            mass=data.get('mass', 0.0),
            mass_unit=data.get('mass_unit', 'kg')
        )

        # Add attachment points
        for ap_data in data.get('attachment_points', []):
            position = ap_data['position']
            name = ap_data['name']
            unit = ap_data.get('unit', 'mm')
            rigid_body.add_attachment_point(name, position, unit=unit)

        return rigid_body

    def __repr__(self) -> str:
        centroid_str = f"{self.centroid} mm" if self.centroid is not None else "None"
        com_str = f"{self.center_of_mass} mm" if self.center_of_mass is not None else "None"
        return (f"RigidBody('{self.name}',\n"
                f"  attachments={len(self.attachment_points)},\n"
                f"  mass={self.mass:.3f} kg,\n"
                f"  centroid={centroid_str},\n"
                f"  center_of_mass={com_str}\n"
                f")")


if __name__ == "__main__":
    print("=" * 70)
    print("RIGID BODY BASE CLASS TEST")
    print("=" * 70)

    # Create a simple rigid body with 4 attachment points
    body = RigidBody(name="test_body", mass=10.0, mass_unit='kg')

    # Add attachment points (corners of a square in XY plane)
    body.add_attachment_point("corner1", [0, 0, 0], unit='mm')
    body.add_attachment_point("corner2", [100, 0, 0], unit='mm')
    body.add_attachment_point("corner3", [100, 100, 0], unit='mm')
    body.add_attachment_point("corner4", [0, 100, 0], unit='mm')

    print(f"\n{body}")

    print("\n--- Initial State ---")
    print(f"Centroid: {body.get_centroid()} mm")
    print(f"Center of mass: {body.get_center_of_mass()} mm")
    print(f"Number of attachment points: {len(body.attachment_points)}")

    print("\n--- Testing Translation ---")
    body.translate([50, 50, 100], unit='mm')
    print(f"After translation [50, 50, 100] mm:")
    print(f"  Centroid: {body.get_centroid()} mm")
    print(f"  Corner1: {body.get_attachment_point('corner1').get_position()} mm")

    print("\n--- Testing Rotation ---")
    # Rotate 45 degrees about Z axis
    angle = np.radians(45)
    R_z = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    body.rotate_about_center(R_z)
    print(f"After 45° rotation about center:")
    print(f"  Centroid: {body.get_centroid()} mm")

    print("\n--- Testing Fit to Targets ---")
    # Create target positions (translate by [10, 20, 30])
    original_positions = body.get_all_attachment_positions()
    target_positions = [pos + np.array([10, 20, 30]) for pos in original_positions]

    rms_error = body.fit_to_attachment_targets(target_positions, unit='mm')
    print(f"RMS error: {rms_error:.6f} mm")
    print(f"New centroid: {body.get_centroid()} mm")

    print("\n--- Testing Reset ---")
    body.reset_to_origin()
    print(f"After reset:")
    print(f"  Centroid: {body.get_centroid()} mm")
    print(f"  Corner1: {body.get_attachment_point('corner1').get_position()} mm")

    print("\n✓ All tests completed successfully!")
