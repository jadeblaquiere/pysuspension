import numpy as np
from typing import List, Tuple, Union, Optional
from suspension_link import SuspensionLink
from attachment_point import AttachmentPoint
from units import to_mm, from_mm, to_kg


class ControlArm:
    """
    Represents a control arm consisting of multiple suspension links and attachment points
    that move together as a rigid body.

    All positions are stored internally in millimeters (mm).
    """

    def __init__(self, name: str = "control_arm", mass: float = 0.0, mass_unit: str = 'kg'):
        """
        Initialize a control arm.

        Args:
            name: Identifier for the control arm
            mass: Mass of the control arm (default: 0.0)
            mass_unit: Unit of input mass (default: 'kg')
        """
        self.name = name
        self.mass = to_kg(mass, mass_unit)
        self.links: List[SuspensionLink] = []
        self.attachment_points: List[AttachmentPoint] = []  # All attachment points (not from links)

        # Rigid body transformation (identity initially)
        self.centroid = None
        self.center_of_mass = None
        self.rotation_matrix = np.eye(3)

        # Store original state for reset (updated when links/attachments are added)
        self._original_state = {
            'link_endpoints': [],  # List of (endpoint1, endpoint2) tuples
            'attachment_points': [],  # List of AttachmentPoint copies
            'centroid': None,
            'center_of_mass': None,
            'rotation_matrix': self.rotation_matrix.copy(),
        }

        # Flag to freeze original state after first transformation
        self._original_state_frozen = False
        
    def add_link(self, link: SuspensionLink) -> None:
        """
        Add a suspension link to the control arm.

        Args:
            link: SuspensionLink to add
        """
        self.links.append(link)
        # Store original endpoint positions
        self._original_state['link_endpoints'].append((link.endpoint1.position.copy(), link.endpoint2.position.copy()))
        self._update_centroid()
    
    def add_attachment_point(self, name: str, position: Union[np.ndarray, Tuple[float, float, float]],
                            unit: str = 'mm') -> AttachmentPoint:
        """
        Add an attachment point to the control arm.

        Args:
            name: Identifier for the attachment point
            position: 3D position [x, y, z]
            unit: Unit of input position (default: 'mm')

        Returns:
            The created AttachmentPoint object
        """
        attachment = AttachmentPoint(
            name=name,
            position=position,
            is_relative=False,  # Control arm attachment points are in absolute coordinates
            unit=unit,
            parent_component=self
        )
        self.attachment_points.append(attachment)
        # Store original attachment point (copy without connections)
        self._original_state['attachment_points'].append(attachment.copy())
        self._update_centroid()
        return attachment
    
    def _update_centroid(self) -> None:
        """Update the centroid and center of mass of all attachment points.
        The center of mass is located at the geometric center of all attachment points."""
        all_points = []

        # Collect all link endpoints
        for link in self.links:
            all_points.append(link.endpoint1.position)
            all_points.append(link.endpoint2.position)

        # Collect attachment point positions
        for attachment in self.attachment_points:
            all_points.append(attachment.position)

        if all_points:
            self.centroid = np.mean(all_points, axis=0)
            self.center_of_mass = self.centroid.copy()
        else:
            self.centroid = np.zeros(3)
            self.center_of_mass = np.zeros(3)

        # Update original state only if not frozen (i.e., during setup, before transformations)
        # This ensures original state reflects all attachments added during setup
        if not self._original_state_frozen:
            self._original_state['centroid'] = self.centroid.copy() if self.centroid is not None else None
            self._original_state['center_of_mass'] = self.center_of_mass.copy() if self.center_of_mass is not None else None
    
    def get_all_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get all unique attachment point positions.
        Link endpoints and additional attachments are included, with duplicates removed.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of unique attachment positions in specified unit
        """
        positions = []
        position_set = []  # For duplicate checking (in mm)

        # Add link endpoints (already in mm internally)
        for link in self.links:
            for endpoint in link.get_endpoints(unit='mm'):
                # Check if this position is already in the list (within tolerance)
                is_duplicate = False
                for existing_pos in position_set:
                    if np.linalg.norm(endpoint - existing_pos) < 1e-6:  # Adjusted tolerance for mm
                        is_duplicate = True
                        break

                if not is_duplicate:
                    positions.append(endpoint.copy())
                    position_set.append(endpoint.copy())

        # Add attachment points (already in mm)
        for attachment in self.attachment_points:
            pos = attachment.position
            # Check for duplicates
            is_duplicate = False
            for existing_pos in position_set:
                if np.linalg.norm(pos - existing_pos) < 1e-6:  # Adjusted tolerance for mm
                    is_duplicate = True
                    break

            if not is_duplicate:
                positions.append(pos.copy())
                position_set.append(pos.copy())

        # Convert all positions to requested unit
        return [from_mm(pos, unit) for pos in positions]

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

    def set_center_of_mass(self, position: Union[np.ndarray, Tuple[float, float, float]], unit: str = 'mm') -> None:
        """
        Set the center of mass position.

        Args:
            position: New center of mass position [x, y, z]
            unit: Unit of input position (default: 'mm')
        """
        self.center_of_mass = to_mm(np.array(position, dtype=float), unit)

    def fit_to_attachment_targets(self, target_positions: List[np.ndarray], unit: str = 'mm') -> float:
        """
        Fit the control arm to target attachment positions using rigid body transformation.
        Updates all link positions and center of mass based on the optimal rigid body fit.
        The center of mass moves as a rigid body with the control arm.

        Args:
            target_positions: List of target positions in same order as get_all_attachment_positions()
            unit: Unit of input target positions (default: 'mm')

        Returns:
            RMS error of the fit (in mm)
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

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = centroid_target - R @ centroid_current

        # Transform center of mass (if it exists)
        if self.center_of_mass is not None:
            self.center_of_mass = R @ self.center_of_mass + t

        # Update all link positions
        for link in self.links:
            new_endpoint1 = R @ link.endpoint1.position + t
            new_endpoint2 = R @ link.endpoint2.position + t
            link.endpoint1.set_position(new_endpoint1, unit='mm')
            link.endpoint2.set_position(new_endpoint2, unit='mm')
            link._update_local_frame()

        # Update attachment point positions
        for attachment in self.attachment_points:
            new_pos = R @ attachment.position + t
            attachment.set_position(new_pos, unit='mm')

        # Update control arm transformation
        self.rotation_matrix = R
        self._update_centroid()

        # Calculate RMS error
        transformed_points = (R @ current_points.T).T + t
        errors = target_points - transformed_points
        rms_error = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

        return rms_error

    def reset_to_origin(self) -> None:
        """
        Reset the control arm to its originally defined position.

        This restores:
        - All link endpoint positions
        - All additional attachment positions
        - Centroid and center of mass
        - Rotation matrix
        """
        # Restore link endpoints
        for i, link in enumerate(self.links):
            if i < len(self._original_state['link_endpoints']):
                endpoint1, endpoint2 = self._original_state['link_endpoints'][i]
                link.endpoint1.set_position(endpoint1.copy(), unit='mm')
                link.endpoint2.set_position(endpoint2.copy(), unit='mm')
                link._update_local_frame()

        # Restore attachment point positions from original state
        for i, attachment in enumerate(self.attachment_points):
            if i < len(self._original_state['attachment_points']):
                original = self._original_state['attachment_points'][i]
                attachment.set_position(original.position, unit='mm')

        # Restore transformation
        self.rotation_matrix = self._original_state['rotation_matrix'].copy()

        # Restore centroid and center of mass
        if self._original_state['centroid'] is not None:
            self.centroid = self._original_state['centroid'].copy()
        if self._original_state['center_of_mass'] is not None:
            self.center_of_mass = self._original_state['center_of_mass'].copy()

        # Unfreeze original state to allow subsequent transformations
        self._original_state_frozen = False

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

    def to_dict(self) -> dict:
        """
        Serialize the control arm to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'name': self.name,
            'mass': float(self.mass),  # Store in kg
            'mass_unit': 'kg',
            'links': [link.to_dict() for link in self.links],
            'attachment_points': [ap.to_dict() for ap in self.attachment_points]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ControlArm':
        """
        Deserialize a control arm from a dictionary.

        Args:
            data: Dictionary containing control arm data

        Returns:
            New ControlArm instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # Create the control arm
        control_arm = cls(
            name=data['name'],
            mass=data.get('mass', 0.0),
            mass_unit=data.get('mass_unit', 'kg')
        )

        # Add links
        for link_data in data.get('links', []):
            link = SuspensionLink.from_dict(link_data)
            control_arm.add_link(link)

        # Add attachment points
        for ap_data in data.get('attachment_points', []):
            # Get position from the attachment point data
            position = ap_data['position']
            unit = ap_data.get('unit', 'mm')
            name = ap_data['name']
            control_arm.add_attachment_point(name, position, unit=unit)

        return control_arm

    def __repr__(self) -> str:
        centroid_str = f"{self.centroid} mm" if self.centroid is not None else "None"
        com_str = f"{self.center_of_mass} mm" if self.center_of_mass is not None else "None"
        return (f"ControlArm('{self.name}',\n"
                f"  links={len(self.links)},\n"
                f"  additional_attachments={len(self.additional_attachments)},\n"
                f"  total_unique_attachments={len(self.get_all_attachment_positions())},\n"
                f"  mass={self.mass:.3f} kg,\n"
                f"  centroid={centroid_str},\n"
                f"  center_of_mass={com_str}\n"
                f")")


if __name__ == "__main__":
    print("=" * 60)
    print("CONTROL ARM TEST (with unit support)")
    print("=" * 60)

    # Create a control arm with multiple links (using meters as input, stored as mm internally)
    control_arm = ControlArm(name="test_control_arm")

    # Add two links (forming a triangulated control arm)
    link1 = SuspensionLink(
        endpoint1=[1.3, 0.4, 0.55],   # Front chassis mount
        endpoint2=[1.5, 0.75, 0.6],   # Ball joint
        name="front_link",
        unit='m'  # Input in meters
    )

    link2 = SuspensionLink(
        endpoint1=[1.2, 0.4, 0.55],   # Rear chassis mount
        endpoint2=[1.5, 0.75, 0.6],   # Ball joint (shared with link1)
        name="rear_link",
        unit='m'  # Input in meters
    )

    control_arm.add_link(link1)
    control_arm.add_link(link2)

    # Add an additional attachment point (e.g., for a sway bar link)
    control_arm.add_attachment_point("sway_bar_mount", [1.25, 0.5, 0.5], unit='m')

    print(f"\n{control_arm}")

    print("\nAll attachment positions (mm):")
    all_positions = control_arm.get_all_attachment_positions(unit='mm')
    for i, pos in enumerate(all_positions):
        print(f"  Point {i}: {pos}")

    print("\nAll attachment positions (m):")
    all_positions_m = control_arm.get_all_attachment_positions(unit='m')
    for i, pos in enumerate(all_positions_m):
        print(f"  Point {i}: {pos}")

    print(f"\nNote: Ball joint is shared between links, so only counted once")
    print(f"Total unique attachments: {len(all_positions)}")

    # Test fitting control arm to new targets
    print("\n--- Testing control arm fit to targets ---")

    # Create target positions in meters (e.g., suspension compressed and rotated)
    original_positions = control_arm.get_all_attachment_positions(unit='m')
    # Simulate movement: translate by 20mm x, -10mm y, -40mm z
    target_positions = [pos + np.array([0.02, -0.01, -0.04]) for pos in original_positions]

    print(f"Original centroid (mm): {control_arm.centroid}")
    print(f"Original centroid (m): {from_mm(control_arm.centroid, 'm')}")

    rms_error = control_arm.fit_to_attachment_targets(target_positions, unit='m')

    print(f"\nAfter fitting:")
    print(f"New centroid (mm): {control_arm.centroid}")
    print(f"New centroid (m): {from_mm(control_arm.centroid, 'm')}")
    print(f"RMS error (mm): {rms_error:.3f}")
    print(f"RMS error (m): {from_mm(rms_error, 'm'):.6f}")

    print(f"\nLink lengths maintained:")
    for link in control_arm.links:
        print(f"  {link.name} (mm): {link.get_length():.3f}")
        print(f"  {link.name} (m): {link.get_length('m'):.6f}")

    print("\n✓ All tests completed successfully!")
