import numpy as np
from typing import List, Tuple, Union, Optional
from .rigid_body import RigidBody
from .suspension_link import SuspensionLink
from .attachment_point import AttachmentPoint
from .units import to_mm, from_mm


class ControlArm(RigidBody):
    """
    Represents a control arm consisting of multiple suspension links and attachment points
    that move together as a rigid body.

    Extends RigidBody to provide rigid body transformation behavior.
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
        super().__init__(name=name, mass=mass, mass_unit=mass_unit)

        self.links: List[SuspensionLink] = []

        # Store original link endpoints for reset
        self._original_link_endpoints = []

    def add_link(self, link: SuspensionLink) -> None:
        """
        Add a suspension link to the control arm.

        Args:
            link: SuspensionLink to add
        """
        self.links.append(link)

        # Store original endpoint positions
        if not self._original_state_frozen:
            self._original_link_endpoints.append((
                link.endpoint1.position.copy(),
                link.endpoint2.position.copy()
            ))

        self._update_centroid()

    def _update_centroid(self) -> None:
        """
        Update the centroid and center of mass from all attachment points and link endpoints.

        Overrides parent to gather points from both links and additional attachments.
        """
        all_points = []

        # Collect all link endpoints
        for link in self.links:
            all_points.append(link.endpoint1.position)
            all_points.append(link.endpoint2.position)

        # Collect additional attachment point positions
        for attachment in self.attachment_points:
            all_points.append(attachment.position)

        if all_points:
            self.centroid = np.mean(all_points, axis=0)
            self.center_of_mass = self.centroid.copy()
        else:
            self.centroid = np.zeros(3)
            self.center_of_mass = np.zeros(3)

        # Update original state only if not frozen
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
                    if np.linalg.norm(endpoint - existing_pos) < 1e-6:
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
                if np.linalg.norm(pos - existing_pos) < 1e-6:
                    is_duplicate = True
                    break

            if not is_duplicate:
                positions.append(pos.copy())
                position_set.append(pos.copy())

        # Convert all positions to requested unit
        return [from_mm(pos, unit) for pos in positions]

    def _apply_transformation(self, R: np.ndarray, t: np.ndarray) -> None:
        """
        Apply rigid body transformation to links and attachments.

        Overrides parent to also transform link endpoints.

        Args:
            R: 3x3 rotation matrix
            t: 3D translation vector (in mm)
        """
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
            if i < len(self._original_link_endpoints):
                endpoint1, endpoint2 = self._original_link_endpoints[i]
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

        # Unfreeze original state to allow subsequent transformations
        self._original_state_frozen = False

        # Recalculate centroid and center of mass
        self._update_centroid()

    def copy(self, copy_joints: bool = False) -> 'ControlArm':
        """
        Create a deep copy of this control arm.

        Creates new copies of all links and attachment points. The copied
        control arm will have the same name, mass, and geometry but will
        be a completely independent object.

        Args:
            copy_joints: If True, preserves joint references on copied attachment points.
                        If False, copied points have no joint connections (default).

        Returns:
            New ControlArm instance with copied links and attachment points

        Note:
            The copy will have the same rotation_matrix and centroid as the original.
            Original state tracking (_original_link_endpoints, etc.) is reset for the copy.
        """
        # Create new control arm with same name and mass
        arm_copy = ControlArm(name=self.name, mass=self.mass, mass_unit='kg')

        # Copy all links
        # Need to track endpoint mappings to handle shared endpoints between links
        endpoint_mapping = {}  # id(original_endpoint) -> copied_endpoint

        for link in self.links:
            # Check if we've already copied the endpoints (shared between links)
            endpoint1_id = id(link.endpoint1)
            endpoint2_id = id(link.endpoint2)

            # Copy endpoint1 if not already copied
            if endpoint1_id in endpoint_mapping:
                endpoint1_copy = endpoint_mapping[endpoint1_id]
            else:
                endpoint1_copy = AttachmentPoint(
                    name=link.endpoint1.name,
                    position=link.endpoint1.position.copy(),
                    unit='mm',
                    parent_component=None,
                    joint=link.endpoint1.joint if copy_joints else None
                )
                endpoint_mapping[endpoint1_id] = endpoint1_copy

            # Copy endpoint2 if not already copied
            if endpoint2_id in endpoint_mapping:
                endpoint2_copy = endpoint_mapping[endpoint2_id]
            else:
                endpoint2_copy = AttachmentPoint(
                    name=link.endpoint2.name,
                    position=link.endpoint2.position.copy(),
                    unit='mm',
                    parent_component=None,
                    joint=link.endpoint2.joint if copy_joints else None
                )
                endpoint_mapping[endpoint2_id] = endpoint_mapping[endpoint2_id] = endpoint2_copy

            # Create copied link
            link_copy = SuspensionLink(
                endpoint1=endpoint1_copy,
                endpoint2=endpoint2_copy,
                name=link.name,
                unit='mm'
            )

            arm_copy.add_link(link_copy)

        # Copy additional attachment points (not part of links)
        for ap in self.attachment_points:
            ap_id = id(ap)

            # Check if this attachment point was already copied as part of a link
            if ap_id in endpoint_mapping:
                # Already copied, just need to add reference
                # (It's already part of the control arm through the link)
                continue
            else:
                # Copy this attachment point
                ap_copy = AttachmentPoint(
                    name=ap.name,
                    position=ap.position.copy(),
                    unit='mm',
                    parent_component=arm_copy,
                    joint=ap.joint if copy_joints else None
                )
                arm_copy.attachment_points.append(ap_copy)
                endpoint_mapping[ap_id] = ap_copy

        # Copy transformation state
        arm_copy.rotation_matrix = self.rotation_matrix.copy()

        # Update centroid
        arm_copy._update_centroid()

        return arm_copy

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
                f"  additional_attachments={len(self.attachment_points)},\n"
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

    # Test inheritance
    print("\n--- Testing inheritance ---")
    print(f"isinstance(control_arm, RigidBody): {isinstance(control_arm, RigidBody)}")

    print("\n✓ All tests completed successfully!")
