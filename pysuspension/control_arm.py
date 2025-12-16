import numpy as np
from typing import List, Tuple, Union, Optional
from .rigid_body import RigidBody
from .attachment_point import AttachmentPoint
from .units import to_mm, from_mm


class ControlArm(RigidBody):
    """
    Represents a control arm consisting of attachment points that move together as a rigid body.

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

    def _update_centroid(self) -> None:
        """
        Update the centroid and center of mass from all attachment points.

        Overrides parent to calculate centroid from attachment point positions.
        """
        if self.attachment_points:
            all_points = [ap.position for ap in self.attachment_points]
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
        Get all attachment point positions.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of attachment positions in specified unit
        """
        positions = [from_mm(ap.position.copy(), unit) for ap in self.attachment_points]
        return positions

    def _apply_transformation(self, R: np.ndarray, t: np.ndarray) -> None:
        """
        Apply rigid body transformation to attachment points.

        Overrides parent to transform all attachment points.

        Args:
            R: 3x3 rotation matrix
            t: 3D translation vector (in mm)
        """
        # Transform center of mass (if it exists)
        if self.center_of_mass is not None:
            self.center_of_mass = R @ self.center_of_mass + t

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
        - All attachment point positions
        - Centroid and center of mass
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

        # Recalculate centroid and center of mass
        self._update_centroid()

    def copy(self, copy_joints: bool = False) -> 'ControlArm':
        """
        Create a deep copy of this control arm.

        Creates new copies of all attachment points. The copied
        control arm will have the same name, mass, and geometry but will
        be a completely independent object.

        Args:
            copy_joints: If True, preserves joint references on copied attachment points.
                        If False, copied points have no joint connections (default).

        Returns:
            New ControlArm instance with copied attachment points

        Note:
            The copy will have the same rotation_matrix and centroid as the original.
        """
        # Create new control arm with same name and mass
        arm_copy = ControlArm(name=self.name, mass=self.mass, mass_unit='kg')

        # Copy all attachment points
        for ap in self.attachment_points:
            ap_copy = AttachmentPoint(
                name=ap.name,
                position=ap.position.copy(),
                unit='mm',
                parent_component=arm_copy,
                joint=ap.joint if copy_joints else None
            )
            arm_copy.attachment_points.append(ap_copy)

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
                f"  attachments={len(self.attachment_points)},\n"
                f"  mass={self.mass:.3f} kg,\n"
                f"  centroid={centroid_str},\n"
                f"  center_of_mass={com_str}\n"
                f")")


if __name__ == "__main__":
    print("=" * 60)
    print("CONTROL ARM TEST (with unit support)")
    print("=" * 60)

    # Create a control arm with attachment points (using meters as input, stored as mm internally)
    control_arm = ControlArm(name="test_control_arm", mass=2.5)

    # Add attachment points for a triangulated control arm
    control_arm.add_attachment_point("front_chassis_mount", [1.3, 0.4, 0.55], unit='m')
    control_arm.add_attachment_point("rear_chassis_mount", [1.2, 0.4, 0.55], unit='m')
    control_arm.add_attachment_point("ball_joint", [1.5, 0.75, 0.6], unit='m')
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

    print(f"\nTotal attachments: {len(all_positions)}")

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

    # Test inheritance
    print("\n--- Testing inheritance ---")
    print(f"isinstance(control_arm, RigidBody): {isinstance(control_arm, RigidBody)}")

    print("\n✓ All tests completed successfully!")
