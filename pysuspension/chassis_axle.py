import numpy as np
from typing import List, Tuple, Union, Optional, TYPE_CHECKING
from .attachment_point import AttachmentPoint
from .units import to_mm, from_mm

if TYPE_CHECKING:
    from chassis import Chassis


class ChassisAxle:
    """
    Represents an axle assembly connected to chassis corners.

    The ChassisAxle is a logical grouping of attachment points that are rigidly
    connected to specific chassis corners. It moves as a rigid body with the chassis.
    Common use cases include mounting points for steering racks and anti-sway bars.

    All positions are stored internally in millimeters (mm).

    Note: ChassisAxle is a logical object with no mass or center of mass.
    """

    def __init__(self, name: str, chassis: 'Chassis', corner_names: List[str]):
        """
        Initialize a chassis axle.

        Args:
            name: Identifier for the axle (e.g., "front_axle", "rear_axle")
            chassis: Reference to the parent Chassis object
            corner_names: List of corner names this axle is connected to

        Raises:
            ValueError: If any corner name doesn't exist in the chassis
        """
        self.name = name
        self.chassis = chassis
        self.corner_names = corner_names.copy()

        # Validate that all corners exist
        for corner_name in self.corner_names:
            if corner_name not in chassis.corners:
                raise ValueError(f"Corner '{corner_name}' not found in chassis")

        # Additional attachment points (beyond corner attachments)
        self.attachment_points: List[AttachmentPoint] = []

        # Store original state for reset
        self._original_attachment_points: List[AttachmentPoint] = []

    def add_attachment_point(self, name: str, position: Union[np.ndarray, Tuple[float, float, float]],
                            unit: str = 'mm') -> AttachmentPoint:
        """
        Add an additional attachment point to this axle.

        These are attachment points that are not part of the corner attachments,
        such as steering rack mounts or anti-sway bar mounts.

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
            is_relative=False,  # Axle attachment points are in absolute coordinates
            unit=unit,
            parent_component=self
        )
        self.attachment_points.append(attachment)
        # Store original attachment point (copy without connections)
        self._original_attachment_points.append(attachment.copy())
        return attachment

    def get_attachment_position(self, name: str, unit: str = 'mm') -> np.ndarray:
        """
        Get the position of a specific attachment point.

        Args:
            name: Name of the attachment point
            unit: Unit for output (default: 'mm')

        Returns:
            3D position vector in specified unit

        Raises:
            ValueError: If attachment point not found
        """
        attachment = self.get_attachment_point(name)
        if attachment is None:
            raise ValueError(f"Attachment point '{name}' not found in axle '{self.name}'")
        return attachment.get_position(unit)

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
        Get all additional attachment point positions for this axle.

        Note: This returns only the additional attachment points, not the corner attachments.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of attachment positions in specified unit
        """
        return [attachment.get_position(unit) for attachment in self.attachment_points]

    def get_corner_names(self) -> List[str]:
        """
        Get the names of chassis corners this axle is connected to.

        Returns:
            List of corner names
        """
        return self.corner_names.copy()

    def get_corner_attachment_positions(self, unit: str = 'mm') -> dict:
        """
        Get all attachment positions from the corners this axle is connected to.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Dictionary mapping corner names to lists of attachment positions
        """
        result = {}
        for corner_name in self.corner_names:
            corner = self.chassis.get_corner(corner_name)
            result[corner_name] = corner.get_attachment_positions(unit=unit)
        return result

    def reset_to_origin(self) -> None:
        """
        Reset the axle attachment points to their originally defined positions.

        Note: This only resets the additional attachment points.
        Corner attachment points are managed by their respective corners.
        """
        for i, attachment in enumerate(self.attachment_points):
            if i < len(self._original_attachment_points):
                original = self._original_attachment_points[i]
                attachment.set_position(original.position, unit='mm')

    def to_dict(self) -> dict:
        """
        Serialize the chassis axle to a dictionary.

        Note: The chassis reference is not serialized to avoid circular references.
        It must be provided when deserializing.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'name': self.name,
            'corner_names': self.corner_names.copy(),
            'attachment_points': [ap.to_dict() for ap in self.attachment_points]
        }

    @classmethod
    def from_dict(cls, data: dict, chassis: 'Chassis') -> 'ChassisAxle':
        """
        Deserialize a chassis axle from a dictionary.

        Args:
            data: Dictionary containing chassis axle data
            chassis: Reference to the parent Chassis object

        Returns:
            New ChassisAxle instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid or corners don't exist
        """
        axle = cls(
            name=data['name'],
            chassis=chassis,
            corner_names=data['corner_names']
        )

        # Add attachment points
        for ap_data in data.get('attachment_points', []):
            position = ap_data['position']
            name = ap_data['name']
            unit = ap_data.get('unit', 'mm')
            axle.add_attachment_point(name, position, unit=unit)

        return axle

    def __repr__(self) -> str:
        return (f"ChassisAxle('{self.name}',\n"
                f"  corners={self.corner_names},\n"
                f"  attachment_points={len(self.attachment_points)}\n"
                f")")


if __name__ == "__main__":
    print("=" * 60)
    print("CHASSIS AXLE TEST")
    print("=" * 60)

    # Note: Full testing requires Chassis class
    # This is just a minimal test to verify the class structure
    print("\n✓ ChassisAxle class loaded successfully!")
    print("Run chassis.py tests for full integration testing.")
