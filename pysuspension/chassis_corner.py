import numpy as np
from typing import List, Tuple, Union, Optional
from .attachment_point import AttachmentPoint
from .units import to_mm, from_mm


class ChassisCorner:
    """
    Represents a corner of the chassis with multiple attachment points.
    All attachment points in a corner are linked to a single suspension knuckle.

    All positions are stored internally in millimeters (mm).
    """

    def __init__(self, name: str):
        """
        Initialize a chassis corner.

        Args:
            name: Identifier for the corner (e.g., "front_left", "rear_right")
        """
        self.name = name
        self.attachment_points: List[AttachmentPoint] = []

        # Store original state for reset
        self._original_attachment_points: List[AttachmentPoint] = []

    def add_attachment_point(self,
                            name_or_attachment: Union[str, AttachmentPoint],
                            position: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None,
                            unit: str = 'mm') -> AttachmentPoint:
        """
        Add an attachment point to this corner.

        Can be called in two ways:
        1. Pass an existing AttachmentPoint object:
           add_attachment_point(attachment_point)
        2. Create a new AttachmentPoint from parameters:
           add_attachment_point(name, position, unit='mm')

        Args:
            name_or_attachment: Either an AttachmentPoint object or a name string
            position: 3D position [x, y, z] (required if name_or_attachment is a string)
            unit: Unit of input position (default: 'mm')

        Returns:
            The AttachmentPoint object (either the one passed in or newly created)

        Raises:
            ValueError: If name_or_attachment is a string but position is not provided
        """
        if isinstance(name_or_attachment, AttachmentPoint):
            # Accepting an existing AttachmentPoint object
            attachment = name_or_attachment
            # Update parent_component reference to this corner
            attachment.parent_component = self
            self.attachment_points.append(attachment)
            # Store original attachment point (copy without connections)
            self._original_attachment_points.append(attachment.copy())
            return attachment
        else:
            # Creating a new AttachmentPoint from name/position/unit
            if position is None:
                raise ValueError("position argument is required when passing a name string")

            name = name_or_attachment
            attachment = AttachmentPoint(
                name=name,
                position=position,
                is_relative=False,  # Chassis attachment points are in absolute coordinates
                unit=unit,
                parent_component=self
            )
            self.attachment_points.append(attachment)
            # Store original attachment point (copy without connections)
            self._original_attachment_points.append(attachment.copy())
            return attachment

    def get_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get all attachment point positions for this corner.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of attachment positions in specified unit
        """
        return [attachment.get_position(unit) for attachment in self.attachment_points]

    def get_attachment_names(self) -> List[str]:
        """
        Get all attachment point names for this corner.

        Returns:
            List of attachment point names
        """
        return [attachment.name for attachment in self.attachment_points]

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

    def reset_to_origin(self) -> None:
        """
        Reset the chassis corner attachment points to their originally defined positions.
        """
        for i, attachment in enumerate(self.attachment_points):
            if i < len(self._original_attachment_points):
                original = self._original_attachment_points[i]
                attachment.set_position(original.position, unit='mm')

    def to_dict(self) -> dict:
        """
        Serialize the chassis corner to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'name': self.name,
            'attachment_points': [ap.to_dict() for ap in self.attachment_points]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChassisCorner':
        """
        Deserialize a chassis corner from a dictionary.

        Args:
            data: Dictionary containing chassis corner data

        Returns:
            New ChassisCorner instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        corner = cls(name=data['name'])

        # Add attachment points
        for ap_data in data.get('attachment_points', []):
            position = ap_data['position']
            name = ap_data['name']
            unit = ap_data.get('unit', 'mm')
            corner.add_attachment_point(name, position, unit=unit)

        return corner

    def __repr__(self) -> str:
        return f"ChassisCorner('{self.name}', attachments={len(self.attachment_points)})"
