import numpy as np
from typing import List, Tuple, Union
from units import to_mm, from_mm


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
        self.attachment_points: List[Tuple[str, np.ndarray]] = []

    def add_attachment_point(self, name: str, position: Union[np.ndarray, Tuple[float, float, float]],
                            unit: str = 'mm') -> None:
        """
        Add an attachment point to this corner.

        Args:
            name: Identifier for the attachment point
            position: 3D position [x, y, z]
            unit: Unit of input position (default: 'mm')
        """
        pos = to_mm(np.array(position, dtype=float), unit)
        if pos.shape != (3,):
            raise ValueError("Position must be a 3-element array [x, y, z]")
        self.attachment_points.append((name, pos))

    def get_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get all attachment point positions for this corner.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of attachment positions in specified unit
        """
        return [from_mm(pos.copy(), unit) for _, pos in self.attachment_points]

    def get_attachment_names(self) -> List[str]:
        """
        Get all attachment point names for this corner.

        Returns:
            List of attachment point names
        """
        return [name for name, _ in self.attachment_points]

    def __repr__(self) -> str:
        return f"ChassisCorner('{self.name}', attachments={len(self.attachment_points)})"
