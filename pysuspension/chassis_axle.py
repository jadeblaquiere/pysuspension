import numpy as np
from typing import List, Tuple, Union, TYPE_CHECKING
from units import to_mm, from_mm

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
        # Stored as (name, position) tuples in mm
        self.additional_attachments: List[Tuple[str, np.ndarray]] = []

        # Store original state for reset
        self._original_additional_attachments: List[Tuple[str, np.ndarray]] = []

    def add_attachment_point(self, name: str, position: Union[np.ndarray, Tuple[float, float, float]],
                            unit: str = 'mm') -> None:
        """
        Add an additional attachment point to this axle.

        These are attachment points that are not part of the corner attachments,
        such as steering rack mounts or anti-sway bar mounts.

        Args:
            name: Identifier for the attachment point
            position: 3D position [x, y, z]
            unit: Unit of input position (default: 'mm')
        """
        pos = to_mm(np.array(position, dtype=float), unit)
        if pos.shape != (3,):
            raise ValueError("Position must be a 3-element array [x, y, z]")
        self.additional_attachments.append((name, pos))
        # Store original position
        self._original_additional_attachments.append((name, pos.copy()))

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
        for attachment_name, pos in self.additional_attachments:
            if attachment_name == name:
                return from_mm(pos.copy(), unit)

        raise ValueError(f"Attachment point '{name}' not found in axle '{self.name}'")

    def get_all_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get all additional attachment point positions for this axle.

        Note: This returns only the additional attachment points, not the corner attachments.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of attachment positions in specified unit
        """
        return [from_mm(pos.copy(), unit) for _, pos in self.additional_attachments]

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
        self.additional_attachments = [(name, pos.copy()) for name, pos in self._original_additional_attachments]

    def __repr__(self) -> str:
        return (f"ChassisAxle('{self.name}',\n"
                f"  corners={self.corner_names},\n"
                f"  additional_attachments={len(self.additional_attachments)}\n"
                f")")


if __name__ == "__main__":
    print("=" * 60)
    print("CHASSIS AXLE TEST")
    print("=" * 60)

    # Note: Full testing requires Chassis class
    # This is just a minimal test to verify the class structure
    print("\n✓ ChassisAxle class loaded successfully!")
    print("Run chassis.py tests for full integration testing.")
