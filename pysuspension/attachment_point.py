import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional, TYPE_CHECKING
from units import to_mm, from_mm

if TYPE_CHECKING:
    from typing import Set


@dataclass
class AttachmentPoint:
    """
    Represents a suspension attachment point that can be connected to other attachment points.

    All positions are stored internally in millimeters (mm).

    Attributes:
        name: Identifier for the attachment point
        position: 3D position vector [x, y, z]
        is_relative: True if relative to parent component, False if absolute
        unit: Unit of input position
        parent_component: Optional reference to the component this attachment point belongs to
    """
    name: str
    position: Union[np.ndarray, Tuple[float, float, float]]  # 3D position vector [x, y, z]
    is_relative: bool = True  # True if relative to parent component, False if absolute
    unit: str = 'mm'  # Unit of input position
    parent_component: Optional[object] = None  # Reference to the owning component
    _position_mm: np.ndarray = field(init=False, repr=False)
    _connected_points: List['AttachmentPoint'] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        # Convert position to mm and store internally
        pos_array = np.array(self.position, dtype=float)
        if pos_array.shape != (3,):
            raise ValueError("Position must be a 3-element array [x, y, z]")
        self._position_mm = to_mm(pos_array, self.unit)
        # Keep position attribute for backward compatibility, but store in mm
        self.position = self._position_mm
        # Initialize connected points list
        self._connected_points = []

    def get_position(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the position in the specified unit.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Position in specified unit
        """
        return from_mm(self._position_mm.copy(), unit)

    def set_position(self, position: Union[np.ndarray, Tuple[float, float, float]], unit: str = 'mm') -> None:
        """
        Update the position of this attachment point.

        Args:
            position: New 3D position vector [x, y, z]
            unit: Unit of input position (default: 'mm')
        """
        pos_array = np.array(position, dtype=float)
        if pos_array.shape != (3,):
            raise ValueError("Position must be a 3-element array [x, y, z]")
        self._position_mm = to_mm(pos_array, unit)
        self.position = self._position_mm

    def connect_to(self, other: 'AttachmentPoint', bidirectional: bool = True) -> None:
        """
        Connect this attachment point to another attachment point.

        Args:
            other: The attachment point to connect to
            bidirectional: If True, also add this point to other's connections (default: True)
        """
        if other not in self._connected_points:
            self._connected_points.append(other)

        if bidirectional and self not in other._connected_points:
            other._connected_points.append(self)

    def disconnect_from(self, other: 'AttachmentPoint', bidirectional: bool = True) -> None:
        """
        Disconnect this attachment point from another attachment point.

        Args:
            other: The attachment point to disconnect from
            bidirectional: If True, also remove this point from other's connections (default: True)
        """
        if other in self._connected_points:
            self._connected_points.remove(other)

        if bidirectional and self in other._connected_points:
            other._connected_points.remove(self)

    def get_connected_points(self) -> List['AttachmentPoint']:
        """
        Get all attachment points connected to this one.

        Returns:
            List of connected AttachmentPoint objects
        """
        return self._connected_points.copy()

    def is_connected_to(self, other: 'AttachmentPoint') -> bool:
        """
        Check if this attachment point is connected to another.

        Args:
            other: The attachment point to check

        Returns:
            True if connected, False otherwise
        """
        return other in self._connected_points

    def clear_connections(self, bidirectional: bool = True) -> None:
        """
        Clear all connections from this attachment point.

        Args:
            bidirectional: If True, also remove this point from all connected points (default: True)
        """
        if bidirectional:
            for point in self._connected_points:
                if self in point._connected_points:
                    point._connected_points.remove(self)

        self._connected_points.clear()

    def copy(self) -> 'AttachmentPoint':
        """
        Create a copy of this attachment point without connections.

        Returns:
            New AttachmentPoint with same name, position, and properties but no connections
        """
        return AttachmentPoint(
            name=self.name,
            position=self._position_mm.copy(),
            is_relative=self.is_relative,
            unit='mm',
            parent_component=self.parent_component
        )


if __name__ == "__main__":
    print("=" * 60)
    print("ATTACHMENT POINT TEST (with unit support)")
    print("=" * 60)

    # Create an attachment point in meters
    ap = AttachmentPoint("test_point", [0.1, 0.2, 0.3], is_relative=True, unit='m')
    print(f"\nAttachment Point: {ap.name}")
    print(f"Position (mm): {ap.get_position()}")
    print(f"Position (m): {ap.get_position('m')}")
    print(f"Position (in): {ap.get_position('in')}")
    print(f"Is relative: {ap.is_relative}")

    # Test with absolute positioning in millimeters
    ap_abs = AttachmentPoint("chassis_mount", [1500.0, 600.0, 400.0], is_relative=False, unit='mm')
    print(f"\nAbsolute Attachment Point: {ap_abs.name}")
    print(f"Position (mm): {ap_abs.get_position()}")
    print(f"Position (m): {ap_abs.get_position('m')}")
    print(f"Is relative: {ap_abs.is_relative}")

    # Test error handling
    print("\n--- Testing error handling ---")
    try:
        bad_ap = AttachmentPoint("bad_point", [0.1, 0.2])
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")

    # Test numpy array conversion
    print("\n--- Testing numpy array input ---")
    np_array = np.array([500.0, 600.0, 700.0])
    ap_numpy = AttachmentPoint("numpy_point", np_array, unit='mm')
    print(f"Created from numpy array (mm): {ap_numpy.get_position()}")
    print(f"Created from numpy array (m): {ap_numpy.get_position('m')}")
    print(f"✓ Numpy array conversion successful")

    print("\n✓ All tests completed successfully!")
