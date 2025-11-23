import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple
from units import to_mm, from_mm


@dataclass
class AttachmentPoint:
    """
    Represents a suspension attachment point on the knuckle.

    All positions are stored internally in millimeters (mm).
    """
    name: str
    position: Union[np.ndarray, Tuple[float, float, float]]  # 3D position vector [x, y, z]
    is_relative: bool = True  # True if relative to tire center, False if absolute
    unit: str = 'mm'  # Unit of input position
    _position_mm: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Convert position to mm and store internally
        pos_array = np.array(self.position, dtype=float)
        if pos_array.shape != (3,):
            raise ValueError("Position must be a 3-element array [x, y, z]")
        self._position_mm = to_mm(pos_array, self.unit)
        # Keep position attribute for backward compatibility, but store in mm
        self.position = self._position_mm

    def get_position(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the position in the specified unit.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Position in specified unit
        """
        return from_mm(self._position_mm.copy(), unit)


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
