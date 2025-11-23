import numpy as np
from dataclasses import dataclass


@dataclass
class AttachmentPoint:
    """Represents a suspension attachment point on the knuckle."""
    name: str
    position: np.ndarray  # 3D position vector [x, y, z]
    is_relative: bool = True  # True if relative to tire center, False if absolute
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        if self.position.shape != (3,):
            raise ValueError("Position must be a 3-element array [x, y, z]")


if __name__ == "__main__":
    print("=" * 60)
    print("ATTACHMENT POINT TEST")
    print("=" * 60)
    
    # Create an attachment point
    ap = AttachmentPoint("test_point", [0.1, 0.2, 0.3], is_relative=True)
    print(f"\nAttachment Point: {ap.name}")
    print(f"Position: {ap.position}")
    print(f"Is relative: {ap.is_relative}")
    
    # Test with absolute positioning
    ap_abs = AttachmentPoint("chassis_mount", [1.5, 0.6, 0.4], is_relative=False)
    print(f"\nAbsolute Attachment Point: {ap_abs.name}")
    print(f"Position: {ap_abs.position}")
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
    np_array = np.array([0.5, 0.6, 0.7])
    ap_numpy = AttachmentPoint("numpy_point", np_array)
    print(f"Created from numpy array: {ap_numpy.position}")
    print(f"✓ Numpy array conversion successful")
