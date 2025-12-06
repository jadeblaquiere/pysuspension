import numpy as np
from typing import List, Tuple, Union
from .attachment_point import AttachmentPoint
from .units import to_mm, from_mm
from .joint_types import JointType, JOINT_STIFFNESS


class SuspensionLink:
    """
    Represents a rigid link between two attachment points in a suspension system.
    The link maintains a constant length once specified.

    All positions are stored internally in millimeters (mm).
    """

    def __init__(self,
                 endpoint1: Union[np.ndarray, Tuple[float, float, float], AttachmentPoint],
                 endpoint2: Union[np.ndarray, Tuple[float, float, float], AttachmentPoint],
                 name: str = "link",
                 unit: str = 'mm'):
        """
        Initialize a suspension link with two endpoints.

        Args:
            endpoint1: 3D position of first endpoint [x, y, z] or AttachmentPoint
            endpoint2: 3D position of second endpoint [x, y, z] or AttachmentPoint
            name: Identifier for the link
            unit: Unit of input positions (default: 'mm'), ignored if AttachmentPoint objects provided
        """
        self.name = name

        # Create AttachmentPoint objects if not provided
        if isinstance(endpoint1, AttachmentPoint):
            self.endpoint1 = endpoint1
            endpoint1.parent_component = self
        else:
            endpoint1_array = np.array(endpoint1, dtype=float)
            if endpoint1_array.shape != (3,):
                raise ValueError("Endpoint1 must be a 3-element array [x, y, z]")
            self.endpoint1 = AttachmentPoint(
                name=f"{name}_endpoint1",
                position=endpoint1_array,
                unit=unit,
                parent_component=self
            )

        if isinstance(endpoint2, AttachmentPoint):
            self.endpoint2 = endpoint2
            endpoint2.parent_component=self
        else:
            endpoint2_array = np.array(endpoint2, dtype=float)
            if endpoint2_array.shape != (3,):
                raise ValueError("Endpoint2 must be a 3-element array [x, y, z]")
            self.endpoint2 = AttachmentPoint(
                name=f"{name}_endpoint2",
                position=endpoint2_array,
                unit=unit,
                parent_component=self
            )

        # Calculate and store the link length (constant, in mm)
        self.length = np.linalg.norm(self.endpoint2.position - self.endpoint1.position)

        if self.length < 1e-6:  # Adjusted tolerance for mm
            raise ValueError("Link endpoints are too close together (zero length)")

        # Calculate link properties in local frame
        self._update_local_frame()
    
    def _update_local_frame(self):
        """Update the local coordinate frame of the link."""
        # Link axis (unit vector from endpoint1 to endpoint2)
        self.axis = (self.endpoint2.position - self.endpoint1.position) / self.length

        # Link center
        self.center = (self.endpoint1.position + self.endpoint2.position) / 2.0

    def get_endpoint1(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the position of the first endpoint.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Position in specified unit
        """
        return self.endpoint1.get_position(unit)

    def get_endpoint2(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the position of the second endpoint.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Position in specified unit
        """
        return self.endpoint2.get_position(unit)

    def get_endpoint1_attachment(self) -> AttachmentPoint:
        """
        Get the AttachmentPoint object for endpoint1.

        Returns:
            AttachmentPoint object for endpoint1
        """
        return self.endpoint1

    def get_endpoint2_attachment(self) -> AttachmentPoint:
        """
        Get the AttachmentPoint object for endpoint2.

        Returns:
            AttachmentPoint object for endpoint2
        """
        return self.endpoint2

    def get_endpoints(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get both endpoint positions as a list.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of positions in specified unit
        """
        return [self.endpoint1.get_position(unit), self.endpoint2.get_position(unit)]

    def get_center(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the center point of the link.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Position in specified unit
        """
        return from_mm(self.center.copy(), unit)

    def get_axis(self) -> np.ndarray:
        """
        Get the unit vector along the link axis (from endpoint1 to endpoint2).
        Unit vectors are dimensionless.

        Returns:
            Unit vector (dimensionless)
        """
        return self.axis.copy()

    def get_length(self, unit: str = 'mm') -> float:
        """
        Get the length of the link (constant).

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Length in specified unit
        """
        return from_mm(self.length, unit)
    
    def set_endpoints(self,
                     endpoint1: Union[np.ndarray, Tuple[float, float, float]],
                     endpoint2: Union[np.ndarray, Tuple[float, float, float]],
                     unit: str = 'mm') -> None:
        """
        Set new endpoint positions. The distance between endpoints must match the link length.

        Args:
            endpoint1: New position of first endpoint
            endpoint2: New position of second endpoint
            unit: Unit of input positions (default: 'mm')

        Raises:
            ValueError: If the new endpoints don't maintain the link length
        """
        new_endpoint1 = to_mm(np.array(endpoint1, dtype=float), unit)
        new_endpoint2 = to_mm(np.array(endpoint2, dtype=float), unit)

        new_length = np.linalg.norm(new_endpoint2 - new_endpoint1)

        if abs(new_length - self.length) > 1e-3:  # Adjusted tolerance for mm
            raise ValueError(f"New endpoints must maintain link length of {self.length:.3f} mm, "
                           f"got {new_length:.3f} mm")

        self.endpoint1.set_position(new_endpoint1, unit='mm')
        self.endpoint2.set_position(new_endpoint2, unit='mm')
        self._update_local_frame()

    def set_endpoint1(self, position: Union[np.ndarray, Tuple[float, float, float]], unit: str = 'mm') -> None:
        """
        Set the position of endpoint1, adjusting endpoint2 to maintain link length and direction.

        Args:
            position: New position of first endpoint
            unit: Unit of input position (default: 'mm')
        """
        new_endpoint1 = to_mm(np.array(position, dtype=float), unit)
        # Keep the same axis direction, just translate
        new_endpoint2 = new_endpoint1 + self.axis * self.length
        self.endpoint1.set_position(new_endpoint1, unit='mm')
        self.endpoint2.set_position(new_endpoint2, unit='mm')
        self._update_local_frame()

    def set_endpoint2(self, position: Union[np.ndarray, Tuple[float, float, float]], unit: str = 'mm') -> None:
        """
        Set the position of endpoint2, adjusting endpoint1 to maintain link length and direction.

        Args:
            position: New position of second endpoint
            unit: Unit of input position (default: 'mm')
        """
        new_endpoint2 = to_mm(np.array(position, dtype=float), unit)
        # Keep the same axis direction, just translate
        new_endpoint1 = new_endpoint2 - self.axis * self.length
        self.endpoint1.set_position(new_endpoint1, unit='mm')
        self.endpoint2.set_position(new_endpoint2, unit='mm')
        self._update_local_frame()

    def translate(self, translation: Union[np.ndarray, Tuple[float, float, float]], unit: str = 'mm') -> None:
        """
        Translate the link by a given vector, maintaining orientation.

        Args:
            translation: Translation vector [dx, dy, dz]
            unit: Unit of input translation (default: 'mm')
        """
        t = to_mm(np.array(translation, dtype=float), unit)
        self.endpoint1.set_position(self.endpoint1.position + t, unit='mm')
        self.endpoint2.set_position(self.endpoint2.position + t, unit='mm')
        self.center += t

    def rotate_about_center(self, rotation_matrix: np.ndarray) -> None:
        """
        Rotate the link about its center point.

        Args:
            rotation_matrix: 3x3 rotation matrix
        """
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")

        # Rotate endpoints about center
        new_endpoint1 = self.center + rotation_matrix @ (self.endpoint1.position - self.center)
        new_endpoint2 = self.center + rotation_matrix @ (self.endpoint2.position - self.center)
        self.endpoint1.set_position(new_endpoint1, unit='mm')
        self.endpoint2.set_position(new_endpoint2, unit='mm')
        self._update_local_frame()

    def rotate_about_endpoint1(self, rotation_matrix: np.ndarray) -> None:
        """
        Rotate the link about endpoint1.

        Args:
            rotation_matrix: 3x3 rotation matrix
        """
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")

        # Rotate endpoint2 about endpoint1
        new_endpoint2 = self.endpoint1.position + rotation_matrix @ (self.endpoint2.position - self.endpoint1.position)
        self.endpoint2.set_position(new_endpoint2, unit='mm')
        self._update_local_frame()

    def rotate_about_endpoint2(self, rotation_matrix: np.ndarray) -> None:
        """
        Rotate the link about endpoint2.

        Args:
            rotation_matrix: 3x3 rotation matrix
        """
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")

        # Rotate endpoint1 about endpoint2
        new_endpoint1 = self.endpoint2.position + rotation_matrix @ (self.endpoint1.position - self.endpoint2.position)
        self.endpoint1.set_position(new_endpoint1, unit='mm')
        self._update_local_frame()
    
    def fit_to_attachment_targets(self, target_endpoints: List[Union[np.ndarray, Tuple[float, float, float]]],
                                 unit: str = 'mm') -> float:
        """
        Fit the link to target endpoint positions while maintaining constant length.
        Updates the link position to minimize weighted RMS error from targets.

        Uses joint stiffness weighting so that stiffer joints (like ball joints) have
        less error tolerance, while compliant joints (like soft bushings) can absorb
        more geometric mismatch.

        The algorithm finds the best position by:
        1. Computing the weighted center of the target endpoints
        2. Computing the direction from target1 to target2
        3. Placing the link at the weighted target center with the link length maintained

        Args:
            target_endpoints: List of two target positions [target_endpoint1, target_endpoint2]
            unit: Unit of input target positions (default: 'mm')

        Returns:
            Weighted RMS error between actual endpoints and targets after fitting (in mm)
        """
        if len(target_endpoints) != 2:
            raise ValueError("Expected list of 2 target endpoints")

        target1 = to_mm(np.array(target_endpoints[0], dtype=float), unit)
        target2 = to_mm(np.array(target_endpoints[1], dtype=float), unit)

        # Get joint stiffness for each endpoint
        if self.endpoint1.joint is not None:
            stiffness1 = JOINT_STIFFNESS[self.endpoint1.joint.joint_type]
        else:
            stiffness1 = JOINT_STIFFNESS[JointType.RIGID]

        if self.endpoint2.joint is not None:
            stiffness2 = JOINT_STIFFNESS[self.endpoint2.joint.joint_type]
        else:
            stiffness2 = JOINT_STIFFNESS[JointType.RIGID]

        total_weight = stiffness1 + stiffness2

        # Compute weighted target center
        target_center = (stiffness1 * target1 + stiffness2 * target2) / total_weight

        # Compute target direction and distance
        target_vector = target2 - target1
        target_distance = np.linalg.norm(target_vector)

        if target_distance < 1e-6:  # Adjusted tolerance for mm
            raise ValueError("Target endpoints are too close together")

        # Compute target axis (unit vector)
        target_axis = target_vector / target_distance

        # Position the link at the weighted target center with correct orientation
        # maintaining the fixed link length
        self.center = target_center.copy()
        self.axis = target_axis.copy()
        new_endpoint1 = self.center - (self.length / 2.0) * self.axis
        new_endpoint2 = self.center + (self.length / 2.0) * self.axis
        self.endpoint1.set_position(new_endpoint1, unit='mm')
        self.endpoint2.set_position(new_endpoint2, unit='mm')

        # Calculate weighted RMS error (in mm)
        error1 = target1 - self.endpoint1.position
        error2 = target2 - self.endpoint2.position
        squared_error1 = np.sum(error1**2)
        squared_error2 = np.sum(error2**2)
        weighted_rms_error = np.sqrt((stiffness1 * squared_error1 + stiffness2 * squared_error2) / total_weight)

        return weighted_rms_error

    def to_dict(self) -> dict:
        """
        Serialize the suspension link to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'name': self.name,
            'endpoint1': self.endpoint1.to_dict(),
            'endpoint2': self.endpoint2.to_dict(),
            'length': float(self.length)  # Store length in mm for validation
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SuspensionLink':
        """
        Deserialize a suspension link from a dictionary.

        Args:
            data: Dictionary containing suspension link data

        Returns:
            New SuspensionLink instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # Recreate AttachmentPoint objects
        endpoint1 = AttachmentPoint.from_dict(data['endpoint1'])
        endpoint2 = AttachmentPoint.from_dict(data['endpoint2'])

        # Create the link with the AttachmentPoint objects
        link = cls(
            endpoint1=endpoint1,
            endpoint2=endpoint2,
            name=data['name'],
            unit='mm'  # AttachmentPoints already in mm
        )

        # Validate that the deserialized length matches
        expected_length = data.get('length')
        if expected_length is not None and abs(link.length - expected_length) > 1e-3:
            raise ValueError(f"Deserialized link length {link.length:.3f} mm doesn't match "
                           f"stored length {expected_length:.3f} mm")

        return link

    def __repr__(self) -> str:
        return (f"SuspensionLink('{self.name}',\n"
                f"  endpoint1={self.endpoint1.position} mm,\n"
                f"  endpoint2={self.endpoint2.position} mm,\n"
                f"  length={self.length:.3f} mm\n"
                f")")


if __name__ == "__main__":
    print("=" * 60)
    print("SUSPENSION LINK TEST (with unit support)")
    print("=" * 60)

    # Create a suspension link (using meters as input, stored as mm internally)
    link = SuspensionLink(
        endpoint1=[1.4, 0.5, 0.6],
        endpoint2=[1.5, 0.75, 0.6],
        name="test_link",
        unit='m'  # Input in meters
    )

    print(f"\n{link}")
    print(f"Link center (mm): {link.get_center()}")
    print(f"Link center (m): {link.get_center('m')}")
    print(f"Link axis: {link.get_axis()}")
    print(f"Link length (mm): {link.get_length():.3f}")
    print(f"Link length (m): {link.get_length('m'):.6f}")
    print(f"Link length (in): {link.get_length('in'):.6f}")

    # Test translation
    print("\n--- Testing translation ---")
    original_ep1 = link.get_endpoint1('m')
    print(f"Original endpoint1 (m): {original_ep1}")
    link.translate([100.0, 0.0, -50.0], unit='mm')  # Translate 100mm, 0mm, -50mm
    print(f"After translation [100mm, 0, -50mm]:")
    print(f"  Endpoint1 (m): {link.get_endpoint1('m')}")
    print(f"  Endpoint1 (mm): {link.get_endpoint1()}")
    print(f"Length maintained (mm): {link.get_length():.3f}")

    # Test rotation about center
    print("\n--- Testing rotation about center ---")
    angle = np.radians(15)
    R_z = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    link.rotate_about_center(R_z)
    print(f"After 15° rotation about center:")
    print(f"  Endpoint1 (mm): {link.get_endpoint1()}")
    print(f"  Endpoint2 (mm): {link.get_endpoint2()}")
    print(f"  Length maintained (mm): {link.get_length():.3f}")

    # Test fitting to target positions
    print("\n--- Testing fit to attachment targets ---")
    # Create targets in meters
    target_endpoints = [
        [1.45, 0.52, 0.58],
        [1.52, 0.78, 0.62]
    ]

    print(f"Target endpoint1 (m): {target_endpoints[0]}")
    print(f"Target endpoint2 (m): {target_endpoints[1]}")

    rms_error = link.fit_to_attachment_targets(target_endpoints, unit='m')

    print(f"\nAfter fitting:")
    fitted_endpoints = link.get_endpoints('m')
    print(f"  Endpoint1 (m): {fitted_endpoints[0]}")
    print(f"  Endpoint2 (m): {fitted_endpoints[1]}")
    print(f"  Length maintained (mm): {link.get_length():.3f}")
    print(f"  Length maintained (m): {link.get_length('m'):.6f}")
    print(f"  RMS error (mm): {rms_error:.3f}")
    print(f"  RMS error (m): {from_mm(rms_error, 'm'):.6f}")

    print("\n✓ All tests completed successfully!")
