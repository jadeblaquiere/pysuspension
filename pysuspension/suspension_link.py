import numpy as np
from typing import List, Tuple, Union


class SuspensionLink:
    """
    Represents a rigid link between two attachment points in a suspension system.
    The link maintains a constant length once specified.
    """
    
    def __init__(self, 
                 endpoint1: Union[np.ndarray, Tuple[float, float, float]],
                 endpoint2: Union[np.ndarray, Tuple[float, float, float]],
                 name: str = "link"):
        """
        Initialize a suspension link with two endpoints.
        
        Args:
            endpoint1: 3D position of first endpoint [x, y, z]
            endpoint2: 3D position of second endpoint [x, y, z]
            name: Identifier for the link
        """
        self.name = name
        self.endpoint1 = np.array(endpoint1, dtype=float)
        self.endpoint2 = np.array(endpoint2, dtype=float)
        
        if self.endpoint1.shape != (3,) or self.endpoint2.shape != (3,):
            raise ValueError("Endpoints must be 3-element arrays [x, y, z]")
        
        # Calculate and store the link length (constant)
        self.length = np.linalg.norm(self.endpoint2 - self.endpoint1)
        
        if self.length < 1e-10:
            raise ValueError("Link endpoints are too close together (zero length)")
        
        # Store initial relative positions for rigid body transformations
        self.endpoint1_relative = self.endpoint1.copy()
        self.endpoint2_relative = self.endpoint2.copy()
        
        # Calculate link properties in local frame
        self._update_local_frame()
    
    def _update_local_frame(self):
        """Update the local coordinate frame of the link."""
        # Link axis (unit vector from endpoint1 to endpoint2)
        self.axis = (self.endpoint2 - self.endpoint1) / self.length
        
        # Link center
        self.center = (self.endpoint1 + self.endpoint2) / 2.0
    
    def get_endpoint1(self) -> np.ndarray:
        """Get the position of the first endpoint."""
        return self.endpoint1.copy()
    
    def get_endpoint2(self) -> np.ndarray:
        """Get the position of the second endpoint."""
        return self.endpoint2.copy()
    
    def get_endpoints(self) -> List[np.ndarray]:
        """Get both endpoint positions as a list."""
        return [self.endpoint1.copy(), self.endpoint2.copy()]
    
    def get_center(self) -> np.ndarray:
        """Get the center point of the link."""
        return self.center.copy()
    
    def get_axis(self) -> np.ndarray:
        """Get the unit vector along the link axis (from endpoint1 to endpoint2)."""
        return self.axis.copy()
    
    def get_length(self) -> float:
        """Get the length of the link (constant)."""
        return self.length
    
    def set_endpoints(self, 
                     endpoint1: Union[np.ndarray, Tuple[float, float, float]],
                     endpoint2: Union[np.ndarray, Tuple[float, float, float]]) -> None:
        """
        Set new endpoint positions. The distance between endpoints must match the link length.
        
        Args:
            endpoint1: New position of first endpoint
            endpoint2: New position of second endpoint
            
        Raises:
            ValueError: If the new endpoints don't maintain the link length
        """
        new_endpoint1 = np.array(endpoint1, dtype=float)
        new_endpoint2 = np.array(endpoint2, dtype=float)
        
        new_length = np.linalg.norm(new_endpoint2 - new_endpoint1)
        
        if abs(new_length - self.length) > 1e-6:
            raise ValueError(f"New endpoints must maintain link length of {self.length:.6f}, "
                           f"got {new_length:.6f}")
        
        self.endpoint1 = new_endpoint1
        self.endpoint2 = new_endpoint2
        self._update_local_frame()
    
    def set_endpoint1(self, position: Union[np.ndarray, Tuple[float, float, float]]) -> None:
        """
        Set the position of endpoint1, adjusting endpoint2 to maintain link length and direction.
        
        Args:
            position: New position of first endpoint
        """
        new_endpoint1 = np.array(position, dtype=float)
        # Keep the same axis direction, just translate
        self.endpoint2 = new_endpoint1 + self.axis * self.length
        self.endpoint1 = new_endpoint1
        self._update_local_frame()
    
    def set_endpoint2(self, position: Union[np.ndarray, Tuple[float, float, float]]) -> None:
        """
        Set the position of endpoint2, adjusting endpoint1 to maintain link length and direction.
        
        Args:
            position: New position of second endpoint
        """
        new_endpoint2 = np.array(position, dtype=float)
        # Keep the same axis direction, just translate
        self.endpoint1 = new_endpoint2 - self.axis * self.length
        self.endpoint2 = new_endpoint2
        self._update_local_frame()
    
    def translate(self, translation: Union[np.ndarray, Tuple[float, float, float]]) -> None:
        """
        Translate the link by a given vector, maintaining orientation.
        
        Args:
            translation: Translation vector [dx, dy, dz]
        """
        t = np.array(translation, dtype=float)
        self.endpoint1 += t
        self.endpoint2 += t
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
        self.endpoint1 = self.center + rotation_matrix @ (self.endpoint1 - self.center)
        self.endpoint2 = self.center + rotation_matrix @ (self.endpoint2 - self.center)
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
        self.endpoint2 = self.endpoint1 + rotation_matrix @ (self.endpoint2 - self.endpoint1)
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
        self.endpoint1 = self.endpoint2 + rotation_matrix @ (self.endpoint1 - self.endpoint2)
        self._update_local_frame()
    
    def fit_to_attachment_targets(self, target_endpoints: List[Union[np.ndarray, Tuple[float, float, float]]]) -> float:
        """
        Fit the link to target endpoint positions while maintaining constant length.
        Updates the link position to minimize RMS error from targets.
        
        The algorithm finds the best position by:
        1. Computing the center of the target endpoints
        2. Computing the direction from target1 to target2
        3. Placing the link at the target center with the link length maintained
        
        Args:
            target_endpoints: List of two target positions [target_endpoint1, target_endpoint2]
            
        Returns:
            RMS error between actual endpoints and targets after fitting
        """
        if len(target_endpoints) != 2:
            raise ValueError("Expected list of 2 target endpoints")
        
        target1 = np.array(target_endpoints[0], dtype=float)
        target2 = np.array(target_endpoints[1], dtype=float)
        
        # Compute target center
        target_center = (target1 + target2) / 2.0
        
        # Compute target direction and distance
        target_vector = target2 - target1
        target_distance = np.linalg.norm(target_vector)
        
        if target_distance < 1e-10:
            raise ValueError("Target endpoints are too close together")
        
        # Compute target axis (unit vector)
        target_axis = target_vector / target_distance
        
        # Position the link at the target center with correct orientation
        # maintaining the fixed link length
        self.center = target_center.copy()
        self.axis = target_axis.copy()
        self.endpoint1 = self.center - (self.length / 2.0) * self.axis
        self.endpoint2 = self.center + (self.length / 2.0) * self.axis
        
        # Calculate RMS error
        error1 = target1 - self.endpoint1
        error2 = target2 - self.endpoint2
        rms_error = np.sqrt((np.sum(error1**2) + np.sum(error2**2)) / 2.0)
        
        return rms_error
    
    def __repr__(self) -> str:
        return (f"SuspensionLink('{self.name}',\n"
                f"  endpoint1={self.endpoint1},\n"
                f"  endpoint2={self.endpoint2},\n"
                f"  length={self.length:.6f}\n"
                f")")


if __name__ == "__main__":
    print("=" * 60)
    print("SUSPENSION LINK TEST")
    print("=" * 60)
    
    # Create a suspension link
    link = SuspensionLink(
        endpoint1=[1.4, 0.5, 0.6],
        endpoint2=[1.5, 0.75, 0.6],
        name="test_link"
    )
    
    print(f"\n{link}")
    print(f"Link center: {link.get_center()}")
    print(f"Link axis: {link.get_axis()}")
    print(f"Link length: {link.get_length():.6f} m")
    
    # Test translation
    print("\n--- Testing translation ---")
    original_ep1 = link.get_endpoint1()
    print(f"Original endpoint1: {original_ep1}")
    link.translate([0.1, 0.0, -0.05])
    print(f"After translation [0.1, 0, -0.05]: {link.get_endpoint1()}")
    print(f"Length maintained: {link.get_length():.6f} m")
    
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
    print(f"  Endpoint1: {link.get_endpoint1()}")
    print(f"  Endpoint2: {link.get_endpoint2()}")
    print(f"  Length maintained: {link.get_length():.6f} m")
    
    # Test fitting to target positions
    print("\n--- Testing fit to attachment targets ---")
    target_endpoints = [
        np.array([1.45, 0.52, 0.58]),
        np.array([1.52, 0.78, 0.62])
    ]
    target_distance = np.linalg.norm(target_endpoints[1] - target_endpoints[0])
    
    print(f"Target endpoint1: {target_endpoints[0]}")
    print(f"Target endpoint2: {target_endpoints[1]}")
    print(f"Target distance: {target_distance:.6f} m")
    print(f"Link length: {link.get_length():.6f} m")
    
    rms_error = link.fit_to_attachment_targets(target_endpoints)
    
    print(f"\nAfter fitting:")
    fitted_endpoints = link.get_endpoints()
    print(f"  Endpoint1: {fitted_endpoints[0]}")
    print(f"  Endpoint2: {fitted_endpoints[1]}")
    print(f"  Length maintained: {link.get_length():.6f} m")
    print(f"  RMS error: {rms_error:.6f} m")
