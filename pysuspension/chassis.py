import numpy as np
from typing import List, Tuple, Union, Dict


class ChassisCorner:
    """
    Represents a corner of the chassis with multiple attachment points.
    All attachment points in a corner are linked to a single suspension knuckle.
    """
    
    def __init__(self, name: str):
        """
        Initialize a chassis corner.
        
        Args:
            name: Identifier for the corner (e.g., "front_left", "rear_right")
        """
        self.name = name
        self.attachment_points: List[Tuple[str, np.ndarray]] = []
    
    def add_attachment_point(self, name: str, position: Union[np.ndarray, Tuple[float, float, float]]) -> None:
        """
        Add an attachment point to this corner.
        
        Args:
            name: Identifier for the attachment point
            position: 3D position [x, y, z]
        """
        pos = np.array(position, dtype=float)
        if pos.shape != (3,):
            raise ValueError("Position must be a 3-element array [x, y, z]")
        self.attachment_points.append((name, pos))
    
    def get_attachment_positions(self) -> List[np.ndarray]:
        """
        Get all attachment point positions for this corner.
        
        Returns:
            List of attachment positions
        """
        return [pos.copy() for _, pos in self.attachment_points]
    
    def get_attachment_names(self) -> List[str]:
        """
        Get all attachment point names for this corner.
        
        Returns:
            List of attachment point names
        """
        return [name for name, _ in self.attachment_points]
    
    def __repr__(self) -> str:
        return f"ChassisCorner('{self.name}', attachments={len(self.attachment_points)})"


class Chassis:
    """
    Represents a vehicle chassis with multiple corners.
    The chassis behaves as a rigid body with all corners moving together.
    """
    
    def __init__(self, name: str = "chassis"):
        """
        Initialize a chassis.
        
        Args:
            name: Identifier for the chassis
        """
        self.name = name
        self.corners: Dict[str, ChassisCorner] = {}
        self.centroid = None
        self.rotation_matrix = np.eye(3)
    
    def add_corner(self, corner: ChassisCorner) -> None:
        """
        Add a corner to the chassis.
        
        Args:
            corner: ChassisCorner to add
        """
        if corner.name in self.corners:
            raise ValueError(f"Corner '{corner.name}' already exists")
        self.corners[corner.name] = corner
        self._update_centroid()
    
    def create_corner(self, name: str) -> ChassisCorner:
        """
        Create and add a new corner to the chassis.
        
        Args:
            name: Name for the new corner
            
        Returns:
            The newly created ChassisCorner
        """
        corner = ChassisCorner(name)
        self.add_corner(corner)
        return corner
    
    def get_corner(self, name: str) -> ChassisCorner:
        """
        Get a corner by name.
        
        Args:
            name: Name of the corner
            
        Returns:
            ChassisCorner object
        """
        if name not in self.corners:
            raise ValueError(f"Corner '{name}' not found")
        return self.corners[name]
    
    def _update_centroid(self) -> None:
        """Update the centroid of all attachment points."""
        all_points = []
        
        for corner in self.corners.values():
            all_points.extend(corner.get_attachment_positions())
        
        if all_points:
            self.centroid = np.mean(all_points, axis=0)
        else:
            self.centroid = np.zeros(3)
    
    def get_all_attachment_positions(self) -> List[np.ndarray]:
        """
        Get all attachment point positions from all corners.
        
        Returns:
            List of all attachment positions (ordered by corner, then by attachment within corner)
        """
        positions = []
        
        for corner_name in sorted(self.corners.keys()):
            corner = self.corners[corner_name]
            positions.extend(corner.get_attachment_positions())
        
        return positions
    
    def get_corner_attachment_positions(self, corner_name: str) -> List[np.ndarray]:
        """
        Get attachment positions for a specific corner.
        
        Args:
            corner_name: Name of the corner
            
        Returns:
            List of attachment positions for the specified corner
        """
        return self.get_corner(corner_name).get_attachment_positions()
    
    def fit_to_attachment_targets(self, target_positions: List[np.ndarray]) -> float:
        """
        Fit the chassis to target attachment positions using rigid body transformation.
        Updates all corner attachment positions based on the optimal rigid body fit.
        
        Args:
            target_positions: List of target positions in same order as get_all_attachment_positions()
            
        Returns:
            RMS error of the fit
        """
        current_positions = self.get_all_attachment_positions()
        
        if len(target_positions) != len(current_positions):
            raise ValueError(f"Expected {len(current_positions)} target positions, got {len(target_positions)}")
        
        if len(current_positions) < 3:
            raise ValueError("Need at least 3 attachment points for rigid body fit")
        
        # Convert to numpy arrays
        current_points = np.array(current_positions)
        target_points = np.array([np.array(p) for p in target_positions])
        
        # Compute centroids
        centroid_current = np.mean(current_points, axis=0)
        centroid_target = np.mean(target_points, axis=0)
        
        # Center the point sets
        current_centered = current_points - centroid_current
        target_centered = target_points - centroid_target
        
        # Compute the cross-covariance matrix H
        H = current_centered.T @ target_centered
        
        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = centroid_target - R @ centroid_current
        
        # Update all corner attachment points
        for corner in self.corners.values():
            updated_attachments = []
            for name, pos in corner.attachment_points:
                new_pos = R @ pos + t
                updated_attachments.append((name, new_pos))
            corner.attachment_points = updated_attachments
        
        # Update chassis transformation
        self.rotation_matrix = R
        self._update_centroid()
        
        # Calculate RMS error
        transformed_points = (R @ current_points.T).T + t
        errors = target_points - transformed_points
        rms_error = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        
        return rms_error
    
    def translate(self, translation: Union[np.ndarray, Tuple[float, float, float]]) -> None:
        """
        Translate the entire chassis by a given vector.
        
        Args:
            translation: Translation vector [dx, dy, dz]
        """
        t = np.array(translation, dtype=float)
        
        for corner in self.corners.values():
            updated_attachments = []
            for name, pos in corner.attachment_points:
                new_pos = pos + t
                updated_attachments.append((name, new_pos))
            corner.attachment_points = updated_attachments
        
        self._update_centroid()
    
    def rotate_about_centroid(self, rotation_matrix: np.ndarray) -> None:
        """
        Rotate the chassis about its centroid.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
        """
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        
        for corner in self.corners.values():
            updated_attachments = []
            for name, pos in corner.attachment_points:
                new_pos = self.centroid + rotation_matrix @ (pos - self.centroid)
                updated_attachments.append((name, new_pos))
            corner.attachment_points = updated_attachments
        
        self.rotation_matrix = rotation_matrix @ self.rotation_matrix
        self._update_centroid()
    
    def __repr__(self) -> str:
        total_attachments = sum(len(c.attachment_points) for c in self.corners.values())
        return (f"Chassis('{self.name}',\n"
                f"  corners={len(self.corners)},\n"
                f"  total_attachments={total_attachments},\n"
                f"  centroid={self.centroid}\n"
                f")")


if __name__ == "__main__":
    print("=" * 60)
    print("CHASSIS TEST")
    print("=" * 60)
    
    # Create a chassis with four corners
    chassis = Chassis(name="test_chassis")
    
    print("\n--- Creating four corners ---")
    
    # Front left corner
    fl_corner = chassis.create_corner("front_left")
    fl_corner.add_attachment_point("upper_front_mount", [1.4, 0.5, 0.6])
    fl_corner.add_attachment_point("upper_rear_mount", [1.3, 0.5, 0.58])
    fl_corner.add_attachment_point("lower_front_mount", [1.45, 0.45, 0.35])
    fl_corner.add_attachment_point("lower_rear_mount", [1.35, 0.45, 0.33])
    
    # Front right corner
    fr_corner = chassis.create_corner("front_right")
    fr_corner.add_attachment_point("upper_front_mount", [1.4, -0.5, 0.6])
    fr_corner.add_attachment_point("upper_rear_mount", [1.3, -0.5, 0.58])
    fr_corner.add_attachment_point("lower_front_mount", [1.45, -0.45, 0.35])
    fr_corner.add_attachment_point("lower_rear_mount", [1.35, -0.45, 0.33])
    
    # Rear left corner
    rl_corner = chassis.create_corner("rear_left")
    rl_corner.add_attachment_point("upper_front_mount", [-1.3, 0.5, 0.6])
    rl_corner.add_attachment_point("upper_rear_mount", [-1.4, 0.5, 0.58])
    rl_corner.add_attachment_point("lower_front_mount", [-1.25, 0.45, 0.35])
    rl_corner.add_attachment_point("lower_rear_mount", [-1.35, 0.45, 0.33])
    
    # Rear right corner
    rr_corner = chassis.create_corner("rear_right")
    rr_corner.add_attachment_point("upper_front_mount", [-1.3, -0.5, 0.6])
    rr_corner.add_attachment_point("upper_rear_mount", [-1.4, -0.5, 0.58])
    rr_corner.add_attachment_point("lower_front_mount", [-1.25, -0.45, 0.35])
    rr_corner.add_attachment_point("lower_rear_mount", [-1.35, -0.45, 0.33])
    
    print(f"\n{chassis}")
    
    print("\nCorners:")
    for corner_name, corner in chassis.corners.items():
        print(f"  {corner}")
    
    # Test getting all attachment positions
    print("\n--- Testing get_all_attachment_positions ---")
    all_positions = chassis.get_all_attachment_positions()
    print(f"Total attachment points: {len(all_positions)}")
    print(f"First attachment: {all_positions[0]}")
    print(f"Last attachment: {all_positions[-1]}")
    
    # Test getting corner-specific positions
    print("\n--- Testing get_corner_attachment_positions ---")
    fl_positions = chassis.get_corner_attachment_positions("front_left")
    print(f"Front left corner has {len(fl_positions)} attachments")
    for i, pos in enumerate(fl_positions):
        print(f"  {fl_corner.get_attachment_names()[i]}: {pos}")
    
    # Test translation
    print("\n--- Testing translation ---")
    original_centroid = chassis.centroid.copy()
    print(f"Original centroid: {original_centroid}")
    
    chassis.translate([0.1, 0.0, 0.05])
    
    print(f"After translation [0.1, 0, 0.05]:")
    print(f"New centroid: {chassis.centroid}")
    print(f"Centroid change: {chassis.centroid - original_centroid}")
    
    # Test rotation
    print("\n--- Testing rotation about centroid ---")
    angle = np.radians(5)
    R_z = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    before_rotation = chassis.get_corner_attachment_positions("front_left")[0].copy()
    chassis.rotate_about_centroid(R_z)
    after_rotation = chassis.get_corner_attachment_positions("front_left")[0].copy()
    
    print(f"Front left first attachment before rotation: {before_rotation}")
    print(f"Front left first attachment after 5° rotation: {after_rotation}")
    print(f"Position change: {np.linalg.norm(after_rotation - before_rotation):.6f} m")
    
    # Test fitting to targets
    print("\n--- Testing fit_to_attachment_targets ---")
    
    # Create target positions (simulate chassis pitch and heave)
    original_positions = chassis.get_all_attachment_positions()
    
    # Simulate: 20mm heave up, 2° pitch (nose down)
    pitch_angle = np.radians(-2)
    R_pitch = np.array([
        [np.cos(pitch_angle), 0, np.sin(pitch_angle)],
        [0, 1, 0],
        [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]
    ])
    
    centroid = chassis.centroid.copy()
    target_positions = []
    for pos in original_positions:
        # Apply pitch about centroid
        rotated = centroid + R_pitch @ (pos - centroid)
        # Apply heave
        target = rotated + np.array([0, 0, 0.02])
        target_positions.append(target)
    
    print(f"Original centroid: {centroid}")
    
    rms_error = chassis.fit_to_attachment_targets(target_positions)
    
    print(f"\nAfter fitting to targets:")
    print(f"New centroid: {chassis.centroid}")
    print(f"RMS error: {rms_error:.9f} m")
    print(f"Centroid vertical change: {(chassis.centroid - centroid)[2]:.6f} m")
    
    print("\n✓ All tests completed successfully!")
