import numpy as np
from typing import List, Tuple, Union, Dict
from units import to_mm, from_mm, to_kg
from chassis_corner import ChassisCorner
from chassis_axle import ChassisAxle


class Chassis:
    """
    Represents a vehicle chassis with multiple corners.
    The chassis behaves as a rigid body with all corners moving together.

    All positions are stored internally in millimeters (mm).
    """

    def __init__(self, name: str = "chassis", mass: float = 0.0, mass_unit: str = 'kg'):
        """
        Initialize a chassis.

        Args:
            name: Identifier for the chassis
            mass: Mass of the chassis (default: 0.0)
            mass_unit: Unit of input mass (default: 'kg')
        """
        self.name = name
        self.mass = to_kg(mass, mass_unit)
        self.corners: Dict[str, ChassisCorner] = {}
        self.axles: Dict[str, ChassisAxle] = {}
        self.centroid = None
        self.center_of_mass = None
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

    def add_axle(self, axle: ChassisAxle) -> None:
        """
        Add an axle to the chassis.

        Args:
            axle: ChassisAxle to add

        Raises:
            ValueError: If axle name already exists or axle belongs to different chassis
        """
        if axle.name in self.axles:
            raise ValueError(f"Axle '{axle.name}' already exists")
        if axle.chassis is not self:
            raise ValueError(f"Axle '{axle.name}' belongs to a different chassis")
        self.axles[axle.name] = axle

    def create_axle(self, name: str, corner_names: List[str]) -> ChassisAxle:
        """
        Create and add a new axle to the chassis.

        Args:
            name: Name for the new axle
            corner_names: List of corner names this axle connects to

        Returns:
            The newly created ChassisAxle
        """
        axle = ChassisAxle(name, self, corner_names)
        self.add_axle(axle)
        return axle

    def get_axle(self, name: str) -> ChassisAxle:
        """
        Get an axle by name.

        Args:
            name: Name of the axle

        Returns:
            ChassisAxle object

        Raises:
            ValueError: If axle not found
        """
        if name not in self.axles:
            raise ValueError(f"Axle '{name}' not found")
        return self.axles[name]

    def _update_centroid(self) -> None:
        """Update the centroid and center of mass of all attachment points.
        The center of mass is located at the centroid."""
        all_points = []

        for corner in self.corners.values():
            all_points.extend(corner.get_attachment_positions())

        if all_points:
            self.centroid = np.mean(all_points, axis=0)
            self.center_of_mass = self.centroid.copy()
        else:
            self.centroid = np.zeros(3)
            self.center_of_mass = np.zeros(3)
    
    def get_all_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get all attachment point positions from all corners.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of all attachment positions in specified unit (ordered by corner, then by attachment within corner)
        """
        positions = []

        for corner_name in sorted(self.corners.keys()):
            corner = self.corners[corner_name]
            positions.extend(corner.get_attachment_positions(unit=unit))

        return positions

    def get_corner_attachment_positions(self, corner_name: str, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get attachment positions for a specific corner.

        Args:
            corner_name: Name of the corner
            unit: Unit for output (default: 'mm')

        Returns:
            List of attachment positions for the specified corner in specified unit
        """
        return self.get_corner(corner_name).get_attachment_positions(unit=unit)

    def get_center_of_mass(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the center of mass position.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            3D position vector in specified unit
        """
        if self.center_of_mass is None:
            return from_mm(np.zeros(3), unit)
        return from_mm(self.center_of_mass.copy(), unit)

    def set_center_of_mass(self, position: Union[np.ndarray, Tuple[float, float, float]], unit: str = 'mm') -> None:
        """
        Set the center of mass position.

        Args:
            position: New center of mass position [x, y, z]
            unit: Unit of input position (default: 'mm')
        """
        self.center_of_mass = to_mm(np.array(position, dtype=float), unit)

    def fit_to_attachment_targets(self, target_positions: List[np.ndarray], unit: str = 'mm') -> float:
        """
        Fit the chassis to target attachment positions using rigid body transformation.
        Updates all corner attachment positions and center of mass based on the optimal rigid body fit.
        The center of mass moves as a rigid body with the chassis.

        Args:
            target_positions: List of target positions in same order as get_all_attachment_positions()
            unit: Unit of input target positions (default: 'mm')

        Returns:
            RMS error of the fit (in mm)
        """
        current_positions = self.get_all_attachment_positions(unit='mm')

        if len(target_positions) != len(current_positions):
            raise ValueError(f"Expected {len(current_positions)} target positions, got {len(target_positions)}")

        if len(current_positions) < 3:
            raise ValueError("Need at least 3 attachment points for rigid body fit")

        # Convert to numpy arrays and to mm
        current_points = np.array(current_positions)
        target_points = np.array([to_mm(np.array(p, dtype=float), unit) for p in target_positions])

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

        # Transform center of mass (if it exists)
        if self.center_of_mass is not None:
            self.center_of_mass = R @ self.center_of_mass + t

        # Update all corner attachment points
        for corner in self.corners.values():
            updated_attachments = []
            for name, pos in corner.attachment_points:
                new_pos = R @ pos + t
                updated_attachments.append((name, new_pos))
            corner.attachment_points = updated_attachments

        # Update all axle attachment points
        for axle in self.axles.values():
            updated_attachments = []
            for name, pos in axle.additional_attachments:
                new_pos = R @ pos + t
                updated_attachments.append((name, new_pos))
            axle.additional_attachments = updated_attachments

        # Update chassis transformation
        self.rotation_matrix = R
        self._update_centroid()

        # Calculate RMS error
        transformed_points = (R @ current_points.T).T + t
        errors = target_points - transformed_points
        rms_error = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

        return rms_error
    
    def translate(self, translation: Union[np.ndarray, Tuple[float, float, float]], unit: str = 'mm') -> None:
        """
        Translate the entire chassis by a given vector.
        The center of mass moves with the chassis.

        Args:
            translation: Translation vector [dx, dy, dz]
            unit: Unit of input translation (default: 'mm')
        """
        t = to_mm(np.array(translation, dtype=float), unit)

        # Translate center of mass (if it exists)
        if self.center_of_mass is not None:
            self.center_of_mass = self.center_of_mass + t

        # Translate all corner attachment points
        for corner in self.corners.values():
            updated_attachments = []
            for name, pos in corner.attachment_points:
                new_pos = pos + t
                updated_attachments.append((name, new_pos))
            corner.attachment_points = updated_attachments

        # Translate all axle attachment points
        for axle in self.axles.values():
            updated_attachments = []
            for name, pos in axle.additional_attachments:
                new_pos = pos + t
                updated_attachments.append((name, new_pos))
            axle.additional_attachments = updated_attachments

        self._update_centroid()
    
    def rotate_about_centroid(self, rotation_matrix: np.ndarray) -> None:
        """
        Rotate the chassis about its centroid.
        The center of mass rotates with the chassis about the centroid.

        Args:
            rotation_matrix: 3x3 rotation matrix
        """
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")

        # Rotate center of mass about centroid (if it exists)
        if self.center_of_mass is not None:
            self.center_of_mass = self.centroid + rotation_matrix @ (self.center_of_mass - self.centroid)

        # Rotate all corner attachment points
        for corner in self.corners.values():
            updated_attachments = []
            for name, pos in corner.attachment_points:
                new_pos = self.centroid + rotation_matrix @ (pos - self.centroid)
                updated_attachments.append((name, new_pos))
            corner.attachment_points = updated_attachments

        # Rotate all axle attachment points
        for axle in self.axles.values():
            updated_attachments = []
            for name, pos in axle.additional_attachments:
                new_pos = self.centroid + rotation_matrix @ (pos - self.centroid)
                updated_attachments.append((name, new_pos))
            axle.additional_attachments = updated_attachments

        self.rotation_matrix = rotation_matrix @ self.rotation_matrix
        self._update_centroid()
    
    def __repr__(self) -> str:
        total_attachments = sum(len(c.attachment_points) for c in self.corners.values())
        centroid_str = f"{self.centroid} mm" if self.centroid is not None else "None"
        com_str = f"{self.center_of_mass} mm" if self.center_of_mass is not None else "None"
        return (f"Chassis('{self.name}',\n"
                f"  corners={len(self.corners)},\n"
                f"  axles={len(self.axles)},\n"
                f"  total_attachments={total_attachments},\n"
                f"  mass={self.mass:.3f} kg,\n"
                f"  centroid={centroid_str},\n"
                f"  center_of_mass={com_str}\n"
                f")")


if __name__ == "__main__":
    print("=" * 60)
    print("CHASSIS TEST (with unit support)")
    print("=" * 60)

    # Create a chassis with four corners (using meters as input, stored as mm internally)
    chassis = Chassis(name="test_chassis")

    print("\n--- Creating four corners ---")

    # Front left corner
    fl_corner = chassis.create_corner("front_left")
    fl_corner.add_attachment_point("upper_front_mount", [1.4, 0.5, 0.6], unit='m')
    fl_corner.add_attachment_point("upper_rear_mount", [1.3, 0.5, 0.58], unit='m')
    fl_corner.add_attachment_point("lower_front_mount", [1.45, 0.45, 0.35], unit='m')
    fl_corner.add_attachment_point("lower_rear_mount", [1.35, 0.45, 0.33], unit='m')

    # Front right corner
    fr_corner = chassis.create_corner("front_right")
    fr_corner.add_attachment_point("upper_front_mount", [1.4, -0.5, 0.6], unit='m')
    fr_corner.add_attachment_point("upper_rear_mount", [1.3, -0.5, 0.58], unit='m')
    fr_corner.add_attachment_point("lower_front_mount", [1.45, -0.45, 0.35], unit='m')
    fr_corner.add_attachment_point("lower_rear_mount", [1.35, -0.45, 0.33], unit='m')

    # Rear left corner
    rl_corner = chassis.create_corner("rear_left")
    rl_corner.add_attachment_point("upper_front_mount", [-1.3, 0.5, 0.6], unit='m')
    rl_corner.add_attachment_point("upper_rear_mount", [-1.4, 0.5, 0.58], unit='m')
    rl_corner.add_attachment_point("lower_front_mount", [-1.25, 0.45, 0.35], unit='m')
    rl_corner.add_attachment_point("lower_rear_mount", [-1.35, 0.45, 0.33], unit='m')

    # Rear right corner
    rr_corner = chassis.create_corner("rear_right")
    rr_corner.add_attachment_point("upper_front_mount", [-1.3, -0.5, 0.6], unit='m')
    rr_corner.add_attachment_point("upper_rear_mount", [-1.4, -0.5, 0.58], unit='m')
    rr_corner.add_attachment_point("lower_front_mount", [-1.25, -0.45, 0.35], unit='m')
    rr_corner.add_attachment_point("lower_rear_mount", [-1.35, -0.45, 0.33], unit='m')

    print(f"\n{chassis}")

    print("\nCorners:")
    for corner_name, corner in chassis.corners.items():
        print(f"  {corner}")

    # Test getting all attachment positions
    print("\n--- Testing get_all_attachment_positions ---")
    all_positions_mm = chassis.get_all_attachment_positions(unit='mm')
    all_positions_m = chassis.get_all_attachment_positions(unit='m')
    print(f"Total attachment points: {len(all_positions_mm)}")
    print(f"First attachment (mm): {all_positions_mm[0]}")
    print(f"First attachment (m): {all_positions_m[0]}")
    print(f"Last attachment (mm): {all_positions_mm[-1]}")
    print(f"Last attachment (m): {all_positions_m[-1]}")

    # Test getting corner-specific positions
    print("\n--- Testing get_corner_attachment_positions ---")
    fl_positions_mm = chassis.get_corner_attachment_positions("front_left", unit='mm')
    fl_positions_m = chassis.get_corner_attachment_positions("front_left", unit='m')
    print(f"Front left corner has {len(fl_positions_mm)} attachments")
    for i, (pos_mm, pos_m) in enumerate(zip(fl_positions_mm, fl_positions_m)):
        print(f"  {fl_corner.get_attachment_names()[i]} (mm): {pos_mm}")
        print(f"  {fl_corner.get_attachment_names()[i]} (m): {pos_m}")

    # Test translation
    print("\n--- Testing translation ---")
    original_centroid = chassis.centroid.copy()
    print(f"Original centroid (mm): {original_centroid}")
    print(f"Original centroid (m): {from_mm(original_centroid, 'm')}")

    chassis.translate([0.1, 0.0, 0.05], unit='m')  # Translate 100mm x, 0 y, 50mm z

    print(f"After translation [0.1m, 0, 0.05m]:")
    print(f"New centroid (mm): {chassis.centroid}")
    print(f"New centroid (m): {from_mm(chassis.centroid, 'm')}")
    print(f"Centroid change (mm): {chassis.centroid - original_centroid}")

    # Test rotation
    print("\n--- Testing rotation about centroid ---")
    angle = np.radians(5)
    R_z = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    before_rotation = chassis.get_corner_attachment_positions("front_left", unit='mm')[0].copy()
    chassis.rotate_about_centroid(R_z)
    after_rotation = chassis.get_corner_attachment_positions("front_left", unit='mm')[0].copy()

    print(f"Front left first attachment before rotation (mm): {before_rotation}")
    print(f"Front left first attachment after 5° rotation (mm): {after_rotation}")
    print(f"Position change (mm): {np.linalg.norm(after_rotation - before_rotation):.3f}")
    print(f"Position change (m): {from_mm(np.linalg.norm(after_rotation - before_rotation), 'm'):.6f}")

    # Test fitting to targets
    print("\n--- Testing fit_to_attachment_targets ---")

    # Create target positions (simulate chassis pitch and heave)
    original_positions = chassis.get_all_attachment_positions(unit='mm')

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
        # Apply heave (20mm up)
        target = rotated + np.array([0, 0, 20.0])
        target_positions.append(target)

    print(f"Original centroid (mm): {centroid}")
    print(f"Original centroid (m): {from_mm(centroid, 'm')}")

    rms_error = chassis.fit_to_attachment_targets(target_positions, unit='mm')

    print(f"\nAfter fitting to targets:")
    print(f"New centroid (mm): {chassis.centroid}")
    print(f"New centroid (m): {from_mm(chassis.centroid, 'm')}")
    print(f"RMS error (mm): {rms_error:.6f}")
    print(f"RMS error (m): {from_mm(rms_error, 'm'):.9f}")
    print(f"Centroid vertical change (mm): {(chassis.centroid - centroid)[2]:.3f}")
    print(f"Centroid vertical change (m): {from_mm((chassis.centroid - centroid)[2], 'm'):.6f}")

    print("\n✓ All tests completed successfully!")
