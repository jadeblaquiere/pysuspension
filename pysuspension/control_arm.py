import numpy as np
from typing import List, Tuple, Union
from suspension_link import SuspensionLink


class ControlArm:
    """
    Represents a control arm consisting of multiple suspension links and attachment points
    that move together as a rigid body.
    """
    
    def __init__(self, name: str = "control_arm"):
        """
        Initialize a control arm.
        
        Args:
            name: Identifier for the control arm
        """
        self.name = name
        self.links: List[SuspensionLink] = []
        self.additional_attachments: List[Tuple[str, np.ndarray]] = []
        
        # Rigid body transformation (identity initially)
        self.centroid = None
        self.rotation_matrix = np.eye(3)
        
    def add_link(self, link: SuspensionLink) -> None:
        """
        Add a suspension link to the control arm.
        
        Args:
            link: SuspensionLink to add
        """
        self.links.append(link)
        self._update_centroid()
    
    def add_attachment_point(self, name: str, position: Union[np.ndarray, Tuple[float, float, float]]) -> None:
        """
        Add an additional attachment point to the control arm.
        
        Args:
            name: Identifier for the attachment point
            position: 3D position [x, y, z]
        """
        pos = np.array(position, dtype=float)
        if pos.shape != (3,):
            raise ValueError("Position must be a 3-element array [x, y, z]")
        self.additional_attachments.append((name, pos))
        self._update_centroid()
    
    def _update_centroid(self) -> None:
        """Update the centroid of all attachment points."""
        all_points = []
        
        # Collect all link endpoints
        for link in self.links:
            all_points.append(link.endpoint1)
            all_points.append(link.endpoint2)
        
        # Collect additional attachment points
        for _, pos in self.additional_attachments:
            all_points.append(pos)
        
        if all_points:
            self.centroid = np.mean(all_points, axis=0)
        else:
            self.centroid = np.zeros(3)
    
    def get_all_attachment_positions(self) -> List[np.ndarray]:
        """
        Get all unique attachment point positions.
        Link endpoints and additional attachments are included, with duplicates removed.
        
        Returns:
            List of unique attachment positions
        """
        positions = []
        position_set = []  # For duplicate checking
        
        # Add link endpoints
        for link in self.links:
            for endpoint in link.get_endpoints():
                # Check if this position is already in the list (within tolerance)
                is_duplicate = False
                for existing_pos in position_set:
                    if np.linalg.norm(endpoint - existing_pos) < 1e-9:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    positions.append(endpoint.copy())
                    position_set.append(endpoint.copy())
        
        # Add additional attachment points
        for _, pos in self.additional_attachments:
            # Check for duplicates
            is_duplicate = False
            for existing_pos in position_set:
                if np.linalg.norm(pos - existing_pos) < 1e-9:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                positions.append(pos.copy())
                position_set.append(pos.copy())
        
        return positions
    
    def fit_to_attachment_targets(self, target_positions: List[np.ndarray]) -> float:
        """
        Fit the control arm to target attachment positions using rigid body transformation.
        Updates all link positions based on the optimal rigid body fit.
        
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
        
        # Update all link positions
        for link in self.links:
            new_endpoint1 = R @ link.endpoint1 + t
            new_endpoint2 = R @ link.endpoint2 + t
            link.endpoint1 = new_endpoint1
            link.endpoint2 = new_endpoint2
            link._update_local_frame()
        
        # Update additional attachment points
        updated_attachments = []
        for name, pos in self.additional_attachments:
            new_pos = R @ pos + t
            updated_attachments.append((name, new_pos))
        self.additional_attachments = updated_attachments
        
        # Update control arm transformation
        self.rotation_matrix = R
        self._update_centroid()
        
        # Calculate RMS error
        transformed_points = (R @ current_points.T).T + t
        errors = target_points - transformed_points
        rms_error = np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        
        return rms_error
    
    def __repr__(self) -> str:
        return (f"ControlArm('{self.name}',\n"
                f"  links={len(self.links)},\n"
                f"  additional_attachments={len(self.additional_attachments)},\n"
                f"  total_unique_attachments={len(self.get_all_attachment_positions())}\n"
                f")")


if __name__ == "__main__":
    print("=" * 60)
    print("CONTROL ARM TEST")
    print("=" * 60)
    
    # Create a control arm with multiple links
    control_arm = ControlArm(name="test_control_arm")
    
    # Add two links (forming a triangulated control arm)
    link1 = SuspensionLink(
        endpoint1=[1.3, 0.4, 0.55],   # Front chassis mount
        endpoint2=[1.5, 0.75, 0.6],   # Ball joint
        name="front_link"
    )
    
    link2 = SuspensionLink(
        endpoint1=[1.2, 0.4, 0.55],   # Rear chassis mount
        endpoint2=[1.5, 0.75, 0.6],   # Ball joint (shared with link1)
        name="rear_link"
    )
    
    control_arm.add_link(link1)
    control_arm.add_link(link2)
    
    # Add an additional attachment point (e.g., for a sway bar link)
    control_arm.add_attachment_point("sway_bar_mount", [1.25, 0.5, 0.5])
    
    print(f"\n{control_arm}")
    
    print("\nAll attachment positions:")
    all_positions = control_arm.get_all_attachment_positions()
    for i, pos in enumerate(all_positions):
        print(f"  Point {i}: {pos}")
    
    print(f"\nNote: Ball joint is shared between links, so only counted once")
    print(f"Total unique attachments: {len(all_positions)}")
    
    # Test fitting control arm to new targets
    print("\n--- Testing control arm fit to targets ---")
    
    # Create target positions (e.g., suspension compressed and rotated)
    original_positions = control_arm.get_all_attachment_positions()
    # Simulate movement: translate and add some variation
    target_positions = [pos + np.array([0.02, -0.01, -0.04]) for pos in original_positions]
    
    print(f"Original centroid: {control_arm.centroid}")
    
    rms_error = control_arm.fit_to_attachment_targets(target_positions)
    
    print(f"\nAfter fitting:")
    print(f"New centroid: {control_arm.centroid}")
    print(f"RMS error: {rms_error:.6f} m")
    
    print(f"\nLink lengths maintained:")
    for link in control_arm.links:
        print(f"  {link.name}: {link.get_length():.6f} m")
