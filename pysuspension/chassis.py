import numpy as np
from typing import List, Tuple, Union, Dict, Optional, TYPE_CHECKING
from .rigid_body import RigidBody
from .units import to_mm, from_mm
from .chassis_corner import ChassisCorner
from .chassis_axle import ChassisAxle

if TYPE_CHECKING:
    from .suspension_joint import SuspensionJoint
    from .attachment_point import AttachmentPoint

# Component type registry for deserialization
# Import these only when needed to avoid circular imports
def _get_component_type_registry():
    """
    Get the component type registry for serialization/deserialization.

    Imports are done inside the function to avoid circular import issues.

    Returns:
        Dictionary mapping type names to component classes
    """
    from .control_arm import ControlArm
    from .suspension_knuckle import SuspensionKnuckle
    from .suspension_link import SuspensionLink
    from .steering_rack import SteeringRack
    from .coil_spring import CoilSpring

    return {
        'ControlArm': ControlArm,
        'SuspensionKnuckle': SuspensionKnuckle,
        'SuspensionLink': SuspensionLink,
        'SteeringRack': SteeringRack,
        'CoilSpring': CoilSpring,
    }


class Chassis(RigidBody):
    """
    Represents a vehicle chassis with multiple corners.
    The chassis behaves as a rigid body with all corners moving together.

    Extends RigidBody to provide rigid body transformation behavior.
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
        super().__init__(name=name, mass=mass, mass_unit=mass_unit)

        self.corners: Dict[str, ChassisCorner] = {}
        self.axles: Dict[str, ChassisAxle] = {}

        # Component tracking for comprehensive serialization
        self.components: Dict[str, RigidBody] = {}  # All suspension components
        self.joints: Dict[str, 'SuspensionJoint'] = {}  # All joints in system

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

    def add_component(self, component: RigidBody) -> None:
        """
        Register a suspension component with this chassis.

        This enables comprehensive serialization of the entire suspension system.
        The component's name attribute is used as the identifier.

        Args:
            component: Suspension component (ControlArm, SuspensionKnuckle, etc.)

        Raises:
            ValueError: If a component with this name already exists
        """
        if component.name in self.components:
            raise ValueError(f"Component '{component.name}' already exists")
        self.components[component.name] = component

    def add_joint(self, joint: 'SuspensionJoint') -> None:
        """
        Register a suspension joint with this chassis.

        This enables comprehensive serialization of all joint connections.

        Args:
            joint: SuspensionJoint connecting components

        Raises:
            ValueError: If joint name already exists
        """
        if joint.name in self.joints:
            raise ValueError(f"Joint '{joint.name}' already exists")
        self.joints[joint.name] = joint

    def get_all_components(self) -> Dict[str, RigidBody]:
        """
        Get all registered suspension components.

        Returns:
            Dictionary mapping component names to component objects
        """
        return self.components.copy()

    def get_all_joints(self) -> Dict[str, 'SuspensionJoint']:
        """
        Get all registered suspension joints.

        Returns:
            Dictionary mapping joint names to joint objects
        """
        return self.joints.copy()

    def _update_centroid(self) -> None:
        """
        Update the centroid and center of mass from all corner attachment points.

        Overrides parent to gather points from all corners.
        """
        all_points = []

        for corner in self.corners.values():
            all_points.extend(corner.get_attachment_positions())

        if all_points:
            self.centroid = np.mean(all_points, axis=0)
            self.center_of_mass = self.centroid.copy()
        else:
            self.centroid = np.zeros(3)
            self.center_of_mass = np.zeros(3)

        # Update original state only if not frozen
        if not self._original_state_frozen:
            self._original_state['centroid'] = self.centroid.copy() if self.centroid is not None else None
            self._original_state['center_of_mass'] = self.center_of_mass.copy() if self.center_of_mass is not None else None

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

    def _apply_transformation(self, R: np.ndarray, t: np.ndarray) -> None:
        """
        Apply rigid body transformation to all corners and axles.

        Overrides parent to transform corner and axle attachment points.

        Args:
            R: 3x3 rotation matrix
            t: 3D translation vector (in mm)
        """
        # Transform center of mass (if it exists)
        if self.center_of_mass is not None:
            self.center_of_mass = R @ self.center_of_mass + t

        # Update all corner attachment points
        for corner in self.corners.values():
            for attachment in corner.attachment_points:
                new_pos = R @ attachment.position + t
                attachment.set_position(new_pos, unit='mm')

        # Update all axle attachment points
        for axle in self.axles.values():
            for attachment in axle.attachment_points:
                new_pos = R @ attachment.position + t
                attachment.set_position(new_pos, unit='mm')

        # Update chassis transformation
        self.rotation_matrix = R
        self._update_centroid()

    def rotate_about_centroid(self, rotation_matrix: np.ndarray) -> None:
        """
        Rotate the chassis about its centroid.

        This is an alias for rotate_about_center() for backward compatibility.

        Args:
            rotation_matrix: 3x3 rotation matrix
        """
        self.rotate_about_center(rotation_matrix)

    def reset_to_origin(self) -> None:
        """
        Reset the chassis and all its components to their originally defined positions.

        This resets:
        - All corner attachment points
        - All axle attachment points
        - Chassis centroid and center of mass
        - Chassis rotation matrix
        """
        # Reset all corners
        for corner in self.corners.values():
            corner.reset_to_origin()

        # Reset all axles
        for axle in self.axles.values():
            axle.reset_to_origin()

        # Reset chassis transformation
        self.rotation_matrix = self._original_state['rotation_matrix'].copy()

        # Unfreeze original state to allow subsequent transformations
        self._original_state_frozen = False

        # Recalculate centroid and center of mass
        self._update_centroid()

    def to_dict(self) -> dict:
        """
        Serialize the chassis and all its components to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'name': self.name,
            'mass': float(self.mass),  # Store in kg
            'mass_unit': 'kg',
            'corners': {name: corner.to_dict() for name, corner in self.corners.items()},
            'axles': {name: axle.to_dict() for name, axle in self.axles.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Chassis':
        """
        Deserialize a chassis from a dictionary.

        Args:
            data: Dictionary containing chassis data

        Returns:
            New Chassis instance with all components

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # Create the chassis
        chassis = cls(
            name=data['name'],
            mass=data.get('mass', 0.0),
            mass_unit=data.get('mass_unit', 'kg')
        )

        # Add corners
        for corner_name, corner_data in data.get('corners', {}).items():
            corner = ChassisCorner.from_dict(corner_data)
            chassis.add_corner(corner)

        # Add axles (after corners, since axles reference corners)
        for axle_name, axle_data in data.get('axles', {}).items():
            axle = ChassisAxle.from_dict(axle_data, chassis)
            chassis.add_axle(axle)

        return chassis

    def to_dict_full(self) -> dict:
        """
        Serialize the entire suspension system including all components and joints.

        This provides comprehensive serialization that captures:
        - Chassis with corners and axles
        - All registered suspension components
        - All suspension joints connecting components

        Returns:
            Dictionary representation of the complete suspension system

        Note:
            Uses to_dict_full() for comprehensive serialization.
            Use to_dict() for chassis-only serialization (backward compatible).
        """
        from .suspension_joint import SuspensionJoint

        # Serialize chassis (corners and axles)
        chassis_data = self.to_dict()

        # Serialize all registered components
        components_data = {}
        for comp_name, component in self.components.items():
            component_type = type(component).__name__
            components_data[comp_name] = {
                'type': component_type,
                'data': component.to_dict()
            }

        # Serialize all registered joints
        joints_data = {}
        for joint_name, joint in self.joints.items():
            # Build connection references
            connections = []
            for point in joint.get_all_attachment_points():
                # Find which component owns this point
                component_ref = self._find_component_reference(point)
                if component_ref:
                    connections.append({
                        'component': component_ref,
                        'point_name': point.name
                    })

            joints_data[joint_name] = {
                'joint_type': joint.joint_type.value,
                'connections': connections
            }

        return {
            'chassis': chassis_data,
            'components': components_data,
            'joints': joints_data
        }

    def _find_component_reference(self, point: 'AttachmentPoint') -> Optional[str]:
        """
        Find the reference path to a component owning an attachment point.

        Args:
            point: AttachmentPoint to locate

        Returns:
            Component reference string (e.g., 'chassis.corners.front_left' or 'upper_arm_fl')
            None if not found
        """
        # Check registered components
        for comp_name, component in self.components.items():
            if hasattr(component, 'attachment_points'):
                for ap in component.attachment_points:
                    if ap is point:
                        return comp_name
            # Check SuspensionLink endpoints
            if hasattr(component, 'endpoint1') and component.endpoint1 is point:
                return comp_name
            if hasattr(component, 'endpoint2') and component.endpoint2 is point:
                return comp_name
            # Check SteeringRack housing attachment points
            if hasattr(component, 'housing'):
                for ap in component.housing.attachment_points:
                    if ap is point:
                        return component.housing.name
            if hasattr(component, 'left_inner_pivot') and component.left_inner_pivot is point:
                return comp_name
            if hasattr(component, 'right_inner_pivot') and component.right_inner_pivot is point:
                return comp_name

        # Check chassis corners
        for corner_name, corner in self.corners.items():
            for ap in corner.attachment_points:
                if ap is point:
                    return f"chassis.corners.{corner_name}"

        # Check chassis axles
        for axle_name, axle in self.axles.items():
            for ap in axle.attachment_points:
                if ap is point:
                    return f"chassis.axles.{axle_name}"

        # Check chassis attachment points (if any)
        if hasattr(self, 'attachment_points'):
            for ap in self.attachment_points:
                if ap is point:
                    return "chassis"

        return None

    @classmethod
    def from_dict_full(cls, data: dict) -> 'Chassis':
        """
        Deserialize a complete suspension system from a dictionary.

        Reconstructs:
        - Chassis with corners and axles
        - All suspension components
        - All suspension joints with proper connections

        Args:
            data: Dictionary containing complete suspension system data

        Returns:
            Chassis instance with all components and joints registered

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid or component types unknown
        """
        from .suspension_joint import SuspensionJoint
        from .joint_types import JointType

        # Get component type registry
        component_registry = _get_component_type_registry()

        # Phase 1: Deserialize chassis
        chassis = cls.from_dict(data['chassis'])

        # Phase 2: Deserialize all components
        for comp_name, comp_data in data.get('components', {}).items():
            comp_type = comp_data['type']
            comp_dict = comp_data['data']

            if comp_type not in component_registry:
                raise ValueError(f"Unknown component type: {comp_type}")

            component_class = component_registry[comp_type]
            component = component_class.from_dict(comp_dict)

            # Register component with chassis (uses component.name automatically)
            chassis.add_component(component)

        # Phase 3: Build lookup table for attachment points
        point_lookup = chassis._build_attachment_point_lookup()

        # Phase 4: Reconstruct all joints
        for joint_name, joint_data in data.get('joints', {}).items():
            joint_type = JointType(joint_data['joint_type'])
            joint = SuspensionJoint(joint_name, joint_type)

            # Reconnect all attachment points
            for conn in joint_data['connections']:
                component_ref = conn['component']
                point_name = conn['point_name']

                # Lookup the attachment point
                point = point_lookup.get(f"{component_ref}.{point_name}")
                if point is None:
                    raise ValueError(
                        f"Could not find attachment point '{point_name}' "
                        f"in component '{component_ref}' for joint '{joint_name}'"
                    )

                joint.add_attachment_point(point)

            # Register joint with chassis
            chassis.add_joint(joint)

        return chassis

    def _build_attachment_point_lookup(self) -> Dict[str, 'AttachmentPoint']:
        """
        Build a lookup table mapping component references to attachment points.

        Returns:
            Dictionary mapping 'component_ref.point_name' to AttachmentPoint objects
        """
        lookup = {}

        # Add registered components
        for comp_name, component in self.components.items():
            if hasattr(component, 'attachment_points'):
                for ap in component.attachment_points:
                    lookup[f"{comp_name}.{ap.name}"] = ap
            # Handle SuspensionLink endpoints
            if hasattr(component, 'endpoint1'):
                lookup[f"{comp_name}.{component.endpoint1.name}"] = component.endpoint1
            if hasattr(component, 'endpoint2'):
                lookup[f"{comp_name}.{component.endpoint2.name}"] = component.endpoint2

        # Add chassis corners
        for corner_name, corner in self.corners.items():
            for ap in corner.attachment_points:
                lookup[f"chassis.corners.{corner_name}.{ap.name}"] = ap

        # Add chassis axles
        for axle_name, axle in self.axles.items():
            for ap in axle.attachment_points:
                lookup[f"chassis.axles.{axle_name}.{ap.name}"] = ap

        # Add chassis attachment points (if any)
        if hasattr(self, 'attachment_points'):
            for ap in self.attachment_points:
                lookup[f"chassis.{ap.name}"] = ap

        return lookup

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

    # Test inheritance
    print("\n--- Testing inheritance ---")
    print(f"isinstance(chassis, RigidBody): {isinstance(chassis, RigidBody)}")

    print("\n✓ All tests completed successfully!")
