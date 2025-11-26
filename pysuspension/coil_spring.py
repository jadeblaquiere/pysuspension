import numpy as np
from typing import List, Tuple, Union
from attachment_point import AttachmentPoint
from units import to_mm, from_mm, to_kg, from_kg, to_kg_per_mm, from_kg_per_mm


class CoilSpring:
    """
    Represents a coil spring with two attachment points, mass, spring rate, and preload.

    All positions are stored internally in millimeters (mm).
    Spring rate is stored internally in kg/mm (kilogram-force per millimeter).
    Forces are stored internally in Newtons (N).

    The spring has:
    - Two attachment points (endpoints)
    - Spring rate (stiffness)
    - Initial length (distance between endpoints at construction)
    - Preload force (reaction force at initial length)
    - Mass (concentrated at the centroid)
    - Reaction force that varies linearly with length change
    """

    def __init__(self,
                 endpoint1: Union[np.ndarray, Tuple[float, float, float], AttachmentPoint],
                 endpoint2: Union[np.ndarray, Tuple[float, float, float], AttachmentPoint],
                 spring_rate: float,
                 preload_force: float = 0.0,
                 mass: float = 0.0,
                 name: str = "coil_spring",
                 unit: str = 'mm',
                 spring_rate_unit: str = 'kg/mm',
                 force_unit: str = 'N',
                 mass_unit: str = 'kg'):
        """
        Initialize a coil spring.

        Args:
            endpoint1: 3D position of first endpoint [x, y, z] or AttachmentPoint
            endpoint2: 3D position of second endpoint [x, y, z] or AttachmentPoint
            spring_rate: Spring rate (stiffness)
            preload_force: Preload force at initial length (default: 0.0)
            mass: Mass of the spring (default: 0.0)
            name: Identifier for the spring
            unit: Unit of input positions (default: 'mm')
            spring_rate_unit: Unit of spring rate (default: 'kg/mm')
            force_unit: Unit of force inputs/outputs (default: 'N')
            mass_unit: Unit of mass input (default: 'kg')
        """
        self.name = name

        # Create AttachmentPoint objects if not provided
        if isinstance(endpoint1, AttachmentPoint):
            self.endpoint1 = endpoint1
        else:
            endpoint1_array = np.array(endpoint1, dtype=float)
            if endpoint1_array.shape != (3,):
                raise ValueError("Endpoint1 must be a 3-element array [x, y, z]")
            self.endpoint1 = AttachmentPoint(
                name=f"{name}_endpoint1",
                position=endpoint1_array,
                is_relative=False,
                unit=unit,
                parent_component=self
            )

        if isinstance(endpoint2, AttachmentPoint):
            self.endpoint2 = endpoint2
        else:
            endpoint2_array = np.array(endpoint2, dtype=float)
            if endpoint2_array.shape != (3,):
                raise ValueError("Endpoint2 must be a 3-element array [x, y, z]")
            self.endpoint2 = AttachmentPoint(
                name=f"{name}_endpoint2",
                position=endpoint2_array,
                is_relative=False,
                unit=unit,
                parent_component=self
            )

        # Calculate initial length (constant, in mm)
        self.initial_length = np.linalg.norm(self.endpoint2.position - self.endpoint1.position)

        if self.initial_length < 1e-6:
            raise ValueError("Spring endpoints are too close together (zero length)")

        # Store spring rate in kg/mm (kgf/mm)
        self.spring_rate = to_kg_per_mm(spring_rate, spring_rate_unit)

        # Store preload force in Newtons
        # Convert from input force unit to N
        if force_unit.lower() in ['n', 'newtons', 'newton']:
            self.preload_force = preload_force
        elif force_unit.lower() in ['kgf', 'kg', 'kilogram-force']:
            self.preload_force = preload_force * 9.80665  # kgf to N
        elif force_unit.lower() in ['lbf', 'lb', 'pound-force', 'pounds']:
            self.preload_force = preload_force * 4.44822  # lbf to N
        else:
            raise ValueError(f"Unknown force unit '{force_unit}'. Use 'N', 'kgf', or 'lbf'")

        # Store mass in kg
        self.mass = to_kg(mass, mass_unit)

        # Calculate spring properties in local frame
        self._update_local_frame()

        # Store original state for reset
        self._original_state = {
            'endpoint1': self.endpoint1.position.copy(),
            'endpoint2': self.endpoint2.position.copy(),
            'initial_length': self.initial_length,
            'preload_force': self.preload_force,
        }

    def _update_local_frame(self):
        """Update the local coordinate frame of the spring."""
        # Current length
        self.current_length = np.linalg.norm(self.endpoint2.position - self.endpoint1.position)

        # Spring axis (unit vector from endpoint1 to endpoint2)
        if self.current_length > 1e-6:
            self.axis = (self.endpoint2.position - self.endpoint1.position) / self.current_length
        else:
            self.axis = np.array([0, 0, 1], dtype=float)  # Default direction if collapsed

        # Spring center (centroid) - this is where the mass is located
        self.center = (self.endpoint1.position + self.endpoint2.position) / 2.0
        self.center_of_mass = self.center.copy()

        # Calculate length change from initial
        self.length_change = self.current_length - self.initial_length

        # Calculate current reaction force
        # Force = preload + spring_rate * length_change
        # Note: spring_rate is in kgf/mm, need to convert to N
        force_kgf = self.preload_force / 9.80665 + self.spring_rate * self.length_change
        self.reaction_force_magnitude = force_kgf * 9.80665  # Convert to N

        # Reaction force vector (along axis, positive in compression)
        # If spring is compressed (length_change < 0), force pushes outward
        self.reaction_force = -self.reaction_force_magnitude * self.axis

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

    def get_endpoints(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get both endpoint positions as a list.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of positions in specified unit
        """
        return [self.endpoint1.get_position(unit), self.endpoint2.get_position(unit)]

    def get_all_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
        """
        Get all attachment point positions (both endpoints).

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            List of attachment positions in specified unit
        """
        return self.get_endpoints(unit)

    def get_center(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the center point of the spring (centroid, where mass is located).

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Position in specified unit
        """
        return from_mm(self.center.copy(), unit)

    def get_center_of_mass(self, unit: str = 'mm') -> np.ndarray:
        """
        Get the center of mass position (same as center for a spring).

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            3D position vector in specified unit
        """
        return from_mm(self.center_of_mass.copy(), unit)

    def get_axis(self) -> np.ndarray:
        """
        Get the unit vector along the spring axis (from endpoint1 to endpoint2).

        Returns:
            Unit vector (dimensionless)
        """
        return self.axis.copy()

    def get_current_length(self, unit: str = 'mm') -> float:
        """
        Get the current length of the spring.

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Length in specified unit
        """
        return from_mm(self.current_length, unit)

    def get_initial_length(self, unit: str = 'mm') -> float:
        """
        Get the initial length of the spring (at construction).

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Length in specified unit
        """
        return from_mm(self.initial_length, unit)

    def get_length_change(self, unit: str = 'mm') -> float:
        """
        Get the change in length from initial (negative = compression, positive = extension).

        Args:
            unit: Unit for output (default: 'mm')

        Returns:
            Length change in specified unit
        """
        return from_mm(self.length_change, unit)

    def get_spring_rate(self, unit: str = 'kg/mm') -> float:
        """
        Get the spring rate.

        Args:
            unit: Unit for output (default: 'kg/mm')

        Returns:
            Spring rate in specified unit
        """
        return from_kg_per_mm(self.spring_rate, unit)

    def get_preload_force(self, unit: str = 'N') -> float:
        """
        Get the preload force (force at initial length).

        Args:
            unit: Unit for output (default: 'N')

        Returns:
            Force in specified unit
        """
        return self._convert_force(self.preload_force, 'N', unit)

    def get_reaction_force_magnitude(self, unit: str = 'N') -> float:
        """
        Get the magnitude of the current reaction force.

        Args:
            unit: Unit for output (default: 'N')

        Returns:
            Force magnitude in specified unit
        """
        return self._convert_force(self.reaction_force_magnitude, 'N', unit)

    def get_reaction_force_vector(self, unit: str = 'N') -> np.ndarray:
        """
        Get the reaction force as a 3D vector.

        Args:
            unit: Unit for output (default: 'N')

        Returns:
            Force vector in specified unit
        """
        magnitude_in_unit = self._convert_force(self.reaction_force_magnitude, 'N', unit)
        return -magnitude_in_unit * self.axis

    def _convert_force(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert force between units."""
        from_unit_lower = from_unit.lower()
        to_unit_lower = to_unit.lower()

        # Convert to N first
        if from_unit_lower in ['n', 'newtons', 'newton']:
            value_n = value
        elif from_unit_lower in ['kgf', 'kg', 'kilogram-force']:
            value_n = value * 9.80665
        elif from_unit_lower in ['lbf', 'lb', 'pound-force', 'pounds']:
            value_n = value * 4.44822
        else:
            raise ValueError(f"Unknown force unit '{from_unit}'")

        # Convert from N to target unit
        if to_unit_lower in ['n', 'newtons', 'newton']:
            return value_n
        elif to_unit_lower in ['kgf', 'kg', 'kilogram-force']:
            return value_n / 9.80665
        elif to_unit_lower in ['lbf', 'lb', 'pound-force', 'pounds']:
            return value_n / 4.44822
        else:
            raise ValueError(f"Unknown force unit '{to_unit}'")

    def fit_to_attachment_targets(self, target_positions: List[Union[np.ndarray, Tuple[float, float, float]]],
                                   unit: str = 'mm') -> float:
        """
        Fit the spring to target endpoint positions.
        Updates the spring position to match exact target positions, changing the spring length.

        Args:
            target_positions: List of two target positions [target_endpoint1, target_endpoint2]
            unit: Unit of input target positions (default: 'mm')

        Returns:
            RMS error (should be near zero since we match exactly)
        """
        if len(target_positions) != 2:
            raise ValueError("Expected list of 2 target endpoints")

        target1 = to_mm(np.array(target_positions[0], dtype=float), unit)
        target2 = to_mm(np.array(target_positions[1], dtype=float), unit)

        # Set endpoints to exact target positions
        self.endpoint1.set_position(target1, unit='mm')
        self.endpoint2.set_position(target2, unit='mm')

        # Update geometry (this will recalculate length, reaction force, etc.)
        self._update_local_frame()

        # Calculate RMS error (should be near zero)
        error1 = target1 - self.endpoint1.position
        error2 = target2 - self.endpoint2.position
        rms_error = np.sqrt((np.sum(error1**2) + np.sum(error2**2)) / 2.0)

        return rms_error

    def reset_to_origin(self) -> None:
        """
        Reset the spring to its originally defined position.
        """
        self.endpoint1.set_position(self._original_state['endpoint1'], unit='mm')
        self.endpoint2.set_position(self._original_state['endpoint2'], unit='mm')
        self._update_local_frame()

    def to_dict(self) -> dict:
        """
        Serialize the coil spring to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'name': self.name,
            'endpoint1': self.endpoint1.to_dict(),
            'endpoint2': self.endpoint2.to_dict(),
            'spring_rate': float(self.spring_rate),  # Store in kg/mm
            'spring_rate_unit': 'kg/mm',
            'preload_force': float(self.preload_force),  # Store in N
            'force_unit': 'N',
            'mass': float(self.mass),  # Store in kg
            'mass_unit': 'kg',
            'initial_length': float(self.initial_length)  # Store in mm for validation
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CoilSpring':
        """
        Deserialize a coil spring from a dictionary.

        Args:
            data: Dictionary containing coil spring data

        Returns:
            New CoilSpring instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        # Recreate AttachmentPoint objects
        endpoint1 = AttachmentPoint.from_dict(data['endpoint1'])
        endpoint2 = AttachmentPoint.from_dict(data['endpoint2'])

        # Create the spring
        spring = cls(
            endpoint1=endpoint1,
            endpoint2=endpoint2,
            spring_rate=data['spring_rate'],
            preload_force=data.get('preload_force', 0.0),
            mass=data.get('mass', 0.0),
            name=data['name'],
            unit='mm',  # AttachmentPoints already in mm
            spring_rate_unit=data.get('spring_rate_unit', 'kg/mm'),
            force_unit=data.get('force_unit', 'N'),
            mass_unit=data.get('mass_unit', 'kg')
        )

        # Validate that the deserialized initial length matches
        expected_length = data.get('initial_length')
        if expected_length is not None and abs(spring.initial_length - expected_length) > 1e-3:
            raise ValueError(f"Deserialized spring initial length {spring.initial_length:.3f} mm "
                           f"doesn't match stored length {expected_length:.3f} mm")

        return spring

    def __repr__(self) -> str:
        return (f"CoilSpring('{self.name}',\n"
                f"  endpoint1={self.endpoint1.position} mm,\n"
                f"  endpoint2={self.endpoint2.position} mm,\n"
                f"  current_length={self.current_length:.3f} mm,\n"
                f"  initial_length={self.initial_length:.3f} mm,\n"
                f"  length_change={self.length_change:.3f} mm,\n"
                f"  spring_rate={self.spring_rate:.3f} kg/mm,\n"
                f"  preload_force={self.preload_force:.3f} N,\n"
                f"  reaction_force={self.reaction_force_magnitude:.3f} N,\n"
                f"  mass={self.mass:.3f} kg\n"
                f")")


if __name__ == "__main__":
    print("=" * 70)
    print("COIL SPRING TEST")
    print("=" * 70)

    # Create a coil spring (using mm as input)
    spring = CoilSpring(
        endpoint1=[1000, 500, 300],
        endpoint2=[1000, 500, 500],
        spring_rate=10.0,  # 10 kg/mm
        preload_force=500.0,  # 500 N preload
        mass=2.0,  # 2 kg
        name="test_spring",
        unit='mm',
        spring_rate_unit='kg/mm',
        force_unit='N',
        mass_unit='kg'
    )

    print(f"\n{spring}")

    print("\n--- Initial state ---")
    print(f"Initial length: {spring.get_initial_length()} mm")
    print(f"Current length: {spring.get_current_length()} mm")
    print(f"Length change: {spring.get_length_change()} mm")
    print(f"Reaction force: {spring.get_reaction_force_magnitude()} N")
    print(f"Center of mass: {spring.get_center_of_mass()} mm")

    # Test compression
    print("\n--- Testing compression (move endpoint2 down 50mm) ---")
    new_targets = [
        [1000, 500, 300],  # endpoint1 unchanged
        [1000, 500, 450]   # endpoint2 compressed 50mm
    ]
    spring.fit_to_attachment_targets(new_targets, unit='mm')

    print(f"Current length: {spring.get_current_length()} mm")
    print(f"Length change: {spring.get_length_change()} mm (negative = compression)")
    print(f"Reaction force: {spring.get_reaction_force_magnitude()} N")
    print(f"Reaction force increase: {spring.get_reaction_force_magnitude() - 500.0:.3f} N")

    # Test extension
    print("\n--- Testing extension (move endpoint2 up 30mm from initial) ---")
    new_targets = [
        [1000, 500, 300],  # endpoint1 unchanged
        [1000, 500, 530]   # endpoint2 extended 30mm
    ]
    spring.fit_to_attachment_targets(new_targets, unit='mm')

    print(f"Current length: {spring.get_current_length()} mm")
    print(f"Length change: {spring.get_length_change()} mm (positive = extension)")
    print(f"Reaction force: {spring.get_reaction_force_magnitude()} N")
    print(f"Reaction force change: {spring.get_reaction_force_magnitude() - 500.0:.3f} N")

    # Test unit conversions
    print("\n--- Testing unit conversions ---")
    print(f"Spring rate: {spring.get_spring_rate('kg/mm'):.3f} kg/mm")
    print(f"Spring rate: {spring.get_spring_rate('N/mm'):.3f} N/mm")
    print(f"Spring rate: {spring.get_spring_rate('lbs/in'):.3f} lbs/in")

    print(f"\nReaction force: {spring.get_reaction_force_magnitude('N'):.3f} N")
    print(f"Reaction force: {spring.get_reaction_force_magnitude('kgf'):.3f} kgf")
    print(f"Reaction force: {spring.get_reaction_force_magnitude('lbf'):.3f} lbf")

    # Test serialization (reset first to initial state)
    print("\n--- Testing serialization ---")
    spring.reset_to_origin()
    import json
    data = spring.to_dict()
    json_str = json.dumps(data, indent=2)
    print(f"JSON serialization successful ({len(json_str)} chars)")

    spring_restored = CoilSpring.from_dict(data)
    print(f"Deserialized spring matches: {abs(spring_restored.current_length - spring.current_length) < 1e-6}")

    # Test reset
    print("\n--- Testing reset ---")
    spring.reset_to_origin()
    print(f"After reset - length: {spring.get_current_length()} mm")
    print(f"After reset - reaction force: {spring.get_reaction_force_magnitude()} N")

    print("\n✓ All tests completed successfully!")
