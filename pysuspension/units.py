"""
Unit conversion utilities for suspension modeling.

All internal calculations are performed in:
- millimeters (mm) for length
- kilograms (kg) for mass

This module provides conversion functions to/from various units.
"""
import numpy as np
from typing import Union


# Length conversion factors to millimeters (base unit)
UNIT_TO_MM = {
    'mm': 1.0,
    'millimeter': 1.0,
    'millimeters': 1.0,
    'cm': 10.0,
    'centimeter': 10.0,
    'centimeters': 10.0,
    'm': 1000.0,
    'meter': 1000.0,
    'meters': 1000.0,
    'in': 25.4,
    'inch': 25.4,
    'inches': 25.4,
    'ft': 304.8,
    'foot': 304.8,
    'feet': 304.8,
}

# Conversion factors from millimeters
MM_TO_UNIT = {unit: 1.0 / factor for unit, factor in UNIT_TO_MM.items()}


# Mass conversion factors to kilograms (base unit)
MASS_UNIT_TO_KG = {
    'kg': 1.0,
    'kilogram': 1.0,
    'kilograms': 1.0,
    'g': 0.001,
    'gram': 0.001,
    'grams': 0.001,
    'mg': 0.000001,
    'milligram': 0.000001,
    'milligrams': 0.000001,
    'lb': 0.453592,
    'lbs': 0.453592,
    'pound': 0.453592,
    'pounds': 0.453592,
    'oz': 0.0283495,
    'ounce': 0.0283495,
    'ounces': 0.0283495,
    'ton': 1000.0,
    'tonne': 1000.0,
    'metric_ton': 1000.0,
}

# Conversion factors from kilograms
KG_TO_MASS_UNIT = {unit: 1.0 / factor for unit, factor in MASS_UNIT_TO_KG.items()}


# Spring rate conversion factors to kg/mm (base unit)
# Note: kg here refers to kilogram-force (kgf), not mass
SPRING_RATE_UNIT_TO_KG_PER_MM = {
    'kg/mm': 1.0,
    'kgf/mm': 1.0,
    'n/mm': 1.0 / 9.80665,  # 1 kgf = 9.80665 N
    'lbs/in': 0.453592 / 25.4,  # 1 lbf = 0.453592 kgf, 1 in = 25.4 mm
    'lbf/in': 0.453592 / 25.4,
}

# Conversion factors from kg/mm
KG_PER_MM_TO_SPRING_RATE_UNIT = {unit: 1.0 / factor for unit, factor in SPRING_RATE_UNIT_TO_KG_PER_MM.items()}


def validate_spring_rate_unit(unit: str) -> str:
    """
    Validate and normalize spring rate unit string.

    Args:
        unit: Spring rate unit string (e.g., 'kg/mm', 'N/mm', 'lbs/in')

    Returns:
        Normalized spring rate unit string

    Raises:
        ValueError: If unit is not recognized
    """
    unit_lower = unit.lower().strip()
    if unit_lower not in SPRING_RATE_UNIT_TO_KG_PER_MM:
        valid_units = sorted(set(['kg/mm', 'N/mm', 'lbs/in']))
        raise ValueError(f"Unknown spring rate unit '{unit}'. Valid units: {valid_units}")
    return unit_lower


def to_kg_per_mm(value: Union[float, np.ndarray], from_unit: str = 'kg/mm') -> Union[float, np.ndarray]:
    """
    Convert a spring rate value from the specified unit to kg/mm (base unit).

    Args:
        value: Value or array to convert
        from_unit: Source unit (default: 'kg/mm')

    Returns:
        Value in kg/mm
    """
    unit = validate_spring_rate_unit(from_unit)
    return value * SPRING_RATE_UNIT_TO_KG_PER_MM[unit]


def from_kg_per_mm(value: Union[float, np.ndarray], to_unit: str = 'kg/mm') -> Union[float, np.ndarray]:
    """
    Convert a spring rate value from kg/mm (base unit) to the specified unit.

    Args:
        value: Value or array in kg/mm
        to_unit: Target unit (default: 'kg/mm')

    Returns:
        Value in target unit
    """
    unit = validate_spring_rate_unit(to_unit)
    return value * KG_PER_MM_TO_SPRING_RATE_UNIT[unit]


def convert_spring_rate(value: Union[float, np.ndarray], from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    Convert a spring rate value from one unit to another.

    Args:
        value: Value or array to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Value in target unit
    """
    kg_per_mm_value = to_kg_per_mm(value, from_unit)
    return from_kg_per_mm(kg_per_mm_value, to_unit)


def validate_unit(unit: str) -> str:
    """
    Validate and normalize length unit string.

    Args:
        unit: Unit string (e.g., 'mm', 'm', 'in')

    Returns:
        Normalized unit string

    Raises:
        ValueError: If unit is not recognized
    """
    unit_lower = unit.lower().strip()
    if unit_lower not in UNIT_TO_MM:
        valid_units = sorted(set(['mm', 'cm', 'm', 'in', 'ft']))
        raise ValueError(f"Unknown unit '{unit}'. Valid units: {valid_units}")
    return unit_lower


def to_mm(value: Union[float, np.ndarray], from_unit: str = 'mm') -> Union[float, np.ndarray]:
    """
    Convert a value from the specified unit to millimeters (base unit).

    Args:
        value: Value or array to convert
        from_unit: Source unit (default: 'mm')

    Returns:
        Value in millimeters
    """
    unit = validate_unit(from_unit)
    return value * UNIT_TO_MM[unit]


def from_mm(value: Union[float, np.ndarray], to_unit: str = 'mm') -> Union[float, np.ndarray]:
    """
    Convert a value from millimeters (base unit) to the specified unit.

    Args:
        value: Value or array in millimeters
        to_unit: Target unit (default: 'mm')

    Returns:
        Value in target unit
    """
    unit = validate_unit(to_unit)
    return value * MM_TO_UNIT[unit]


def convert(value: Union[float, np.ndarray], from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    Convert a value from one unit to another.

    Args:
        value: Value or array to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Value in target unit
    """
    mm_value = to_mm(value, from_unit)
    return from_mm(mm_value, to_unit)


def format_length(value: float, unit: str = 'mm', precision: int = 3) -> str:
    """
    Format a length value with units for display.

    Args:
        value: Value in the specified unit
        unit: Unit string
        precision: Number of decimal places

    Returns:
        Formatted string (e.g., "123.456 mm")
    """
    unit = validate_unit(unit)
    return f"{value:.{precision}f} {unit}"


def validate_mass_unit(unit: str) -> str:
    """
    Validate and normalize mass unit string.

    Args:
        unit: Mass unit string (e.g., 'kg', 'g', 'lb')

    Returns:
        Normalized mass unit string

    Raises:
        ValueError: If unit is not recognized
    """
    unit_lower = unit.lower().strip()
    if unit_lower not in MASS_UNIT_TO_KG:
        valid_units = sorted(set(['kg', 'g', 'mg', 'lb', 'oz', 'ton']))
        raise ValueError(f"Unknown mass unit '{unit}'. Valid units: {valid_units}")
    return unit_lower


def to_kg(value: Union[float, np.ndarray], from_unit: str = 'kg') -> Union[float, np.ndarray]:
    """
    Convert a mass value from the specified unit to kilograms (base unit).

    Args:
        value: Value or array to convert
        from_unit: Source unit (default: 'kg')

    Returns:
        Value in kilograms
    """
    unit = validate_mass_unit(from_unit)
    return value * MASS_UNIT_TO_KG[unit]


def from_kg(value: Union[float, np.ndarray], to_unit: str = 'kg') -> Union[float, np.ndarray]:
    """
    Convert a mass value from kilograms (base unit) to the specified unit.

    Args:
        value: Value or array in kilograms
        to_unit: Target unit (default: 'kg')

    Returns:
        Value in target unit
    """
    unit = validate_mass_unit(to_unit)
    return value * KG_TO_MASS_UNIT[unit]


def convert_mass(value: Union[float, np.ndarray], from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    Convert a mass value from one unit to another.

    Args:
        value: Value or array to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Value in target unit
    """
    kg_value = to_kg(value, from_unit)
    return from_kg(kg_value, to_unit)


def format_mass(value: float, unit: str = 'kg', precision: int = 3) -> str:
    """
    Format a mass value with units for display.

    Args:
        value: Value in the specified unit
        unit: Unit string
        precision: Number of decimal places

    Returns:
        Formatted string (e.g., "12.345 kg")
    """
    unit = validate_mass_unit(unit)
    return f"{value:.{precision}f} {unit}"


if __name__ == "__main__":
    print("=" * 60)
    print("UNITS MODULE TEST")
    print("=" * 60)

    # Test basic conversions
    print("\n--- Testing basic conversions ---")

    test_value = 100.0  # mm
    print(f"Original: {test_value} mm")
    print(f"To cm: {from_mm(test_value, 'cm')} cm")
    print(f"To m: {from_mm(test_value, 'm')} m")
    print(f"To inches: {from_mm(test_value, 'in')} in")
    print(f"To feet: {from_mm(test_value, 'ft')} ft")

    # Test array conversions
    print("\n--- Testing array conversions ---")
    position_mm = np.array([100.0, 200.0, 300.0])
    print(f"Position (mm): {position_mm}")
    print(f"Position (m): {from_mm(position_mm, 'm')}")
    print(f"Position (in): {from_mm(position_mm, 'in')}")

    # Test round-trip conversion
    print("\n--- Testing round-trip conversion ---")
    original = 1.5  # meters
    in_mm = to_mm(original, 'm')
    back_to_m = from_mm(in_mm, 'm')
    print(f"Original: {original} m")
    print(f"To mm: {in_mm} mm")
    print(f"Back to m: {back_to_m} m")
    print(f"Match: {np.isclose(original, back_to_m)}")

    # Test unit validation
    print("\n--- Testing unit validation ---")
    try:
        to_mm(100, 'invalid_unit')
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")

    # Test formatting
    print("\n--- Testing formatting ---")
    length_mm = 1234.56789
    print(f"Default: {format_length(length_mm)}")
    print(f"2 decimals: {format_length(length_mm, precision=2)}")
    print(f"In meters: {format_length(from_mm(length_mm, 'm'), 'm', precision=4)}")

    # Test mass conversions
    print("\n--- Testing mass conversions ---")
    test_mass = 10.0  # kg
    print(f"Original: {test_mass} kg")
    print(f"To g: {from_kg(test_mass, 'g')} g")
    print(f"To lb: {from_kg(test_mass, 'lb')} lb")
    print(f"To oz: {from_kg(test_mass, 'oz')} oz")

    # Test mass round-trip conversion
    print("\n--- Testing mass round-trip conversion ---")
    original_lb = 22.0  # pounds
    in_kg = to_kg(original_lb, 'lb')
    back_to_lb = from_kg(in_kg, 'lb')
    print(f"Original: {original_lb} lb")
    print(f"To kg: {in_kg} kg")
    print(f"Back to lb: {back_to_lb} lb")
    print(f"Match: {np.isclose(original_lb, back_to_lb)}")

    # Test mass unit validation
    print("\n--- Testing mass unit validation ---")
    try:
        to_kg(100, 'invalid_unit')
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")

    # Test mass formatting
    print("\n--- Testing mass formatting ---")
    mass_kg = 5.678
    print(f"Default: {format_mass(mass_kg)}")
    print(f"In grams: {format_mass(from_kg(mass_kg, 'g'), 'g', precision=1)}")
    print(f"In pounds: {format_mass(from_kg(mass_kg, 'lb'), 'lb', precision=2)}")

    print("\n✓ All tests completed successfully!")
