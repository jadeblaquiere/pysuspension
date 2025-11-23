"""
Unit conversion utilities for suspension modeling.

All internal calculations are performed in millimeters (mm).
This module provides conversion functions to/from various units.
"""
import numpy as np
from typing import Union


# Conversion factors to millimeters (base unit)
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


def validate_unit(unit: str) -> str:
    """
    Validate and normalize unit string.

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

    print("\n✓ All tests completed successfully!")
