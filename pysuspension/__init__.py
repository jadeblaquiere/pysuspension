"""
pysuspension - A Python library for suspension geometry modeling and analysis.

This package provides classes and utilities for modeling vehicle suspension systems,
including rigid body dynamics, constraint-based solving, and unit conversions.

All internal calculations use:
- millimeters (mm) for length
- kilograms (kg) for mass
- kg/mm for spring rates

Basic usage:
    >>> from pysuspension import Chassis, ControlArm, SuspensionKnuckle
    >>> chassis = Chassis(name="my_chassis", mass=1000.0, mass_unit='kg')
    >>> corner = chassis.create_corner("front_left")
    >>> corner.add_attachment_point("upper_ball_joint", [0, 600, 400], unit='mm')
"""

# Version information
__version__ = "0.1.0"
__author__ = "pysuspension contributors"

# Core rigid body components
from .rigid_body import RigidBody
from .chassis import Chassis
from .chassis_corner import ChassisCorner
from .chassis_axle import ChassisAxle
from .control_arm import ControlArm
from .suspension_knuckle import SuspensionKnuckle
from .steering_rack import SteeringRack

# Links and attachment points
from .suspension_link import SuspensionLink
from .attachment_point import AttachmentPoint
from .coil_spring import CoilSpring

# Joints - connections between attachment points
from .suspension_joint import SuspensionJoint

# Joint types and stiffness
from .joint_types import JointType, JOINT_STIFFNESS

# Constraints for solving
from .constraints import (
    Constraint,
    GeometricConstraint,
    DistanceConstraint,
    FixedPointConstraint,
    CoincidentPointConstraint,
    PartialPositionConstraint,
)

# Solvers
from .solver import SuspensionSolver, SolverResult
from .corner_solver import CornerSolver
from .solver_state import SolverState, DOFSpecification

# Unit conversion utilities
from .units import (
    # Length conversion
    UNIT_TO_MM,
    MM_TO_UNIT,
    validate_unit,
    to_mm,
    from_mm,
    convert,
    format_length,

    # Mass conversion
    MASS_UNIT_TO_KG,
    KG_TO_MASS_UNIT,
    validate_mass_unit,
    to_kg,
    from_kg,
    convert_mass,
    format_mass,

    # Spring rate conversion
    SPRING_RATE_UNIT_TO_KG_PER_MM,
    KG_PER_MM_TO_SPRING_RATE_UNIT,
    validate_spring_rate_unit,
    to_kg_per_mm,
    from_kg_per_mm,
    convert_spring_rate,
)

# Define public API
__all__ = [
    # Version
    '__version__',
    '__author__',

    # Core components
    'RigidBody',
    'Chassis',
    'ChassisCorner',
    'ChassisAxle',
    'ControlArm',
    'SuspensionKnuckle',
    'SteeringRack',

    # Links and attachment
    'SuspensionLink',
    'AttachmentPoint',
    'CoilSpring',

    # Joints
    'SuspensionJoint',

    # Joint types
    'JointType',
    'JOINT_STIFFNESS',

    # Constraints
    'Constraint',
    'GeometricConstraint',
    'DistanceConstraint',
    'FixedPointConstraint',
    'CoincidentPointConstraint',
    'PartialPositionConstraint',

    # Solvers
    'SuspensionSolver',
    'SolverResult',
    'CornerSolver',
    'SolverState',
    'DOFSpecification',

    # Unit conversion - Length
    'UNIT_TO_MM',
    'MM_TO_UNIT',
    'validate_unit',
    'to_mm',
    'from_mm',
    'convert',
    'format_length',

    # Unit conversion - Mass
    'MASS_UNIT_TO_KG',
    'KG_TO_MASS_UNIT',
    'validate_mass_unit',
    'to_kg',
    'from_kg',
    'convert_mass',
    'format_mass',

    # Unit conversion - Spring rate
    'SPRING_RATE_UNIT_TO_KG_PER_MM',
    'KG_PER_MM_TO_SPRING_RATE_UNIT',
    'validate_spring_rate_unit',
    'to_kg_per_mm',
    'from_kg_per_mm',
    'convert_spring_rate',
]
