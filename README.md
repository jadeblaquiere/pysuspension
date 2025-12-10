# pysuspension

A Python library for vehicle suspension geometry modeling and analysis. Model suspension systems with rigid body dynamics, constraint-based solving, and comprehensive unit conversion support.

## Features

- **Rigid Body Modeling**: Create and manipulate chassis, control arms, knuckles, and other suspension components
- **Constraint-Based Solving**: Use geometric constraints to solve suspension kinematics
- **Unit Conversion Support**: Work in your preferred units (mm, m, inches, etc.) with automatic internal conversion
- **ISO-8855 Compliant**: Follows standard automotive axis conventions (x=longitudinal, y=lateral, z=vertical)
- **Component Library**: Pre-built components for common suspension elements
  - Control arms (upper/lower A-arms)
  - Suspension knuckles
  - Steering racks
  - Coil springs
  - Suspension links and joints

## Installation

Install from source:

```bash
git clone https://github.com/jadeblaquiere/pysuspension.git
cd pysuspension
pip install -e .
```

For development with testing dependencies:

```bash
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## Quick Start

```python
from pysuspension import Chassis, SuspensionKnuckle, ControlArm, SuspensionLink

# Create a chassis with a corner
chassis = Chassis(name="my_chassis", mass=1000.0, mass_unit='kg')
corner = chassis.create_corner("front_left")

# Add attachment points (in meters)
corner.add_attachment_point("upper_front_mount", [1.4, 0.5, 0.6], unit='m')
corner.add_attachment_point("upper_rear_mount", [1.3, 0.5, 0.58], unit='m')
corner.add_attachment_point("lower_front_mount", [1.45, 0.45, 0.35], unit='m')

# Create a suspension knuckle
knuckle = SuspensionKnuckle(
    tire_center_x=1.5,
    tire_center_y=0.75,
    rolling_radius=0.35,
    toe_angle=0.5,
    camber_angle=-1.0,
    unit='m'
)

# Add attachment points relative to knuckle center
knuckle.add_attachment_point("upper_ball_joint", [0, 0, 0.25], relative=True, unit='m')
knuckle.add_attachment_point("lower_ball_joint", [0, 0, -0.25], relative=True, unit='m')

# Get attachment positions for linking
chassis_positions = chassis.get_corner_attachment_positions("front_left")
knuckle_positions = knuckle.get_all_attachment_positions(absolute=True)

# Create links connecting chassis to knuckle
upper_link = SuspensionLink(
    endpoint1=chassis_positions[0],
    endpoint2=knuckle_positions["upper_ball_joint"],
    name="upper_front_link",
    unit='mm'  # Internal calculations use mm
)
```

## Key Concepts

### Unit System

Internally, all calculations use:
- **millimeters (mm)** for length
- **kilograms (kg)** for mass
- **kg/mm** for spring rates

However, you can input and retrieve values in any supported unit:
- Length: `'mm'`, `'m'`, `'cm'`, `'in'`, `'ft'`
- Mass: `'kg'`, `'g'`, `'lb'`, `'oz'`
- Spring rate: `'kg/mm'`, `'N/mm'`, `'lbf/in'`

```python
from pysuspension import to_mm, from_mm, convert

# Convert to internal units
length_mm = to_mm(1.5, 'm')  # 1500.0

# Convert from internal units
length_m = from_mm(1500.0, 'm')  # 1.5

# Direct conversion
length_in = convert(1.5, 'm', 'in')  # 59.055...
```

### Axis Convention (ISO-8855)

- **x-axis**: Longitudinal (positive = forward)
- **y-axis**: Lateral (positive = left)
- **z-axis**: Vertical (positive = up)

### Components

**Rigid Bodies**: Base class for all physical components
- `Chassis`: Vehicle frame with corners and axles
- `ControlArm`: Upper or lower control arms
- `SuspensionKnuckle`: Hub carrier/upright
- `SteeringRack`: Steering mechanism

**Links and Connections**:
- `SuspensionLink`: Connects two attachment points
- `AttachmentPoint`: Named point on a rigid body
- `CoilSpring`: Spring element with rate and preload
- `SuspensionJoint`: Defines joint type and behavior

**Constraints**:
- `DistanceConstraint`: Maintains fixed distance between points
- `FixedPointConstraint`: Locks a point in space
- `CoincidentPointConstraint`: Forces two points to same location
- `PartialPositionConstraint`: Constrains specific DOF

**Solvers**:
- `SuspensionSolver`: General-purpose constraint solver
- `CornerSolver`: Specialized solver for individual suspension corners

## Project Structure

```
pysuspension/
├── pysuspension/           # Main package
│   ├── __init__.py         # Package exports and API
│   ├── rigid_body.py       # Base rigid body class
│   ├── chassis.py          # Chassis component
│   ├── chassis_corner.py   # Corner management
│   ├── chassis_axle.py     # Axle management
│   ├── control_arm.py      # Control arm component
│   ├── suspension_knuckle.py  # Knuckle component
│   ├── steering_rack.py    # Steering component
│   ├── suspension_link.py  # Link elements
│   ├── attachment_point.py # Attachment points
│   ├── coil_spring.py      # Spring elements
│   ├── suspension_joint.py # Joint definitions
│   ├── joint_types.py      # Joint type enumerations
│   ├── constraints.py      # Constraint definitions
│   ├── solver.py           # Main solver
│   ├── corner_solver.py    # Corner-specific solver
│   ├── solver_state.py     # Solver state management
│   └── units.py            # Unit conversion utilities
├── tests/                  # Test suite
├── setup.py               # Package configuration
└── README.md              # This file
```

## Testing

Run the test suite:

```bash
pytest
```

With coverage:

```bash
pytest --cov=pysuspension
```

## Examples

See the `tests/` directory for comprehensive examples:
- `test_integrated_suspension.py`: Complete suspension system setup
- `test_constraints.py`: Using geometric constraints
- `test_corner_solver.py`: Solving suspension kinematics
- `test_coil_spring_refactor.py`: Working with springs

## Development Status

This project is in **Alpha** status (v0.1.0). The API may change between releases.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

Apache License 2.0

## Links

- **Repository**: https://github.com/jadeblaquiere/pysuspension
- **Issues**: https://github.com/jadeblaquiere/pysuspension/issues

## Citation

If you use this library in your research, please cite:

```
pysuspension contributors. (2024). pysuspension: Python foundations for suspension modeling.
https://github.com/jadeblaquiere/pysuspension
```
