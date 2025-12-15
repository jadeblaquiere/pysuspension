# CornerSolver Model Integration Guide

## Overview

This guide explains the new CornerSolver integration with suspension models. This feature allows you to:

- **Automatically configure** CornerSolver from a modeled suspension
- **Discover components** by traversing from knuckle to chassis
- **Safely copy** all components for solving (originals unchanged)
- **Solve** with intuitive knuckle-centric methods

## Quick Start

```python
from pysuspension import CornerSolver, SuspensionKnuckle

# 1. Build your suspension model (knuckle, control arms, joints, etc.)
knuckle = create_my_suspension()  # Your suspension model

# 2. Create solver automatically from the model
solver = CornerSolver.from_suspension_knuckle(knuckle)

# 3. Solve for heave displacement
result = solver.solve_for_heave(25, unit='mm')

# 4. Get solved knuckle position
solved_knuckle = solver.get_solved_knuckle()
print(f"New position: {solved_knuckle.tire_center}")
```

## The Old Way vs The New Way

### Old Way (Manual Setup)

```python
# Had to manually configure everything
solver = CornerSolver("corner")

# Register every joint
solver.add_joint("upper_ball", [point1, point2], JointType.BALL_JOINT)
solver.add_joint("lower_ball", [point3, point4], JointType.BALL_JOINT)
# ... many more joints ...

# Add every control arm
solver.add_control_arm(upper_arm, chassis_mount_points=[...], knuckle_mount_points=[...])
solver.add_control_arm(lower_arm, chassis_mount_points=[...], knuckle_mount_points=[...])

# Add every link
solver.add_link(damper, end1_mount_point=..., end2_mount_point=...)
# ... many more links ...

# Set wheel center manually
solver.set_wheel_center(wheel_center_point)
```

### New Way (Automatic Discovery)

```python
# Just pass the knuckle - everything else is automatic!
solver = CornerSolver.from_suspension_knuckle(knuckle)

# That's it! Solver is fully configured and ready to use.
```

## Building a Suspension Model

### Step 1: Create Chassis Corner

```python
from pysuspension import ChassisCorner

chassis_corner = ChassisCorner("front_left")

# Add chassis attachment points
uf_mount = chassis_corner.add_attachment_point("upper_front", [1400, 0, 600], unit='mm')
ur_mount = chassis_corner.add_attachment_point("upper_rear", [1200, 0, 600], unit='mm')
lf_mount = chassis_corner.add_attachment_point("lower_front", [1500, 0, 300], unit='mm')
lr_mount = chassis_corner.add_attachment_point("lower_rear", [1100, 0, 300], unit='mm')
```

### Step 2: Create Control Arms

```python
from pysuspension import ControlArm, SuspensionLink

upper_arm = ControlArm("upper_control_arm")

# Add links to form the control arm
upper_front_link = SuspensionLink(
    endpoint1=[1400, 0, 600],      # Chassis mount
    endpoint2=[1400, 650, 580],    # Ball joint
    name="upper_front_link",
    unit='mm'
)
upper_arm.add_link(upper_front_link)

upper_rear_link = SuspensionLink(
    endpoint1=[1200, 0, 600],      # Chassis mount
    endpoint2=[1400, 650, 580],    # Ball joint (shared)
    name="upper_rear_link",
    unit='mm'
)
upper_arm.add_link(upper_rear_link)

# Same for lower control arm...
```

### Step 3: Create Suspension Knuckle

```python
from pysuspension import SuspensionKnuckle

knuckle = SuspensionKnuckle(
    tire_center_x=1400,       # mm
    tire_center_y=750,        # mm
    rolling_radius=390,       # mm
    toe_angle=0.0,            # degrees
    camber_angle=-1.0,        # degrees
    unit='mm'
)

# Add attachment points for ball joints
upper_ball = knuckle.add_attachment_point("upper_ball", [1400, 650, 580], unit='mm')
lower_ball = knuckle.add_attachment_point("lower_ball", [1400, 700, 200], unit='mm')
```

### Step 4: Create Joints Connecting Components

```python
from pysuspension import SuspensionJoint
from pysuspension.joint_types import JointType

# Bushing connecting chassis to control arm
uf_bushing = SuspensionJoint("upper_front_bushing", JointType.BUSHING_SOFT)
uf_bushing.add_attachment_point(uf_mount)  # Chassis
uf_bushing.add_attachment_point(upper_front_link.endpoint1)  # Control arm

# Ball joint connecting control arm to knuckle
upper_ball_joint = SuspensionJoint("upper_ball_joint", JointType.BALL_JOINT)
upper_ball_joint.add_attachment_point(upper_front_link.endpoint2)
upper_ball_joint.add_attachment_point(upper_rear_link.endpoint2)
upper_ball_joint.add_attachment_point(upper_ball)  # Knuckle

# ... create all other joints ...
```

## Using CornerSolver.from_suspension_knuckle()

### Basic Usage

```python
solver = CornerSolver.from_suspension_knuckle(
    knuckle,                    # Your suspension knuckle
    name="my_solver",           # Optional: solver name
    copy_components=True        # Default: True (safe, originals unchanged)
)
```

### What Happens Automatically

1. **Graph Discovery**: Traverses from knuckle through joints to find all components
2. **Component Copying**: Creates working copies of all components
3. **Relationship Preservation**: Maintains joint connections between copies
4. **Chassis Detection**: Automatically identifies and fixes chassis mount points
5. **Solver Configuration**: Sets up all constraints and degrees of freedom
6. **Wheel Center**: Creates wheel center from knuckle tire_center

### Output During Setup

```
Discovering suspension graph from knuckle 'front_left_knuckle'...
  Found 2 control arms
  Found 0 standalone links
  Found 6 joints
  Found 4 chassis mount points
Creating working copies of components...
Configuring CornerSolver 'my_solver'...
Registering joints...
Setting up chassis mount constraints...
Adding control arms...
Adding standalone links...
✓ CornerSolver configured successfully
  Total DOF: 3
  Total constraints: 18
```

## Solving with the Knuckle-Centric API

### solve_for_knuckle_heave()

Solve for vertical displacement of the knuckle:

```python
# Solve for +25mm heave
result = solver.solve_for_knuckle_heave(25, unit='mm')

print(f"Converged: {result.success}")
print(f"RMS error: {result.get_rms_error():.6f} mm")
```

### get_solved_knuckle()

Access the solved knuckle with updated positions:

```python
solved_knuckle = solver.get_solved_knuckle()

print(f"New tire center: {solved_knuckle.tire_center}")
print(f"New camber: {np.degrees(solved_knuckle.camber_angle):.2f}°")

# Access attachment points
upper_ball = solved_knuckle.get_attachment_point("upper_ball")
print(f"Upper ball joint position: {upper_ball.position}")
```

### Original Components Remain Unchanged

```python
# After solving, the original knuckle is unchanged
print(f"Original position: {knuckle.tire_center}")        # Still at original
print(f"Solved position: {solved_knuckle.tire_center}")   # New position
```

## Advanced Usage

### update_original_from_solved()

**⚠️ WARNING: This modifies the original suspension model!**

If you need to transfer solved positions back to the originals:

```python
solver = CornerSolver.from_suspension_knuckle(knuckle, copy_components=True)
result = solver.solve_for_heave(25, unit='mm')

# Original still unchanged here
print(f"Before: {knuckle.tire_center}")

# Update original from solved (WARNING: modifies original!)
solver.update_original_from_solved()

# Now original has new positions
print(f"After: {knuckle.tire_center}")
```

### Using Original Components (No Copying)

For cases where you want to work directly with originals:

```python
solver = CornerSolver.from_suspension_knuckle(
    knuckle,
    copy_components=False  # Work with originals directly
)

# Solving now modifies the original components!
result = solver.solve_for_heave(25, unit='mm')
```

### Heave Travel Analysis

```python
solver = CornerSolver.from_suspension_knuckle(knuckle)
solver.save_initial_state()

heave_values = [-40, -20, 0, 20, 40]
results = []

for heave in heave_values:
    # Reset to initial state
    solver.reset_to_initial_state()

    # Solve for this heave
    result = solver.solve_for_heave(heave, unit='mm')

    # Get camber from solved state
    solved_knuckle = solver.get_solved_knuckle()
    upper_ball = solved_knuckle.get_attachment_point("upper_ball")
    lower_ball = solved_knuckle.get_attachment_point("lower_ball")
    camber = solver.get_camber(upper_ball, lower_ball, unit='deg')

    results.append({
        'heave': heave,
        'camber': camber,
        'rms_error': result.get_rms_error()
    })

    print(f"Heave {heave:+3d}mm: Camber = {camber:+6.2f}°")
```

## Architecture and Design

### Graph Discovery

The `discover_suspension_graph()` function:
- Starts at knuckle attachment points
- Follows joint connections to find components
- Stops at chassis attachment points (boundaries)
- Returns a `SuspensionGraph` with all discovered components

### Component Copying

The `create_working_copies()` function uses a two-pass algorithm:

**Pass 1**: Copy all components
- Creates new instances of knuckle, control arms, links
- Creates new attachment points with same positions
- Builds mapping: `{id(original): copy}`

**Pass 2**: Reconstruct joint connections
- Creates new joints with same types
- Reconnects copied attachment points using mapping
- Preserves joint stiffness and properties

### Solver Configuration

The `from_suspension_knuckle()` method:
1. Discovers the suspension graph
2. Creates working copies (if requested)
3. Registers all joints
4. Fixes chassis attachment points
5. Adds control arms and links
6. Creates wheel center from tire_center
7. Returns fully configured solver

## Complete Example

See `examples/corner_solver_from_model.py` for a complete working example.

## Testing

Run the comprehensive test suite:

```bash
python tests/test_corner_solver_from_knuckle.py
```

Tests include:
- Graph discovery validation
- Component copying with relationship preservation
- Automatic solver configuration
- Solving with heave constraints
- Original component isolation
- Full integration workflow

## Implementation Files

**Core Implementation:**
- `pysuspension/suspension_graph.py`: Graph discovery and copying
- `pysuspension/corner_solver.py`: Integration methods
- `pysuspension/suspension_knuckle.py`: copy() method
- `pysuspension/control_arm.py`: copy() method
- `pysuspension/suspension_link.py`: copy() method
- `pysuspension/attachment_point.py`: Enhanced copy() method

**Tests and Examples:**
- `tests/test_corner_solver_from_knuckle.py`: Comprehensive tests
- `examples/corner_solver_from_model.py`: Usage example

## Benefits Summary

✅ **Automatic Discovery**: No manual solver setup required
✅ **Safe Copying**: Original components remain unchanged
✅ **Relationship Preservation**: Joints maintained between copies
✅ **Intuitive API**: Knuckle-centric workflow matches mental model
✅ **Type Safety**: Uses component types to identify chassis points
✅ **Comprehensive**: Handles control arms, links, joints automatically
✅ **Backwards Compatible**: Existing manual setup still works

## Migration Guide

### Existing Code (Manual Setup)

If you have existing code using manual CornerSolver setup:

**Before:**
```python
solver = CornerSolver("corner")
solver.add_joint("upper_ball", [p1, p2], JointType.BALL_JOINT)
solver.add_control_arm(upper_arm, chassis_mount_points=[...])
# ... many more manual steps ...
```

**After:**
```python
solver = CornerSolver.from_suspension_knuckle(knuckle)
# That's it!
```

### Benefits of Migration

- **Less Code**: ~5-10x fewer lines for setup
- **Less Error-Prone**: No manual tracking of mount points
- **More Maintainable**: Changes to model automatically reflected
- **Easier to Understand**: Clear separation between model and analysis

## Troubleshooting

### "Joint ... not found in solver"

Make sure all components are connected via joints before calling `from_suspension_knuckle()`.

### "No chassis attachment points found"

Ensure your suspension components connect to `ChassisCorner` attachment points.

### "Solver should converge but RMS error is high"

Check that joint types are appropriate and links have correct lengths.

### Graph Discovery finds wrong components

Verify that `parent_component` is set correctly on all attachment points.

## Future Enhancements

Potential future additions:
- Multi-corner solving (full vehicle)
- Incremental solving (update only changed components)
- Automatic optimization of pickup points
- Real-time visualization during solving
- Serialization of configured solvers

## Support

For questions or issues:
- Review the example: `examples/corner_solver_from_model.py`
- Run tests: `tests/test_corner_solver_from_knuckle.py`
- Check implementation plan: `CORNER_SOLVER_INTEGRATION_PLAN.md`

## License

Same as pysuspension main project.
