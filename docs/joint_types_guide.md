# Suspension Joint Types Guide

This guide explains the different joint types available in pysuspension and when to use each one.

## Overview

Suspension joints model the compliance (flexibility) of connections between suspension components. Different joint types have different stiffness values, affecting how forces are transmitted through the suspension.

## Available Joint Types

### RIGID
**Stiffness:** 1,000,000,000 N/mm (1e9)

The stiffest joint type, modeling an infinitely rigid connection.

**Use cases:**
- Direct welded connections
- Theoretical analysis where compliance doesn't matter
- Components that should move as one rigid body

**Example:**
```python
from pysuspension.joint_types import JointType

solver.add_joint(
    name="rigid_link",
    points=[component1.mount, component2.mount],
    joint_type=JointType.RIGID
)
```

### BALL_JOINT
**Stiffness:** 100,000 N/mm

Very stiff spherical joint allowing rotation but minimal translation.

**Use cases:**
- Control arm to knuckle connections
- Tie rod ends
- Most suspension ball joints in production vehicles
- Analysis where ball joint compliance is negligible

**Example:**
```python
solver.add_joint(
    name="upper_ball_joint",
    points=[upper_arm.endpoint, knuckle.upper_mount],
    joint_type=JointType.BALL_JOINT
)
```

**Physical interpretation:** A typical automotive ball joint can withstand 10+ kN of force with < 0.1mm deflection.

### BUSHING_HARD
**Stiffness:** 10,000 N/mm

Hard rubber or polyurethane bushing.

**Use cases:**
- Performance-oriented chassis mounts
- Hard polyurethane bushings
- Sport suspension applications
- Race car suspension with minimal compliance

**Example:**
```python
solver.add_joint(
    name="chassis_mount_hard",
    points=[control_arm.chassis_end, chassis.mount_point],
    joint_type=JointType.BUSHING_HARD
)
```

**Physical interpretation:** 1 kN force causes ~0.1mm deflection.

### BUSHING_MEDIUM
**Stiffness:** 1,000 N/mm

Medium rubber bushing, typical for OEM performance vehicles.

**Use cases:**
- Factory sport suspension
- Balanced comfort/performance setups
- Most aftermarket suspension bushings
- Default choice for realistic modeling

**Example:**
```python
solver.add_joint(
    name="chassis_mount_medium",
    points=[control_arm.chassis_end, chassis.mount_point],
    joint_type=JointType.BUSHING_MEDIUM
)
```

**Physical interpretation:** 1 kN force causes ~1mm deflection.

### BUSHING_SOFT
**Stiffness:** 100 N/mm

Soft rubber bushing prioritizing comfort.

**Use cases:**
- Comfort-oriented vehicles
- Luxury car suspensions
- OEM suspensions prioritizing NVH (Noise, Vibration, Harshness) isolation
- Modeling maximum compliance effects

**Example:**
```python
solver.add_joint(
    name="chassis_mount_soft",
    points=[control_arm.chassis_end, chassis.mount_point],
    joint_type=JointType.BUSHING_SOFT
)
```

**Physical interpretation:** 1 kN force causes ~10mm deflection.

### CUSTOM
**Stiffness:** User-defined

For specialized applications or measured bushing data.

**Use cases:**
- Matching measured bushing stiffness
- Modeling non-standard materials
- Research applications
- Calibrating to test data

**Example:**
```python
# Custom stiffness of 5000 N/mm
solver.add_joint(
    name="custom_bushing",
    points=[control_arm.chassis_end, chassis.mount_point],
    joint_type=JointType.CUSTOM,
    stiffness=5000.0
)
```

## Realistic Suspension Example

A typical double wishbone suspension uses **mixed joint types**:

```python
from pysuspension import CornerSolver, ControlArm, SuspensionLink, AttachmentPoint
from pysuspension.joint_types import JointType

solver = CornerSolver("front_left")

# Create upper control arm
upper_arm = ControlArm("upper_control_arm")
front_link = SuspensionLink([1300, 0, 600], [1400, 650, 580], "front_link", unit='mm')
rear_link = SuspensionLink([1200, 0, 600], [1400, 650, 580], "rear_link", unit='mm')
upper_arm.add_link(front_link)
upper_arm.add_link(rear_link)

# Create attachment points
knuckle_upper = AttachmentPoint("knuckle_upper", [1400, 650, 580], unit='mm')
chassis_front = AttachmentPoint("chassis_front", [1300, 0, 600], unit='mm')
chassis_rear = AttachmentPoint("chassis_rear", [1200, 0, 600], unit='mm')

# Define joints with different types:
# 1. Stiff ball joint at knuckle (minimal deflection)
solver.add_joint(
    name="upper_ball_joint",
    points=[front_link.endpoint2, knuckle_upper],
    joint_type=JointType.BALL_JOINT  # 100,000 N/mm
)

# 2. Soft bushings at chassis (compliance for NVH isolation)
solver.add_joint(
    name="front_chassis_bushing",
    points=[front_link.endpoint1, chassis_front],
    joint_type=JointType.BUSHING_SOFT  # 100 N/mm
)

solver.add_joint(
    name="rear_chassis_bushing",
    points=[rear_link.endpoint1, chassis_rear],
    joint_type=JointType.BUSHING_SOFT  # 100 N/mm
)

# Add control arm to solver
solver.add_control_arm(
    control_arm=upper_arm,
    chassis_mount_points=[front_link.endpoint1, rear_link.endpoint1],
    knuckle_mount_points=[front_link.endpoint2]
)
```

This creates a realistic model where:
- **Ball joint** (100,000 N/mm) provides precise wheel location
- **Soft bushings** (100 N/mm) isolate road vibrations from chassis
- Stiffness ratio of 1000:1 matches real-world suspensions

## Joint Stiffness Hierarchy

From stiffest to most compliant:

| Joint Type      | Stiffness (N/mm) | Relative Stiffness | Typical Deflection (1 kN) |
|-----------------|------------------|--------------------|-----------------------------|
| RIGID           | 1,000,000,000    | 10,000,000x        | 0.000001 mm                |
| BALL_JOINT      | 100,000          | 1,000x             | 0.01 mm                    |
| BUSHING_HARD    | 10,000           | 100x               | 0.1 mm                     |
| BUSHING_MEDIUM  | 1,000            | 10x                | 1 mm                       |
| BUSHING_SOFT    | 100              | 1x (baseline)      | 10 mm                      |

## When NOT to Use Explicit Joints

For simple kinematics where all joints are similar, you can skip explicit joint definitions:

```python
# Simple geometry-based approach (no explicit joints)
solver = CornerSolver("front_left")

# Mark chassis mounts
solver.chassis_mounts.extend([
    upper_front_link.endpoint1,
    upper_rear_link.endpoint1
])

# Add links - geometry naturally constrains ball joint location
solver.add_link(upper_front_link, end1_mount_point=upper_front_link.endpoint1)
solver.add_link(upper_rear_link, end1_mount_point=upper_rear_link.endpoint1)
```

When two links from the same control arm meet at a point, the geometry (fixed chassis mounts + link lengths) naturally constrains that point. Adding an explicit joint between them creates unnecessary constraints.

## Design Guidelines

### For Performance Suspensions
- **Ball joints** at knuckle (maximize precision)
- **BUSHING_HARD** or **BUSHING_MEDIUM** at chassis (balance performance/comfort)
- Minimize total compliance for responsive handling

### For Comfort Suspensions
- **Ball joints** at knuckle (maintain safety/control)
- **BUSHING_SOFT** at chassis (maximize NVH isolation)
- Accept more compliance for better ride quality

### For Race Applications
- **BALL_JOINT** or **RIGID** everywhere
- Spherical bearings instead of rubber bushings
- Maximize stiffness for predictability

### For Compliance Analysis
- Use **measured stiffness** values (CUSTOM type)
- Model the actual bushing characteristics
- Validate against physical testing

## Advanced Topics

### Compliance Effects on Kinematics

Softer bushings allow suspension geometry to change under load:
- **Caster change** under braking/acceleration
- **Camber change** in cornering
- **Toe change** (bump steer) under lateral load

Example comparing stiff vs. soft:
```python
# Stiff suspension (minimal geometry change)
solver_stiff.add_joint("mount", [...], JointType.BUSHING_HARD)

# Soft suspension (significant geometry change)
solver_soft.add_joint("mount", [...], JointType.BUSHING_SOFT)

# Compare kinematics under 5000N lateral load
# Soft bushings will show more camber/toe change
```

### Temperature Effects

Real bushings soften with temperature. For temperature-sensitive analysis:
```python
# Cold rubber: stiffer
solver.add_joint("cold_bushing", [...], JointType.CUSTOM, stiffness=2000)

# Hot rubber: softer
solver.add_joint("hot_bushing", [...], JointType.CUSTOM, stiffness=500)
```

### Frequency-Dependent Stiffness

Rubber bushings exhibit frequency-dependent behavior. For quasi-static analysis (steady-state cornering), use lower stiffness. For high-frequency vibration analysis, use higher stiffness.

## References

- ISO 8855: Road vehicles - Vehicle dynamics and road-holding ability
- Milliken & Milliken, "Race Car Vehicle Dynamics"
- Dixon, "Suspension Geometry and Computation"

## Summary

- Use **BALL_JOINT** for spherical joints (control arm to knuckle)
- Use **BUSHING_SOFT/MEDIUM/HARD** for rubber/polyurethane bushings (control arm to chassis)
- Use **RIGID** for theoretical analysis or welded connections
- Use **CUSTOM** for measured bushing data
- Mix joint types to model realistic suspensions with compliance
- Skip explicit joints for simple geometry-based kinematics
