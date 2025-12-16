# SteeringRack RigidBody Inheritance Refactoring Plan

## Executive Summary

This plan outlines the refactoring of the `SteeringRack` class to inherit from `RigidBody`, allowing it to be treated as a single rigid body object externally while maintaining its internal differentiation between the housing and rack inner pivots.

## Current Architecture

### Current SteeringRack Structure
- **Does NOT** inherit from `RigidBody`
- Contains a `housing` member which IS a `RigidBody`
- Has its own attachment points: `left_inner_pivot` and `right_inner_pivot` (separate from housing)
- Steering-specific functionality:
  - `set_turn_angle()` - moves rack along rack axis
  - Rack displacement tracking
  - Travel parameters (travel_per_rotation, max_displacement)
- Delegates housing operations:
  - `get_chassis_attachment_points()` → `housing.get_all_attachment_points()`
  - `fit_chassis_to_attachment_targets()` → applies transformation to both housing and rack
- Custom transformation logic applies to both housing and rack pivots
- Custom reset and serialization logic

### RigidBody Base Class
- Base class providing:
  - Attachment point management
  - Transformation methods: `fit_to_attachment_targets()`, `transform()`, `translate()`, `rotate_about_center()`
  - Properties: `centroid`, `center_of_mass`, `rotation_matrix`, `attachment_points`
  - State management: `reset_to_origin()`, `save_state()`
  - Serialization: `to_dict()`, `from_dict()`
  - Protected method `_apply_transformation()` that subclasses can override

### Reference: Existing RigidBody Subclasses
Examples of successful inheritance patterns:
- **ControlArm**: Overrides `_apply_transformation()` to transform link endpoints in addition to attachment points
- **SuspensionKnuckle**: Extends with tire geometry
- **Chassis**: Extends with corners and axles

## Proposed Architecture

### Design Decision: Include Inner Pivots in attachment_points (REVISED)

**Revision Note**: The original plan kept inner pivots separate from `attachment_points`. After review, we've decided to include inner pivots in the `attachment_points` list for a more complete representation.

**Key Insight**: `SteeringRack` inherits from `RigidBody` with ALL attachment points (housing + inner pivots) in the `attachment_points` list. However, only housing points are used for fitting transformations, while inner pivots follow the transformation and maintain their steering displacement.

### Inheritance Strategy

```python
class SteeringRack(RigidBody):
    """
    Steering rack with housing and inner pivots as attachment points.

    All attachment points (housing + inner pivots) are in self.attachment_points.
    Housing points (first N) are used for rigid body fitting.
    Inner pivots (last 2) follow transformations and maintain steering displacement.
    """
```

### Architecture Benefits

1. **Polymorphism**: SteeringRack can be treated as a RigidBody in generic code
2. **Complete Representation**: All attachment points (housing + inner pivots) accessible via standard API
3. **Simpler Graph Discovery**: Inner pivots naturally discovered through attachment_points iteration
4. **Consistent Pattern**: Follows ControlArm pattern of extending RigidBody with additional state
5. **Accurate Centroid**: Centroid includes entire rack unit (housing + rack), not just housing
6. **Encapsulation**: Inner pivots accessed via getter methods, not direct attributes

### Design Trade-offs

**Advantages:**
- ✅ More complete representation - all attachment points truly ARE attachment points
- ✅ External APIs (`get_all_attachment_points()`) return everything
- ✅ Simpler for generic code that iterates attachment points
- ✅ Graph discovery naturally finds inner pivots

**Challenges:**
- ⚠️ `fit_chassis_to_attachment_targets()` must be overridden to fit only housing points
- ⚠️ Target position count (housing only) ≠ total attachment count (housing + inner pivots)
- ⚠️ Need to track partition between housing and inner pivot points
- ⚠️ Centroid includes inner pivots (more accurate, but different from housing-only centroid)

**Solution**: Track `_housing_attachment_count` to partition the list, override fitting method to use only housing points

## Implementation Plan

### Phase 1: Core Class Refactoring

#### 1.1 Update Class Declaration
```python
# Current
class SteeringRack:

# New
class SteeringRack(RigidBody):
    """
    Represents a steering rack unit with housing and inner tie rod pivots.

    Inherits from RigidBody to provide rigid body transformation behavior
    for the housing. The housing attachment points are managed by the
    RigidBody base class, while the rack inner pivots are maintained as
    internal state that transforms along with the housing.
    """
```

#### 1.2 Refactor Constructor

**Current Pattern**:
```python
def __init__(self, housing_attachments, left_inner_pivot, right_inner_pivot, ...):
    self.name = name
    self.housing = RigidBody(...)
    for point in housing_points:
        self.housing.add_attachment_point(point)
    self.left_inner_pivot = ...
    self.right_inner_pivot = ...
```

**New Pattern (REVISED)**:
```python
def __init__(self, housing_attachments, left_inner_pivot, right_inner_pivot,
             travel_per_rotation, max_displacement, name="steering_rack",
             unit='mm', mass=0.0, mass_unit='kg'):
    # Initialize RigidBody base
    super().__init__(name=name, mass=mass, mass_unit=mass_unit)

    # Add housing attachment points to self.attachment_points
    housing_points = [
        self._ensure_attachment_point(pos, f"{name}_mount_{i}", unit)
        for i, pos in enumerate(housing_attachments)
    ]
    for point in housing_points:
        self.add_attachment_point(point)

    # Track housing attachment count (for selective fitting)
    self._housing_attachment_count = len(housing_points)

    # Initialize rack pivots and ADD to self.attachment_points
    self._left_inner_pivot = self._ensure_attachment_point(
        left_inner_pivot, f"{name}_left_inner", unit
    )
    self._left_inner_pivot.parent_component = self
    self.add_attachment_point(self._left_inner_pivot)

    self._right_inner_pivot = self._ensure_attachment_point(
        right_inner_pivot, f"{name}_right_inner", unit
    )
    self._right_inner_pivot.parent_component = self
    self.add_attachment_point(self._right_inner_pivot)

    # Calculate rack parameters
    rack_vector = self._right_inner_pivot.position - self._left_inner_pivot.position
    self.rack_length = np.linalg.norm(rack_vector)
    if self.rack_length < 1e-6:
        raise ValueError("Inner pivot points are too close together")
    self.rack_axis = rack_vector / self.rack_length
    self.rack_axis_initial = self.rack_axis.copy()

    # Calculate rack center position
    self.rack_center_initial = (self._left_inner_pivot.position +
                                self._right_inner_pivot.position) / 2.0
    self.rack_center = self.rack_center_initial.copy()

    # Store initial positions for reference (used by set_turn_angle)
    self.left_inner_pivot_initial = self._left_inner_pivot.position.copy()
    self.right_inner_pivot_initial = self._right_inner_pivot.position.copy()

    # Store original positions for reset
    self._original_left_inner_pivot = self._left_inner_pivot.position.copy()
    self._original_right_inner_pivot = self._right_inner_pivot.position.copy()
    self._original_rack_center = self.rack_center.copy()
    self._original_rack_axis = self.rack_axis.copy()

    # Travel parameters (in mm)
    self.travel_per_rotation = to_mm(travel_per_rotation, unit)
    self.max_displacement = to_mm(max_displacement, unit)

    # Current state
    self.current_displacement = 0.0
    self.current_angle = 0.0
```

**Key Changes**:
- Call `super().__init__()` first
- Housing attachment points added to `self.attachment_points` (not `self.housing`)
- Remove `self.housing = RigidBody(...)` entirely
- **NEW**: Track `_housing_attachment_count` for partitioning
- **NEW**: Inner pivots stored in private attributes (`_left_inner_pivot`, `_right_inner_pivot`)
- **NEW**: Inner pivots ALSO added to `self.attachment_points` via `add_attachment_point()`
- **NEW**: Result is `attachment_points = [housing_0, housing_1, housing_2, left_inner, right_inner]`

#### 1.3 Add Getter Methods for Inner Pivots

**NEW**: Since inner pivots are now private attributes, provide getter methods:

```python
def get_left_inner_pivot(self) -> AttachmentPoint:
    """
    Get the left inner pivot attachment point.

    Returns:
        AttachmentPoint object for left inner pivot
    """
    return self._left_inner_pivot

def get_right_inner_pivot(self) -> AttachmentPoint:
    """
    Get the right inner pivot attachment point.

    Returns:
        AttachmentPoint object for right inner pivot
    """
    return self._right_inner_pivot

def get_housing_attachment_points(self) -> List[AttachmentPoint]:
    """
    Get only the housing attachment points (used for chassis mounting).

    Returns:
        List of housing AttachmentPoint objects (excludes inner pivots)
    """
    return self.attachment_points[:self._housing_attachment_count]

def get_housing_attachment_positions(self, unit: str = 'mm') -> List[np.ndarray]:
    """
    Get only the housing attachment positions (used for chassis mounting).

    Args:
        unit: Unit for output (default: 'mm')

    Returns:
        List of housing attachment positions in specified unit
    """
    housing_points = self.get_housing_attachment_points()
    return [ap.get_position(unit) for ap in housing_points]
```

#### 1.4 Override `_apply_transformation()`

Follow the ControlArm pattern. **Note**: The base class will transform ALL attachment points (including inner pivots), so we only need to handle rack-specific state:

```python
def _apply_transformation(self, R: np.ndarray, t: np.ndarray) -> None:
    """
    Apply rigid body transformation to housing attachments and rack pivots.

    The base class handles transforming all attachment_points (housing + inner pivots).
    We only need to transform additional rack-specific state:
    - Rack center position
    - Rack axis direction
    - Initial positions for steering reference

    Args:
        R: 3x3 rotation matrix
        t: 3D translation vector (in mm)
    """
    # Call parent to handle ALL attachment points (housing + inner pivots) and center of mass
    super()._apply_transformation(R, t)

    # Transform rack axis (rotation only - it's a direction vector)
    self.rack_axis = R @ self.rack_axis
    self.rack_axis_initial = self.rack_axis.copy()

    # Transform rack center and its initial position
    self.rack_center = R @ self.rack_center + t
    self.rack_center_initial = R @ self.rack_center_initial + t

    # Transform initial positions (used as reference for set_turn_angle)
    self.left_inner_pivot_initial = R @ self.left_inner_pivot_initial + t
    self.right_inner_pivot_initial = R @ self.right_inner_pivot_initial + t

    # Note: Steering angle and displacement are NOT reset here
    # They represent user input, not transformation state
```

**Key Point**: The parent `_apply_transformation()` will handle transforming the inner pivot `AttachmentPoint` objects since they're in `self.attachment_points`. We only need to transform the additional state (rack axis, rack center, initial positions).

#### 1.5 Override `reset_to_origin()`

```python
def reset_to_origin(self) -> None:
    """
    Reset the steering rack to its originally defined position.

    Resets both housing (via parent) and rack-specific state:
    - Housing attachment points (via parent)
    - Inner pivot positions (via parent)
    - Rack center and axis
    - Steering angle and displacement
    """
    # Reset ALL attachment points (housing + inner pivots) via parent
    super().reset_to_origin()

    # Reset rack-specific state to original positions
    self.rack_center = self._original_rack_center.copy()
    self.rack_center_initial = self._original_rack_center.copy()
    self.rack_axis = self._original_rack_axis.copy()
    self.rack_axis_initial = self._original_rack_axis.copy()

    # Reset initial positions for steering reference
    self.left_inner_pivot_initial = self._original_left_inner_pivot.copy()
    self.right_inner_pivot_initial = self._original_right_inner_pivot.copy()

    # Reset steering angle and displacement
    self.current_displacement = 0.0
    self.current_angle = 0.0
```

**Key Change**: Parent's `reset_to_origin()` now handles inner pivots since they're in `attachment_points`. We only reset rack-specific state.

#### 1.6 Update `fit_chassis_to_attachment_targets()` Method

**CRITICAL**: This method must be completely overridden because we only fit housing points, not inner pivots. The base class `fit_to_attachment_targets()` expects ALL attachment points, but we only want to fit the housing subset.

```python
def fit_chassis_to_attachment_targets(self, target_positions: List[np.ndarray],
                                      unit: str = 'mm') -> float:
    """
    Fit the steering rack housing to target chassis attachment positions.

    This method fits ONLY the housing attachment points (not inner pivots).
    The inner pivots follow the housing transformation while maintaining their
    steering displacement along the rack axis.

    Args:
        target_positions: List of target positions for HOUSING attachments only
                         (length must match housing attachment count, not total)
        unit: Unit of input positions (default: 'mm')

    Returns:
        RMS error of the fit (in mm)

    Raises:
        ValueError: If number of target positions doesn't match housing attachment count
    """
    if len(target_positions) != self._housing_attachment_count:
        raise ValueError(
            f"Expected {self._housing_attachment_count} target positions for housing, "
            f"got {len(target_positions)}"
        )

    # Freeze original state on first transformation
    self._original_state_frozen = True

    # Get current housing positions only
    housing_points = self.get_housing_attachment_points()
    current_positions = [ap.position.copy() for ap in housing_points]

    # Convert target positions to mm
    target_points = np.array([to_mm(np.array(p, dtype=float), unit) for p in target_positions])
    current_points = np.array(current_positions)

    # Get joint stiffness for housing points only
    from .joint_types import JointType, JOINT_STIFFNESS
    stiffness_weights = []
    for ap in housing_points:
        if ap.joint is not None:
            stiffness = JOINT_STIFFNESS[ap.joint.joint_type]
        else:
            stiffness = JOINT_STIFFNESS[JointType.RIGID]
        stiffness_weights.append(stiffness)
    stiffness_weights = np.array(stiffness_weights)
    total_weight = np.sum(stiffness_weights)

    # Compute weighted centroids
    centroid_current = np.sum(current_points * stiffness_weights[:, np.newaxis], axis=0) / total_weight
    centroid_target = np.sum(target_points * stiffness_weights[:, np.newaxis], axis=0) / total_weight

    # Center the point sets
    current_centered = current_points - centroid_current
    target_centered = target_points - centroid_target

    # Compute weighted cross-covariance matrix H
    weighted_current = current_centered * np.sqrt(stiffness_weights[:, np.newaxis])
    weighted_target = target_centered * np.sqrt(stiffness_weights[:, np.newaxis])
    H = weighted_current.T @ weighted_target

    # SVD (Kabsch algorithm)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_target - R @ centroid_current

    # Apply transformation to ALL attachment points (housing + inner pivots) and rack state
    self._apply_transformation(R, t)

    # Calculate RMS error (housing points only)
    transformed_points = (R @ current_points.T).T + t
    errors = target_points - transformed_points
    squared_errors = np.sum(errors**2, axis=1)
    weighted_rms_error = np.sqrt(np.sum(stiffness_weights * squared_errors) / total_weight)

    # Reset steering angle and displacement after transformation
    # (transformation moves entire unit, not steering input)
    self.current_displacement = 0.0
    self.current_angle = 0.0

    return weighted_rms_error
```

**Why we can't call parent's `fit_to_attachment_targets()`**: The parent method expects target positions for ALL attachment points (housing + inner pivots). But we only have targets for housing. So we must reimplement the Kabsch algorithm for the housing subset only.

#### 1.7 Update Serialization Methods

**Note**: Inner pivots are now in `attachment_points`, but we serialize them separately to maintain clear distinction and backward compatibility.

```python
def to_dict(self) -> dict:
    """
    Serialize the steering rack to a dictionary.

    Combines RigidBody data (housing + inner pivots in attachment_points)
    with SteeringRack-specific data.
    """
    # Get base RigidBody serialization (includes ALL attachment points)
    data = super().to_dict()

    # Add SteeringRack-specific fields
    data.update({
        'type': 'SteeringRack',  # For type identification
        '_housing_attachment_count': self._housing_attachment_count,  # For partitioning
        'travel_per_rotation': float(self.travel_per_rotation),
        'max_displacement': float(self.max_displacement),
        'current_angle': float(self.current_angle),
        'current_displacement': float(self.current_displacement),
    })

    return data

@classmethod
def from_dict(cls, data: dict) -> 'SteeringRack':
    """
    Deserialize a steering rack from a dictionary.

    Supports both new format (inherits RigidBody) and old format (housing member)
    for backward compatibility.
    """
    # Handle backward compatibility with old format
    if 'housing' in data:
        # Old format: housing was a separate RigidBody member
        housing = RigidBody.from_dict(data['housing'])
        housing_positions = housing.get_all_attachment_points()

        # Extract inner pivot data from old format
        if 'left_inner_pivot' in data and 'right_inner_pivot' in data:
            left_inner = AttachmentPoint.from_dict(data['left_inner_pivot'])
            right_inner = AttachmentPoint.from_dict(data['right_inner_pivot'])
        else:
            raise KeyError("Missing inner pivot data in serialized SteeringRack")
    else:
        # New format: attachment_points includes housing + inner pivots
        all_attachment_points = [
            AttachmentPoint.from_dict(ap_data)
            for ap_data in data.get('attachment_points', [])
        ]

        # Partition based on _housing_attachment_count
        housing_count = data.get('_housing_attachment_count')
        if housing_count is None:
            raise KeyError("Missing _housing_attachment_count in new format")

        housing_positions = all_attachment_points[:housing_count]
        left_inner = all_attachment_points[housing_count]
        right_inner = all_attachment_points[housing_count + 1]

    # Create the steering rack
    rack = cls(
        housing_attachments=housing_positions,
        left_inner_pivot=left_inner,
        right_inner_pivot=right_inner,
        travel_per_rotation=data['travel_per_rotation'],
        max_displacement=data['max_displacement'],
        name=data['name'],
        unit='mm',
        mass=data.get('mass', 0.0),
        mass_unit=data.get('mass_unit', 'kg')
    )

    # Restore current state if present
    if 'current_angle' in data:
        rack.set_turn_angle(data['current_angle'])

    return rack
```

#### 1.8 Update `copy()` Method

```python
def copy(self, copy_joints: bool = False) -> 'SteeringRack':
    """
    Create a copy of the steering rack.

    Args:
        copy_joints: If True, copy joint references; if False, set joints to None

    Returns:
        New SteeringRack instance with copied state
    """
    # Copy housing attachment points only (first N attachment points)
    housing_attachments_copy = []
    for ap in self.get_housing_attachment_points():
        ap_copy = ap.copy(copy_joint=copy_joints, copy_parent=False)
        housing_attachments_copy.append(ap_copy)

    # Copy inner pivot points
    left_inner_copy = self._left_inner_pivot.copy(copy_joint=copy_joints, copy_parent=False)
    right_inner_copy = self._right_inner_pivot.copy(copy_joint=copy_joints, copy_parent=False)

    # Create new steering rack
    rack_copy = SteeringRack(
        housing_attachments=housing_attachments_copy,
        left_inner_pivot=left_inner_copy,
        right_inner_pivot=right_inner_copy,
        travel_per_rotation=self.travel_per_rotation,
        max_displacement=self.max_displacement,
        name=self.name,
        unit='mm',
        mass=self.mass,
        mass_unit='kg'
    )

    # Copy current state
    rack_copy.set_turn_angle(self.current_angle)

    return rack_copy
```

#### 1.9 Update `__repr__()` Method

```python
def __repr__(self) -> str:
    """String representation showing housing and rack state."""
    centroid_str = f"{self.centroid} mm" if self.centroid is not None else "None"
    return (f"SteeringRack('{self.name}',\n"
            f"  housing_centroid={centroid_str},\n"
            f"  housing_attachments={len(self.attachment_points)},\n"
            f"  rack_center={self.rack_center} mm,\n"
            f"  rack_length={self.rack_length:.3f} mm,\n"
            f"  current_angle={self.current_angle:.2f}°,\n"
            f"  current_displacement={self.current_displacement:.3f} mm,\n"
            f"  travel_per_rotation={self.travel_per_rotation:.3f} mm/deg\n"
            f")")
```

### Phase 2: Update Dependent Code

#### 2.1 Update `chassis.py`

**Current code** (lines 452-460):
```python
# Check SteeringRack housing attachment points
if hasattr(component, 'housing'):
    for ap in component.housing.attachment_points:
        if ap is point:
            return component.housing.name
if hasattr(component, 'left_inner_pivot') and component.left_inner_pivot is point:
    return comp_name
if hasattr(component, 'right_inner_pivot') and component.right_inner_pivot is point:
    return comp_name
```

**New code**:
```python
# Check SteeringRack inner pivots (housing points handled by attachment_points check above)
if hasattr(component, 'left_inner_pivot') and component.left_inner_pivot is point:
    return comp_name
if hasattr(component, 'right_inner_pivot') and component.right_inner_pivot is point:
    return comp_name
```

**Rationale**: Housing attachment points are now in `component.attachment_points`, which is already checked earlier in the method. Only need special handling for rack pivots.

Similar change needed around line 573-576 in `_build_attachment_lookup()`.

#### 2.2 Update `suspension_graph.py`

**Current comment** (line 235-236):
```python
# The housing attachment points belong to the housing RigidBody,
# not the SteeringRack itself, so we only handle the inner pivots
```

**New comment**:
```python
# The housing attachment points are now part of the SteeringRack RigidBody.
# Here we only handle the inner pivots, which are internal rack state.
```

No code changes needed - the graph discovery logic already handles this correctly via parent_component references.

#### 2.3 Update `rigid_body.py` Documentation

**Current** (line 36):
```python
# - SteeringRack: Housing is a rigid body
```

**New**:
```python
# - SteeringRack: Rigid body with movable internal rack pivots
```

### Phase 3: Testing and Validation

#### 3.1 Update Existing Tests

File: `tests/test_serialization.py`

**Changes needed**:
- Update `test_steering_rack_serialization()` to verify inheritance
- Test that `isinstance(rack, RigidBody)` returns True
- Test that housing attachment methods work correctly
- Verify backward compatibility with old serialized data

#### 3.2 New Test Cases

Create `tests/test_steering_rack_inheritance.py`:

```python
def test_steering_rack_is_rigid_body():
    """Test that SteeringRack inherits from RigidBody."""
    rack = create_test_steering_rack()
    assert isinstance(rack, RigidBody)

def test_housing_attachment_points_accessible():
    """Test that housing attachment points are accessible via RigidBody interface."""
    rack = create_test_steering_rack()

    # Should have 5 total attachment points (3 housing + 2 inner pivots)
    assert len(rack.get_all_attachment_points()) == 5
    assert len(rack.attachment_points) == 5

    # Should have 3 housing attachment points specifically
    assert len(rack.get_housing_attachment_points()) == 3

    # Inner pivots SHOULD be in attachment_points (REVISED DESIGN)
    assert rack.get_left_inner_pivot() in rack.attachment_points
    assert rack.get_right_inner_pivot() in rack.attachment_points

def test_transformation_applies_to_all():
    """Test that transformations apply to both housing and rack pivots."""
    rack = create_test_steering_rack()

    # Record initial positions
    housing_initial = rack.get_all_attachment_positions()
    left_pivot_initial = rack.get_left_inner_pivot().position.copy()
    rack_center_initial = rack.rack_center.copy()

    # Apply transformation
    translation = np.array([100.0, 50.0, 0.0])
    rack.translate(translation, unit='mm')

    # Verify ALL attachment points moved (housing + inner pivots)
    housing_new = rack.get_all_attachment_positions()
    for i, pos in enumerate(housing_new):
        expected = housing_initial[i] + translation
        assert np.allclose(pos, expected)

    # Verify rack pivot moved (it's in attachment_points, so parent handles it)
    assert np.allclose(rack.get_left_inner_pivot().position, left_pivot_initial + translation)

    # Verify rack center moved
    assert np.allclose(rack.rack_center, rack_center_initial + translation)

def test_fit_to_targets_transforms_rack():
    """Test that fit_chassis_to_attachment_targets transforms rack pivots."""
    rack = create_test_steering_rack()

    # Get initial rack pivot distance from housing centroid
    centroid_initial = rack.get_centroid()
    pivot_offset_initial = rack.get_left_inner_pivot().position - centroid_initial

    # Create target positions for HOUSING ONLY (rotate housing)
    angle = np.radians(45)
    R = rotation_matrix_z(angle)
    current_housing_positions = rack.get_housing_attachment_positions()
    target_positions = [(R @ p) for p in current_housing_positions]

    # Fit to targets (only housing points)
    rack.fit_chassis_to_attachment_targets(target_positions, unit='mm')

    # Verify rack pivot rotated with housing
    centroid_new = rack.get_centroid()
    pivot_offset_new = rack.get_left_inner_pivot().position - centroid_new
    expected_offset = R @ pivot_offset_initial
    assert np.allclose(pivot_offset_new, expected_offset)

def test_reset_resets_everything():
    """Test that reset_to_origin resets both housing and rack."""
    rack = create_test_steering_rack()

    # Record originals (all attachment points including inner pivots)
    all_original = [p.copy() for p in rack.get_all_attachment_positions()]
    left_pivot_original = rack.get_left_inner_pivot().position.copy()

    # Transform and steer
    rack.translate([100, 0, 0], unit='mm')
    rack.set_turn_angle(30.0)

    # Reset
    rack.reset_to_origin()

    # Verify everything restored (all attachment points)
    all_reset = rack.get_all_attachment_positions()
    for i, pos in enumerate(all_reset):
        assert np.allclose(pos, all_original[i])

    assert np.allclose(rack.get_left_inner_pivot().position, left_pivot_original)
    assert rack.current_angle == 0.0

def test_polymorphic_usage():
    """Test that SteeringRack can be used polymorphically as RigidBody."""
    rack = create_test_steering_rack()

    # Should be usable in code that expects RigidBody
    def transform_rigid_body(rb: RigidBody):
        rb.translate([10, 20, 30], unit='mm')
        return rb.get_centroid()

    result = transform_rigid_body(rack)
    assert result is not None
```

#### 3.3 Integration Tests

- Test with `CornerSolver` to ensure steering rack integration works
- Test with `suspension_graph.py` discovery
- Test with `Chassis` component management

#### 3.4 Validation Checklist

- [ ] All existing tests pass
- [ ] New inheritance tests pass
- [ ] `isinstance(steering_rack, RigidBody)` returns True
- [ ] Housing attachment points accessible via RigidBody interface
- [ ] Transformations apply to both housing and rack pivots correctly
- [ ] Reset restores both housing and rack state
- [ ] Serialization/deserialization works (both new and old formats)
- [ ] Copy method creates proper independent copies
- [ ] Integration with suspension graph works
- [ ] Integration with chassis works
- [ ] CornerSolver integration works (if applicable)

## Migration Guide

### For Users of SteeringRack

#### Breaking Changes

1. **`steering_rack.housing` no longer exists**
   ```python
   # Old
   housing_points = steering_rack.housing.get_all_attachment_points()

   # New
   housing_points = steering_rack.get_all_attachment_points()
   ```

2. **Method renames**
   ```python
   # Old
   chassis_points = steering_rack.get_chassis_attachment_points()
   chassis_positions = steering_rack.get_chassis_attachment_positions()

   # New (inherited from RigidBody)
   chassis_points = steering_rack.get_all_attachment_points()
   chassis_positions = steering_rack.get_all_attachment_positions()
   ```

3. **Type checking for housing attribute**
   ```python
   # Old
   if hasattr(component, 'housing'):
       # Handle steering rack housing

   # New
   if isinstance(component, SteeringRack):
       # Housing points are in component.attachment_points (or use component.get_housing_attachment_points())
       # Rack pivots accessed via component.get_left_inner_pivot(), component.get_right_inner_pivot()
   ```

4. **Inner pivot access** - Use getter methods
   ```python
   # Old (direct attribute access)
   left_pivot = steering_rack.left_inner_pivot
   right_pivot = steering_rack.right_inner_pivot

   # New (getter methods)
   left_pivot = steering_rack.get_left_inner_pivot()
   right_pivot = steering_rack.get_right_inner_pivot()

   # Or add properties for backward compatibility (see migration section)
   ```

#### Non-Breaking Changes (Backward Compatible)

1. **Type checking with isinstance**
   ```python
   # Still works
   if isinstance(component, SteeringRack):
       rack.set_turn_angle(30.0)

   # Now also works
   if isinstance(component, RigidBody):
       component.translate([10, 0, 0])
   ```

2. **Serialization** - Old format can still be deserialized
   ```python
   # Old JSON format with 'housing' field - still works
   old_data = load_old_steering_rack_json()
   rack = SteeringRack.from_dict(old_data)  # Handles backward compatibility
   ```

3. **Rack pivot access** - Use getter methods (CHANGED)
   ```python
   # Old (direct attribute access)
   left_pos = steering_rack.left_inner_pivot.position
   right_pos = steering_rack.right_inner_pivot.position

   # New (getter methods - RECOMMENDED)
   left_pos = steering_rack.get_left_inner_pivot().position
   right_pos = steering_rack.get_right_inner_pivot().position

   # Steering angle setting unchanged
   steering_rack.set_turn_angle(45.0)
   ```

   **Note**: For backward compatibility during transition, consider adding properties:
   ```python
   @property
   def left_inner_pivot(self):
       return self._left_inner_pivot

   @property
   def right_inner_pivot(self):
       return self._right_inner_pivot
   ```

### For Library Maintainers

#### Update Patterns

1. **Generic RigidBody handling now includes SteeringRack**
   ```python
   def transform_all_rigid_bodies(components: List[RigidBody]):
       for component in components:
           component.translate([10, 0, 0])
       # Now works with SteeringRack too!
   ```

2. **Discovery and graph building**
   ```python
   # Pattern still works - isinstance check is the same
   if isinstance(component, SteeringRack):
       graph.steering_racks.append(component)
   ```

3. **Handling special attachment points**
   ```python
   # Need to distinguish housing points from rack pivots
   if isinstance(component, SteeringRack):
       # Housing attachment points
       housing_points = component.attachment_points

       # Rack pivot points (not in attachment_points list)
       rack_pivots = [component.left_inner_pivot, component.right_inner_pivot]
   ```

## Risks and Mitigation

### Risk 1: Breaking Changes for `housing` Attribute Access

**Impact**: HIGH - Code that directly accesses `steering_rack.housing` will break

**Mitigation**:
1. Search entire codebase for `.housing` access on SteeringRack
2. Update all occurrences to use direct methods
3. Add deprecation warnings before full removal (optional transition period)
4. Consider adding a property that raises a helpful error:
   ```python
   @property
   def housing(self):
       raise AttributeError(
           "SteeringRack.housing no longer exists. "
           "Use SteeringRack methods directly (SteeringRack now inherits from RigidBody). "
           "For housing attachment points: use get_all_attachment_points()"
       )
   ```

### Risk 2: Serialization Format Changes

**Impact**: MEDIUM - Old saved files might not load

**Mitigation**:
- `from_dict()` supports both old and new formats
- Add version field to new format for future changes
- Provide migration script for batch conversion if needed

### Risk 3: Unexpected Behavior in Generic RigidBody Code

**Impact**: LOW - SteeringRack might be processed where not expected

**Mitigation**:
- SteeringRack has always been a rigid component
- This change makes the type system reflect reality
- Generic RigidBody handling should work correctly
- Add tests for polymorphic usage

### Risk 4: Performance Impact

**Impact**: NEGLIGIBLE - Inheritance adds minimal overhead

**Mitigation**:
- Benchmark transformation performance before/after
- Python method dispatch overhead is minimal
- Most time spent in numpy operations (unchanged)

## Timeline and Phases

### Phase 1: Preparation (1 day)
- [ ] Search codebase for all `steering_rack.housing` usage
- [ ] Document all call sites that need updating
- [ ] Create feature branch
- [ ] Set up new test file

### Phase 2: Core Implementation (2-3 days)
- [ ] Update SteeringRack class declaration
- [ ] Refactor constructor
- [ ] Override `_apply_transformation()`
- [ ] Override `reset_to_origin()`
- [ ] Update serialization methods
- [ ] Update other methods (copy, repr, etc.)

### Phase 3: Update Dependencies (1 day)
- [ ] Update chassis.py
- [ ] Update suspension_graph.py
- [ ] Update rigid_body.py docs
- [ ] Fix any other usage sites

### Phase 4: Testing (2 days)
- [ ] Write new inheritance tests
- [ ] Update existing tests
- [ ] Run full test suite
- [ ] Manual testing with example suspensions
- [ ] Performance benchmarks

### Phase 5: Documentation (1 day)
- [ ] Update API documentation
- [ ] Update class docstrings
- [ ] Create migration guide
- [ ] Update examples

### Total Estimated Time: 7-8 days

## Success Criteria

1. ✅ `isinstance(steering_rack, RigidBody)` returns `True`
2. ✅ Housing attachment points accessible via RigidBody methods
3. ✅ Transformations correctly affect both housing and rack pivots
4. ✅ All existing tests pass (with updates)
5. ✅ New inheritance tests pass
6. ✅ Serialization backward compatibility works
7. ✅ Integration tests pass (suspension graph, chassis, corner solver)
8. ✅ No performance regression
9. ✅ Documentation complete
10. ✅ Zero references to `steering_rack.housing` in codebase

## Conclusion

This refactoring aligns SteeringRack with the existing RigidBody inheritance pattern used by ControlArm and other components. The change is architecturally sound and provides significant benefits:

- **Polymorphism**: Can treat SteeringRack as RigidBody where appropriate
- **Cleaner API**: Direct access to housing attachment points
- **Consistent Pattern**: Follows established ControlArm pattern
- **Maintainability**: Single inheritance hierarchy reduces complexity

The main breaking change is removal of the `housing` member, which requires updating code that directly accesses it. However, this is mitigated by clear error messages and the fact that the new API is simpler (just call methods directly on the SteeringRack).

The implementation follows the proven pattern of ControlArm, which successfully extends RigidBody with additional state (link endpoints). SteeringRack will similarly extend RigidBody with rack pivot state.
