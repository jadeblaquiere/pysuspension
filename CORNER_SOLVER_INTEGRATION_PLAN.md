# CornerSolver Integration with Suspension Model - Implementation Plan

## Overview

This plan details the integration of CornerSolver with the existing suspension model (SuspensionKnuckle, ChassisCorner, ControlArm, SuspensionLink, SuspensionJoint) to enable constraint-based solving initialized directly from modeled suspension components.

## Goals

1. Initialize a CornerSolver instance from a SuspensionKnuckle reference
2. Automatically discover and traverse the component/joint graph from knuckle to chassis attachment points
3. Auto-configure chassis attachment points as fixed constraints
4. Create working copies of all components for solving (original objects remain unchanged)
5. Apply constraints to the knuckle (e.g., heave displacement)
6. Solve for new positions using constraint-based algorithm
7. Results available in the copied component objects

## Current Architecture Analysis

### Existing Components

1. **CornerSolver** (`corner_solver.py`):
   - Extends SuspensionSolver
   - Currently requires manual setup of joints, control arms, links
   - Has methods: `add_joint()`, `add_control_arm()`, `add_link()`, `solve_for_heave()`
   - Tracks: joints dict, control_arms list, links list, chassis_mounts list

2. **SuspensionKnuckle** (`suspension_knuckle.py`):
   - Extends RigidBody
   - Has attachment points for ball joints, tie rod, etc.
   - Each attachment point can have a joint reference
   - Has methods: `add_attachment_point()`, `get_attachment_point()`

3. **SuspensionJoint** (`suspension_joint.py`):
   - Connects multiple AttachmentPoints
   - Has joint_type (BALL_JOINT, BUSHING_SOFT, etc.)
   - Has methods: `add_attachment_point()`, `get_connected_points()`
   - Each AttachmentPoint has a back-reference to its joint

4. **AttachmentPoint** (`attachment_point.py`):
   - Has: name, position, parent_component, joint
   - parent_component can be: SuspensionKnuckle, ControlArm, ChassisCorner, etc.
   - joint is a SuspensionJoint connecting this point to others

5. **ControlArm** (`control_arm.py`):
   - Extends RigidBody
   - Contains multiple SuspensionLinks
   - Has methods: `add_link()`, `get_all_attachment_points()`

6. **SuspensionLink** (`suspension_link.py`):
   - Rigid link with two endpoints (both AttachmentPoints)
   - Maintains constant length
   - Endpoints can connect to joints

7. **ChassisCorner** (`chassis_corner.py`):
   - Contains attachment points for one corner
   - Attachment points here should be fixed in solving

8. **Chassis** (`chassis.py`):
   - Contains multiple ChassisCorners
   - Can register components and joints

## Detailed Design

### Phase 1: Graph Traversal Algorithm

**Objective:** Discover all components and joints connected to a SuspensionKnuckle

**Algorithm:**
```python
def discover_suspension_graph(knuckle: SuspensionKnuckle) -> SuspensionGraph:
    """
    Traverse from knuckle attachment points to discover all connected components.

    Returns:
        SuspensionGraph containing:
        - knuckle: The starting knuckle
        - control_arms: List of discovered ControlArms
        - links: List of discovered SuspensionLinks
        - joints: Dict of discovered SuspensionJoints
        - chassis_points: List of ChassisCorner attachment points
        - all_attachment_points: Set of all discovered points
    """
```

**Traversal Steps:**
1. Start with knuckle attachment points
2. For each attachment point:
   - If it has a joint, get all connected points from that joint
   - For each connected point:
     - Check parent_component type
     - If ChassisCorner: add to chassis_points (stop traversal here)
     - If ControlArm: add to control_arms, continue with its points
     - If SuspensionLink: add to links, continue with other endpoint
     - Track visited points to avoid cycles
3. Collect all discovered joints
4. Return structured graph

**Data Structure:**
```python
@dataclass
class SuspensionGraph:
    knuckle: SuspensionKnuckle
    control_arms: List[ControlArm]
    links: List[SuspensionLink]
    joints: Dict[str, SuspensionJoint]
    chassis_points: List[AttachmentPoint]
    knuckle_points: List[AttachmentPoint]
    all_attachment_points: Set[AttachmentPoint]
```

### Phase 2: Deep Copy Strategy

**Objective:** Create working copies of all components while maintaining relationships

**Challenge:** Objects reference each other (joints reference points, points reference components, etc.)

**Solution:** Two-pass copying with reference mapping

**Algorithm:**
```python
def create_working_copies(graph: SuspensionGraph) -> Tuple[SuspensionGraph, Dict]:
    """
    Create deep copies of all components with maintained relationships.

    Returns:
        Tuple of:
        - New SuspensionGraph with copied components
        - Mapping dict: {id(original): copy}
    """
```

**Copy Process:**

**Pass 1: Create object copies**
1. Copy knuckle → creates new attachment points
2. Copy each control arm → creates new links and attachment points
3. Copy each standalone link → creates new attachment points
4. Build mapping: `{id(original_point): copied_point, id(original_component): copied_component}`

**Pass 2: Reconstruct joint connections**
1. For each original joint:
   - Create new joint with same name and type
   - For each attached point in original:
     - Find copied point using mapping
     - Add to new joint
   - Store in new joints dict

**Key Implementation Details:**
- Use `copy.deepcopy()` with custom `__deepcopy__` methods if needed
- Or implement explicit `copy()` methods on each class
- Maintain a single mapping dict throughout the process
- Ensure copied AttachmentPoints reference copied components

### Phase 3: CornerSolver Initialization from Model

**Objective:** Add factory method to create CornerSolver from SuspensionKnuckle

**New Method:**
```python
@classmethod
def from_suspension_knuckle(cls,
                            knuckle: SuspensionKnuckle,
                            wheel_center_point: Optional[AttachmentPoint] = None,
                            name: str = "corner_solver",
                            copy_components: bool = True) -> 'CornerSolver':
    """
    Create a CornerSolver from a suspension knuckle.

    This method:
    1. Discovers all connected components via graph traversal
    2. Creates working copies of all components (if copy_components=True)
    3. Configures solver with discovered geometry
    4. Fixes chassis attachment points
    5. Sets up all joints and constraints

    Args:
        knuckle: SuspensionKnuckle to build solver from
        wheel_center_point: Optional specific point to use as wheel center
                          If None, uses knuckle.tire_center
        name: Name for the solver
        copy_components: If True, creates copies; if False, uses originals

    Returns:
        Configured CornerSolver ready for solving

    Example:
        >>> knuckle = SuspensionKnuckle(...)
        >>> # ... set up control arms, links, joints ...
        >>> solver = CornerSolver.from_suspension_knuckle(knuckle)
        >>> result = solver.solve_for_heave(25, unit='mm')
        >>> # Original knuckle unchanged, solver has copies
    """
```

**Implementation Steps:**
1. Discover suspension graph from knuckle
2. If copy_components=True:
   - Create working copies with maintained relationships
   - Use copied graph for solver setup
3. Create new CornerSolver instance
4. Register all copied joints using `add_joint()`
5. For each control arm: call `add_control_arm()` with appropriate mount points
6. For each standalone link: call `add_link()` with appropriate mount points
7. Identify chassis_mounts from graph.chassis_points
8. Set wheel center (use wheel_center_point or create from tire_center)
9. Store reference to original knuckle and copied knuckle

**Additional Attributes:**
```python
class CornerSolver(SuspensionSolver):
    # ... existing attributes ...

    # New attributes for model integration
    self.original_knuckle: Optional[SuspensionKnuckle] = None
    self.copied_knuckle: Optional[SuspensionKnuckle] = None
    self.component_mapping: Dict = {}  # original_id -> copy
    self.reverse_mapping: Dict = {}    # copy_id -> original
```

### Phase 4: Enhanced Constraint Methods

**Objective:** Add knuckle-centric constraint methods

**New Methods:**

```python
def solve_for_knuckle_heave(self,
                           displacement: float,
                           unit: str = 'mm',
                           initial_guess: Optional[np.ndarray] = None) -> SolverResult:
    """
    Solve for suspension position with knuckle heaved vertically.

    Similar to solve_for_heave() but specifically works with the knuckle's
    tire center position.

    Args:
        displacement: Vertical displacement from current knuckle position
        unit: Unit of displacement
        initial_guess: Initial guess for solver

    Returns:
        SolverResult with new positions in copied components
    """
    # Use copied_knuckle.tire_center for constraint application
    # Call solve_for_heave() with appropriate wheel_center

def update_original_from_solved(self) -> None:
    """
    Update original components with solved positions from copies.

    WARNING: This modifies the original suspension model!
    Use with caution.

    Updates:
    - Original knuckle position and orientation
    - Original control arm positions
    - Original link endpoint positions
    """
    # Use component_mapping to update originals
    # For each copied component, update corresponding original

def get_solved_knuckle(self) -> SuspensionKnuckle:
    """
    Get the solved knuckle with updated position/orientation.

    Returns:
        Copied knuckle with solved positions
    """
    return self.copied_knuckle
```

### Phase 5: Component Copy Methods

**Objective:** Ensure all component classes support proper copying

**Required Methods for Each Component:**

1. **AttachmentPoint** (already has `copy()` method):
   - Returns new point without joint connection
   - Needs enhancement to optionally copy parent_component reference

2. **SuspensionLink**:
   - Add `copy()` method that creates new link with copied endpoints
   - Should maintain length

3. **ControlArm** (extends RigidBody):
   - Add `copy()` method that copies all links and attachment points
   - Should maintain rigid body properties

4. **SuspensionKnuckle** (extends RigidBody):
   - Add `copy()` method that copies attachment points and tire geometry
   - Should maintain orientation and tire properties

5. **SuspensionJoint**:
   - Already has basic structure
   - Enhance to support copying without attachment points (for two-pass copy)

## Implementation Phases

### Phase 1: Graph Discovery (Priority: HIGH)

**Files to Modify:**
- Create new file: `pysuspension/suspension_graph.py`
  - Define `SuspensionGraph` dataclass
  - Implement `discover_suspension_graph()` function

**Estimated Complexity:** Medium
**Dependencies:** None

### Phase 2: Component Copy Methods (Priority: HIGH)

**Files to Modify:**
- `pysuspension/suspension_link.py`: Add `copy()` method
- `pysuspension/control_arm.py`: Add `copy()` method
- `pysuspension/suspension_knuckle.py`: Add `copy()` method
- `pysuspension/attachment_point.py`: Enhance `copy()` method

**Estimated Complexity:** Medium
**Dependencies:** None

### Phase 3: Deep Copy with Relationships (Priority: HIGH)

**Files to Modify:**
- `pysuspension/suspension_graph.py`: Add `create_working_copies()` function

**Estimated Complexity:** High (complex relationship management)
**Dependencies:** Phase 1, Phase 2

### Phase 4: CornerSolver Integration (Priority: HIGH)

**Files to Modify:**
- `pysuspension/corner_solver.py`:
  - Add `from_suspension_knuckle()` class method
  - Add new instance attributes
  - Add helper methods for auto-configuration

**Estimated Complexity:** High
**Dependencies:** Phase 1, Phase 2, Phase 3

### Phase 5: Enhanced Constraint Methods (Priority: MEDIUM)

**Files to Modify:**
- `pysuspension/corner_solver.py`:
  - Add `solve_for_knuckle_heave()` method
  - Add `update_original_from_solved()` method
  - Add `get_solved_knuckle()` method

**Estimated Complexity:** Low
**Dependencies:** Phase 4

### Phase 6: Testing (Priority: HIGH)

**Files to Create:**
- `tests/test_corner_solver_from_knuckle.py`
  - Test graph discovery
  - Test component copying
  - Test solver initialization from knuckle
  - Test solving with heave constraint
  - Test that originals are unchanged

**Estimated Complexity:** Medium
**Dependencies:** All previous phases

### Phase 7: Documentation (Priority: MEDIUM)

**Files to Modify:**
- Update `pysuspension/corner_solver.py` docstrings
- Update README.md with new usage examples
- Create example script demonstrating the new workflow

**Estimated Complexity:** Low
**Dependencies:** All previous phases

## Usage Example (Target API)

```python
from pysuspension import (
    SuspensionKnuckle, ControlArm, SuspensionLink,
    SuspensionJoint, ChassisCorner, CornerSolver
)
from pysuspension.joint_types import JointType

# 1. Set up suspension model
chassis_corner = ChassisCorner("front_left")
chassis_corner.add_attachment_point("upper_front", [1400, 0, 600], unit='mm')
chassis_corner.add_attachment_point("upper_rear", [1200, 0, 600], unit='mm')
chassis_corner.add_attachment_point("lower_front", [1500, 0, 300], unit='mm')
chassis_corner.add_attachment_point("lower_rear", [1100, 0, 300], unit='mm')

# Create control arms
upper_arm = ControlArm("upper_arm")
upper_front_link = SuspensionLink(
    chassis_corner.get_attachment_point("upper_front"),
    [1400, 650, 580],  # Upper ball joint position
    name="upper_front",
    unit='mm'
)
upper_arm.add_link(upper_front_link)
# ... add rear link ...

# Create knuckle with attachment points
knuckle = SuspensionKnuckle(
    tire_center_x=1400, tire_center_y=750, rolling_radius=390,
    camber_angle=-1.0, unit='mm'
)
knuckle.add_attachment_point("upper_ball_joint", [1400, 650, 580], unit='mm')
knuckle.add_attachment_point("lower_ball_joint", [1400, 700, 200], unit='mm')

# Define joints connecting components
upper_ball_joint = SuspensionJoint("upper_ball", JointType.BALL_JOINT)
upper_ball_joint.add_attachment_point(upper_front_link.endpoint2)
upper_ball_joint.add_attachment_point(knuckle.get_attachment_point("upper_ball_joint"))

# ... set up lower arm, joints, etc. ...

# 2. Initialize solver from knuckle (NEW!)
solver = CornerSolver.from_suspension_knuckle(knuckle)

# 3. Apply heave constraint and solve (NEW!)
result = solver.solve_for_knuckle_heave(25, unit='mm')

# 4. Get results from copied components (NEW!)
solved_knuckle = solver.get_solved_knuckle()
print(f"New knuckle position: {solved_knuckle.tire_center}")
print(f"New camber: {np.degrees(solved_knuckle.camber_angle)}")

# 5. Original knuckle is unchanged
print(f"Original knuckle position: {knuckle.tire_center}")  # Still at original
```

## Key Design Decisions

### Decision 1: Copy by Default
**Choice:** `copy_components=True` by default in `from_suspension_knuckle()`
**Rationale:**
- Safer: originals remain unchanged
- Allows multiple solves with different constraints
- Clearer separation between model and analysis

**Alternative:** Could modify originals in place
**Rejected because:** Would make it hard to reset, could lose original geometry

### Decision 2: Two-Pass Copying
**Choice:** Copy objects first, then reconstruct joint connections
**Rationale:**
- Simpler than trying to maintain all references during copy
- Clear separation of concerns
- Easier to debug

**Alternative:** Single-pass deep copy with custom `__deepcopy__`
**Rejected because:** More complex, harder to maintain

### Decision 3: Graph Discovery via Parent Component
**Choice:** Use `parent_component` to identify chassis attachment points
**Rationale:**
- ChassisCorner already tracks its attachment points
- Natural stopping condition for traversal
- Clear semantic meaning

**Alternative:** Mark chassis points with a flag
**Rejected because:** Requires additional bookkeeping, less explicit

### Decision 4: Wheel Center Handling
**Choice:** Create AttachmentPoint from knuckle.tire_center if not provided
**Rationale:**
- Knuckle already tracks tire center position
- Natural connection between model and solver
- Allows override if needed

**Alternative:** Require explicit wheel_center_point
**Rejected because:** Less convenient, redundant with knuckle data

## Validation Criteria

The implementation will be considered successful when:

1. ✓ Can initialize CornerSolver from a SuspensionKnuckle with connected components
2. ✓ Graph traversal discovers all control arms, links, and joints
3. ✓ Chassis attachment points are automatically identified and fixed
4. ✓ Component copying maintains all relationships correctly
5. ✓ Solving with heave constraint produces correct results
6. ✓ Original components remain unchanged after solving
7. ✓ Copied components contain solved positions
8. ✓ Can access solved knuckle position and orientation
9. ✓ All existing CornerSolver functionality still works
10. ✓ Comprehensive tests pass for new functionality

## Risk Assessment

### Risk 1: Circular References in Copying
**Probability:** Medium
**Impact:** High
**Mitigation:** Use two-pass copying, extensive testing

### Risk 2: Missing Components in Traversal
**Probability:** Medium
**Impact:** Medium
**Mitigation:** Thorough traversal algorithm, validation checks

### Risk 3: Performance with Large Graphs
**Probability:** Low
**Impact:** Low
**Mitigation:** Graph sizes expected to be small (single corner)

### Risk 4: Breaking Existing API
**Probability:** Low
**Impact:** High
**Mitigation:** Add new methods only, keep existing methods unchanged

## Future Enhancements

1. **Multi-corner solving:** Extend to handle complete vehicle with all four corners
2. **Incremental solving:** Update only changed components
3. **Serialization:** Save/load configured solvers
4. **Visualization:** Show discovered component graph
5. **Optimization:** Cache graph discovery for repeated solves
6. **Validation:** Check for invalid suspension configurations

## Summary

This implementation plan provides a clear path to integrate CornerSolver with the existing suspension model. The key innovations are:

1. **Automatic discovery** of suspension geometry from a knuckle reference
2. **Deep copying** with maintained relationships for safe solving
3. **Intuitive API** that matches the user's mental model
4. **Backwards compatibility** with existing CornerSolver usage

The phased approach allows for incremental development and testing, reducing risk and ensuring quality at each step.
