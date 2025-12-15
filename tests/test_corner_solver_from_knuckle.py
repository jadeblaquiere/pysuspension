"""
Test CornerSolver integration with suspension model.

Tests the new from_suspension_knuckle() functionality including:
- Graph discovery from knuckle
- Component copying with relationship preservation
- Automatic solver configuration
- Solving with knuckle-centric methods
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension.corner_solver import CornerSolver
from pysuspension.suspension_knuckle import SuspensionKnuckle
from pysuspension.control_arm import ControlArm
from pysuspension.suspension_link import SuspensionLink
from pysuspension.suspension_joint import SuspensionJoint
from pysuspension.chassis_corner import ChassisCorner
from pysuspension.attachment_point import AttachmentPoint
from pysuspension.joint_types import JointType
from pysuspension.suspension_graph import discover_suspension_graph, create_working_copies


def setup_double_wishbone_suspension():
    """
    Set up a complete double-wishbone suspension for testing.

    Returns:
        Tuple of (knuckle, upper_arm, lower_arm, chassis_corner, joints_dict)
    """
    print("\n--- Setting up double-wishbone suspension ---")

    # Create chassis corner with attachment points
    chassis_corner = ChassisCorner("front_left")
    uf_chassis = chassis_corner.add_attachment_point("upper_front", [1400, 0, 600], unit='mm')
    ur_chassis = chassis_corner.add_attachment_point("upper_rear", [1200, 0, 600], unit='mm')
    lf_chassis = chassis_corner.add_attachment_point("lower_front", [1500, 0, 300], unit='mm')
    lr_chassis = chassis_corner.add_attachment_point("lower_rear", [1100, 0, 300], unit='mm')

    # Create upper control arm
    upper_arm = ControlArm("upper_arm")
    upper_front_link = SuspensionLink(
        endpoint1=[1400, 0, 600],     # Will connect to chassis
        endpoint2=[1400, 650, 580],   # Ball joint
        name="upper_front",
        unit='mm'
    )
    upper_rear_link = SuspensionLink(
        endpoint1=[1200, 0, 600],     # Will connect to chassis
        endpoint2=[1400, 650, 580],   # Ball joint (shared)
        name="upper_rear",
        unit='mm'
    )
    upper_arm.add_link(upper_front_link)
    upper_arm.add_link(upper_rear_link)

    # Create lower control arm
    lower_arm = ControlArm("lower_arm")
    lower_front_link = SuspensionLink(
        endpoint1=[1500, 0, 300],     # Will connect to chassis
        endpoint2=[1400, 700, 200],   # Ball joint
        name="lower_front",
        unit='mm'
    )
    lower_rear_link = SuspensionLink(
        endpoint1=[1100, 0, 300],     # Will connect to chassis
        endpoint2=[1400, 700, 200],   # Ball joint (shared)
        name="lower_rear",
        unit='mm'
    )
    lower_arm.add_link(lower_front_link)
    lower_arm.add_link(lower_rear_link)

    # Create suspension knuckle
    knuckle = SuspensionKnuckle(
        tire_center_x=1400,
        tire_center_y=750,
        rolling_radius=390,
        toe_angle=0.0,
        camber_angle=-1.0,
        unit='mm',
        name='front_left_knuckle'
    )

    # Add attachment points to knuckle
    upper_ball = knuckle.add_attachment_point("upper_ball_joint", [1400, 650, 580], unit='mm')
    lower_ball = knuckle.add_attachment_point("lower_ball_joint", [1400, 700, 200], unit='mm')

    # Create joints dictionary for tracking
    joints = {}

    # Upper front bushing (chassis to control arm)
    joints['uf_bushing'] = SuspensionJoint("uf_bushing", JointType.BUSHING_SOFT)
    joints['uf_bushing'].add_attachment_point(uf_chassis)
    joints['uf_bushing'].add_attachment_point(upper_front_link.endpoint1)

    # Upper rear bushing
    joints['ur_bushing'] = SuspensionJoint("ur_bushing", JointType.BUSHING_SOFT)
    joints['ur_bushing'].add_attachment_point(ur_chassis)
    joints['ur_bushing'].add_attachment_point(upper_rear_link.endpoint1)

    # Upper ball joint (control arm to knuckle)
    joints['upper_ball'] = SuspensionJoint("upper_ball", JointType.BALL_JOINT)
    joints['upper_ball'].add_attachment_point(upper_front_link.endpoint2)
    joints['upper_ball'].add_attachment_point(upper_rear_link.endpoint2)
    joints['upper_ball'].add_attachment_point(upper_ball)

    # Lower front bushing
    joints['lf_bushing'] = SuspensionJoint("lf_bushing", JointType.BUSHING_SOFT)
    joints['lf_bushing'].add_attachment_point(lf_chassis)
    joints['lf_bushing'].add_attachment_point(lower_front_link.endpoint1)

    # Lower rear bushing
    joints['lr_bushing'] = SuspensionJoint("lr_bushing", JointType.BUSHING_SOFT)
    joints['lr_bushing'].add_attachment_point(lr_chassis)
    joints['lr_bushing'].add_attachment_point(lower_rear_link.endpoint1)

    # Lower ball joint
    joints['lower_ball'] = SuspensionJoint("lower_ball", JointType.BALL_JOINT)
    joints['lower_ball'].add_attachment_point(lower_front_link.endpoint2)
    joints['lower_ball'].add_attachment_point(lower_rear_link.endpoint2)
    joints['lower_ball'].add_attachment_point(lower_ball)

    print(f"✓ Created {len(joints)} joints")
    print(f"✓ Created 2 control arms with 4 links")
    print(f"✓ Created knuckle with {len(knuckle.attachment_points)} attachment points")

    return knuckle, upper_arm, lower_arm, chassis_corner, joints


def test_graph_discovery():
    """Test suspension graph discovery from knuckle."""
    print("\n" + "=" * 70)
    print("TEST: Graph Discovery")
    print("=" * 70)

    knuckle, upper_arm, lower_arm, chassis_corner, joints = setup_double_wishbone_suspension()

    # Discover the graph
    print("\nDiscovering suspension graph...")
    graph = discover_suspension_graph(knuckle)

    # Validate discovery
    print(f"\nDiscovered components:")
    print(f"  Control arms: {len(graph.control_arms)}")
    print(f"  Standalone links: {len(graph.links)}")
    print(f"  Joints: {len(graph.joints)}")
    print(f"  Chassis points: {len(graph.chassis_points)}")
    print(f"  Knuckle points: {len(graph.knuckle_points)}")
    print(f"  Total attachment points: {len(graph.all_attachment_points)}")

    # Assertions
    assert len(graph.control_arms) == 2, f"Expected 2 control arms, got {len(graph.control_arms)}"
    assert len(graph.joints) == 6, f"Expected 6 joints, got {len(graph.joints)}"
    assert len(graph.chassis_points) == 4, f"Expected 4 chassis points, got {len(graph.chassis_points)}"
    assert len(graph.knuckle_points) == 2, f"Expected 2 knuckle points, got {len(graph.knuckle_points)}"
    assert graph.knuckle is knuckle, "Graph knuckle should be the original knuckle"

    # Check that control arms were found
    arm_names = {arm.name for arm in graph.control_arms}
    assert 'upper_arm' in arm_names, "Upper arm not found"
    assert 'lower_arm' in arm_names, "Lower arm not found"

    print("\n✓ Graph discovery test passed!")
    return graph


def test_component_copying():
    """Test component copying with relationship preservation."""
    print("\n" + "=" * 70)
    print("TEST: Component Copying")
    print("=" * 70)

    knuckle, upper_arm, lower_arm, chassis_corner, joints = setup_double_wishbone_suspension()

    # Discover graph
    graph = discover_suspension_graph(knuckle)

    # Create copies
    print("\nCreating working copies...")
    copied_graph, mapping = create_working_copies(graph)

    print(f"Created {len(mapping)} object mappings")

    # Validate copying
    assert copied_graph.knuckle is not graph.knuckle, "Knuckle should be copied"
    assert copied_graph.knuckle.name == graph.knuckle.name, "Knuckle name should match"

    # Check that positions are the same but objects are different
    orig_pos = graph.knuckle.tire_center
    copy_pos = copied_graph.knuckle.tire_center
    assert np.allclose(orig_pos, copy_pos), "Copied knuckle position should match"
    assert copied_graph.knuckle.tire_center is not graph.knuckle.tire_center, "Position arrays should be different objects"

    # Check control arms
    assert len(copied_graph.control_arms) == len(graph.control_arms), "Same number of control arms"
    for orig_arm, copy_arm in zip(graph.control_arms, copied_graph.control_arms):
        assert copy_arm is not orig_arm, "Control arms should be different objects"
        assert copy_arm.name == orig_arm.name, "Control arm names should match"
        assert len(copy_arm.links) == len(orig_arm.links), "Same number of links"

    # Check joints are reconstructed
    assert len(copied_graph.joints) == len(graph.joints), "Same number of joints"
    for joint_name in graph.joints:
        assert joint_name in copied_graph.joints, f"Joint {joint_name} should be copied"
        orig_joint = graph.joints[joint_name]
        copy_joint = copied_graph.joints[joint_name]
        assert copy_joint is not orig_joint, "Joints should be different objects"
        assert copy_joint.joint_type == orig_joint.joint_type, "Joint types should match"
        assert len(copy_joint.attachment_points) == len(orig_joint.attachment_points), "Same number of connected points"

    # Verify that modifying copy doesn't affect original
    print("\nVerifying isolation (modifying copy)...")
    original_tire_center = graph.knuckle.tire_center.copy()
    copied_graph.knuckle.tire_center[2] += 50.0  # Move copy up by 50mm

    assert not np.allclose(graph.knuckle.tire_center, copied_graph.knuckle.tire_center), "Original should be unchanged"
    assert np.allclose(graph.knuckle.tire_center, original_tire_center), "Original should remain at original position"

    print("\n✓ Component copying test passed!")
    return copied_graph, mapping


def test_from_suspension_knuckle():
    """Test CornerSolver.from_suspension_knuckle() method."""
    print("\n" + "=" * 70)
    print("TEST: CornerSolver.from_suspension_knuckle()")
    print("=" * 70)

    knuckle, upper_arm, lower_arm, chassis_corner, joints = setup_double_wishbone_suspension()

    # Create solver from knuckle
    print("\nCreating CornerSolver from knuckle...")
    solver = CornerSolver.from_suspension_knuckle(knuckle, name="test_solver")

    # Validate solver configuration
    print(f"\nSolver configuration:")
    print(f"  Control arms: {len(solver.control_arms)}")
    print(f"  Links: {len(solver.links)}")
    print(f"  Joints: {len(solver.joints)}")
    print(f"  Chassis mounts: {len(solver.chassis_mounts)}")
    print(f"  Constraints: {len(solver.constraints)}")
    print(f"  DOF: {solver.state.get_dof()}")
    print(f"  Wheel center: {solver.wheel_center is not None}")

    # Assertions
    assert solver.original_knuckle is knuckle, "Original knuckle should be stored"
    assert solver.copied_knuckle is not None, "Copied knuckle should exist"
    assert solver.copied_knuckle is not knuckle, "Copied knuckle should be different object"
    assert len(solver.joints) == 6, f"Expected 6 joints, got {len(solver.joints)}"
    assert len(solver.chassis_mounts) == 4, f"Expected 4 chassis mounts, got {len(solver.chassis_mounts)}"
    assert solver.wheel_center is not None, "Wheel center should be set"
    assert len(solver.constraints) > 0, "Should have constraints"

    # Chassis mounts should be in the fixed points list
    # (We can verify they exist in the state)
    for chassis_mount in solver.chassis_mounts:
        point_name = chassis_mount.name
        assert point_name in solver.state.points, f"Chassis point {point_name} should be in state"

    print("\n✓ from_suspension_knuckle() test passed!")
    return solver


def test_solve_for_knuckle_heave():
    """Test solving with knuckle heave constraint."""
    print("\n" + "=" * 70)
    print("TEST: solve_for_knuckle_heave()")
    print("=" * 70)

    knuckle, upper_arm, lower_arm, chassis_corner, joints = setup_double_wishbone_suspension()

    # Create solver
    solver = CornerSolver.from_suspension_knuckle(knuckle)

    # Save initial state
    solver.save_initial_state()
    initial_tire_center = solver.copied_knuckle.tire_center.copy()

    print(f"\nInitial tire center: {initial_tire_center} mm")

    # Solve for +25mm heave
    print("\nSolving for +25mm heave...")
    result = solver.solve_for_knuckle_heave(25, unit='mm')

    print(f"Solver converged: {result.success}")
    print(f"RMS error: {result.get_rms_error():.6f} mm")

    # Get solved wheel center position from the solver state
    solved_wheel_center = result.get_position(solver.wheel_center.name)

    print(f"Solved wheel center: {solved_wheel_center} mm")
    print(f"Z displacement: {solved_wheel_center[2] - initial_tire_center[2]:.3f} mm")

    # Assertions
    assert result.success, "Solver should converge"
    assert result.get_rms_error() < 1.0, f"RMS error {result.get_rms_error():.6f} should be < 1.0 mm"

    # Check that Z moved by approximately 25mm (allowing for suspension geometry)
    z_displacement = solved_wheel_center[2] - initial_tire_center[2]
    assert abs(z_displacement - 25.0) < 5.0, f"Z displacement {z_displacement:.3f} should be close to 25mm"

    # Check that original knuckle is unchanged
    assert np.allclose(knuckle.tire_center, initial_tire_center), "Original knuckle should be unchanged"

    # Test solving at different heave positions
    print("\nTesting multiple heave positions...")
    heave_positions = [-25, -10, 0, 10, 25, 40]
    results = []

    for heave in heave_positions:
        solver.reset_to_initial_state()
        result = solver.solve_for_heave(heave, unit='mm')
        solved_wheel_pos = result.get_position(solver.wheel_center.name)
        solved_z = solved_wheel_pos[2]
        z_disp = solved_z - initial_tire_center[2]
        rms = result.get_rms_error()
        results.append((heave, z_disp, rms))
        print(f"  Heave {heave:+4d}mm: Z_disp={z_disp:+7.3f}mm, RMS={rms:.6f}mm")

    # Check all converged
    for heave, z_disp, rms in results:
        assert rms < 1.0, f"RMS error at heave={heave} should be < 1.0 mm"

    print("\n✓ solve_for_knuckle_heave() test passed!")
    return solver, results


def test_update_original_from_solved():
    """Test updating original components from solved positions."""
    print("\n" + "=" * 70)
    print("TEST: update_original_from_solved()")
    print("=" * 70)

    knuckle, upper_arm, lower_arm, chassis_corner, joints = setup_double_wishbone_suspension()

    # Create solver with copying
    solver = CornerSolver.from_suspension_knuckle(knuckle, copy_components=True)

    # Store original positions
    original_tire_center = knuckle.tire_center.copy()
    original_upper_ball = knuckle.get_attachment_point("upper_ball_joint").position.copy()

    print(f"Original tire center: {original_tire_center} mm")

    # Debug: Check if knuckle points are in solver state
    print(f"\nKnuckle points in solver state:")
    for kp in solver.knuckle_points:
        print(f"  {kp.name}: in state = {kp.name in solver.state.points}")

    # Debug: Check constraints involving knuckle points
    print(f"\nConstraints involving knuckle points:")
    for constraint in solver.constraints:
        constraint_name = constraint.name
        if 'upper_ball' in constraint_name or 'lower_ball' in constraint_name:
            print(f"  {constraint_name}")

    # Solve for heave
    print("\nSolving for +30mm heave...")
    result = solver.solve_for_heave(30, unit='mm')

    solved_tire_center = solver.copied_knuckle.tire_center.copy()
    print(f"Solved tire center: {solved_tire_center} mm")

    # Debug: Check if control arm endpoints moved
    print(f"\nControl arm endpoint positions after solving:")
    for arm in solver.control_arms:
        for link in arm.links:
            ep1_pos = result.get_position(link.endpoint1.name)
            ep2_pos = result.get_position(link.endpoint2.name)
            print(f"  {arm.name}.{link.name}.endpoint1: {ep1_pos}")
            print(f"  {arm.name}.{link.name}.endpoint2: {ep2_pos}")

    # Debug: Check knuckle point positions in solver state
    print(f"\nKnuckle point positions in solver state:")
    for kp in solver.knuckle_points:
        pos_in_state = result.get_position(kp.name)
        pos_in_knuckle = kp.position
        print(f"  {kp.name}:")
        print(f"    In solver state: {pos_in_state}")
        print(f"    In knuckle object: {pos_in_knuckle}")

    # Verify original is still unchanged
    assert np.allclose(knuckle.tire_center, original_tire_center), "Original should still be unchanged before update"

    # Update original from solved
    print("\nUpdating original from solved positions...")
    solver.update_original_from_solved()

    updated_tire_center = knuckle.tire_center
    updated_upper_ball = knuckle.get_attachment_point("upper_ball_joint").position

    print(f"Updated tire center: {updated_tire_center} mm")
    print(f"Original upper ball: {original_upper_ball} mm")
    print(f"Updated upper ball: {updated_upper_ball} mm")
    print(f"Copied upper ball: {solver.copied_knuckle.get_attachment_point('upper_ball_joint').position} mm")

    # Assertions
    assert np.allclose(updated_tire_center, solved_tire_center), "Original should now match solved"
    assert not np.allclose(updated_tire_center, original_tire_center), "Original should be different from initial"

    # Check attachment points updated
    assert not np.allclose(updated_upper_ball, original_upper_ball), "Attachment points should be updated"

    print("\n✓ update_original_from_solved() test passed!")


def test_full_integration():
    """Full integration test with complete workflow."""
    print("\n" + "=" * 70)
    print("TEST: Full Integration")
    print("=" * 70)

    # Set up complete suspension
    knuckle, upper_arm, lower_arm, chassis_corner, joints = setup_double_wishbone_suspension()

    print("\n--- Workflow Test ---")

    # Step 1: Create solver from knuckle
    print("\n1. Creating solver from knuckle...")
    solver = CornerSolver.from_suspension_knuckle(knuckle, name="integration_test")
    print(f"   ✓ Solver created with {solver.state.get_dof()} DOF")

    # Step 2: Solve initial configuration
    print("\n2. Solving initial configuration...")
    solver.save_initial_state()
    result = solver.solve()
    print(f"   ✓ Initial solve: RMS error = {result.get_rms_error():.6f} mm")

    # Step 3: Solve for heave travel
    print("\n3. Solving heave travel sweep...")
    heave_values = [-40, -20, 0, 20, 40]
    camber_values = []

    for heave in heave_values:
        solver.reset_to_initial_state()
        result = solver.solve_for_heave(heave, unit='mm')

        # Calculate camber from solved positions
        solved_knuckle = solver.get_solved_knuckle()
        upper_ball = solved_knuckle.get_attachment_point("upper_ball_joint")
        lower_ball = solved_knuckle.get_attachment_point("lower_ball_joint")
        camber = solver.get_camber(upper_ball, lower_ball, unit='deg')
        camber_values.append(camber)

        print(f"   Heave {heave:+3d}mm: Camber = {camber:+6.2f}°, RMS = {result.get_rms_error():.6f} mm")

    # Step 4: Verify camber change
    print("\n4. Verifying camber gain...")
    camber_change = camber_values[-1] - camber_values[0]
    heave_range = heave_values[-1] - heave_values[0]
    camber_gain = camber_change / heave_range
    print(f"   Camber change: {camber_change:.3f}° over {heave_range}mm")
    print(f"   Camber gain: {camber_gain:.4f}°/mm")

    # Step 5: Test component isolation
    print("\n5. Verifying original components unchanged...")
    assert np.allclose(knuckle.tire_center[2], 390.0), "Original knuckle should be at initial position"
    print(f"   ✓ Original knuckle position: {knuckle.tire_center} mm (unchanged)")

    # Step 6: Test geometry calculations
    print("\n6. Testing geometry calculations...")
    solved_knuckle = solver.get_solved_knuckle()
    upper_ball = solved_knuckle.get_attachment_point("upper_ball_joint")
    lower_ball = solved_knuckle.get_attachment_point("lower_ball_joint")

    camber = solver.get_camber(upper_ball, lower_ball, unit='deg')
    caster = solver.get_caster(upper_ball, lower_ball, unit='deg')

    print(f"   Camber: {camber:.2f}°")
    print(f"   Caster: {caster:.2f}°")

    print("\n✓ Full integration test passed!")
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        # Run all tests
        print("\n" + "=" * 70)
        print("CORNERSOLVER INTEGRATION TEST SUITE")
        print("=" * 70)

        test_graph_discovery()
        test_component_copying()
        test_from_suspension_knuckle()
        test_solve_for_knuckle_heave()
        test_update_original_from_solved()
        test_full_integration()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
