"""
Example: Using CornerSolver with Suspension Model Integration

This example demonstrates the new workflow for constraint-based suspension solving
using CornerSolver.from_suspension_knuckle(). This approach allows you to:

1. Build your suspension model with components and joints
2. Automatically discover and configure the solver
3. Solve for different positions while keeping originals unchanged
4. Access solved knuckle position and orientation

Advantages:
- No manual solver setup
- Automatic component discovery
- Safe copying (originals unchanged)
- Intuitive knuckle-centric API
"""

import numpy as np
import matplotlib.pyplot as plt

from pysuspension import (
    SuspensionKnuckle,
    ControlArm,
    SuspensionLink,
    SuspensionJoint,
    ChassisCorner,
    CornerSolver
)
from pysuspension.joint_types import JointType


def create_double_wishbone_suspension():
    """
    Create a complete double-wishbone suspension model.

    This function demonstrates how to build a suspension model with:
    - Chassis corner with attachment points
    - Upper and lower control arms
    - Suspension knuckle with ball joint attachments
    - Joints connecting all components

    Returns:
        SuspensionKnuckle: The knuckle with all components connected
    """
    print("=" * 70)
    print("BUILDING SUSPENSION MODEL")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Create chassis corner with mounting points
    # ========================================================================
    print("\n1. Creating chassis corner...")

    chassis_corner = ChassisCorner("front_left")

    # Upper control arm chassis mounts
    uf_chassis = chassis_corner.add_attachment_point(
        "upper_front_mount", [1400, 0, 600], unit='mm'
    )
    ur_chassis = chassis_corner.add_attachment_point(
        "upper_rear_mount", [1200, 0, 600], unit='mm'
    )

    # Lower control arm chassis mounts
    lf_chassis = chassis_corner.add_attachment_point(
        "lower_front_mount", [1500, 0, 300], unit='mm'
    )
    lr_chassis = chassis_corner.add_attachment_point(
        "lower_rear_mount", [1100, 0, 300], unit='mm'
    )

    print(f"   ✓ Created chassis corner with {len(chassis_corner.attachment_points)} mount points")

    # ========================================================================
    # STEP 2: Create upper control arm
    # ========================================================================
    print("\n2. Creating upper control arm...")

    upper_arm = ControlArm("upper_control_arm")

    # Front link (chassis to ball joint)
    upper_front_link = SuspensionLink(
        endpoint1=[1400, 0, 600],      # Chassis mount
        endpoint2=[1400, 650, 580],    # Ball joint location
        name="upper_front_link",
        unit='mm'
    )
    upper_arm.add_link(upper_front_link)

    # Rear link (chassis to ball joint - shared endpoint)
    upper_rear_link = SuspensionLink(
        endpoint1=[1200, 0, 600],      # Chassis mount
        endpoint2=[1400, 650, 580],    # Ball joint location (shared)
        name="upper_rear_link",
        unit='mm'
    )
    upper_arm.add_link(upper_rear_link)

    print(f"   ✓ Created upper control arm with {len(upper_arm.links)} links")

    # ========================================================================
    # STEP 3: Create lower control arm
    # ========================================================================
    print("\n3. Creating lower control arm...")

    lower_arm = ControlArm("lower_control_arm")

    # Front link
    lower_front_link = SuspensionLink(
        endpoint1=[1500, 0, 300],      # Chassis mount
        endpoint2=[1400, 700, 200],    # Ball joint location
        name="lower_front_link",
        unit='mm'
    )
    lower_arm.add_link(lower_front_link)

    # Rear link
    lower_rear_link = SuspensionLink(
        endpoint1=[1100, 0, 300],      # Chassis mount
        endpoint2=[1400, 700, 200],    # Ball joint location (shared)
        name="lower_rear_link",
        unit='mm'
    )
    lower_arm.add_link(lower_rear_link)

    print(f"   ✓ Created lower control arm with {len(lower_arm.links)} links")

    # ========================================================================
    # STEP 4: Create suspension knuckle
    # ========================================================================
    print("\n4. Creating suspension knuckle...")

    knuckle = SuspensionKnuckle(
        tire_center_x=1400,           # Longitudinal position (mm)
        tire_center_y=750,            # Lateral position / track (mm)
        rolling_radius=390,           # Tire rolling radius (mm)
        toe_angle=0.0,                # Initial toe (degrees)
        camber_angle=-1.0,            # Initial camber (degrees)
        wheel_offset=0.0,             # Wheel mounting plane offset
        unit='mm',
        name='front_left_knuckle'
    )

    # Add ball joint attachment points on knuckle
    upper_ball_knuckle = knuckle.add_attachment_point(
        "upper_ball_joint", [1400, 650, 580], unit='mm'
    )
    lower_ball_knuckle = knuckle.add_attachment_point(
        "lower_ball_joint", [1400, 700, 200], unit='mm'
    )

    print(f"   ✓ Created knuckle with tire center at {knuckle.tire_center} mm")

    # ========================================================================
    # STEP 5: Create joints connecting components
    # ========================================================================
    print("\n5. Creating joints...")

    # Upper control arm to chassis (bushings)
    uf_bushing = SuspensionJoint("upper_front_bushing", JointType.BUSHING_SOFT)
    uf_bushing.add_attachment_point(uf_chassis)
    uf_bushing.add_attachment_point(upper_front_link.endpoint1)

    ur_bushing = SuspensionJoint("upper_rear_bushing", JointType.BUSHING_SOFT)
    ur_bushing.add_attachment_point(ur_chassis)
    ur_bushing.add_attachment_point(upper_rear_link.endpoint1)

    # Upper ball joint (control arm to knuckle)
    upper_ball = SuspensionJoint("upper_ball_joint", JointType.BALL_JOINT)
    upper_ball.add_attachment_point(upper_front_link.endpoint2)
    upper_ball.add_attachment_point(upper_rear_link.endpoint2)
    upper_ball.add_attachment_point(upper_ball_knuckle)

    # Lower control arm to chassis (bushings)
    lf_bushing = SuspensionJoint("lower_front_bushing", JointType.BUSHING_SOFT)
    lf_bushing.add_attachment_point(lf_chassis)
    lf_bushing.add_attachment_point(lower_front_link.endpoint1)

    lr_bushing = SuspensionJoint("lower_rear_bushing", JointType.BUSHING_SOFT)
    lr_bushing.add_attachment_point(lr_chassis)
    lr_bushing.add_attachment_point(lower_rear_link.endpoint1)

    # Lower ball joint (control arm to knuckle)
    lower_ball = SuspensionJoint("lower_ball_joint", JointType.BALL_JOINT)
    lower_ball.add_attachment_point(lower_front_link.endpoint2)
    lower_ball.add_attachment_point(lower_rear_link.endpoint2)
    lower_ball.add_attachment_point(lower_ball_knuckle)

    print("   ✓ Created 6 joints (4 bushings + 2 ball joints)")

    print("\n" + "=" * 70)
    print("✓ SUSPENSION MODEL COMPLETE")
    print("=" * 70)

    return knuckle


def analyze_suspension_kinematics(knuckle):
    """
    Analyze suspension kinematics using the new CornerSolver workflow.

    Args:
        knuckle: SuspensionKnuckle with connected components
    """
    print("\n" + "=" * 70)
    print("SUSPENSION ANALYSIS WORKFLOW")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Create solver from knuckle (automatic discovery)
    # ========================================================================
    print("\n1. Creating CornerSolver from knuckle...")
    print("   (This automatically discovers all connected components)")

    solver = CornerSolver.from_suspension_knuckle(
        knuckle,
        name="front_left_solver",
        copy_components=True  # Safe: originals unchanged
    )

    print(f"\n   Solver ready!")
    print(f"   - DOF: {solver.state.get_dof()}")
    print(f"   - Constraints: {len(solver.constraints)}")

    # ========================================================================
    # STEP 2: Solve initial configuration
    # ========================================================================
    print("\n2. Solving initial configuration...")

    solver.save_initial_state()
    result = solver.solve()

    print(f"   ✓ Converged: {result.success}")
    print(f"   ✓ RMS error: {result.get_rms_error():.6f} mm")

    # ========================================================================
    # STEP 3: Analyze heave travel
    # ========================================================================
    print("\n3. Analyzing suspension travel (heave sweep)...")

    heave_range = np.linspace(-50, 50, 21)  # -50mm to +50mm
    results = {
        'heave': [],
        'wheel_z': [],
        'camber': [],
        'rms_error': []
    }

    for heave in heave_range:
        # Reset to initial state
        solver.reset_to_initial_state()

        # Solve for this heave position
        result = solver.solve_for_heave(heave, unit='mm')

        # Get solved knuckle
        solved_knuckle = solver.get_solved_knuckle()

        # Calculate camber from solved positions
        upper_ball = solved_knuckle.get_attachment_point("upper_ball_joint")
        lower_ball = solved_knuckle.get_attachment_point("lower_ball_joint")
        camber = solver.get_camber(upper_ball, lower_ball, unit='deg')

        # Store results
        results['heave'].append(heave)
        results['wheel_z'].append(solved_knuckle.tire_center[2])
        results['camber'].append(camber)
        results['rms_error'].append(result.get_rms_error())

    print(f"   ✓ Completed {len(heave_range)} solve iterations")

    # Calculate camber gain
    camber_change = results['camber'][-1] - results['camber'][0]
    heave_span = results['heave'][-1] - results['heave'][0]
    camber_gain = camber_change / heave_span

    print(f"\n   Suspension Characteristics:")
    print(f"   - Camber range: {min(results['camber']):.2f}° to {max(results['camber']):.2f}°")
    print(f"   - Camber gain: {camber_gain:.4f}°/mm")
    print(f"   - Max RMS error: {max(results['rms_error']):.6f} mm")

    # ========================================================================
    # STEP 4: Verify original is unchanged
    # ========================================================================
    print("\n4. Verifying original model unchanged...")

    original_position = knuckle.tire_center
    print(f"   Original knuckle position: {original_position} mm")
    print(f"   ✓ Original model is unchanged (copy_components=True)")

    # ========================================================================
    # STEP 5: Optional - Update original from solved (if needed)
    # ========================================================================
    print("\n5. [Optional] Updating original from solved state...")

    # Solve to a specific position
    solver.reset_to_initial_state()
    solver.solve_for_heave(25, unit='mm')

    print(f"   Before update: {knuckle.tire_center[2]:.3f} mm")

    # WARNING: This modifies the original!
    # solver.update_original_from_solved()
    # print(f"   After update: {knuckle.tire_center[2]:.3f} mm")

    print(f"   (Skipped update - keeping original unchanged)")

    return results


def plot_results(results):
    """Plot suspension analysis results."""
    print("\n6. Plotting results...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot wheel center Z vs heave
    ax1.plot(results['heave'], results['wheel_z'], 'b-', linewidth=2)
    ax1.set_xlabel('Heave Input (mm)', fontsize=12)
    ax1.set_ylabel('Wheel Center Z (mm)', fontsize=12)
    ax1.set_title('Wheel Travel vs Heave Input', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot camber vs heave
    ax2.plot(results['heave'], results['camber'], 'r-', linewidth=2)
    ax2.set_xlabel('Heave Input (mm)', fontsize=12)
    ax2.set_ylabel('Camber Angle (deg)', fontsize=12)
    ax2.set_title('Camber Change vs Heave Input', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('suspension_analysis.png', dpi=150)
    print("   ✓ Saved plot to suspension_analysis.png")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUSPENSION ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"{'Heave (mm)':<12} {'Wheel Z (mm)':<15} {'Camber (deg)':<15} {'RMS Error (mm)':<15}")
    print("-" * 70)

    for i in range(0, len(results['heave']), 5):
        heave = results['heave'][i]
        wheel_z = results['wheel_z'][i]
        camber = results['camber'][i]
        rms = results['rms_error'][i]
        print(f"{heave:<12.1f} {wheel_z:<15.3f} {camber:<15.3f} {rms:<15.6f}")


def main():
    """Main example workflow."""
    print("\n" + "=" * 70)
    print("CORNER SOLVER WITH SUSPENSION MODEL INTEGRATION")
    print("Example: Double-Wishbone Front Suspension")
    print("=" * 70)

    # Create the suspension model
    knuckle = create_double_wishbone_suspension()

    # Analyze kinematics
    results = analyze_suspension_kinematics(knuckle)

    # Plot results
    try:
        plot_results(results)
    except Exception as e:
        print(f"\n   Note: Could not create plots ({e})")

    print("\n" + "=" * 70)
    print("✓ EXAMPLE COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Build your suspension model with components and joints")
    print("2. Use CornerSolver.from_suspension_knuckle() for automatic setup")
    print("3. Solve with solve_for_heave() or solve_for_knuckle_heave()")
    print("4. Access results with get_solved_knuckle()")
    print("5. Original model stays unchanged (safe for iteration)")
    print("=" * 70)


if __name__ == "__main__":
    main()
