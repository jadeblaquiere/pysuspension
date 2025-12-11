"""
Example: Instant Center Analysis for Suspension Design

This example demonstrates how to calculate roll and pitch instant centers
for a suspension corner using the pysuspension library with constraint-based
kinematics.

Instant centers are important geometric properties that describe the
instantaneous center of rotation for suspension motion. They affect:
- Roll resistance and handling characteristics
- Ride quality and comfort
- Anti-squat and anti-dive behavior

The analysis works by:
1. Setting up suspension linkage (control arms, links, wheel center)
2. Using constraint-based solver to simulate heave motion
3. Capturing the wheel center positions through the suspension travel
4. Projecting to ground level to get tire contact patches
5. Fitting circles to the projected trajectories
6. Extracting instant center positions from circle centers
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension import CornerSolver, SuspensionLink, AttachmentPoint, SuspensionKnuckle


def create_double_wishbone_suspension(name="front_corner", track_width=1500):
    """
    Create a double wishbone suspension for analysis.

    Args:
        name: Name for the corner solver
        track_width: Lateral distance from centerline to wheel center (mm)

    Returns:
        Configured CornerSolver ready for analysis
    """
    solver = CornerSolver(name)

    # Upper control arm links (shorter, at top)
    upper_front_link = SuspensionLink(
        endpoint1=[1400, 0, 600],           # Chassis mount (front)
        endpoint2=[1400, track_width-100, 580],  # Ball joint
        name="upper_front",
        unit='mm'
    )
    upper_rear_link = SuspensionLink(
        endpoint1=[1200, 0, 600],           # Chassis mount (rear)
        endpoint2=[1400, track_width-100, 580],  # Ball joint (shared)
        name="upper_rear",
        unit='mm'
    )

    # Lower control arm links (longer, at bottom)
    lower_front_link = SuspensionLink(
        endpoint1=[1500, 0, 300],           # Chassis mount (front)
        endpoint2=[1400, track_width-50, 200],   # Ball joint
        name="lower_front",
        unit='mm'
    )
    lower_rear_link = SuspensionLink(
        endpoint1=[1100, 0, 300],           # Chassis mount (rear)
        endpoint2=[1400, track_width-50, 200],   # Ball joint (shared)
        name="lower_rear",
        unit='mm'
    )

    # Wheel center (on knuckle, between ball joints)
    wheel_center = AttachmentPoint("wheel_center", [1400, track_width, 390], unit='mm')

    # Mark chassis mount points
    solver.chassis_mounts.extend([
        upper_front_link.endpoint1,
        upper_rear_link.endpoint1,
        lower_front_link.endpoint1,
        lower_rear_link.endpoint1
    ])

    # Add control arm links to solver
    solver.add_link(upper_front_link, end1_mount_point=upper_front_link.endpoint1)
    solver.add_link(upper_rear_link, end1_mount_point=upper_rear_link.endpoint1)
    solver.add_link(lower_front_link, end1_mount_point=lower_front_link.endpoint1)
    solver.add_link(lower_rear_link, end1_mount_point=lower_rear_link.endpoint1)

    # Connect wheel center to ball joints (simulates knuckle)
    upper_ball_joint = upper_front_link.endpoint2
    lower_ball_joint = lower_front_link.endpoint2

    # Create knuckle links
    upper_to_wheel = SuspensionLink(
        upper_ball_joint,
        wheel_center,
        name="upper_knuckle_link",
        unit='mm'
    )
    lower_to_wheel = SuspensionLink(
        lower_ball_joint,
        wheel_center,
        name="lower_knuckle_link",
        unit='mm'
    )

    solver.add_link(upper_to_wheel)
    solver.add_link(lower_to_wheel)

    # Set wheel center for heave analysis
    solver.set_wheel_center(wheel_center)

    # Solve initial configuration
    solver.save_initial_state()
    result = solver.solve()

    if not result.success:
        print(f"Warning: Initial solve failed: {result.message}")

    return solver


def example_basic_instant_center():
    """
    Basic example: Calculate instant centers for a double wishbone suspension.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Instant Center Analysis")
    print("=" * 70)

    # Create a double wishbone suspension
    solver = create_double_wishbone_suspension("front_left")

    # Create suspension knuckle with tire geometry
    knuckle = SuspensionKnuckle(
        tire_center_x=1400,
        tire_center_y=1500,
        rolling_radius=390,
        camber_angle=-1.0,  # -1° negative camber
        unit='mm'
    )

    print(f"\nSuspension Configuration:")
    print(f"  Type: Double wishbone")
    print(f"  Wheel center: {solver.wheel_center.position} mm")
    print(f"  Number of links: {len(solver.links)}")
    print(f"  Number of constraints: {len(solver.constraints)}")
    print(f"  Tire camber: {np.degrees(knuckle.camber_angle):.1f}°")

    # Calculate instant centers using default motion range (±10mm)
    print(f"\nCalculating instant centers...")
    print(f"  Motion range: ±10mm vertical travel")
    print(f"  Sample points: 5 positions [0, +5, +10, -5, -10] mm")

    result = solver.calculate_instant_centers(knuckle, unit='mm')

    # Display results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\nRoll Instant Center:")
    print(f"  Position: X={result['roll_center'][0]:.1f} mm, "
          f"Y={result['roll_center'][1]:.1f} mm, "
          f"Z={result['roll_center'][2]:.1f} mm")
    print(f"  Arc radius: {result['roll_radius']:.2f} mm")
    print(f"  Fit quality: {result['roll_fit_quality']:.6f}")

    print(f"\nPitch Instant Center:")
    print(f"  Position: X={result['pitch_center'][0]:.1f} mm, "
          f"Y={result['pitch_center'][1]:.1f} mm, "
          f"Z={result['pitch_center'][2]:.1f} mm")
    print(f"  Arc radius: {result['pitch_radius']:.2f} mm")
    print(f"  Fit quality: {result['pitch_fit_quality']:.6f}")

    print(f"\nSolver Quality:")
    print(f"  Max solve error: {max(result['solve_errors']):.6f} mm")
    print(f"  All solves converged: {all(e < 0.01 for e in result['solve_errors'])}")

    print(f"\nInterpretation:")
    print(f"  - Roll center height: {result['roll_center'][2]:.1f} mm above ground")
    print(f"  - Small arc radius indicates nearly circular motion")
    print(f"  - Lower roll center = more body roll, better grip")
    print(f"  - Higher roll center = less body roll, more responsive")

    print(f"\n✓ Basic instant center analysis complete!\n")

    return result


def example_custom_motion_range():
    """
    Example with custom motion range and more sample points.
    """
    print("=" * 70)
    print("EXAMPLE 2: Custom Motion Range")
    print("=" * 70)

    # Create suspension
    solver = create_double_wishbone_suspension("rear_corner", track_width=1600)

    # Create knuckle for rear suspension
    knuckle = SuspensionKnuckle(
        tire_center_x=1400,
        tire_center_y=1600,
        rolling_radius=390,
        camber_angle=-2.0,  # More camber in rear
        unit='mm'
    )

    print(f"\nRear Suspension Configuration:")
    print(f"  Wider track width: {solver.wheel_center.position[1] * 2:.0f} mm total")

    # Use larger motion range
    z_offsets = [-30, -20, -10, 0, 10, 20, 30]

    print(f"\nCustom Analysis Parameters:")
    print(f"  Motion range: ±30mm")
    print(f"  Sample points: {len(z_offsets)}")
    print(f"  Z offsets: {z_offsets} mm")

    result = solver.calculate_instant_centers(
        knuckle,
        z_offsets=z_offsets,
        unit='mm'
    )

    print(f"\nResults:")
    print(f"  Roll center: Z = {result['roll_center'][2]:.1f} mm")
    print(f"  Pitch center: Z = {result['pitch_center'][2]:.1f} mm")
    print(f"  Roll arc radius: {result['roll_radius']:.2f} mm")
    print(f"  Pitch arc radius: {result['pitch_radius']:.2f} mm")

    print(f"\n✓ Custom motion range analysis complete!\n")

    return result


def example_compare_track_widths():
    """
    Compare instant centers for different track widths.
    """
    print("=" * 70)
    print("EXAMPLE 3: Comparing Different Track Widths")
    print("=" * 70)

    track_widths = [
        ("Narrow Track", 1400),
        ("Standard Track", 1500),
        ("Wide Track", 1600)
    ]

    results = []

    print("\nAnalyzing configurations...\n")

    for name, track_width in track_widths:
        print(f"Configuration: {name}")
        print(f"  Track width: {track_width * 2} mm total")

        solver = create_double_wishbone_suspension("comparison", track_width)

        # Create knuckle matching the track width
        knuckle = SuspensionKnuckle(
            tire_center_x=1400,
            tire_center_y=track_width,
            rolling_radius=390,
            camber_angle=-1.0,
            unit='mm'
        )

        result = solver.calculate_instant_centers(knuckle, unit='mm')
        results.append(result)

        print(f"  → Roll center: Y={result['roll_center'][1]:.1f} mm, "
              f"Z={result['roll_center'][2]:.1f} mm")
        print(f"  → Arc radius: {result['roll_radius']:.2f} mm")
        print()

    # Summary comparison
    print(f"{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Configuration':<20} {'Roll Center Y (mm)':<20} {'Arc Radius (mm)':<20}")
    print("-" * 60)

    for (name, _), result in zip(track_widths, results):
        print(f"{name:<20} {result['roll_center'][1]:>18.1f}  "
              f"{result['roll_radius']:>18.2f}")

    print(f"\nObservation: Wider track generally affects roll center lateral position")
    print(f"✓ Track width comparison complete!\n")

    return results


def example_high_resolution_analysis():
    """
    High-resolution analysis with many sample points.
    """
    print("=" * 70)
    print("EXAMPLE 4: High-Resolution Analysis")
    print("=" * 70)

    solver = create_double_wishbone_suspension("high_res")

    # Create knuckle
    knuckle = SuspensionKnuckle(
        tire_center_x=1400,
        tire_center_y=1500,
        rolling_radius=390,
        camber_angle=-1.5,
        unit='mm'
    )

    # Use many sample points for high accuracy
    z_offsets = np.linspace(-20, 20, 21).tolist()

    print(f"\nHigh-resolution configuration:")
    print(f"  Sample points: {len(z_offsets)}")
    print(f"  Motion range: {z_offsets[0]:.1f} to {z_offsets[-1]:.1f} mm")
    print(f"  Point spacing: {z_offsets[1] - z_offsets[0]:.1f} mm")

    result = solver.calculate_instant_centers(
        knuckle,
        z_offsets=z_offsets,
        unit='mm'
    )

    print(f"\nResults:")
    print(f"  Roll center: [{result['roll_center'][0]:.2f}, "
          f"{result['roll_center'][1]:.2f}, "
          f"{result['roll_center'][2]:.2f}] mm")
    print(f"  Pitch center: [{result['pitch_center'][0]:.2f}, "
          f"{result['pitch_center'][1]:.2f}, "
          f"{result['pitch_center'][2]:.2f}] mm")

    print(f"\nFit Quality Metrics:")
    print(f"  Roll fit quality: {result['roll_fit_quality']:.8f}")
    print(f"  Pitch fit quality: {result['pitch_fit_quality']:.8f}")
    print(f"  (Lower values indicate better fits)")

    print(f"\nSolver Performance:")
    print(f"  Average solve error: {np.mean(result['solve_errors']):.6f} mm")
    print(f"  Max solve error: {max(result['solve_errors']):.6f} mm")
    print(f"  Min solve error: {min(result['solve_errors']):.6f} mm")

    print(f"\n  With {len(z_offsets)} points, the circle fit is very accurate!")

    print(f"\n✓ High-resolution analysis complete!\n")

    return result


def example_different_units():
    """
    Example using metric units (meters).
    """
    print("=" * 70)
    print("EXAMPLE 5: Working with Different Units")
    print("=" * 70)

    solver = create_double_wishbone_suspension("metric_test")

    # Create knuckle
    knuckle = SuspensionKnuckle(
        tire_center_x=1400,
        tire_center_y=1500,
        rolling_radius=390,
        camber_angle=-1.0,
        unit='mm'
    )

    # Use meter-based offsets
    z_offsets_m = [0, 0.005, 0.010, -0.005, -0.010]

    print(f"\nUsing meters for input and output:")
    print(f"  Z offsets: {z_offsets_m} m")

    result = solver.calculate_instant_centers(
        knuckle,
        z_offsets=z_offsets_m,
        unit='m'  # Output in meters
    )

    print(f"\nResults (in meters):")
    print(f"  Roll center: [{result['roll_center'][0]:.4f}, "
          f"{result['roll_center'][1]:.4f}, "
          f"{result['roll_center'][2]:.4f}] m")
    print(f"  Arc radius: {result['roll_radius']:.6f} m")
    print(f"  Pitch center: [{result['pitch_center'][0]:.4f}, "
          f"{result['pitch_center'][1]:.4f}, "
          f"{result['pitch_center'][2]:.4f}] m")
    print(f"  Arc radius: {result['pitch_radius']:.6f} m")

    # Convert to mm for comparison
    print(f"\nSame results (converted to mm):")
    print(f"  Roll center Z: {result['roll_center'][2] * 1000:.1f} mm")
    print(f"  Arc radius: {result['roll_radius'] * 1000:.2f} mm")

    print(f"\n✓ Units test complete!\n")

    return result


def run_all_examples():
    """
    Run all instant center analysis examples.
    """
    print("\n" + "=" * 70)
    print("INSTANT CENTER ANALYSIS EXAMPLES")
    print("Demonstrating suspension kinematics analysis with pysuspension")
    print("=" * 70 + "\n")

    # Run examples
    example_basic_instant_center()
    example_custom_motion_range()
    example_compare_track_widths()
    example_high_resolution_analysis()
    example_different_units()

    # Final summary
    print("=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Instant centers describe the center of rotation for suspension motion")
    print("  • Roll center affects body roll and lateral load transfer")
    print("  • Pitch center affects brake dive and acceleration squat")
    print("  • The analysis uses constraint-based kinematics for accurate results")
    print("  • Smaller arc radii indicate the suspension traces a nearly circular path")
    print("  • Lower fit quality values indicate more accurate circle fits")
    print("\nNext Steps:")
    print("  • Analyze instant center migration through full suspension travel")
    print("  • Compare front and rear suspension instant centers")
    print("  • Optimize instant center locations for specific handling goals")
    print("  • Study the effect of suspension geometry changes on instant centers")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_examples()
