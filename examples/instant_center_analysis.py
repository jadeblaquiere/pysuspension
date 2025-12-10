"""
Example: Instant Center Analysis for Suspension Design

This example demonstrates how to calculate roll and pitch instant centers
for a suspension corner using the pysuspension library.

Instant centers are important geometric properties that describe the
instantaneous center of rotation for suspension motion. They affect:
- Roll resistance and handling characteristics
- Ride quality and comfort
- Anti-squat and anti-dive behavior

The analysis works by:
1. Creating a suspension knuckle with tire geometry
2. Simulating small vertical (heave) displacements
3. Capturing the tire contact patch trajectory
4. Fitting circles to the projected trajectories
5. Extracting instant center positions from circle centers
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pysuspension import CornerSolver, SuspensionKnuckle


def example_basic_instant_center():
    """
    Basic example: Calculate instant centers for a front suspension corner.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Instant Center Analysis")
    print("=" * 70)

    # Create a suspension knuckle for a front-left corner
    # Typical dimensions for a passenger car
    knuckle = SuspensionKnuckle(
        tire_center_x=1.5,      # 1.5m forward from origin (m)
        tire_center_y=0.75,     # 0.75m lateral (track width) (m)
        rolling_radius=0.35,    # 350mm tire radius (m)
        toe_angle=0.0,          # 0° toe (degrees)
        camber_angle=-1.0,      # -1° negative camber (degrees)
        unit='m',               # Input units
        name='front_left'
    )

    print(f"\nKnuckle Configuration:")
    print(f"  Tire center: {knuckle.tire_center} mm")
    print(f"  Rolling radius: {knuckle.tire_center[2]:.1f} mm")
    print(f"  Camber: {np.degrees(knuckle.camber_angle):.1f}°")
    print(f"  Toe: {np.degrees(knuckle.toe_angle):.1f}°")

    # Create a corner solver
    solver = CornerSolver(name="front_left_corner")

    # Calculate instant centers using default motion range (±10mm)
    print(f"\nCalculating instant centers...")
    print(f"  Motion range: ±10mm vertical travel")
    print(f"  Sample points: 5 positions [0, +5, +10, -5, -10] mm")

    result = solver.calculate_instant_centers(
        knuckle=knuckle,
        unit='mm'  # Use millimeters for z_offsets and output
    )

    # Display results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\nRoll Instant Center:")
    print(f"  Position: X={result['roll_center'][0]:.1f} mm, "
          f"Y={result['roll_center'][1]:.1f} mm, "
          f"Z={result['roll_center'][2]:.1f} mm")
    print(f"  Radius: {result['roll_radius']:.1f} mm")
    print(f"  Fit quality: {result['roll_fit_quality']:.6f}")
    print(f"  RMS residual: {result['roll_residuals']:.6f} mm")

    print(f"\nPitch Instant Center:")
    print(f"  Position: X={result['pitch_center'][0]:.1f} mm, "
          f"Y={result['pitch_center'][1]:.1f} mm, "
          f"Z={result['pitch_center'][2]:.1f} mm")
    print(f"  Radius: {result['pitch_radius']:.1f} mm")
    print(f"  Fit quality: {result['pitch_fit_quality']:.6f}")
    print(f"  RMS residual: {result['pitch_residuals']:.6f} mm")

    print(f"\nInterpretation:")
    print(f"  - Roll center height: {result['roll_center'][2]:.1f} mm above ground")
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

    # Create knuckle for rear suspension
    knuckle = SuspensionKnuckle(
        tire_center_x=1200,     # mm
        tire_center_y=800,      # mm (wider rear track)
        rolling_radius=360,     # mm (slightly larger rear tire)
        camber_angle=-2.0,      # degrees (more negative camber)
        unit='mm'
    )

    print(f"\nRear Suspension Configuration:")
    print(f"  Track width: {knuckle.tire_center[1] * 2:.0f} mm")
    print(f"  Tire diameter: {knuckle.tire_center[2] * 2:.0f} mm")

    solver = CornerSolver("rear_corner")

    # Use larger motion range for rear suspension analysis
    # Sample more points for better accuracy
    z_offsets = [-30, -20, -10, 0, 10, 20, 30]  # ±30mm range

    print(f"\nCustom Analysis Parameters:")
    print(f"  Motion range: ±30mm")
    print(f"  Sample points: {len(z_offsets)}")
    print(f"  Z offsets: {z_offsets} mm")

    result = solver.calculate_instant_centers(
        knuckle=knuckle,
        z_offsets=z_offsets,
        unit='mm'
    )

    print(f"\nResults:")
    print(f"  Roll center: Z = {result['roll_center'][2]:.1f} mm")
    print(f"  Pitch center: Z = {result['pitch_center'][2]:.1f} mm")
    print(f"  Roll radius: {result['roll_radius']:.1f} mm")
    print(f"  Pitch radius: {result['pitch_radius']:.1f} mm")

    print(f"\n✓ Custom motion range analysis complete!\n")

    return result


def example_compare_configurations():
    """
    Compare instant centers for different suspension configurations.
    """
    print("=" * 70)
    print("EXAMPLE 3: Comparing Different Configurations")
    print("=" * 70)

    configurations = [
        {
            'name': 'Narrow Track',
            'track_width': 1400,  # mm
            'camber': 0.0
        },
        {
            'name': 'Wide Track',
            'track_width': 1600,  # mm
            'camber': 0.0
        },
        {
            'name': 'Negative Camber',
            'track_width': 1500,  # mm
            'camber': -2.0
        }
    ]

    results = []
    solver = CornerSolver("comparison_solver")

    print("\nAnalyzing configurations...\n")

    for config in configurations:
        print(f"Configuration: {config['name']}")
        print(f"  Track width: {config['track_width']} mm")
        print(f"  Camber: {config['camber']}°")

        knuckle = SuspensionKnuckle(
            tire_center_x=1500,
            tire_center_y=config['track_width'] / 2,
            rolling_radius=350,
            camber_angle=config['camber'],
            unit='mm'
        )

        result = solver.calculate_instant_centers(knuckle, unit='mm')
        results.append(result)

        print(f"  → Roll center height: {result['roll_center'][2]:.1f} mm")
        print(f"  → Roll radius: {result['roll_radius']:.1f} mm")
        print()

    # Summary comparison
    print(f"{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Configuration':<20} {'Roll Center Z (mm)':<20} {'Roll Radius (mm)':<20}")
    print("-" * 60)

    for config, result in zip(configurations, results):
        print(f"{config['name']:<20} {result['roll_center'][2]:>18.1f}  "
              f"{result['roll_radius']:>18.1f}")

    print(f"\n✓ Configuration comparison complete!\n")

    return results


def example_using_meters():
    """
    Example using metric units (meters) throughout.
    """
    print("=" * 70)
    print("EXAMPLE 4: Working in Meters")
    print("=" * 70)

    # Define everything in meters
    knuckle = SuspensionKnuckle(
        tire_center_x=1.5,
        tire_center_y=0.75,
        rolling_radius=0.35,
        unit='m'
    )

    solver = CornerSolver("metric_solver")

    # Use meter-based offsets
    z_offsets_m = [0, 0.005, 0.010, -0.005, -0.010]  # ±10mm in meters

    print(f"\nWorking in meters:")
    print(f"  Tire center: [{knuckle.tire_center[0]/1000:.3f}, "
          f"{knuckle.tire_center[1]/1000:.3f}, "
          f"{knuckle.tire_center[2]/1000:.3f}] m")
    print(f"  Z offsets: {z_offsets_m} m")

    result = solver.calculate_instant_centers(
        knuckle=knuckle,
        z_offsets=z_offsets_m,
        unit='m'  # Output in meters
    )

    print(f"\nResults (in meters):")
    print(f"  Roll center: [{result['roll_center'][0]:.4f}, "
          f"{result['roll_center'][1]:.4f}, "
          f"{result['roll_center'][2]:.4f}] m")
    print(f"  Roll radius: {result['roll_radius']:.4f} m")
    print(f"  Pitch center: [{result['pitch_center'][0]:.4f}, "
          f"{result['pitch_center'][1]:.4f}, "
          f"{result['pitch_center'][2]:.4f}] m")
    print(f"  Pitch radius: {result['pitch_radius']:.4f} m")

    print(f"\n✓ Metric analysis complete!\n")

    return result


def example_fine_grained_analysis():
    """
    High-resolution analysis with many sample points.
    """
    print("=" * 70)
    print("EXAMPLE 5: High-Resolution Analysis")
    print("=" * 70)

    knuckle = SuspensionKnuckle(
        tire_center_x=1500,
        tire_center_y=750,
        rolling_radius=350,
        unit='mm'
    )

    solver = CornerSolver("high_res_solver")

    # Use many sample points for high accuracy
    z_offsets = np.linspace(-20, 20, 21).tolist()  # 21 points over ±20mm

    print(f"\nHigh-resolution configuration:")
    print(f"  Sample points: {len(z_offsets)}")
    print(f"  Motion range: {z_offsets[0]:.1f} to {z_offsets[-1]:.1f} mm")
    print(f"  Point spacing: {z_offsets[1] - z_offsets[0]:.1f} mm")

    result = solver.calculate_instant_centers(
        knuckle=knuckle,
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
    print(f"  Roll fit quality: {result['roll_fit_quality']:.10f}")
    print(f"  Pitch fit quality: {result['pitch_fit_quality']:.10f}")
    print(f"  (Lower values indicate better fits)")

    print(f"\n  With {len(z_offsets)} points, the circle fit is very accurate!")
    print(f"  Roll residual: {result['roll_residuals']:.6f} mm")
    print(f"  Pitch residual: {result['pitch_residuals']:.6f} mm")

    print(f"\n✓ High-resolution analysis complete!\n")

    return result


def run_all_examples():
    """
    Run all instant center analysis examples.
    """
    print("\n" + "=" * 70)
    print("INSTANT CENTER ANALYSIS EXAMPLES")
    print("Demonstrating suspension geometry analysis with pysuspension")
    print("=" * 70 + "\n")

    # Run examples
    example_basic_instant_center()
    example_custom_motion_range()
    example_compare_configurations()
    example_using_meters()
    example_fine_grained_analysis()

    # Final summary
    print("=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Instant centers describe the center of rotation for suspension motion")
    print("  • Roll center affects body roll and lateral load transfer")
    print("  • Pitch center affects brake dive and acceleration squat")
    print("  • Lower fit quality values indicate more accurate circle fits")
    print("  • The method works with any unit system (mm, m, inches, etc.)")
    print("\nNext Steps:")
    print("  • Integrate with full suspension linkage models")
    print("  • Analyze instant center migration through full travel")
    print("  • Compare front and rear suspension instant centers")
    print("  • Optimize instant center locations for specific handling goals")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_examples()
