"""
Geometric utility functions for suspension analysis.

This module provides geometric calculation utilities including:
- Circle fitting using algebraic least-squares method
- Point projection onto coordinate planes
- Validation and error checking for geometric operations
"""

import numpy as np
from typing import Tuple, Dict


def project_points_to_plane(points: np.ndarray, plane: str) -> np.ndarray:
    """
    Project 3D points onto a specified coordinate plane.

    Args:
        points: Nx3 array of [x, y, z] coordinates
        plane: Plane identifier - 'YZ' or 'XZ'
            - 'YZ': Projects onto Y-Z plane (extracts y, z coordinates)
            - 'XZ': Projects onto X-Z plane (extracts x, z coordinates)

    Returns:
        Nx2 array of projected 2D coordinates

    Raises:
        ValueError: If plane is not 'YZ' or 'XZ', or if points shape is invalid

    Examples:
        >>> points_3d = np.array([[1, 2, 3], [4, 5, 6]])
        >>> project_points_to_plane(points_3d, 'YZ')
        array([[2, 3],
               [5, 6]])
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be Nx3 array, got shape {points.shape}")

    if points.shape[0] < 3:
        raise ValueError(f"Need at least 3 points for projection, got {points.shape[0]}")

    plane = plane.upper()

    if plane == 'YZ':
        # Extract Y and Z coordinates (columns 1 and 2)
        return points[:, [1, 2]]
    elif plane == 'XZ':
        # Extract X and Z coordinates (columns 0 and 2)
        return points[:, [0, 2]]
    else:
        raise ValueError(f"Plane must be 'YZ' or 'XZ', got '{plane}'")


def fit_circle_2d(points: np.ndarray) -> Dict[str, any]:
    """
    Fit a circle to 2D points using algebraic least-squares method.

    The circle equation (u - a)² + (v - b)² = r² is linearized to:
    u² + v² = 2au + 2bv + c, where c = r² - a² - b²

    This is solved as a linear system: A·x = b
    where x = [a, b, c]ᵀ contains the circle parameters.

    Args:
        points: Nx2 array of 2D coordinates [u, v]

    Returns:
        Dictionary containing:
            - 'center': [a, b] - Circle center coordinates
            - 'radius': r - Circle radius
            - 'residuals': RMS residual error of the fit
            - 'fit_quality': Normalized residual (residual / radius)

    Raises:
        ValueError: If insufficient points provided or points are collinear

    References:
        Coope, I. D. (1993). "Circle fitting by linear and nonlinear least squares."
        Journal of Optimization Theory and Applications, 76(2), 381-388.

    Examples:
        >>> # Perfect circle
        >>> theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
        >>> points = np.column_stack([np.cos(theta), np.sin(theta)])
        >>> result = fit_circle_2d(points)
        >>> np.allclose(result['center'], [0, 0])
        True
        >>> np.allclose(result['radius'], 1.0)
        True
    """
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Points must be Nx2 array, got shape {points.shape}")

    if points.shape[0] < 3:
        raise ValueError(f"Need at least 3 points for circle fitting, got {points.shape[0]}")

    # Extract u and v coordinates
    u = points[:, 0]
    v = points[:, 1]

    # Check for collinearity - compute variance in perpendicular direction
    # If points are collinear, they won't define a unique circle
    if points.shape[0] == 3:
        # For exactly 3 points, check if they're collinear using cross product
        vec1 = points[1] - points[0]
        vec2 = points[2] - points[0]
        cross_prod = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        if abs(cross_prod) < 1e-10:
            raise ValueError("Points are collinear, cannot fit unique circle")

    # Construct the matrix equation: A·x = b
    # A = [2u₁, 2v₁, 1]
    #     [2u₂, 2v₂, 1]
    #     [  ...      ]
    A = np.column_stack([2 * u, 2 * v, np.ones_like(u)])

    # b = [u₁² + v₁²]
    #     [u₂² + v₂²]
    #     [   ...    ]
    b = u**2 + v**2

    # Solve the least-squares problem: minimize ||A·x - b||²
    # Returns x = [a, b, c]ᵀ
    result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Extract circle parameters
    a, b_coef, c = result
    center = np.array([a, b_coef])

    # Calculate radius: r = √(a² + b² + c)
    # From: c = r² - a² - b² => r² = a² + b² + c
    radius = np.sqrt(a**2 + b_coef**2 + c)

    # Calculate RMS residual error
    # Residual = distance from point to circle edge
    distances = np.sqrt((u - a)**2 + (v - b_coef)**2)
    point_residuals = np.abs(distances - radius)
    rms_residual = np.sqrt(np.mean(point_residuals**2))

    # Normalized fit quality (dimensionless)
    fit_quality = rms_residual / radius if radius > 0 else np.inf

    return {
        'center': center,
        'radius': radius,
        'residuals': rms_residual,
        'fit_quality': fit_quality
    }


def calculate_instant_center_from_points(
    contact_points: np.ndarray,
    plane: str
) -> Dict[str, any]:
    """
    Calculate instant center by projecting 3D contact points onto a plane and fitting a circle.

    This is a convenience function that combines projection and circle fitting.

    Args:
        contact_points: Nx3 array of 3D contact point positions
        plane: 'YZ' for roll center or 'XZ' for pitch center

    Returns:
        Dictionary containing:
            - 'center_2d': [u, v] - Center in projected 2D plane
            - 'center_3d': [x, y, z] - Center back-projected to 3D
            - 'radius': Circle radius
            - 'residuals': RMS fit error
            - 'fit_quality': Normalized fit quality
            - 'projected_points': Nx2 array of projected points used for fitting

    Examples:
        >>> # Simulate suspension motion creating circular arc
        >>> points = np.array([[1.5, 0.7, 0], [1.5, 0.71, 5], [1.5, 0.73, 10]])
        >>> result = calculate_instant_center_from_points(points, 'YZ')
        >>> 'center_3d' in result
        True
    """
    # Project points onto the specified plane
    projected_points = project_points_to_plane(contact_points, plane)

    # Fit circle to projected points
    fit_result = fit_circle_2d(projected_points)

    # Back-project center to 3D space
    center_2d = fit_result['center']
    plane = plane.upper()

    if plane == 'YZ':
        # Center is in YZ plane, x coordinate is average of contact points
        x_avg = np.mean(contact_points[:, 0])
        center_3d = np.array([x_avg, center_2d[0], center_2d[1]])
    elif plane == 'XZ':
        # Center is in XZ plane, y coordinate is average of contact points
        y_avg = np.mean(contact_points[:, 1])
        center_3d = np.array([center_2d[0], y_avg, center_2d[1]])
    else:
        raise ValueError(f"Plane must be 'YZ' or 'XZ', got '{plane}'")

    return {
        'center_2d': center_2d,
        'center_3d': center_3d,
        'radius': fit_result['radius'],
        'residuals': fit_result['residuals'],
        'fit_quality': fit_result['fit_quality'],
        'projected_points': projected_points
    }
