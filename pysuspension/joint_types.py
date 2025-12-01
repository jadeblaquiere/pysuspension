"""
Joint type definitions for suspension components.

This module defines standard joint types with typical stiffness characteristics
used throughout the suspension system for compliance modeling.
"""

from enum import Enum


class JointType(Enum):
    """
    Predefined joint types with typical stiffness characteristics.

    Stiffness values represent translational stiffness in N/mm.
    These are typical values and can be overridden with custom stiffness.
    """
    RIGID = "rigid"              # Welded, bolted - effectively infinite stiffness
    BALL_JOINT = "ball_joint"    # Spherical bearing - very stiff
    SPHERICAL_BEARING = "spherical_bearing"  # Similar to ball joint
    BUSHING_HARD = "bushing_hard"    # Polyurethane bushing - moderate compliance
    BUSHING_SOFT = "bushing_soft"    # Rubber bushing - high compliance
    RUBBER_MOUNT = "rubber_mount"    # Engine mount style - very compliant
    CUSTOM = "custom"            # User-defined compliance


# Typical joint stiffness values (N/mm)
# These represent translational stiffness of the joint
JOINT_STIFFNESS = {
    JointType.RIGID: 1e6,           # Essentially infinite (1,000,000 N/mm)
    JointType.BALL_JOINT: 1e5,      # Very stiff (100,000 N/mm)
    JointType.SPHERICAL_BEARING: 1e5,
    JointType.BUSHING_HARD: 1e3,    # Moderate (1,000 N/mm)
    JointType.BUSHING_SOFT: 100,    # Compliant (100 N/mm)
    JointType.RUBBER_MOUNT: 10,     # Very compliant (10 N/mm)
}
