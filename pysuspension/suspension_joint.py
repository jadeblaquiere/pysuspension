"""
Suspension joint class for modeling connections between attachment points.

A SuspensionJoint represents a physical joint (ball joint, bushing, etc.) that
connects multiple attachment points together. The joint has a type that defines
its compliance/elasticity characteristics.
"""

from typing import List, Optional, TYPE_CHECKING
from .joint_types import JointType

if TYPE_CHECKING:
    from .attachment_point import AttachmentPoint


class SuspensionJoint:
    """
    Represents a joint connecting multiple attachment points.

    A suspension joint models a physical connection between attachment points
    with a specific type (ball joint, bushing, etc.) that defines its compliance
    characteristics. This ensures a single source of truth for the joint type
    shared by all connected attachment points.

    Attributes:
        name: Identifier for the joint
        joint_type: Type of joint (defines stiffness/compliance)
        attachment_points: List of all attachment points connected by this joint
    """

    def __init__(self, name: str, joint_type: JointType = JointType.BALL_JOINT):
        """
        Initialize a suspension joint.

        Args:
            name: Identifier for the joint
            joint_type: Type of joint (default: JointType.BALL_JOINT)
        """
        self.name = name
        self.joint_type = joint_type
        self.attachment_points: List['AttachmentPoint'] = []

    def add_attachment_point(self, point: 'AttachmentPoint') -> None:
        """
        Add an attachment point to this joint.

        Args:
            point: AttachmentPoint to connect to this joint
        """
        if point not in self.attachment_points:
            self.attachment_points.append(point)
            # Set the back-reference on the attachment point
            if point.joint is not self:
                point.joint = self

    def remove_attachment_point(self, point: 'AttachmentPoint') -> None:
        """
        Remove an attachment point from this joint.

        Args:
            point: AttachmentPoint to disconnect from this joint
        """
        if point in self.attachment_points:
            self.attachment_points.remove(point)
            # Clear the back-reference on the attachment point
            if point.joint is self:
                point.joint = None

    def get_connected_points(self, point: 'AttachmentPoint') -> List['AttachmentPoint']:
        """
        Get all attachment points connected to the given point through this joint.

        Args:
            point: The attachment point to query connections for

        Returns:
            List of attachment points connected through this joint (excluding the query point)
        """
        return [p for p in self.attachment_points if p != point]

    def get_all_attachment_points(self) -> List['AttachmentPoint']:
        """
        Get all attachment points connected by this joint.

        Returns:
            List of all attachment points in this joint
        """
        return self.attachment_points.copy()

    def clear(self) -> None:
        """
        Remove all attachment points from this joint.
        """
        # Clear back-references
        for point in self.attachment_points:
            if point.joint is self:
                point.joint = None
        self.attachment_points.clear()

    def to_dict(self) -> dict:
        """
        Serialize the suspension joint to a dictionary.

        Note: AttachmentPoint references are not serialized to avoid circular references.
        The joint connections should be reconstructed when deserializing the parent component.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'name': self.name,
            'joint_type': self.joint_type.value,  # Serialize enum as string
            'attachment_point_names': [p.name for p in self.attachment_points]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SuspensionJoint':
        """
        Deserialize a suspension joint from a dictionary.

        Note: This creates the joint but does not reconnect attachment points.
        The parent component should handle reconnecting attachment points by name.

        Args:
            data: Dictionary containing suspension joint data

        Returns:
            New SuspensionJoint instance (without attachment points connected)

        Raises:
            KeyError: If required fields are missing
            ValueError: If data is invalid
        """
        joint_type = JointType(data['joint_type'])
        joint = cls(
            name=data['name'],
            joint_type=joint_type
        )
        # Note: attachment_point_names are stored for reference but not reconnected here
        # The parent component should handle reconnection by looking up points by name
        return joint

    def __repr__(self) -> str:
        return (f"SuspensionJoint('{self.name}', "
                f"type={self.joint_type}, "
                f"points={len(self.attachment_points)})")


if __name__ == "__main__":
    print("=" * 60)
    print("SUSPENSION JOINT TEST")
    print("=" * 60)

    # This test requires AttachmentPoint, so we'll keep it simple
    joint = SuspensionJoint("test_joint", JointType.BALL_JOINT)
    print(f"\nCreated joint: {joint}")
    print(f"Joint type: {joint.joint_type}")
    print(f"Connected points: {len(joint.attachment_points)}")

    # Test serialization
    data = joint.to_dict()
    print(f"\nSerialized: {data}")

    # Test deserialization
    joint2 = SuspensionJoint.from_dict(data)
    print(f"Deserialized: {joint2}")
    print(f"Joint types match: {joint.joint_type == joint2.joint_type}")

    print("\n✓ Basic tests completed successfully!")
