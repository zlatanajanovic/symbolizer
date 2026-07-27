"""
Motion Planning Interface Module.

This module provides abstract interfaces for motion planning that can be
implemented by different backends (PyBullet, MoveIt, etc.).

Part of the Symbolizer project for VLM-guided symbolic planning.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class MotionPlanStatus(Enum):
    """Status of a motion plan."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    COLLISION = "collision"
    IK_FAILED = "ik_failed"
    UNREACHABLE = "unreachable"


@dataclass
class MotionPlan:
    """Result of motion planning.
    
    Attributes:
        waypoints: List of joint configurations along the path
        durations: Time duration for each segment
        success: Whether planning succeeded
        status: Detailed status of the plan
        error_message: Error description if planning failed
        metadata: Additional planning metadata
    """
    waypoints: List[np.ndarray] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)
    success: bool = False
    status: MotionPlanStatus = MotionPlanStatus.FAILURE
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)
    
    @property
    def total_duration(self) -> float:
        return sum(self.durations)
    
    @property
    def path_length(self) -> float:
        """Compute path length in configuration space."""
        if len(self.waypoints) < 2:
            return 0.0
        length = 0.0
        for i in range(len(self.waypoints) - 1):
            length += np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
        return length
    
    def interpolate(self, num_points: int) -> List[np.ndarray]:
        """Interpolate waypoints to get more points."""
        if len(self.waypoints) < 2:
            return self.waypoints
        
        interpolated = []
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i]
            end = self.waypoints[i + 1]
            for t in np.linspace(0, 1, num_points // (len(self.waypoints) - 1)):
                interpolated.append(start + t * (end - start))
        interpolated.append(self.waypoints[-1])
        return interpolated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "waypoints": [wp.tolist() for wp in self.waypoints],
            "durations": self.durations,
            "success": self.success,
            "status": self.status.value,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "num_waypoints": self.num_waypoints,
            "total_duration": self.total_duration,
            "path_length": self.path_length
        }


@dataclass
class Pose:
    """6D pose representation (position + orientation).
    
    Attributes:
        position: 3D position (x, y, z)
        orientation: Quaternion (x, y, z, w) or Euler angles (roll, pitch, yaw)
        is_quaternion: Whether orientation is quaternion or Euler
    """
    position: np.ndarray
    orientation: np.ndarray
    is_quaternion: bool = True
    
    def __post_init__(self):
        self.position = np.array(self.position)
        self.orientation = np.array(self.orientation)
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix."""
        from scipy.spatial.transform import Rotation
        
        mat = np.eye(4)
        mat[:3, 3] = self.position
        
        if self.is_quaternion:
            # scipy uses scalar-last (x, y, z, w)
            mat[:3, :3] = Rotation.from_quat(self.orientation).as_matrix()
        else:
            mat[:3, :3] = Rotation.from_euler('xyz', self.orientation).as_matrix()
        
        return mat
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "Pose":
        """Create Pose from 4x4 transformation matrix."""
        from scipy.spatial.transform import Rotation
        
        position = matrix[:3, 3]
        orientation = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return cls(position=position, orientation=orientation, is_quaternion=True)
    
    def to_euler(self) -> "Pose":
        """Convert to Euler angle representation."""
        if not self.is_quaternion:
            return self
        
        from scipy.spatial.transform import Rotation
        euler = Rotation.from_quat(self.orientation).as_euler('xyz')
        return Pose(position=self.position, orientation=euler, is_quaternion=False)
    
    def to_quaternion(self) -> "Pose":
        """Convert to quaternion representation."""
        if self.is_quaternion:
            return self
        
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_euler('xyz', self.orientation).as_quat()
        return Pose(position=self.position, orientation=quat, is_quaternion=True)
    
    def to_array(self) -> np.ndarray:
        """Convert to 7D array (position + quaternion)."""
        pose = self.to_quaternion()
        return np.concatenate([pose.position, pose.orientation])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Pose":
        """Create from 7D array."""
        return cls(position=arr[:3], orientation=arr[3:7], is_quaternion=True)


@dataclass
class GraspPose:
    """Grasp pose specification.
    
    Attributes:
        approach_pose: Pose to approach from
        grasp_pose: Final grasp pose
        retreat_pose: Pose to retreat to after grasping
        gripper_width: Gripper width for the grasp
        object_id: ID of object being grasped
    """
    approach_pose: Pose
    grasp_pose: Pose
    retreat_pose: Pose
    gripper_width: float = 0.0
    object_id: Optional[int] = None
    
    def get_sequence(self) -> List[Pose]:
        """Get the approach -> grasp -> retreat sequence."""
        return [self.approach_pose, self.grasp_pose, self.retreat_pose]


class MotionPlanner(ABC):
    """Abstract base class for motion planners.
    
    Implementations should provide collision-aware motion planning
    for manipulator arms.
    """
    
    @abstractmethod
    def plan_joint_motion(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        obstacles: Optional[List[int]] = None,
        timeout: float = 5.0
    ) -> MotionPlan:
        """Plan motion in joint space.
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            obstacles: List of obstacle IDs to avoid
            timeout: Planning timeout in seconds
            
        Returns:
            MotionPlan containing waypoints or error
        """
        pass
    
    @abstractmethod
    def plan_to_pose(
        self,
        start_config: np.ndarray,
        goal_pose: Pose,
        obstacles: Optional[List[int]] = None,
        timeout: float = 5.0
    ) -> MotionPlan:
        """Plan motion to end-effector pose.
        
        Args:
            start_config: Starting joint configuration
            goal_pose: Goal end-effector pose
            obstacles: List of obstacle IDs to avoid
            timeout: Planning timeout in seconds
            
        Returns:
            MotionPlan containing waypoints or error
        """
        pass
    
    @abstractmethod
    def plan_cartesian_path(
        self,
        start_config: np.ndarray,
        waypoint_poses: List[Pose],
        obstacles: Optional[List[int]] = None,
        max_step: float = 0.01
    ) -> MotionPlan:
        """Plan Cartesian path through waypoint poses.
        
        Args:
            start_config: Starting joint configuration
            waypoint_poses: List of end-effector poses to pass through
            obstacles: List of obstacle IDs to avoid
            max_step: Maximum step size in Cartesian space
            
        Returns:
            MotionPlan containing waypoints or error
        """
        pass
    
    @abstractmethod
    def check_collision(
        self,
        config: np.ndarray,
        obstacles: Optional[List[int]] = None
    ) -> bool:
        """Check if a configuration is in collision.
        
        Args:
            config: Joint configuration to check
            obstacles: List of obstacle IDs to check against
            
        Returns:
            True if in collision, False otherwise
        """
        pass
    
    @abstractmethod
    def compute_ik(
        self,
        target_pose: Pose,
        seed_config: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Compute inverse kinematics.
        
        Args:
            target_pose: Target end-effector pose
            seed_config: Seed configuration for IK solver
            
        Returns:
            Joint configuration or None if no solution
        """
        pass
    
    @abstractmethod
    def compute_fk(
        self,
        config: np.ndarray
    ) -> Pose:
        """Compute forward kinematics.
        
        Args:
            config: Joint configuration
            
        Returns:
            End-effector pose
        """
        pass


class GraspPlanner(ABC):
    """Abstract base class for grasp planners."""
    
    @abstractmethod
    def plan_grasp(
        self,
        object_id: int,
        grasp_type: str = "top"
    ) -> Optional[GraspPose]:
        """Plan a grasp for an object.
        
        Args:
            object_id: ID of object to grasp
            grasp_type: Type of grasp ("top", "side", etc.)
            
        Returns:
            GraspPose or None if no valid grasp found
        """
        pass
    
    @abstractmethod
    def get_stable_placements(
        self,
        object_id: int,
        surface_id: int
    ) -> List[Pose]:
        """Get stable placement poses on a surface.
        
        Args:
            object_id: ID of object to place
            surface_id: ID of surface to place on
            
        Returns:
            List of stable placement poses
        """
        pass


@dataclass
class PickAction:
    """Pick action specification."""
    object_id: int
    object_name: str
    grasp_pose: GraspPose
    approach_config: np.ndarray
    grasp_config: np.ndarray
    retreat_config: np.ndarray


@dataclass
class PlaceAction:
    """Place action specification."""
    object_id: int
    object_name: str
    target_surface: str
    place_pose: Pose
    approach_config: np.ndarray
    place_config: np.ndarray
    retreat_config: np.ndarray


class SymbolicMotionBridge(ABC):
    """Bridge between symbolic actions and motion primitives.
    
    This class translates high-level symbolic actions (pick, place, etc.)
    into sequences of motion plans.
    """
    
    @abstractmethod
    def refine_pick(
        self,
        object_name: str,
        current_config: np.ndarray
    ) -> Tuple[bool, List[MotionPlan], Optional[PickAction]]:
        """Refine a symbolic pick action into motion plans.
        
        Args:
            object_name: Name of object to pick
            current_config: Current robot configuration
            
        Returns:
            Tuple of (success, motion_plans, pick_action)
        """
        pass
    
    @abstractmethod
    def refine_place(
        self,
        object_name: str,
        target_surface: str,
        current_config: np.ndarray
    ) -> Tuple[bool, List[MotionPlan], Optional[PlaceAction]]:
        """Refine a symbolic place action into motion plans.
        
        Args:
            object_name: Name of object being placed
            target_surface: Name of target surface
            current_config: Current robot configuration
            
        Returns:
            Tuple of (success, motion_plans, place_action)
        """
        pass
    
    @abstractmethod
    def refine_symbolic_plan(
        self,
        symbolic_plan: List[str],
        initial_config: np.ndarray
    ) -> Tuple[bool, List[MotionPlan]]:
        """Refine a full symbolic plan into motion plans.
        
        Args:
            symbolic_plan: List of symbolic actions (PDDL format)
            initial_config: Initial robot configuration
            
        Returns:
            Tuple of (success, motion_plans)
        """
        pass


def test_motion_interface():
    """Test the motion interface module."""
    print("Testing Motion Interface Module")
    print("=" * 50)
    
    # Test 1: MotionPlan creation
    print("\nTest 1: MotionPlan creation...")
    plan = MotionPlan(
        waypoints=[
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        ],
        durations=[0.5, 0.5],
        success=True,
        status=MotionPlanStatus.SUCCESS
    )
    
    assert plan.num_waypoints == 3
    assert plan.total_duration == 1.0
    assert plan.path_length > 0
    print(f"  ✓ MotionPlan: {plan.num_waypoints} waypoints, {plan.total_duration}s duration")
    
    # Test 2: Pose creation and conversion
    print("\nTest 2: Pose creation and conversion...")
    pose = Pose(
        position=np.array([1.0, 2.0, 3.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
        is_quaternion=True
    )
    
    matrix = pose.to_matrix()
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[:3, 3], [1.0, 2.0, 3.0])
    print(f"  ✓ Pose to matrix conversion passed")
    
    pose2 = Pose.from_matrix(matrix)
    assert np.allclose(pose.position, pose2.position)
    print(f"  ✓ Matrix to Pose conversion passed")
    
    # Test 3: Euler conversion
    print("\nTest 3: Euler/Quaternion conversion...")
    euler_pose = pose.to_euler()
    assert not euler_pose.is_quaternion
    quat_pose = euler_pose.to_quaternion()
    assert quat_pose.is_quaternion
    assert np.allclose(pose.orientation, quat_pose.orientation, atol=1e-6)
    print(f"  ✓ Euler/Quaternion round-trip passed")
    
    # Test 4: Path interpolation
    print("\nTest 4: Path interpolation...")
    interpolated = plan.interpolate(10)
    assert len(interpolated) >= 10
    print(f"  ✓ Interpolated to {len(interpolated)} points")
    
    # Test 5: Dictionary serialization
    print("\nTest 5: Dictionary serialization...")
    plan_dict = plan.to_dict()
    assert "waypoints" in plan_dict
    assert "success" in plan_dict
    assert plan_dict["success"] == True
    assert plan_dict["status"] == "success"
    print(f"  ✓ Dictionary serialization passed")
    
    # Test 6: GraspPose
    print("\nTest 6: GraspPose creation...")
    grasp = GraspPose(
        approach_pose=Pose(np.array([0, 0, 0.1]), np.array([0, 0, 0, 1])),
        grasp_pose=Pose(np.array([0, 0, 0]), np.array([0, 0, 0, 1])),
        retreat_pose=Pose(np.array([0, 0, 0.1]), np.array([0, 0, 0, 1])),
        gripper_width=0.04,
        object_id=1
    )
    sequence = grasp.get_sequence()
    assert len(sequence) == 3
    print(f"  ✓ GraspPose with {len(sequence)} poses")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    return True


if __name__ == "__main__":
    test_motion_interface()
