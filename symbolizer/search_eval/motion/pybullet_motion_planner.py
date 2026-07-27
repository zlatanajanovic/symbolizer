"""
PyBullet Motion Planner Implementation.

This module provides a concrete implementation of the MotionPlanner interface
using PyBullet and pybullet-planning for motion planning.

Part of the Symbolizer project for VLM-guided symbolic planning.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .motion_interface import (
    GraspPlanner,
    GraspPose,
    MotionPlan,
    MotionPlanner,
    MotionPlanStatus,
    PickAction,
    PlaceAction,
    Pose,
    SymbolicMotionBridge,
)

logger = logging.getLogger(__name__)


# PyBullet utilities (conditional import)
_PYBULLET_AVAILABLE = False
_PYBULLET_TOOLS_AVAILABLE = False

try:
    import pybullet as p
    _PYBULLET_AVAILABLE = True
except ImportError:
    logger.warning("pybullet not available. PyBullet motion planning disabled.")

try:
    from pybullet_tools.utils import (
        plan_joint_motion as pb_plan_motion,
        get_joint_positions,
        set_joint_positions,
        get_link_pose,
        pairwise_collision,
        get_sample_fn,
        get_movable_joints,
    )
    _PYBULLET_TOOLS_AVAILABLE = True
except ImportError:
    logger.warning("pybullet-tools not available. Advanced motion planning disabled.")


@dataclass
class RobotConfig:
    """Configuration for a robot in PyBullet.
    
    Attributes:
        robot_id: PyBullet body ID of the robot
        arm_joints: List of joint indices for the arm
        gripper_joints: List of joint indices for the gripper
        end_effector_link: Link index of the end effector
        joint_limits_lower: Lower joint limits
        joint_limits_upper: Upper joint limits
    """
    robot_id: int
    arm_joints: List[int]
    gripper_joints: List[int] = None
    end_effector_link: int = None
    joint_limits_lower: np.ndarray = None
    joint_limits_upper: np.ndarray = None
    
    def __post_init__(self):
        if self.gripper_joints is None:
            self.gripper_joints = []
        if self.end_effector_link is None and self.arm_joints:
            self.end_effector_link = self.arm_joints[-1]


class PyBulletMotionPlanner(MotionPlanner):
    """Motion planner using PyBullet and pybullet-planning.
    
    This planner uses RRT-based motion planning from pybullet-tools
    for collision-aware path planning.
    
    Example:
        >>> robot_config = RobotConfig(robot_id=1, arm_joints=[0,1,2,3,4,5])
        >>> planner = PyBulletMotionPlanner(robot_config)
        >>> plan = planner.plan_joint_motion(start, goal, obstacles=[2, 3])
    """
    
    def __init__(
        self,
        robot_config: RobotConfig,
        client_id: int = 0,
        self_collision: bool = True,
        max_iterations: int = 1000,
        resolution: float = 0.05
    ):
        """Initialize the PyBullet motion planner.
        
        Args:
            robot_config: Robot configuration
            client_id: PyBullet client ID
            self_collision: Whether to check self-collision
            max_iterations: Maximum planning iterations
            resolution: Path discretization resolution
        """
        if not _PYBULLET_AVAILABLE:
            raise ImportError("pybullet is required for PyBulletMotionPlanner")
        
        self.robot_config = robot_config
        self.client_id = client_id
        self.self_collision = self_collision
        self.max_iterations = max_iterations
        self.resolution = resolution
        
        # Cache joint limits
        if robot_config.joint_limits_lower is None:
            self._cache_joint_limits()
    
    def _cache_joint_limits(self):
        """Cache joint limits from PyBullet."""
        lower = []
        upper = []
        for joint_idx in self.robot_config.arm_joints:
            joint_info = p.getJointInfo(
                self.robot_config.robot_id,
                joint_idx,
                physicsClientId=self.client_id
            )
            lower.append(joint_info[8])  # Lower limit
            upper.append(joint_info[9])  # Upper limit
        self.robot_config.joint_limits_lower = np.array(lower)
        self.robot_config.joint_limits_upper = np.array(upper)
    
    def plan_joint_motion(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        obstacles: Optional[List[int]] = None,
        timeout: float = 5.0
    ) -> MotionPlan:
        """Plan motion in joint space using RRT.
        
        Args:
            start_config: Starting joint configuration
            goal_config: Goal joint configuration
            obstacles: List of obstacle body IDs
            timeout: Planning timeout in seconds
            
        Returns:
            MotionPlan with waypoints or error
        """
        if not _PYBULLET_TOOLS_AVAILABLE:
            return MotionPlan(
                success=False,
                status=MotionPlanStatus.FAILURE,
                error_message="pybullet-tools not available"
            )
        
        start_time = time.time()
        obstacles = obstacles or []
        
        # Set robot to start configuration
        set_joint_positions(
            self.robot_config.robot_id,
            self.robot_config.arm_joints,
            start_config
        )
        
        # Check start/goal validity
        if self.check_collision(start_config, obstacles):
            return MotionPlan(
                success=False,
                status=MotionPlanStatus.COLLISION,
                error_message="Start configuration is in collision"
            )
        
        if self.check_collision(goal_config, obstacles):
            return MotionPlan(
                success=False,
                status=MotionPlanStatus.COLLISION,
                error_message="Goal configuration is in collision"
            )
        
        try:
            # Plan using pybullet-tools
            path = pb_plan_motion(
                self.robot_config.robot_id,
                self.robot_config.arm_joints,
                goal_config,
                obstacles=obstacles,
                self_collisions=self.self_collision,
                max_iterations=self.max_iterations,
                resolutions=self.resolution
            )
            
            elapsed = time.time() - start_time
            
            if path is None:
                return MotionPlan(
                    success=False,
                    status=MotionPlanStatus.FAILURE,
                    error_message="No collision-free path found",
                    metadata={"planning_time": elapsed}
                )
            
            waypoints = [np.array(config) for config in path]
            durations = [0.1] * (len(waypoints) - 1)  # Default timing
            
            return MotionPlan(
                waypoints=waypoints,
                durations=durations,
                success=True,
                status=MotionPlanStatus.SUCCESS,
                metadata={
                    "planning_time": elapsed,
                    "num_obstacles": len(obstacles)
                }
            )
            
        except Exception as e:
            logger.error(f"Motion planning failed: {e}")
            return MotionPlan(
                success=False,
                status=MotionPlanStatus.FAILURE,
                error_message=str(e)
            )
    
    def plan_to_pose(
        self,
        start_config: np.ndarray,
        goal_pose: Pose,
        obstacles: Optional[List[int]] = None,
        timeout: float = 5.0
    ) -> MotionPlan:
        """Plan motion to end-effector pose.
        
        First computes IK for the target pose, then plans joint motion.
        
        Args:
            start_config: Starting joint configuration
            goal_pose: Goal end-effector pose
            obstacles: List of obstacle body IDs
            timeout: Planning timeout in seconds
            
        Returns:
            MotionPlan with waypoints or error
        """
        # Compute IK for goal pose
        goal_config = self.compute_ik(goal_pose, start_config)
        
        if goal_config is None:
            return MotionPlan(
                success=False,
                status=MotionPlanStatus.IK_FAILED,
                error_message="IK solution not found for goal pose"
            )
        
        # Plan joint motion to IK solution
        return self.plan_joint_motion(start_config, goal_config, obstacles, timeout)
    
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
            waypoint_poses: List of end-effector poses
            obstacles: List of obstacle body IDs
            max_step: Maximum step size in Cartesian space
            
        Returns:
            MotionPlan with waypoints or error
        """
        if not waypoint_poses:
            return MotionPlan(
                success=False,
                status=MotionPlanStatus.FAILURE,
                error_message="No waypoint poses provided"
            )
        
        all_waypoints = [start_config]
        all_durations = []
        current_config = start_config
        
        for pose in waypoint_poses:
            # Compute IK for this pose
            target_config = self.compute_ik(pose, current_config)
            
            if target_config is None:
                return MotionPlan(
                    success=False,
                    status=MotionPlanStatus.IK_FAILED,
                    error_message=f"IK failed for waypoint pose"
                )
            
            # Plan motion to this config
            plan = self.plan_joint_motion(
                current_config,
                target_config,
                obstacles
            )
            
            if not plan.success:
                return plan
            
            # Append waypoints (skip first since it's the current config)
            all_waypoints.extend(plan.waypoints[1:])
            all_durations.extend(plan.durations)
            
            current_config = target_config
        
        return MotionPlan(
            waypoints=all_waypoints,
            durations=all_durations,
            success=True,
            status=MotionPlanStatus.SUCCESS
        )
    
    def check_collision(
        self,
        config: np.ndarray,
        obstacles: Optional[List[int]] = None
    ) -> bool:
        """Check if a configuration is in collision.
        
        Args:
            config: Joint configuration to check
            obstacles: List of obstacle body IDs
            
        Returns:
            True if in collision, False otherwise
        """
        if not _PYBULLET_TOOLS_AVAILABLE:
            return False
        
        obstacles = obstacles or []
        
        # Set robot to configuration
        set_joint_positions(
            self.robot_config.robot_id,
            self.robot_config.arm_joints,
            config
        )
        
        # Check collision with each obstacle
        for obstacle_id in obstacles:
            if pairwise_collision(self.robot_config.robot_id, obstacle_id):
                return True
        
        return False
    
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
        if not _PYBULLET_AVAILABLE:
            return None
        
        # Set seed config if provided
        if seed_config is not None:
            set_joint_positions(
                self.robot_config.robot_id,
                self.robot_config.arm_joints,
                seed_config
            )
        
        # Get quaternion orientation
        pose = target_pose.to_quaternion()
        
        try:
            ik_solution = p.calculateInverseKinematics(
                self.robot_config.robot_id,
                self.robot_config.end_effector_link,
                pose.position.tolist(),
                pose.orientation.tolist(),
                lowerLimits=self.robot_config.joint_limits_lower.tolist(),
                upperLimits=self.robot_config.joint_limits_upper.tolist(),
                jointRanges=(
                    self.robot_config.joint_limits_upper -
                    self.robot_config.joint_limits_lower
                ).tolist(),
                restPoses=seed_config.tolist() if seed_config is not None else None,
                physicsClientId=self.client_id
            )
            
            # Extract only arm joints
            if ik_solution:
                return np.array(ik_solution[:len(self.robot_config.arm_joints)])
            
        except Exception as e:
            logger.error(f"IK failed: {e}")
        
        return None
    
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
        # Set robot configuration
        set_joint_positions(
            self.robot_config.robot_id,
            self.robot_config.arm_joints,
            config
        )
        
        # Get link pose
        if _PYBULLET_TOOLS_AVAILABLE:
            pos, orn = get_link_pose(
                self.robot_config.robot_id,
                self.robot_config.end_effector_link
            )
        else:
            link_state = p.getLinkState(
                self.robot_config.robot_id,
                self.robot_config.end_effector_link,
                physicsClientId=self.client_id
            )
            pos = link_state[4]  # World position
            orn = link_state[5]  # World orientation (quaternion)
        
        return Pose(
            position=np.array(pos),
            orientation=np.array(orn),
            is_quaternion=True
        )


class PyBulletGraspPlanner(GraspPlanner):
    """Grasp planner for PyBullet environments.
    
    Generates grasp poses for objects based on their bounding boxes.
    """
    
    def __init__(
        self,
        client_id: int = 0,
        approach_distance: float = 0.1,
        gripper_depth: float = 0.05
    ):
        """Initialize grasp planner.
        
        Args:
            client_id: PyBullet client ID
            approach_distance: Distance for approach pose
            gripper_depth: Depth of gripper penetration
        """
        self.client_id = client_id
        self.approach_distance = approach_distance
        self.gripper_depth = gripper_depth
    
    def plan_grasp(
        self,
        object_id: int,
        grasp_type: str = "top"
    ) -> Optional[GraspPose]:
        """Plan a grasp for an object.
        
        Args:
            object_id: PyBullet body ID
            grasp_type: Type of grasp ("top", "side")
            
        Returns:
            GraspPose or None
        """
        if not _PYBULLET_AVAILABLE:
            return None
        
        # Get object pose and AABB
        pos, orn = p.getBasePositionAndOrientation(
            object_id,
            physicsClientId=self.client_id
        )
        aabb = p.getAABB(object_id, physicsClientId=self.client_id)
        
        obj_center = np.array(pos)
        obj_height = aabb[1][2] - aabb[0][2]
        
        if grasp_type == "top":
            # Top-down grasp
            grasp_pos = obj_center + np.array([0, 0, obj_height/2 + self.gripper_depth])
            approach_pos = grasp_pos + np.array([0, 0, self.approach_distance])
            retreat_pos = approach_pos
            
            # Point gripper down
            grasp_orn = np.array([0, 1, 0, 0])  # 180 degree rotation around Y
            
        elif grasp_type == "side":
            # Side grasp (approach from +X direction)
            grasp_pos = obj_center + np.array([self.gripper_depth, 0, 0])
            approach_pos = grasp_pos + np.array([self.approach_distance, 0, 0])
            retreat_pos = grasp_pos + np.array([0, 0, self.approach_distance])
            
            # Point gripper toward object
            grasp_orn = np.array([0, 0.707, 0, 0.707])  # 90 degree rotation
            
        else:
            logger.warning(f"Unknown grasp type: {grasp_type}")
            return None
        
        return GraspPose(
            approach_pose=Pose(approach_pos, grasp_orn),
            grasp_pose=Pose(grasp_pos, grasp_orn),
            retreat_pose=Pose(retreat_pos, grasp_orn),
            object_id=object_id
        )
    
    def get_stable_placements(
        self,
        object_id: int,
        surface_id: int
    ) -> List[Pose]:
        """Get stable placement poses on a surface.
        
        Args:
            object_id: ID of object to place
            surface_id: ID of surface
            
        Returns:
            List of stable placement poses
        """
        if not _PYBULLET_AVAILABLE:
            return []
        
        # Get surface AABB
        surface_aabb = p.getAABB(surface_id, physicsClientId=self.client_id)
        surface_top = surface_aabb[1][2]
        surface_center = [
            (surface_aabb[0][0] + surface_aabb[1][0]) / 2,
            (surface_aabb[0][1] + surface_aabb[1][1]) / 2
        ]
        
        # Get object AABB for height
        obj_aabb = p.getAABB(object_id, physicsClientId=self.client_id)
        obj_height = obj_aabb[1][2] - obj_aabb[0][2]
        
        # Generate placement pose at surface center
        place_pos = np.array([surface_center[0], surface_center[1], surface_top + obj_height/2])
        place_orn = np.array([0, 0, 0, 1])  # Identity orientation
        
        return [Pose(place_pos, place_orn)]


class PyBulletSymbolicMotionBridge(SymbolicMotionBridge):
    """Bridge between symbolic actions and PyBullet motion.
    
    Translates symbolic actions (pick, place) into motion plans.
    """
    
    def __init__(
        self,
        motion_planner: PyBulletMotionPlanner,
        grasp_planner: PyBulletGraspPlanner,
        object_map: Dict[str, int],
        surface_map: Dict[str, int]
    ):
        """Initialize the bridge.
        
        Args:
            motion_planner: Motion planner instance
            grasp_planner: Grasp planner instance
            object_map: Map from object names to PyBullet IDs
            surface_map: Map from surface names to PyBullet IDs
        """
        self.motion_planner = motion_planner
        self.grasp_planner = grasp_planner
        self.object_map = object_map
        self.surface_map = surface_map
    
    def refine_pick(
        self,
        object_name: str,
        current_config: np.ndarray
    ) -> Tuple[bool, List[MotionPlan], Optional[PickAction]]:
        """Refine a pick action into motion plans.
        
        Args:
            object_name: Name of object to pick
            current_config: Current robot configuration
            
        Returns:
            Tuple of (success, motion_plans, pick_action)
        """
        if object_name not in self.object_map:
            logger.error(f"Unknown object: {object_name}")
            return False, [], None
        
        object_id = self.object_map[object_name]
        
        # Plan grasp
        grasp_pose = self.grasp_planner.plan_grasp(object_id, "top")
        if grasp_pose is None:
            return False, [], None
        
        motion_plans = []
        
        # Plan motion to approach pose
        approach_plan = self.motion_planner.plan_to_pose(
            current_config,
            grasp_pose.approach_pose,
            obstacles=list(self.object_map.values())
        )
        if not approach_plan.success:
            return False, [approach_plan], None
        motion_plans.append(approach_plan)
        
        # Get approach config (last waypoint)
        approach_config = approach_plan.waypoints[-1]
        
        # Plan motion to grasp pose (excluding the object being grasped)
        other_objects = [oid for name, oid in self.object_map.items() if name != object_name]
        grasp_plan = self.motion_planner.plan_to_pose(
            approach_config,
            grasp_pose.grasp_pose,
            obstacles=other_objects
        )
        if not grasp_plan.success:
            return False, motion_plans + [grasp_plan], None
        motion_plans.append(grasp_plan)
        
        grasp_config = grasp_plan.waypoints[-1]
        
        # Plan retreat motion
        retreat_plan = self.motion_planner.plan_to_pose(
            grasp_config,
            grasp_pose.retreat_pose,
            obstacles=other_objects
        )
        if not retreat_plan.success:
            return False, motion_plans + [retreat_plan], None
        motion_plans.append(retreat_plan)
        
        pick_action = PickAction(
            object_id=object_id,
            object_name=object_name,
            grasp_pose=grasp_pose,
            approach_config=approach_config,
            grasp_config=grasp_config,
            retreat_config=retreat_plan.waypoints[-1]
        )
        
        return True, motion_plans, pick_action
    
    def refine_place(
        self,
        object_name: str,
        target_surface: str,
        current_config: np.ndarray
    ) -> Tuple[bool, List[MotionPlan], Optional[PlaceAction]]:
        """Refine a place action into motion plans.
        
        Args:
            object_name: Name of object being held
            target_surface: Name of target surface
            current_config: Current robot configuration
            
        Returns:
            Tuple of (success, motion_plans, place_action)
        """
        if object_name not in self.object_map:
            logger.error(f"Unknown object: {object_name}")
            return False, [], None
        
        if target_surface not in self.surface_map:
            logger.error(f"Unknown surface: {target_surface}")
            return False, [], None
        
        object_id = self.object_map[object_name]
        surface_id = self.surface_map[target_surface]
        
        # Get placement poses
        placements = self.grasp_planner.get_stable_placements(object_id, surface_id)
        if not placements:
            return False, [], None
        
        place_pose = placements[0]  # Use first placement
        
        # Create approach and retreat poses
        approach_offset = np.array([0, 0, 0.1])
        approach_pose = Pose(
            place_pose.position + approach_offset,
            place_pose.orientation
        )
        
        motion_plans = []
        
        # Plan to approach
        other_objects = [oid for name, oid in self.object_map.items() if name != object_name]
        approach_plan = self.motion_planner.plan_to_pose(
            current_config,
            approach_pose,
            obstacles=other_objects + [surface_id]
        )
        if not approach_plan.success:
            return False, [approach_plan], None
        motion_plans.append(approach_plan)
        
        approach_config = approach_plan.waypoints[-1]
        
        # Plan to place
        place_plan = self.motion_planner.plan_to_pose(
            approach_config,
            place_pose,
            obstacles=other_objects
        )
        if not place_plan.success:
            return False, motion_plans + [place_plan], None
        motion_plans.append(place_plan)
        
        place_config = place_plan.waypoints[-1]
        
        # Plan retreat
        retreat_plan = self.motion_planner.plan_to_pose(
            place_config,
            approach_pose,
            obstacles=other_objects + [object_id]  # Now object is placed
        )
        if not retreat_plan.success:
            return False, motion_plans + [retreat_plan], None
        motion_plans.append(retreat_plan)
        
        place_action = PlaceAction(
            object_id=object_id,
            object_name=object_name,
            target_surface=target_surface,
            place_pose=place_pose,
            approach_config=approach_config,
            place_config=place_config,
            retreat_config=retreat_plan.waypoints[-1]
        )
        
        return True, motion_plans, place_action
    
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
        all_motion_plans = []
        current_config = initial_config
        held_object = None
        
        for action_str in symbolic_plan:
            # Parse action
            parts = action_str.strip("()").split()
            action_name = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            if action_name == "pick" or action_name == "pick-up":
                if len(args) < 1:
                    logger.error(f"Pick action requires object argument: {action_str}")
                    return False, all_motion_plans
                
                object_name = args[0]
                success, plans, pick_action = self.refine_pick(object_name, current_config)
                
                if not success:
                    return False, all_motion_plans + plans
                
                all_motion_plans.extend(plans)
                current_config = pick_action.retreat_config
                held_object = object_name
                
            elif action_name == "place" or action_name == "put-down" or action_name == "stack":
                if held_object is None:
                    logger.error(f"Place action but no object held: {action_str}")
                    return False, all_motion_plans
                
                target = args[0] if args else "table"
                success, plans, place_action = self.refine_place(
                    held_object,
                    target,
                    current_config
                )
                
                if not success:
                    return False, all_motion_plans + plans
                
                all_motion_plans.extend(plans)
                current_config = place_action.retreat_config
                held_object = None
                
            else:
                logger.warning(f"Unknown action type: {action_name}")
        
        return True, all_motion_plans


def test_pybullet_motion_planner():
    """Test the PyBullet motion planner (without actual PyBullet)."""
    print("Testing PyBullet Motion Planner Module")
    print("=" * 50)
    
    # Test 1: RobotConfig creation
    print("\nTest 1: RobotConfig creation...")
    config = RobotConfig(
        robot_id=1,
        arm_joints=[0, 1, 2, 3, 4, 5],
        gripper_joints=[6, 7],
        end_effector_link=5
    )
    
    assert config.robot_id == 1
    assert len(config.arm_joints) == 6
    assert config.end_effector_link == 5
    print("  ✓ RobotConfig creation passed")
    
    # Test 2: Import check
    print("\nTest 2: Import availability check...")
    print(f"  PyBullet available: {_PYBULLET_AVAILABLE}")
    print(f"  PyBullet-tools available: {_PYBULLET_TOOLS_AVAILABLE}")
    print("  ✓ Import check passed")
    
    # Test 3: Module-level interfaces
    print("\nTest 3: Interface classes importable...")
    from .motion_interface import (
        MotionPlan,
        MotionPlanner,
        GraspPlanner,
        SymbolicMotionBridge
    )
    print("  ✓ Interface classes imported")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    return True


if __name__ == "__main__":
    test_pybullet_motion_planner()
