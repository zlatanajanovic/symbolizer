"""
Motion Planning Module.

This module provides interfaces and implementations for motion planning
in manipulation tasks.

Part of the Symbolizer project for VLM-guided symbolic planning.
"""

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

# Conditionally import PyBullet implementations
try:
    from .pybullet_motion_planner import (
        PyBulletMotionPlanner,
        PyBulletGraspPlanner,
        PyBulletSymbolicMotionBridge,
        RobotConfig,
    )
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

__all__ = [
    # Interfaces
    "MotionPlan",
    "MotionPlanner",
    "MotionPlanStatus",
    "Pose",
    "GraspPose",
    "GraspPlanner",
    "PickAction",
    "PlaceAction",
    "SymbolicMotionBridge",
    # PyBullet implementations (may not be available)
    "PyBulletMotionPlanner",
    "PyBulletGraspPlanner",
    "PyBulletSymbolicMotionBridge",
    "RobotConfig",
    "PYBULLET_AVAILABLE",
]
