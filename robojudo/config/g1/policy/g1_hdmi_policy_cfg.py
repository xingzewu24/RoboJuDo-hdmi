"""
G1 HDMI Policy Configuration.

Provides configuration for running HDMI-trained policies on the G1 robot.
Supports checkpoints: G1PushDoorHand, G1RollBall, G1TrackSuitcase.

IMPORTANT: Joint ordering
- MuJoCo order: Depth-first traversal (left_leg -> right_leg -> waist -> left_arm -> right_arm)
- Isaac order: Interleaved left/right (left_hip_pitch, right_hip_pitch, waist_yaw, ...)

The HDMI policy was trained with Isaac order. The policy class handles the conversion.
"""
from pathlib import Path

from robojudo.config import ROOT_DIR
from robojudo.policy.policy_cfgs import HdmiPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


# ============================================================================
# HDMI Isaac Joint Order (what the ONNX policy expects)
# This is the order from policy-xg6644nr-final.yaml: isaac_joint_names
# ============================================================================
HDMI_ISAAC_JOINT_NAMES_29 = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

HDMI_ISAAC_JOINT_NAMES_23 = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
]

# ============================================================================
# MuJoCo Joint Order (what the simulation provides)
# This is the depth-first traversal order from g1_pushdoor.xml
# ============================================================================
MUJOCO_JOINT_NAMES_29 = [
    # Left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

MUJOCO_JOINT_NAMES_23 = [
    # Left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (4)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    # Right arm (4)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


def compute_reorder_indices(src_names: list[str], dst_names: list[str]) -> list[int]:
    """Compute indices to reorder src to dst. Result[i] = src index for dst[i]."""
    return [src_names.index(name) for name in dst_names]


# Precompute reordering indices
MUJOCO_TO_ISAAC_29 = compute_reorder_indices(MUJOCO_JOINT_NAMES_29, HDMI_ISAAC_JOINT_NAMES_29)
ISAAC_TO_MUJOCO_29 = compute_reorder_indices(HDMI_ISAAC_JOINT_NAMES_29, MUJOCO_JOINT_NAMES_29)
MUJOCO_TO_ISAAC_23 = compute_reorder_indices(MUJOCO_JOINT_NAMES_23, HDMI_ISAAC_JOINT_NAMES_23)
ISAAC_TO_MUJOCO_23 = compute_reorder_indices(HDMI_ISAAC_JOINT_NAMES_23, MUJOCO_JOINT_NAMES_23)


# ============================================================================
# Default Standing Poses
# ============================================================================
# From HDMI checkpoint yaml default_joint_pos (regex patterns resolved)
HDMI_DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


def get_default_pos_for_joints(joint_names: list[str]) -> list[float]:
    """Get default positions for a list of joint names."""
    return [HDMI_DEFAULT_JOINT_POS.get(name, 0.0) for name in joint_names]


# Default pose in MuJoCo order (for environment initialization)
DEFAULT_POS_MUJOCO_29 = get_default_pos_for_joints(MUJOCO_JOINT_NAMES_29)
DEFAULT_POS_MUJOCO_23 = get_default_pos_for_joints(MUJOCO_JOINT_NAMES_23)

# Default pose in Isaac order (for policy observation normalization)
DEFAULT_POS_ISAAC_29 = get_default_pos_for_joints(HDMI_ISAAC_JOINT_NAMES_29)
DEFAULT_POS_ISAAC_23 = get_default_pos_for_joints(HDMI_ISAAC_JOINT_NAMES_23)


# ============================================================================
# DoF Configurations
# ============================================================================
class G1HdmiObsDoF(DoFConfig):
    """G1 DoF configuration for HDMI observations.
    
    Uses MuJoCo joint order (what the environment provides).
    The policy handles reordering to Isaac order internally.
    """
    joint_names: list[str] = MUJOCO_JOINT_NAMES_29
    default_pos: list[float] | None = DEFAULT_POS_MUJOCO_29


class G1HdmiActionDoF(DoFConfig):
    """G1 DoF configuration for HDMI action output.
    
    Uses MuJoCo joint order (what the environment expects for PD control).
    The policy handles reordering from Isaac order internally.
    """
    joint_names: list[str] = MUJOCO_JOINT_NAMES_23
    default_pos: list[float] | None = DEFAULT_POS_MUJOCO_23


# Path to hdmi-sim2real folder (relative to project root)
HDMI_BASE_PATH = str(ROOT_DIR / "hdmi-sim2real")


class G1HdmiPolicyCfg(HdmiPolicyCfg):
    """
    G1 HDMI Policy configuration.
    
    Available checkpoints:
    - G1PushDoorHand: policy-xg6644nr-final.onnx
    - G1RollBall: policy-yte3rr8b-final.onnx
    - G1TrackSuitcase: policy-v55m8a23-final.onnx
    """
    robot: str = "g1"
    
    # DoF configuration: 29 obs DoFs, 23 action DoFs (no wrist)
    obs_dof: DoFConfig = G1HdmiObsDoF()
    action_dof: DoFConfig = G1HdmiActionDoF()
    
    # HDMI paths
    hdmi_base_path: str = HDMI_BASE_PATH
    
    # Default to G1PushDoorHand checkpoint
    checkpoint_name: str = "G1PushDoorHand"
    model_file: str = "policy-xg6644nr-final.onnx"
    
    # Policy parameters
    action_scale: float = 0.25
    # Lower beta => stronger smoothing => slower action changes
    action_beta: float = 0.3
    adapt_hx_size: int = 256
    use_residual_action: bool = False


class G1HdmiPushDoorPolicyCfg(G1HdmiPolicyCfg):
    """G1 HDMI Policy for door pushing task."""
    checkpoint_name: str = "G1PushDoorHand"
    model_file: str = "policy-xg6644nr-final.onnx"

    # Use a short warmup so the robot stabilizes before the walking-to-door motion kicks in.
    warmup_steps: int = 100

    # Slightly reduce overall action magnitude for sim stability.
    action_scale: float = 0.2

    # No sign flips in MuJoCo by default; keep symmetric joints untouched.
    action_sign_flip_joints: list[str] = []


class G1HdmiRollBallPolicyCfg(G1HdmiPolicyCfg):
    """G1 HDMI Policy for ball rolling task."""
    checkpoint_name: str = "G1RollBall"
    model_file: str = "policy-yte3rr8b-final.onnx"


class G1HdmiTrackSuitcasePolicyCfg(G1HdmiPolicyCfg):
    """G1 HDMI Policy for suitcase tracking task."""
    checkpoint_name: str = "G1TrackSuitcase"
    model_file: str = "policy-v55m8a23-final.onnx"
