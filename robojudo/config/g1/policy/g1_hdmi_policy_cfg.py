"""
G1 HDMI Policy Configuration.

Provides configuration for running HDMI-trained policies on the G1 robot.
Supports checkpoints: G1PushDoorHand, G1RollBall, G1TrackSuitcase.
"""
from pathlib import Path

from robojudo.config import ROOT_DIR
from robojudo.policy.policy_cfgs import HdmiPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


# HDMI uses 29 DOF for G1 observations but only outputs 23 actions (no wrist joints)
class G1HdmiObsDoF(DoFConfig):
    """G1 DoF configuration for HDMI observations (29 joints)."""
    joint_names: list[str] = [
        # Left leg
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        # Right leg
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        # Waist
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        # Left arm
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        # Right arm
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    # Default standing pose (29 values) - HDMI G1 proper standing configuration
    # Based on HDMI checkpoint yaml: default_joint_pos
    default_pos: list[float] | None = [
        # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        # Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        # Waist: yaw, roll, pitch
        0.0, 0.0, 0.0,
        # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        # Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ]


class G1HdmiActionDoF(DoFConfig):
    """G1 DoF configuration for HDMI action output (23 joints, no wrist)."""
    joint_names: list[str] = [
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
        # Left arm without wrist (4)
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        # Right arm without wrist (4)
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ]

    # Default standing pose (23 values) - HDMI G1 proper standing configuration
    # Based on HDMI checkpoint yaml: default_joint_pos (no wrist joints)
    default_pos: list[float] | None = [
        # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        # Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        # Waist: yaw, roll, pitch
        0.0, 0.0, 0.0,
        # Left arm (no wrist): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
        0.2, 0.2, 0.0, 0.6,
        # Right arm (no wrist): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
        0.2, -0.2, 0.0, 0.6,
    ]


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
    action_beta: float = 0.8
    adapt_hx_size: int = 256
    use_residual_action: bool = False


class G1HdmiPushDoorPolicyCfg(G1HdmiPolicyCfg):
    """G1 HDMI Policy for door pushing task."""
    checkpoint_name: str = "G1PushDoorHand"
    model_file: str = "policy-xg6644nr-final.onnx"


class G1HdmiRollBallPolicyCfg(G1HdmiPolicyCfg):
    """G1 HDMI Policy for ball rolling task."""
    checkpoint_name: str = "G1RollBall"
    model_file: str = "policy-yte3rr8b-final.onnx"


class G1HdmiTrackSuitcasePolicyCfg(G1HdmiPolicyCfg):
    """G1 HDMI Policy for suitcase tracking task."""
    checkpoint_name: str = "G1TrackSuitcase"
    model_file: str = "policy-v55m8a23-final.onnx"
