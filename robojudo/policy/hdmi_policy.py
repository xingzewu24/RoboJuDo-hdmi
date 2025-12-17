"""
HDMI Policy Adapter for RoboJuDo.

This module provides an adapter to use HDMI-trained policies (from hdmi-sim2real)
within the RoboJuDo framework.

Reference checkpoints:
- G1PushDoorHand
- G1RollBall
- G1TrackSuitcase
"""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from collections import deque

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import HdmiPolicyCfg

logger = logging.getLogger(__name__)


def quat_rotate_inverse_numpy(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion.

    NOTE: HDMI/Isaac convention uses quaternions in [w, x, y, z].
    RoboJuDo environment states commonly provide [x, y, z, w], so convert at call sites.

    Args:
        q: Quaternion in (w, x, y, z) format. Shape (..., 4)
        v: Vector to rotate. Shape (..., 3)

    Returns:
        Rotated vector. Shape (..., 3)
    """
    shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0
    
    return (a - b + c).reshape(shape)


@policy_registry.register
class HdmiPolicy(Policy):
    """
    HDMI Policy adapter for RoboJuDo.
    
    Loads ONNX models trained with the HDMI framework and adapts them
    for deployment through RoboJuDo's pipeline.
    
    The HDMI policy expects observations in a specific format based on
    the observation registry classes defined in hdmi-sim2real.
    
    IMPORTANT: Joint Order Conversion
    - MuJoCo provides joints in depth-first tree traversal order
    - HDMI policy expects joints in Isaac order (interleaved left/right)
    - This class handles the conversion automatically
    
    Key observations:
    - projected_gravity_b: Gravity vector in body frame (3,)
    - root_ang_vel_b: Base angular velocity in body frame (3,)
    - joint_pos: Joint positions relative to default (num_dofs,) in Isaac order
    - joint_vel: Joint velocities (num_dofs,)
    - prev_actions: Previous action outputs (num_actions,)
    
    Additional observations may be required depending on the checkpoint:
    - object_pos_b: Object position relative to robot (for manipulation tasks)
    - ref_contact_pos_b: Reference contact positions (for manipulation)
    """
    
    cfg_policy: HdmiPolicyCfg
    
    # Default door position for G1PushDoorHand task (3m in front of robot)
    DOOR_POSITION_WORLD = np.array([3.0, 0.0, 0.0])
    
    # Default warmup configuration (can be overridden by config)
    DEFAULT_WARMUP_STEPS = 100  # 2 seconds at 50Hz
    
    # G1 standing default pose based on HDMI checkpoint (from yaml: default_joint_pos)
    G1_HDMI_DEFAULT_POSE = {
        "left_hip_pitch_joint": -0.312,
        "right_hip_pitch_joint": -0.312,
        "waist_yaw_joint": 0.0,
        "left_hip_roll_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "waist_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "left_knee_joint": 0.669,
        "right_knee_joint": 0.669,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_pitch_joint": 0.2,
        "left_ankle_pitch_joint": -0.363,
        "right_ankle_pitch_joint": -0.363,
        "left_shoulder_roll_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "left_ankle_roll_joint": 0.0,
        "right_ankle_roll_joint": 0.0,
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
    
    def __init__(self, cfg_policy: HdmiPolicyCfg, device: str = "cpu"):
        # Import joint order mapping from config
        from robojudo.config.g1.policy.g1_hdmi_policy_cfg import (
            MUJOCO_TO_ISAAC_29, ISAAC_TO_MUJOCO_29,
            MUJOCO_TO_ISAAC_23, ISAAC_TO_MUJOCO_23,
            HDMI_ISAAC_JOINT_NAMES_29, HDMI_ISAAC_JOINT_NAMES_23,
            DEFAULT_POS_ISAAC_29, DEFAULT_POS_ISAAC_23,
        )
        
        # Store joint reordering indices
        self.mujoco_to_isaac_29 = np.array(MUJOCO_TO_ISAAC_29)
        self.isaac_to_mujoco_29 = np.array(ISAAC_TO_MUJOCO_29)
        self.mujoco_to_isaac_23 = np.array(MUJOCO_TO_ISAAC_23)
        self.isaac_to_mujoco_23 = np.array(ISAAC_TO_MUJOCO_23)
        
        # Store Isaac order joint names and defaults
        self.isaac_joint_names_29 = HDMI_ISAAC_JOINT_NAMES_29
        self.isaac_joint_names_23 = HDMI_ISAAC_JOINT_NAMES_23
        self.default_pos_isaac_29 = np.array(DEFAULT_POS_ISAAC_29, dtype=np.float32)
        self.default_pos_isaac_23 = np.array(DEFAULT_POS_ISAAC_23, dtype=np.float32)
        
        # Setup ONNX session
        sess_options = ort.SessionOptions()
        
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        policy_file = cfg_policy.policy_file
        logger.info(f"[HdmiPolicy] Loading ONNX model from: {policy_file}")
        
        if not os.path.exists(policy_file):
            raise FileNotFoundError(f"ONNX model not found: {policy_file}")
        
        self.session = ort.InferenceSession(policy_file, sess_options, providers=providers)
        
        # Load JSON metadata for input/output keys
        json_path = policy_file.replace(".onnx", ".json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                self.meta = json.load(f)
            self.in_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["in_keys"]]
            self.out_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["out_keys"]]
            logger.info(f"[HdmiPolicy] Input keys: {self.in_keys}")
            logger.info(f"[HdmiPolicy] Output keys: {self.out_keys}")
        else:
            # Fallback: use ONNX input/output names directly
            self.in_keys = [inp.name for inp in self.session.get_inputs()]
            self.out_keys = [out.name for out in self.session.get_outputs()]
            self.meta = None
            logger.warning(f"[HdmiPolicy] JSON metadata not found, using ONNX names: {self.in_keys}")
        
        # Load YAML config for detailed parameters (default joint pos, action scales, etc.)
        yaml_path = policy_file.replace(".onnx", ".yaml")
        self.yaml_config = None
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                self.yaml_config = yaml.safe_load(f)
            logger.info(f"[HdmiPolicy] Loaded YAML config from: {yaml_path}")
        
        # Initialize parent class
        super().__init__(cfg_policy=cfg_policy, device=device)
        
        # Override default_dof_pos with HDMI checkpoint's default pose
        self._setup_default_pose()
        
        # HDMI-specific state
        self.adapt_hx_size = cfg_policy.adapt_hx_size
        self.adapt_hx = np.zeros((1, self.adapt_hx_size), dtype=np.float32)
        
        self.use_residual_action = cfg_policy.use_residual_action
        self.gravity_vec = np.array([0.0, 0.0, -1.0])
        
        # Previous actions buffer (for 3 timesteps of history as per HDMI config)
        self.prev_actions_history = [np.zeros(self.num_actions, dtype=np.float32) for _ in range(3)]
        self.prev_actions_buffer = np.zeros(self.num_actions, dtype=np.float32)

        # Joint position history buffer for HDMI observation (lags [0,1,2,3,4,8]).
        # Store Isaac-order raw joint positions (29,).
        self._joint_pos_hist: deque[np.ndarray] = deque(maxlen=9)
        
        # Observation scaling factors (common IsaacGym/HDMI defaults)
        self.obs_scales = {
            "ang_vel": 0.25,
            "dof_vel": 0.05,
            "dof_pos": 1.0,
        }
        
        # Object tracking state (door at x=3.0m)
        self.door_pos_world = self.DOOR_POSITION_WORLD.copy()
        
        # Motion phase counter (for command observations)
        self.motion_step = 0
        self.motion_duration_steps = 573  # ~11.46 seconds at 50Hz (from yaml)
        
        # Step counter for warmup period
        self.step_count = 0

        # Warmup settings (configurable)
        self.warmup_steps = getattr(cfg_policy, "warmup_steps", self.DEFAULT_WARMUP_STEPS)
        self.warmup_max_action_start = getattr(cfg_policy, "warmup_max_action_start", 0.05)
        self.warmup_max_action_final = getattr(cfg_policy, "warmup_max_action_final", 0.25)

        # Optional action sign flips (joint names are in MuJoCo action order = cfg_policy.action_dof.joint_names)
        self._action_flip_set = set(getattr(cfg_policy, "action_sign_flip_joints", []) or [])
        
        # Safety mode: disabled by default since we now provide proper observations
        self.safety_mode = False
        self.safety_action_scale = 0.1
        
        # Per-joint action scales from HDMI yaml (policy-xg6644nr-final.yaml)
        # Order matches policy_joint_names (23 joints, Isaac order)
        self.action_scales = np.array([
            # hip_pitch, hip_pitch, waist_yaw
            0.55, 0.55, 0.55,
            # hip_roll, hip_roll, waist_roll
            0.35, 0.35, 0.44,
            # hip_yaw, hip_yaw, waist_pitch
            0.55, 0.55, 0.44,
            # knee, knee, shoulder_pitch, shoulder_pitch
            0.35, 0.35, 0.44, 0.44,
            # ankle_pitch, ankle_pitch, shoulder_roll, shoulder_roll
            0.44, 0.44, 0.44, 0.44,
            # ankle_roll, ankle_roll, shoulder_yaw, shoulder_yaw
            0.44, 0.44, 0.44, 0.44,
            # elbow, elbow
            0.44, 0.44,
        ], dtype=np.float32)
        # Global scale multiplier (lets us safely tune amplitude without editing hardcoded YAML scales)
        # NOTE: This is the PolicyCfg.action_scale, not the YAML per-joint action_scale map.
        self.global_action_scale = float(getattr(cfg_policy, "action_scale", 1.0))

        # Fallback single scale (used if action_scales shape doesn't match)
        # Keep this as 1.0; cfg_policy.action_scale is applied via self.global_action_scale.
        self.action_scale = 1.0
        
        # Load motion data for reference trajectory
        self._load_motion_data()
        
        logger.info(f"[HdmiPolicy] Initialized with {self.num_dofs} obs dofs, {self.num_actions} action dofs")
        logger.info(f"[HdmiPolicy] Default standing pose (first 6): {self.default_dof_pos[:6]}")
        logger.info(f"[HdmiPolicy] Door position: {self.door_pos_world}")
        if self.warmup_steps > 0:
            logger.info(f"[HdmiPolicy] Warmup steps: {self.warmup_steps} (~{self.warmup_steps/50:.1f}s)")
        else:
            logger.info("[HdmiPolicy] Warmup disabled (warmup_steps=0): starting policy immediately")
    
    def _load_motion_data(self):
        """Load motion reference data for walking trajectory."""
        from robojudo.config import ROOT_DIR
        
        # Path to motion data
        motion_path = ROOT_DIR / "hdmi-sim2real" / "data" / "motion" / "data_for_sim" / "push_door-hand-0828"
        
        self.motion_data = None
        self.motion_joint_indices = None
        self.motion_body_indices = None
        
        if motion_path.exists():
            try:
                # Load motion npz file
                npz_path = motion_path / "motion.npz"
                meta_path = motion_path / "meta.json"
                
                if npz_path.exists() and meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    
                    motion_raw = dict(np.load(npz_path))
                    
                    # Store motion data
                    self.motion_data = {
                        "joint_pos": motion_raw.get("joint_pos", None),  # (T, num_joints)
                        "body_pos_w": motion_raw.get("body_pos_w", None),  # (T, num_bodies, 3)
                        "body_quat_w": motion_raw.get("body_quat_w", None),  # (T, num_bodies, 4)
                    }
                    self.motion_meta = meta
                    self.motion_fps = meta.get("fps", 50)
                    self.motion_length = self.motion_data["joint_pos"].shape[0] if self.motion_data["joint_pos"] is not None else 0
                    
                    # Map motion joint names to our policy joint names
                    motion_joint_names = meta.get("joint_names", [])
                    action_joint_names = self.cfg_policy.action_dof.joint_names
                    
                    self.motion_joint_indices = []
                    for jname in action_joint_names:
                        if jname in motion_joint_names:
                            self.motion_joint_indices.append(motion_joint_names.index(jname))
                        else:
                            self.motion_joint_indices.append(-1)  # Not found
                    
                    # Map body names for reference body positions
                    motion_body_names = meta.get("body_names", [])
                    # Policy expects 16 bodies for ref_body_pos_future_local
                    policy_body_names = [
                        "pelvis", "left_hip_pitch_link", "right_hip_pitch_link",
                        "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
                        "left_knee_link", "right_knee_link", 
                        "left_shoulder_pitch_link", "right_shoulder_pitch_link",
                        "left_ankle_roll_link", "right_ankle_roll_link",
                        "left_elbow_link", "right_elbow_link",
                        "left_wrist_yaw_link", "right_wrist_yaw_link",
                    ]
                    self.motion_body_indices = []
                    for bname in policy_body_names:
                        if bname in motion_body_names:
                            self.motion_body_indices.append(motion_body_names.index(bname))
                        else:
                            self.motion_body_indices.append(0)  # Default to pelvis
                    
                    logger.info(f"[HdmiPolicy] Loaded motion data: {self.motion_length} frames at {self.motion_fps}Hz")
                else:
                    logger.warning(f"[HdmiPolicy] Motion files not found at {motion_path}")
            except Exception as e:
                logger.warning(f"[HdmiPolicy] Failed to load motion data: {e}")
                self.motion_data = None
        else:
            logger.warning(f"[HdmiPolicy] Motion path not found: {motion_path}")
    
    def _setup_default_pose(self):
        """Setup default standing pose from HDMI checkpoint config.
        
        Sets up two versions of the default pose:
        - default_dof_pos: MuJoCo order (for environment interaction)
        - default_dof_pos_isaac: Isaac order (for policy observation normalization)
        """
        # Get joint names from observation dof config (MuJoCo order)
        obs_joint_names = self.cfg_obs_dof.joint_names
        
        # Build default pose dictionary from yaml or class defaults
        hdmi_defaults = self.G1_HDMI_DEFAULT_POSE.copy()
        
        # If we have yaml config, extract default_joint_pos from it
        if self.yaml_config and "default_joint_pos" in self.yaml_config:
            yaml_defaults = self.yaml_config["default_joint_pos"]
            
            # Parse regex-style defaults from yaml (e.g., ".*_hip_pitch_joint": -0.312)
            import re
            for joint_name in obs_joint_names:
                for pattern, value in yaml_defaults.items():
                    if re.match(pattern, joint_name):
                        hdmi_defaults[joint_name] = value
                        break
        
        # Build default pose array in MuJoCo order (for environment)
        new_default_pos = []
        for joint_name in obs_joint_names:
            if joint_name in hdmi_defaults:
                new_default_pos.append(hdmi_defaults[joint_name])
            else:
                new_default_pos.append(0.0)
        
        self.default_dof_pos = np.array(new_default_pos, dtype=np.float32)
        
        # Also store default pose in Isaac order (for policy observations)
        # This is used for normalizing observations before feeding to ONNX model
        self.default_dof_pos_isaac = self.default_pos_isaac_29.copy()
        
        logger.info(f"[HdmiPolicy] Setup default pose in MuJoCo order ({len(self.default_dof_pos)} dofs)")
        logger.info(f"[HdmiPolicy] MuJoCo default[:6]: {self.default_dof_pos[:6]}")
        logger.info(f"[HdmiPolicy] Isaac default[:6]: {self.default_dof_pos_isaac[:6]}")

    
    def reset(self):
        """Reset policy state between episodes."""
        self.adapt_hx[:] = 0.0
        self.prev_actions_buffer[:] = 0.0
        self.prev_actions_history = [np.zeros(self.num_actions, dtype=np.float32) for _ in range(3)]
        self.last_action = np.zeros(self.num_actions)
        self.motion_step = 0
        self.step_count = 0
        self._joint_pos_hist.clear()
        if self.warmup_steps > 0:
            logger.info("[HdmiPolicy] Reset: starting warmup period")
        else:
            logger.info("[HdmiPolicy] Reset: warmup disabled")
    
    def post_step_callback(self, commands: list[str] | None = None):
        """Handle any post-step commands."""
        for command in commands or []:
            if command == "[POLICY_RESET]":
                self.reset()
    
    def _compute_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        """
        Compute gravity vector projected into body frame.
        
        Args:
            quat: Body quaternion in (w, x, y, z) format. Shape (4,)
        
        Returns:
            Projected gravity vector in body frame. Shape (3,)
        """
        projected_gravity = quat_rotate_inverse_numpy(
            quat[None, :],
            self.gravity_vec[None, :]
        ).squeeze(0)
        return projected_gravity
    
    def _compute_object_obs(self, base_pos: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
        """
        Compute object observations for door pushing task.
        
        The HDMI G1PushDoorHand expects 7 object observation values:
        - object_xy_b (2): Door XY position in body frame
        - object_heading_b (2): Door heading vector in body frame (cos, sin)  
        - ref_contact_pos_b (3): Reference contact position relative to robot
        
        Args:
            base_pos: Robot base position in world frame (3,)
            base_quat: Robot base quaternion in (w, x, y, z) format (4,)
        
        Returns:
            Object observation tensor (7,)
        """
        # Door position in world frame
        door_pos_w = self.door_pos_world
        
        # Get yaw-only quaternion for computing body-frame coordinates.
        # IMPORTANT: base_quat is a general quaternion (w, x, y, z). Normalizing only (w, z)
        # is not a valid yaw extraction once roll/pitch exist. Extract yaw from the full quat.
        w, x, y, z = float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3])
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_quat = np.array([np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)], dtype=np.float32)
        
        # Object XY position in body frame
        door_pos_b = quat_rotate_inverse_numpy(
            yaw_quat[None, :],
            (door_pos_w - base_pos)[None, :]
        ).squeeze(0)
        object_xy_b = door_pos_b[:2]  # (2,)
        
        # Object heading in body frame (door facing direction)
        # Door panel initially faces -X in world frame (perpendicular to wall)
        door_heading_w = np.array([1.0, 0.0, 0.0])  # Door opening direction
        door_heading_b = quat_rotate_inverse_numpy(
            yaw_quat[None, :],
            door_heading_w[None, :]
        ).squeeze(0)
        object_heading_b = door_heading_b[:2]  # (2,)
        
        # Reference contact position (from yaml: offset [0, -0.6, 1.0] relative to door_panel)
        # This represents where the robot should contact the door
        # In body frame, relative to robot
        contact_offset_w = np.array([0.0, -0.6, 1.0])  # Door panel contact point offset
        contact_pos_w = door_pos_w + contact_offset_w
        contact_pos_b = quat_rotate_inverse_numpy(
            yaw_quat[None, :],
            (contact_pos_w - base_pos)[None, :]
        ).squeeze(0)
        ref_contact_pos_b = contact_pos_b  # (3,)
        
        # Concatenate all object observations
        object_obs = np.concatenate([
            object_xy_b,      # 2
            object_heading_b, # 2
            ref_contact_pos_b # 3
        ]).astype(np.float32)
        
        return object_obs
    
    def _compute_command_obs(self) -> np.ndarray:
        """
        Compute command observations for motion reference.
        
        The HDMI G1PushDoorHand expects 356 command observation values:
        - ref_body_pos_future_local: Reference body positions for future timesteps (240)
        - ref_joint_pos_future: Reference joint positions for future timesteps (115)
        - ref_motion_phase: Motion phase (0 to 1) (1)
        
        Behavior:
        - During warmup (step_count < WARMUP_STEPS): Return standing command
        - After warmup: Return motion trajectory command (walking to door)
        
        Returns:
            Command observation tensor (356,)
        """
        # The command observation includes:
        # - ref_body_pos_future_local: 5 future steps * 16 bodies * 3 coords = 240
        # - ref_joint_pos_future: 5 future steps * 23 joints = 115  
        # - ref_motion_phase: 1
        # Total = 356
        
        command_obs = np.zeros(356, dtype=np.float32)
        
        # Get default standing pose for action joints (in Isaac order for policy)
        # Use default_pos_isaac_23 which is already in Isaac order
        default_action_pos = self.default_pos_isaac_23.copy()
        
        # Check if we're in warmup period
        is_warmup = self.warmup_steps > 0 and self.step_count < self.warmup_steps
        
        if is_warmup:
            # WARMUP MODE: Stand still with default pose
            # ref_body_pos_future_local: zeros (stay in place)
            # ref_joint_pos_future: default standing pose
            motion_phase = 0.0
            
            # Fill 5 future timesteps with standing pose reference
            for t in range(5):
                start_idx = 240 + t * 23
                command_obs[start_idx:start_idx + 23] = default_action_pos
            
            if self.warmup_steps > 0 and self.step_count % 50 == 0:
                logger.info(f"[HdmiPolicy] Warmup: {self.step_count}/{self.warmup_steps} steps")
        else:
            # ACTIVE MODE: Use motion trajectory for walking
            motion_t = self.step_count - self.warmup_steps
            
            if self.motion_data is not None and self.motion_data["joint_pos"] is not None:
                # Use loaded motion data
                # Cycle through motion if we exceed length
                motion_idx = motion_t % self.motion_length
                
                # Future timesteps for observation (from yaml: [1, 2, 8, 16, 32])
                future_steps = [1, 2, 8, 16, 32]
                
                # Fill ref_joint_pos_future (indices 240-354)
                for t_idx, future_t in enumerate(future_steps):
                    future_motion_idx = (motion_idx + future_t) % self.motion_length
                    
                    # Get joint positions from motion data
                    motion_joint_pos = self.motion_data["joint_pos"][future_motion_idx]
                    
                    # Map to policy joint order
                    ref_joint_pos = np.zeros(23, dtype=np.float32)
                    for i, motion_idx_j in enumerate(self.motion_joint_indices):
                        if motion_idx_j >= 0 and i < 23:
                            ref_joint_pos[i] = motion_joint_pos[motion_idx_j]
                        elif i < 23:
                            ref_joint_pos[i] = default_action_pos[i]
                    
                    start_idx = 240 + t_idx * 23
                    command_obs[start_idx:start_idx + 23] = ref_joint_pos
                
                # Fill ref_body_pos_future_local (indices 0-239)
                # This requires transforming body positions to local frame
                if self.motion_data["body_pos_w"] is not None:
                    root_pos = self.motion_data["body_pos_w"][motion_idx, 0]  # Pelvis position
                    root_quat = self.motion_data["body_quat_w"][motion_idx, 0] if self.motion_data["body_quat_w"] is not None else np.array([1,0,0,0])
                    
                    # Compute yaw-only quaternion for local transform
                    norm = np.sqrt(root_quat[0]**2 + root_quat[3]**2) + 1e-8
                    yaw_quat = np.array([root_quat[0]/norm, 0.0, 0.0, root_quat[3]/norm])
                    
                    for t_idx, future_t in enumerate(future_steps):
                        future_motion_idx = (motion_idx + future_t) % self.motion_length
                        
                        for b_idx, body_motion_idx in enumerate(self.motion_body_indices):
                            if b_idx < 16:
                                body_pos_w = self.motion_data["body_pos_w"][future_motion_idx, body_motion_idx]
                                # Transform to local frame
                                body_pos_local = quat_rotate_inverse_numpy(
                                    yaw_quat[None, :],
                                    (body_pos_w - root_pos)[None, :]
                                ).squeeze(0)
                                
                                start_idx = t_idx * 16 * 3 + b_idx * 3
                                command_obs[start_idx:start_idx + 3] = body_pos_local
                
                # Motion phase
                motion_phase = (motion_idx % self.motion_duration_steps) / self.motion_duration_steps
                
                if motion_t % 100 == 0:
                    logger.info(f"[HdmiPolicy] Active: motion_t={motion_t}, motion_idx={motion_idx}, phase={motion_phase:.2f}")
            else:
                # No motion data: fall back to standing reference
                motion_phase = (motion_t % self.motion_duration_steps) / self.motion_duration_steps
                
                for t in range(5):
                    start_idx = 240 + t * 23
                    command_obs[start_idx:start_idx + 23] = default_action_pos
                
                if motion_t % 100 == 0:
                    logger.warning(f"[HdmiPolicy] No motion data, using standing reference")
        
        # ref_motion_phase at index 355
        command_obs[355] = motion_phase
        
        return command_obs
    
    def get_observation(self, env_data, ctrl_data) -> tuple[np.ndarray, dict]:
        """
        Build observation tensor from environment data.
        
        The HDMI ONNX model expects 3 observation groups:
        - 'command': (1, 356) - reference motion/trajectory observations
        - 'policy': (1, 249) - proprioceptive observations (joint pos, vel, gravity, actions)
        - 'object': (1, 7) - object tracking observations
        
        IMPORTANT: Joint positions from MuJoCo are reordered to Isaac order before
        being fed to the policy, since the HDMI policy was trained with Isaac order.
        
        Args:
            env_data: Environment state data (dof_pos, dof_vel, base_quat, etc.)
            ctrl_data: Controller data (commands, etc.)
        
        Returns:
            Tuple of (observation_dict, extras_dict)
        """
        # Increment step counter for warmup logic (done here since pipeline calls get_observation directly)
        self.step_count += 1
        
        # Extract state from env_data (MuJoCo order)
        dof_pos_mujoco = env_data.dof_pos  # (num_obs_dofs,) = (29,) for G1, MuJoCo order
        dof_vel_mujoco = env_data.dof_vel  # (num_obs_dofs,) = (29,), MuJoCo order
        # RoboJuDo provides quaternions as [x, y, z, w].
        # IMPORTANT: HDMI "root" signals (projected gravity / yaw-only transforms) should use the
        # *base/free-joint* orientation. Using torso link orientation can introduce a systematic bias
        # (torso may pitch/roll relative to pelvis), which destabilizes the policy.
        base_quat_xyzw = env_data.base_quat
        if base_quat_xyzw is None:
            base_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        base_quat = base_quat_xyzw[[3, 0, 1, 2]]
        base_ang_vel = env_data.base_ang_vel  # (3,) in body frame
        base_pos = getattr(env_data, 'base_pos', None)
        if base_pos is None:
            base_pos = np.array([0.0, 0.0, 0.793], dtype=np.float32)  # Default standing height
        
        # CRITICAL: Reorder joint positions from MuJoCo order to Isaac order
        # The HDMI policy was trained with Isaac order (interleaved left/right)
        dof_pos_isaac = dof_pos_mujoco[self.mujoco_to_isaac_29]  # (29,) Isaac order
        dof_vel_isaac = dof_vel_mujoco[self.mujoco_to_isaac_29]  # (29,) Isaac order
        
        # Compute common observations with scaling
        obs_projected_gravity = self._compute_projected_gravity(base_quat)  # (3,)
        obs_ang_vel = base_ang_vel * self.obs_scales["ang_vel"]  # (3,) scaled
        
        # IMPORTANT: HDMI uses RAW joint positions without subtracting default pose
        # This matches the hdmi-sim2real reference implementation in joint_pos_history
        # The policy was trained with raw positions, not normalized by default
        obs_dof_pos = dof_pos_isaac * self.obs_scales["dof_pos"]  # (29,) - RAW, no normalization
        obs_dof_vel = dof_vel_isaac * self.obs_scales["dof_vel"]  # (29,) scaled
        
        # Debug logging (first few steps only)
        if self.step_count < 5:
            logger.info(f"[HdmiPolicy] Step {self.step_count}: MuJoCo dof_pos[:6]={dof_pos_mujoco[:6]}")
            logger.info(f"[HdmiPolicy] Step {self.step_count}: Isaac dof_pos[:6]={dof_pos_isaac[:6]}")
            logger.info(f"[HdmiPolicy] Step {self.step_count}: RAW obs_dof_pos[:6]={obs_dof_pos[:6]} (raw positions for HDMI)")
            logger.info(f"[HdmiPolicy] Step {self.step_count}: base_ang_vel={base_ang_vel}, gravity={obs_projected_gravity}")
        
        # HDMI uses history of prev_actions (3 steps) - already in Isaac order
        obs_prev_actions = np.concatenate(self.prev_actions_history).astype(np.float32)  # (23*3=69,)
        
        # Build 'policy' observation group (proprioceptive data)
        # Based on yaml config:
        # - projected_gravity_history (history_steps: [0]) -> 3
        # - root_ang_vel_history (history_steps: [0]) -> 3
        # - joint_pos_history (history_steps: [0,1,2,3,4,8]) -> 29*6 = 174  
        # - prev_actions (steps: 3) -> 23*3 = 69
        # Total = 3 + 3 + 174 + 69 = 249
        
        # HDMI uses joint position history steps [0,1,2,3,4,8] (0 = current).
        # Maintain a small buffer so the policy sees a more realistic temporal context.
        self._joint_pos_hist.append(obs_dof_pos.copy())
        while len(self._joint_pos_hist) < 9:
            # Pad initial history with the first observed pose.
            self._joint_pos_hist.appendleft(obs_dof_pos.copy())

        history_lags = [0, 1, 2, 3, 4, 8]
        joint_pos_history = np.concatenate([self._joint_pos_hist[-(lag + 1)] for lag in history_lags], axis=0)
        
        policy_obs = np.concatenate([
            obs_projected_gravity,     # 3
            obs_ang_vel,               # 3  
            joint_pos_history,         # 174 (29 joints * 6 history steps)
            obs_prev_actions,          # 69 (23 actions * 3 steps)
        ]).astype(np.float32)  # = 249 elements
        
        # 'command' observation group (356): reference motion data
        command_obs = self._compute_command_obs()  # (356,)
        
        # 'object' observation group (7): object position/orientation relative to robot
        object_obs = self._compute_object_obs(base_pos, base_quat)  # (7,)
        
        # Build observation dictionary for ONNX model
        obs_dict = {
            "command": command_obs[None, :],  # (1, 356)
            "policy": policy_obs[None, :],    # (1, 249)
            "object": object_obs[None, :],    # (1, 7)
            # Some HDMI ONNX models use this boolean to reset internal recurrent state.
            "is_init": np.array([self.step_count == 1], dtype=bool),
        }
        
        extras = {
            "obs_projected_gravity": obs_projected_gravity,
            "obs_ang_vel": obs_ang_vel,
            "obs_dof_pos": obs_dof_pos,
            "object_obs": object_obs,
        }
        
        return obs_dict, extras
    
    def get_action(self, obs: np.ndarray | dict) -> np.ndarray:
        """
        Run ONNX inference to get action.
        
        Args:
            obs: Observation dict with keys matching ONNX model inputs
        
        Returns:
            Scaled action array
        """
        if isinstance(obs, np.ndarray):
            # Legacy path: obs is a flat array, wrap in dict
            obs_dict = {"obs": obs[None, :].astype(np.float32)}
        else:
            obs_dict = obs
        
        # Build ONNX input dictionary based on model's expected inputs
        ort_inputs = {}
        for inp in self.session.get_inputs():
            inp_name = inp.name
            
            # Map ONNX input names to our observation dict
            if inp_name in obs_dict:
                ort_inputs[inp_name] = obs_dict[inp_name]
            elif inp_name == "observation" and "obs" in obs_dict:
                ort_inputs[inp_name] = obs_dict["obs"]
            elif inp_name == "adapt_hx":
                ort_inputs[inp_name] = self.adapt_hx
            elif inp_name == "is_init":
                ort_inputs[inp_name] = obs_dict.get("is_init", np.zeros(1, dtype=bool))
            else:
                logger.warning(f"[HdmiPolicy] Missing input '{inp_name}', using zeros")
                shape = inp.shape
                # Replace dynamic dims with 1
                shape = [1 if isinstance(s, str) or s is None else s for s in shape]
                ort_inputs[inp_name] = np.zeros(shape, dtype=np.float32)
        
        # Run inference
        try:
            outputs = self.session.run(None, ort_inputs)
        except Exception as e:
            logger.error(f"[HdmiPolicy] ONNX inference failed: {e}")
            logger.error(f"[HdmiPolicy] Input shapes: {[(k, v.shape) for k, v in ort_inputs.items()]}")
            raise
        
        # Parse outputs based on out_keys
        output_dict = {k: v for k, v in zip(self.out_keys, outputs)}
        
        # Extract action
        if "action" in output_dict:
            action = output_dict["action"].squeeze(0)
        elif ("next", "action") in output_dict:
            action = output_dict[("next", "action")].squeeze(0)
        else:
            # Assume first output is action
            action = outputs[0].squeeze(0)
        
        # Update hidden state if present
        for key, val in output_dict.items():
            if isinstance(key, tuple) and key[0] == "next" and key[1] == "adapt_hx":
                self.adapt_hx = val
            elif key == "next_adapt_hx" or key == "adapt_hx_out":
                self.adapt_hx = val
        
        # Clip action
        action = np.clip(action, -100, 100)
        
        # Smooth action (exponential moving average)
        action = (1 - self.action_beta) * self.last_action + self.action_beta * action
        self.last_action = action.copy()
        
        # CRITICAL: Clamp actions during warmup for stability
        # During warmup, the robot should stay near default pose
        if self.warmup_steps > 0 and self.step_count <= self.warmup_steps:
            # Progressively increase action range during warmup
            warmup_progress = min(1.0, self.step_count / max(1, self.warmup_steps))
            max_action = self.warmup_max_action_start + (
                self.warmup_max_action_final - self.warmup_max_action_start
            ) * warmup_progress
            action = np.clip(action, -max_action, max_action)

            if self.step_count % 25 == 0:
                logger.info(
                    f"[HdmiPolicy] Warmup {self.step_count}/{self.warmup_steps}: max_action={max_action:.3f}"
                )
        
        # Store for next step's observation (shift history and add new)
        self.prev_actions_history.pop(0)  # Remove oldest
        self.prev_actions_history.append(action.copy())  # Add newest
        self.prev_actions_buffer = action.copy()
        
        # Apply per-joint action scaling from HDMI yaml
        # Use per-joint scales if available and shape matches, else fallback to single scale
        if hasattr(self, 'action_scales') and len(self.action_scales) == len(action):
            effective_scales = self.action_scales
        else:
            effective_scales = self.action_scale

        # Always apply global scale multiplier from config.
        # This makes cfg_policy.action_scale effective even when YAML per-joint scales exist.
        effective_scales = effective_scales * self.global_action_scale
            
        if self.safety_mode:
            # Reduce action magnitude when in safety mode (no tracking data)
            effective_scales = effective_scales * self.safety_action_scale
        
        scaled_actions = action * effective_scales
        
        # CRITICAL: Reorder actions from Isaac order to MuJoCo order
        # The DoFAdapter in PolicyWrapper expects actions in the same order as cfg_action_dof.joint_names
        # which is MuJoCo order (MUJOCO_JOINT_NAMES_23)
        scaled_actions_mujoco = scaled_actions[self.isaac_to_mujoco_23]

        # Optional: fix sign convention mismatches by flipping selected joints.
        # Names are interpreted in MuJoCo action joint order.
        if self._action_flip_set:
            mujoco_action_joint_names = self.cfg_policy.action_dof.joint_names
            flip_mask = np.array([name in self._action_flip_set for name in mujoco_action_joint_names], dtype=bool)
            if np.any(flip_mask):
                scaled_actions_mujoco = scaled_actions_mujoco.copy()
                scaled_actions_mujoco[flip_mask] *= -1.0
                if self.step_count == 1:
                    flipped = [mujoco_action_joint_names[i] for i in np.where(flip_mask)[0]]
                    logger.info(f"[HdmiPolicy] Action sign flip enabled for: {flipped}")

        # Debug a tiny summary early to confirm scaling/reordering.
        if self.step_count <= 3:
            logger.info(
                f"[HdmiPolicy] Step {self.step_count}: action(isaac)[:6]={action[:6]}, "
                f"scaled_action(mj)[:6]={scaled_actions_mujoco[:6]}"
            )
        
        # Debug: print shapes periodically
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 0
        
        if self._step_count % 100 == 0:
            logger.debug(f"[HdmiPolicy] Step {self._step_count}: action shape={action.shape}, scaled_actions shape={scaled_actions_mujoco.shape}")
        
        return scaled_actions_mujoco
    
    def __call__(self, env_data, ctrl_data) -> dict:
        """
        Main policy interface for RoboJuDo.
        
        The get_action() method returns actions already converted to MuJoCo order.
        
        Args:
            env_data: Environment state
            ctrl_data: Controller data
        
        Returns:
            Dictionary with 'target_q' for joint position targets (in MuJoCo order)
        """
        # Note: step_count is incremented in get_observation()
        obs_dict, extras = self.get_observation(env_data, ctrl_data)
        actions_mujoco = self.get_action(obs_dict)  # Actions already in MuJoCo order (23 DOFs)
        
        # Note: Warmup action clamping is now done in get_action()
        
        # Compute target joint positions (in MuJoCo order)
        # self.default_pos is already in MuJoCo order (from config)
        target_q = self.default_pos.copy() + actions_mujoco
        
        # Debug first few steps
        if self.step_count <= 3:
            logger.info(f"[HdmiPolicy] __call__ step {self.step_count}:")
            logger.info(f"  actions_mujoco[:6]={actions_mujoco[:6]}")
            logger.info(f"  target_q[:6]={target_q[:6]}")
        
        return {"target_q": target_q}
