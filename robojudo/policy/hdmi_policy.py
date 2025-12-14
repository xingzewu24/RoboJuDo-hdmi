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

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import HdmiPolicyCfg

logger = logging.getLogger(__name__)


def quat_rotate_inverse_numpy(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion.
    
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
    
    Key observations:
    - projected_gravity_b: Gravity vector in body frame (3,)
    - root_ang_vel_b: Base angular velocity in body frame (3,)
    - joint_pos: Joint positions relative to default (num_dofs,)
    - joint_vel: Joint velocities (num_dofs,)
    - prev_actions: Previous action outputs (num_actions,)
    
    Additional observations may be required depending on the checkpoint:
    - object_pos_b: Object position relative to robot (for manipulation tasks)
    - ref_contact_pos_b: Reference contact positions (for manipulation)
    """
    
    cfg_policy: HdmiPolicyCfg
    
    # Default door position for G1PushDoorHand task (3m in front of robot)
    DOOR_POSITION_WORLD = np.array([3.0, 0.0, 0.0])
    
    # Warmup configuration: robot stands still for first N steps (~2 seconds at 50Hz)
    WARMUP_STEPS = 100  # 2 seconds at 50Hz
    
    # G1 standing default pose based on HDMI checkpoint (from yaml: default_joint_pos)
    # Order: matches HDMI policy_joint_names (23 joints, Isaac order)
    G1_HDMI_DEFAULT_POSE_ISAAC_ORDER = {
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
    }
    
    def __init__(self, cfg_policy: HdmiPolicyCfg, device: str = "cpu"):
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
        # Fallback single scale (used if action_scales shape doesn't match)
        self.action_scale = 0.5
        
        # Load motion data for reference trajectory
        self._load_motion_data()
        
        logger.info(f"[HdmiPolicy] Initialized with {self.num_dofs} obs dofs, {self.num_actions} action dofs")
        logger.info(f"[HdmiPolicy] Default standing pose (first 6): {self.default_dof_pos[:6]}")
        logger.info(f"[HdmiPolicy] Door position: {self.door_pos_world}")
        logger.info(f"[HdmiPolicy] Warmup steps: {self.WARMUP_STEPS} (~{self.WARMUP_STEPS/50:.1f}s)")
    
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
        """Setup default standing pose from HDMI checkpoint config or class defaults."""
        # Get joint names from observation dof config
        obs_joint_names = self.cfg_obs_dof.joint_names
        
        # Build default pose array in observation order
        hdmi_defaults = self.G1_HDMI_DEFAULT_POSE_ISAAC_ORDER
        
        # If we have yaml config, extract default_joint_pos from it
        if self.yaml_config and "default_joint_pos" in self.yaml_config:
            hdmi_defaults = {}
            yaml_defaults = self.yaml_config["default_joint_pos"]
            
            # Parse regex-style defaults from yaml (e.g., ".*_hip_pitch_joint": -0.312)
            import re
            for joint_name in obs_joint_names:
                for pattern, value in yaml_defaults.items():
                    if re.match(pattern, joint_name):
                        hdmi_defaults[joint_name] = value
                        break
        
        # Build default pose array
        new_default_pos = []
        for joint_name in obs_joint_names:
            if joint_name in hdmi_defaults:
                new_default_pos.append(hdmi_defaults[joint_name])
            else:
                # Use 0.0 for joints not in the HDMI defaults (e.g., wrist joints)
                new_default_pos.append(0.0)
        
        self.default_dof_pos = np.array(new_default_pos, dtype=np.float32)
        logger.info(f"[HdmiPolicy] Setup HDMI default standing pose ({len(self.default_dof_pos)} dofs): {self.default_dof_pos[:6]}...")
        logger.info(f"[HdmiPolicy] Full default pose: {self.default_dof_pos}")

    
    def reset(self):
        """Reset policy state between episodes."""
        self.adapt_hx[:] = 0.0
        self.prev_actions_buffer[:] = 0.0
        self.prev_actions_history = [np.zeros(self.num_actions, dtype=np.float32) for _ in range(3)]
        self.last_action = np.zeros(self.num_actions)
        self.motion_step = 0
        self.step_count = 0
        logger.info("[HdmiPolicy] Reset: starting warmup period")
    
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
        
        # Get yaw-only quaternion for computing body-frame coordinates
        # Extract yaw from quaternion (simplified: assume flat ground)
        yaw_quat = base_quat.copy()
        # Zero out pitch and roll (keep only yaw rotation around Z)
        # For a yaw-only quaternion: q = [cos(yaw/2), 0, 0, sin(yaw/2)]
        # We can extract this by normalizing the w and z components
        norm = np.sqrt(yaw_quat[0]**2 + yaw_quat[3]**2) + 1e-8
        yaw_quat = np.array([yaw_quat[0]/norm, 0.0, 0.0, yaw_quat[3]/norm])
        
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
        
        # Get default standing pose for action joints
        action_joint_names = self.cfg_policy.action_dof.joint_names
        default_action_pos = []
        for jname in action_joint_names:
            if jname in self.G1_HDMI_DEFAULT_POSE_ISAAC_ORDER:
                default_action_pos.append(self.G1_HDMI_DEFAULT_POSE_ISAAC_ORDER[jname])
            else:
                default_action_pos.append(0.0)
        default_action_pos = np.array(default_action_pos, dtype=np.float32)
        
        # Check if we're in warmup period
        is_warmup = self.step_count < self.WARMUP_STEPS
        
        if is_warmup:
            # WARMUP MODE: Stand still with default pose
            # ref_body_pos_future_local: zeros (stay in place)
            # ref_joint_pos_future: default standing pose
            motion_phase = 0.0
            
            # Fill 5 future timesteps with standing pose reference
            for t in range(5):
                start_idx = 240 + t * 23
                command_obs[start_idx:start_idx + 23] = default_action_pos
            
            if self.step_count % 50 == 0:
                logger.info(f"[HdmiPolicy] Warmup: {self.step_count}/{self.WARMUP_STEPS} steps")
        else:
            # ACTIVE MODE: Use motion trajectory for walking
            motion_t = self.step_count - self.WARMUP_STEPS
            
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
        
        Args:
            env_data: Environment state data (dof_pos, dof_vel, base_quat, etc.)
            ctrl_data: Controller data (commands, etc.)
        
        Returns:
            Tuple of (observation_dict, extras_dict)
        """
        # Extract state from env_data
        dof_pos = env_data.dof_pos  # (num_obs_dofs,) = (29,) for G1
        dof_vel = env_data.dof_vel  # (num_obs_dofs,) = (29,)
        base_quat = env_data.torso_quat  # (4,) in (w, x, y, z) format
        base_ang_vel = env_data.base_ang_vel  # (3,) in body frame
        base_pos = getattr(env_data, 'base_pos', np.array([0.0, 0.0, 0.793]))  # Default standing height
        
        # Compute common observations with scaling
        obs_projected_gravity = self._compute_projected_gravity(base_quat)  # (3,)
        obs_ang_vel = base_ang_vel * self.obs_scales["ang_vel"]  # (3,) scaled
        
        # CRITICAL: Normalize joint positions relative to default pose
        # Raw dof_pos causes extreme policy outputs!
        if dof_pos.shape != self.default_dof_pos.shape:
            logger.error(f"[HdmiPolicy] Shape mismatch! dof_pos={dof_pos.shape}, default={self.default_dof_pos.shape}")
        obs_dof_pos = (dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"]  # relative to default (29,)
        obs_dof_vel = dof_vel * self.obs_scales["dof_vel"]  # (29,) scaled
        
        # Debug logging (first few steps only)
        if self.step_count < 5:
            logger.info(f"[HdmiPolicy] Step {self.step_count}: RAW dof_pos[:6]={dof_pos[:6]}")
            logger.info(f"[HdmiPolicy] Step {self.step_count}: RAW dof_vel[:6]={dof_vel[:6]}")
            logger.info(f"[HdmiPolicy] Step {self.step_count}: default_dof_pos[:6]={self.default_dof_pos[:6]}")
            logger.info(f"[HdmiPolicy] Step {self.step_count}: NORMALIZED obs_dof_pos[:6]={obs_dof_pos[:6]} (should be near 0 if standing)")
            logger.info(f"[HdmiPolicy] Step {self.step_count}: base_ang_vel={base_ang_vel}, gravity={obs_projected_gravity}")
        
        # HDMI uses history of prev_actions (3 steps)
        obs_prev_actions = np.concatenate(self.prev_actions_history).astype(np.float32)  # (23*3=69,)
        
        # Build 'policy' observation group (proprioceptive data)
        # Based on yaml config:
        # - projected_gravity_history (history_steps: [0]) -> 3
        # - root_ang_vel_history (history_steps: [0]) -> 3
        # - joint_pos_history (history_steps: [0,1,2,3,4,8]) -> 29*6 = 174  
        # - prev_actions (steps: 3) -> 23*3 = 69
        # Total = 3 + 3 + 174 + 69 = 249
        
        # For now, we use current observations for all history steps
        # This is a simplification; full implementation would track observation history
        joint_pos_history = np.tile(obs_dof_pos, 6)  # 6 history steps -> 174
        
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
            
        if self.safety_mode:
            # Reduce action magnitude when in safety mode (no tracking data)
            effective_scales = effective_scales * self.safety_action_scale
        
        scaled_actions = action * effective_scales
        
        # Debug: print shapes periodically
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 0
        
        if self._step_count % 100 == 0:
            logger.debug(f"[HdmiPolicy] Step {self._step_count}: action shape={action.shape}, scaled_actions shape={scaled_actions.shape}")
        
        return scaled_actions
    
    def __call__(self, env_data, ctrl_data) -> dict:
        """
        Main policy interface for RoboJuDo.
        
        Args:
            env_data: Environment state
            ctrl_data: Controller data
        
        Returns:
            Dictionary with 'target_q' for joint position targets
        """
        # Increment step counter for warmup logic
        self.step_count += 1
        
        obs_dict, extras = self.get_observation(env_data, ctrl_data)
        actions = self.get_action(obs_dict)
        
        # Compute target joint positions
        # NOTE: Use self.default_pos (action-space, 23 DOFs) NOT self.default_dof_pos (obs-space, 29 DOFs)
        target_q = self.default_pos.copy()
        
        # During warmup, dampen actions further for stability
        if self.step_count <= self.WARMUP_STEPS:
            # Clamp actions to small range during warmup
            actions = np.clip(actions, -0.1, 0.1)
        
        target_q += actions
        
        # Debug first few steps
        if self.step_count <= 3:
            logger.info(f"[HdmiPolicy] __call__ step {self.step_count}: actions[:6]={actions[:6]}, target_q[:6]={target_q[:6]}")
        
        return {
            "target_q": target_q,
            "extras": extras,
        }
