from robojudo.config import ASSETS_DIR
from robojudo.environment.env_cfgs import EnvCfg
from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg


class G1_29DoF(DoFConfig):
    # num_dofs as 29
    joint_names: list[str] = [
        *[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ],
        *[
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
        *["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        *[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
        *[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    ]
    default_pos: list[float] | None = [
        *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
        *[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
        *[0, 0, 0],
        *[0, 0, 0, 0, 0, 0, 0],
        *[0, 0, 0, 0, 0, 0, 0],
    ]

    stiffness: list[float] | None = [
        *[100, 100, 100, 150, 40, 40],
        *[100, 100, 100, 150, 40, 40],
        *[200, 200, 200],
        *[40, 40, 40, 40, 20, 20, 20],
        *[40, 40, 40, 40, 20, 20, 20],
    ]

    damping: list[float] | None = [
        *[5, 5, 5, 5, 2, 2],
        *[5, 5, 5, 5, 2, 2],
        *[6, 6, 6],
        *[2, 2, 2, 2, 2, 2, 2],
        *[2, 2, 2, 2, 2, 2, 2],
    ]

    torque_limits: list[float] | None = [
        *[200, 200, 200, 300, 40, 40],
        *[200, 200, 200, 300, 40, 40],
        *[200, 200, 200],
        *[40, 40, 18, 18, 10, 10, 10],
        *[40, 40, 18, 18, 10, 10, 10],
    ]

    position_limits: list[list[float]] | None = [
        *[
            [-2.5307, 2.8798],
            [-0.5236, 2.9671],
            [-2.7576, 2.7576],
            [-0.087267, 2.8798],
            [-0.87267, 0.5236],
            [-0.2618, 0.2618],
        ],
        *[
            [-2.5307, 2.8798],
            [-2.9671, 0.5236],
            [-2.7576, 2.7576],
            [-0.087267, 2.8798],
            [-0.87267, 0.5236],
            [-0.2618, 0.2618],
        ],
        *[[-2.618, 2.618], [-0.52, 0.52], [-0.52, 0.52]],
        *[
            [-3.0892, 2.6704],
            [-1.5882, 2.2515],
            [-2.618, 2.618],
            [-1.0472, 2.0944],
            [-1.972222054, 1.972222054],
            [-1.614429558, 1.614429558],
            [-1.614429558, 1.614429558],
        ],
        *[
            [-3.0892, 2.6704],
            [-2.2515, 1.5882],
            [-2.618, 2.618],
            [-1.0472, 2.0944],
            [-1.972222054, 1.972222054],
            [-1.614429558, 1.614429558],
            [-1.614429558, 1.614429558],
        ],
    ]


class G1_23DoF(G1_29DoF):
    # num_dofs as 23
    _subset: bool = True  # if True, simplely inheritance & pick

    _subset_joint_names: list[str] | None = [
        *[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ],
        *[
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
        *["waist_yaw_joint"],
        *[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
        ],
        *[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ],
    ]


class G1_12DoF(G1_29DoF):
    # num_dofs as 12
    _subset: bool = True  # if True, simplely inheritance & pick

    _subset_joint_names: list[str] | None = [
        *[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ],
        *[
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
    ]


class G1EnvCfg(EnvCfg):
    xml: str = (ASSETS_DIR / "robots/g1/g1_29dof_rev_1_0.xml").as_posix()

    dof: DoFConfig = G1_29DoF()

    forward_kinematic: ForwardKinematicCfg | None = ForwardKinematicCfg(
        xml_path=xml,
        debug_viz=False,
        kinematic_joint_names=dof.joint_names,
    )
    update_with_fk: bool = True
    torso_name: str = "torso_link"


class G1_23EnvCfg(EnvCfg):
    xml: str = (ASSETS_DIR / "robots/g1/g1_23dof_rev_1_0.xml").as_posix()

    dof: DoFConfig = G1_23DoF()

    forward_kinematic: ForwardKinematicCfg | None = ForwardKinematicCfg(
        xml_path=xml,
        debug_viz=False,
        kinematic_joint_names=dof.joint_names,
    )
    update_with_fk: bool = True
    torso_name: str = "torso_link"


class G1_12EnvCfg(EnvCfg):
    xml: str = (ASSETS_DIR / "robots/g1/g1_12dof.xml").as_posix()

    dof: DoFConfig = G1_12DoF()

    forward_kinematic: ForwardKinematicCfg | None = ForwardKinematicCfg(
        xml_path=xml,
        debug_viz=False,
        kinematic_joint_names=dof.joint_names,
    )
    update_with_fk: bool = False
    torso_name: str = "pelvis"  # no torso in 12dof model


class G1HdmiDoF(G1_29DoF):
    """G1 DoF configuration for HDMI policies with proper standing pose and PD gains."""
    # Override default_pos with HDMI standing pose from checkpoint yaml
    default_pos: list[float] | None = [
        # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        *[-0.312, 0.0, 0.0, 0.669, -0.363, 0.0],
        # Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll  
        *[-0.312, 0.0, 0.0, 0.669, -0.363, 0.0],
        # Waist: yaw, roll, pitch
        *[0.0, 0.0, 0.0],
        # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
        *[0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
        # Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
        *[0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
    ]
    
    # Ground truth stiffness (Kp) from HDMI yaml: policy-xg6644nr-final.yaml
    # Order: left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7) = 29 total
    stiffness: list[float] | None = [
        # Left leg: hip_pitch=40.18, hip_roll=99.10, hip_yaw=40.18, knee=99.10, ankle_pitch=28.50, ankle_roll=28.50
        *[40.18, 99.10, 40.18, 99.10, 28.50, 28.50],
        # Right leg: same as left
        *[40.18, 99.10, 40.18, 99.10, 28.50, 28.50],
        # Waist: yaw=40.18, roll=28.50, pitch=28.50
        *[40.18, 28.50, 28.50],
        # Left arm: shoulder_pitch=14.25, shoulder_roll=14.25, shoulder_yaw=14.25, elbow=14.25, wrist_roll=14.25, wrist_pitch=16.78, wrist_yaw=16.78
        *[14.25, 14.25, 14.25, 14.25, 14.25, 16.78, 16.78],
        # Right arm: same as left
        *[14.25, 14.25, 14.25, 14.25, 14.25, 16.78, 16.78],
    ]
    
    # Ground truth damping (Kd) from HDMI yaml
    damping: list[float] | None = [
        # Left leg: hip_pitch=2.56, hip_roll=6.31, hip_yaw=2.56, knee=6.31, ankle_pitch=1.81, ankle_roll=1.81
        *[2.56, 6.31, 2.56, 6.31, 1.81, 1.81],
        # Right leg: same as left
        *[2.56, 6.31, 2.56, 6.31, 1.81, 1.81],
        # Waist: yaw=2.56, roll=1.81, pitch=1.81
        *[2.56, 1.81, 1.81],
        # Left arm: shoulder_pitch=0.91, shoulder_roll=0.91, shoulder_yaw=0.91, elbow=0.91, wrist_roll=0.91, wrist_pitch=1.07, wrist_yaw=1.07
        *[0.91, 0.91, 0.91, 0.91, 0.91, 1.07, 1.07],
        # Right arm: same as left
        *[0.91, 0.91, 0.91, 0.91, 0.91, 1.07, 1.07],
    ]


class G1PushDoorEnvCfg(EnvCfg):
    """G1 environment with door for push door task (HDMI G1PushDoorHand)."""
    xml: str = (ASSETS_DIR / "scenes/g1_pushdoor.xml").as_posix()

    # Use HDMI-specific DoF with proper standing pose
    dof: DoFConfig = G1HdmiDoF()

    forward_kinematic: ForwardKinematicCfg | None = ForwardKinematicCfg(
        xml_path=xml,
        debug_viz=False,
        kinematic_joint_names=dof.joint_names,
    )
    update_with_fk: bool = True
    torso_name: str = "torso_link"
