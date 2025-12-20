from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.tools.tool_cfgs import DoFConfig

from .g1_env_cfg import G1_12EnvCfg, G1_23EnvCfg, G1EnvCfg, G1PushDoorEnvCfg, G1HdmiDoF


class G1HdmiDoFMuJoCo(G1HdmiDoF):
    """MuJoCo-tuned PD gains for HDMI standing/motion.

    HDMI yaml Kp/Kd are calibrated for Isaac-style simulation.
    In MuJoCo, we generally need stronger joint-space PD to reliably hold a pose.
    """

    # NOTE: Don't reference `G1HdmiDoF.stiffness` / `.damping` here.
    # These are Pydantic fields and are not normal class attributes at import time.
    # Explicit values are derived from HDMI yaml gains with a MuJoCo stability scale.

    # Kp scaled by 3.0
    stiffness: list[float] | None = [
        # Left leg
        *[120.54, 297.30, 120.54, 297.30, 85.50, 85.50],
        # Right leg
        *[120.54, 297.30, 120.54, 297.30, 85.50, 85.50],
        # Waist
        *[120.54, 85.50, 85.50],
        # Left arm
        *[42.75, 42.75, 42.75, 42.75, 42.75, 50.34, 50.34],
        # Right arm
        *[42.75, 42.75, 42.75, 42.75, 42.75, 50.34, 50.34],
    ]

    # Kd scaled by 1.7
    damping: list[float] | None = [
        # Left leg
        *[4.352, 10.727, 4.352, 10.727, 3.077, 3.077],
        # Right leg
        *[4.352, 10.727, 4.352, 10.727, 3.077, 3.077],
        # Waist
        *[4.352, 3.077, 3.077],
        # Left arm
        *[1.547, 1.547, 1.547, 1.547, 1.547, 1.819, 1.819],
        # Right arm
        *[1.547, 1.547, 1.547, 1.547, 1.547, 1.819, 1.819],
    ]

    # Torque limits: inherit from base, but allow a bit more ankle authority in MuJoCo.
    # This helps prevent slow tipping when feet need stronger corrective torques.
    torque_limits: list[float] | None = [
        # Left leg (increase ankle_pitch/ankle_roll from 40 -> 60)
        *[200, 200, 200, 300, 60, 60],
        # Right leg
        *[200, 200, 200, 300, 60, 60],
        # Waist
        *[200, 200, 200],
        # Left arm
        *[40, 40, 18, 18, 10, 10, 10],
        # Right arm
        *[40, 40, 18, 18, 10, 10, 10],
    ]


class G1MujocoEnvCfg(G1EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======

    update_with_fk: bool = True


class G1_23MujocoEnvCfg(G1_23EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======
    update_with_fk: bool = True


class G1_12MujocoEnvCfg(G1_12EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======
    update_with_fk: bool = False


class G1PushDoorMujocoEnvCfg(G1PushDoorEnvCfg, MujocoEnvCfg):
    """MuJoCo environment for G1 push door task with door positioned at x=0.6m."""
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======
    update_with_fk: bool = True

    # Stronger MuJoCo PD to improve standing stability
    dof: DoFConfig = G1HdmiDoFMuJoCo()
    
    # Match HDMI policy training frequency (50Hz)
    # MuJoCo dt=0.002s, so decimation=10 -> control_dt=0.02s
    decimation: int = 10
