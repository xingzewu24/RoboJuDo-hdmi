from robojudo.environment.env_cfgs import MujocoEnvCfg

from .g1_env_cfg import G1_12EnvCfg, G1_23EnvCfg, G1EnvCfg, G1PushDoorEnvCfg


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
