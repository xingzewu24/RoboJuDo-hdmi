import logging
import time

import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]

        # Build joint index maps so dof_pos/dof_vel match cfg_env.dof.joint_names.
        # Do NOT assume the last N qpos/qvel correspond to the configured joints.
        self._dof_qpos_adr: np.ndarray | None = None
        self._dof_qvel_adr: np.ndarray | None = None
        self._rebuild_dof_index_maps()
        
        # Initialize robot to keyframe 0 if available (e.g., hdmi_standing pose)
        # This ensures proper initial joint positions for policy stability
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            logger.info(f"[MujocoEnv] Initialized to keyframe 0 (nkey={self.model.nkey})")
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
            diable_key_callbacks=True,
        )

        # === Default camera ===
        # Use a *free* camera (movable with mouse), but initialize it to a side-view-like pose
        # for the push-door scene so the door doesn't block the robot.
        self._auto_lookat = True
        if str(cfg_env.xml).endswith("g1_pushdoor.xml"):
            # Side view: stand on +/-Y side, look towards door/robot area.
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.azimuth = 90.0
            self.viewer.cam.elevation = -15.0
            self.viewer.cam.distance = 4.0
            self.viewer.cam.lookat = np.array([0.6, 0.0, 1.0], dtype=np.float32)
            # Don't override lookat every step, otherwise user panning feels "stuck".
            self._auto_lookat = False
            logger.info("[MujocoEnv] PushDoor: initialized free camera to side view (movable)")
        else:
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -10.0
            self.viewer.cam.azimuth = 180.0
        # self.viewer._paused = True

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()

        # Diagnostics
        self._step_counter = 0

        self.update()  # get initial state

    def update_dof_cfg(self, override_cfg=None):
        super().update_dof_cfg(override_cfg=override_cfg)
        # If model already exists (i.e., after __init__), refresh index maps.
        if hasattr(self, "model") and self.model is not None:
            self._rebuild_dof_index_maps()

    def _rebuild_dof_index_maps(self):
        qpos_adrs: list[int] = []
        qvel_adrs: list[int] = []
        for joint_name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)  # pyright: ignore[reportAttributeAccessIssue]
            if jid < 0:
                raise ValueError(f"[MujocoEnv] Joint '{joint_name}' not found in model")
            qpos_adrs.append(int(self.model.jnt_qposadr[jid]))
            qvel_adrs.append(int(self.model.jnt_dofadr[jid]))

        self._dof_qpos_adr = np.asarray(qpos_adrs, dtype=np.int32)
        self._dof_qvel_adr = np.asarray(qvel_adrs, dtype=np.int32)

    def reborn(self, init_qpos=None, init_joint_pos=None):
        """Reset the robot to initial state.
        
        Args:
            init_qpos: Optional base pose [x, y, z, qw, qx, qy, qz] (7 values)
            init_joint_pos: Optional joint positions array. If None and keyframe 0 exists,
                            uses keyframe. This allows proper standing pose initialization.
        """
        if init_qpos is not None or init_joint_pos is not None:
            if init_qpos is not None:
                self.data.qpos[0:7] = init_qpos
            if init_joint_pos is not None:
                assert self._dof_qpos_adr is not None
                n_joints = min(len(init_joint_pos), len(self._dof_qpos_adr))
                self.data.qpos[self._dof_qpos_adr[:n_joints]] = init_joint_pos[:n_joints]
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            # Use keyframe 0 if available (e.g., hdmi_standing pose)
            if self.model.nkey > 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                mujoco.mj_resetData(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        assert self._dof_qpos_adr is not None and self._dof_qvel_adr is not None
        dof_pos = self.data.qpos.astype(np.float32)[self._dof_qpos_adr]
        dof_vel = self.data.qvel.astype(np.float32)[self._dof_qvel_adr]

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        # Convert world-frame velocities to body frame (RoboJuDo expects body-frame velocities).
        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        ang_vel = quat_rotate_inverse_np(quat, ang_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        # Optionally keep the robot centered. For PushDoor we disable this so the user
        # can freely move/pan the camera without it snapping back every step.
        if self._auto_lookat:
            self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            self.viewer.render()

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            # Periodic saturation diagnostics (helps debug slow tipping / foot instability).
            # Keep this very lightweight and low-frequency to avoid log spam.
            if self._step_counter % 200 == 0:
                ratio = np.abs(torque) / (self.torque_limits + 1e-9)
                worst_i = int(np.argmax(ratio))
                worst_ratio = float(ratio[worst_i])
                logger.info(
                    f"[MujocoEnv] torque/limit worst={worst_ratio:.2f} on {self.joint_names[worst_i]}"
                )

                sat = ratio >= 0.95
                sat_count = int(np.count_nonzero(sat))
                if sat_count > 0:
                    sat_idx = np.where(sat)[0]
                    joints = [self.joint_names[i] for i in sat_idx[:6]]
                    logger.info(f"[MujocoEnv] near-saturation: {sat_count}/{self.num_dofs}; top={joints}")

            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
            self._step_counter += 1
        self.update(simple=False)

    def shutdown(self):
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
