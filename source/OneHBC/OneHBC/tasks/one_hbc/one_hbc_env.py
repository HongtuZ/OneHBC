from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
from .one_hbc_env_cfg import OneHBCEnvCfg
from OneHBC.utils.motion_loader import MotionLoader


class OneHBCEnv(ManagerBasedRLEnv):
    cfg: OneHBCEnvCfg

    def __init__(self, cfg: OneHBCEnvCfg, render_mode: str | None = None, **kwargs):
        self.motion_loader = MotionLoader(
            cfg.motion_loader.motion_data_dir,
            cfg.motion_loader.motion_data_weights,
            device=cfg.sim.device,
        )
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
