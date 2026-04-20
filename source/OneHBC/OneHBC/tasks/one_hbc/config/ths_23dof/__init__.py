# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="OneHBC-RL-Rough-THS23Dof-v0",
    entry_point="OneHBC.tasks.one_hbc.one_hbc_env:OneHBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_env_cfg:RLRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RLRoughPPORunnerCfg",
    },
)


gym.register(
    id="OneHBC-RL-Rough-THS23Dof-Play-v0",
    entry_point="OneHBC.tasks.one_hbc.one_hbc_env:OneHBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_env_cfg:RLRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RLRoughPPORunnerCfg",
    },
)


gym.register(
    id="OneHBC-RL-Flat-THS23Dof-v0",
    entry_point="OneHBC.tasks.one_hbc.one_hbc_env:OneHBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_env_cfg:RLFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RLFlatPPORunnerCfg",
    },
)


gym.register(
    id="OneHBC-RL-Flat-THS23Dof-Play-v0",
    entry_point="OneHBC.tasks.one_hbc.one_hbc_env:OneHBCEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_env_cfg:RLFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RLFlatPPORunnerCfg",
    },
)
