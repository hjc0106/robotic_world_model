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
    id="MBRL-Velocity-Flat-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
    },
)

gym.register(
    id="MBRL-Velocity-Rough-Unitree-A1-WMP-v0",
    entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.a1.envs.unitree_a1_manager_based_mbrl_env:UnitreeA1ManagerBasedAMPRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wm_flat_env_cfg:UnitreeA1RoughEnvCfg_WMP",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_wmp_ppo_cfg:UnitreeA1RoughWMPPPORunnerCfg",
    },
)

gym.register(
    id="MBRL-Velocity-Flat-Unitree-A1-WMP-v0",
    entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.a1.envs.unitree_a1_manager_based_mbrl_env:UnitreeA1ManagerBasedAMPRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wm_flat_env_cfg:UnitreeA1FlatEnvCfg_WMP",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_wmp_ppo_cfg:UnitreeA1RoughWMPPPORunnerCfg",
    },
)

gym.register(
    id="MBRL-Velocity-Flat-Unitree-A1-AMP-v0",
    entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.a1.envs.unitree_a1_manager_based_mbrl_env:UnitreeA1ManagerBasedAMPRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg_AMP",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_ppo_cfg:UnitreeA1FlatAMPPPORunnerCfg",
    },
)

gym.register(
    id="MBRL-Velocity-Flat-Unitree-A1-AMP-Play-v0",
    entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.a1.envs.unitree_a1_manager_based_mbrl_env:UnitreeA1ManagerBasedAMPRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg_AMP_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_ppo_cfg:UnitreeA1FlatAMPPPORunnerCfg",
    },
)

gym.register(
    id="MBRL-Velocity-Flat-Unitree-A1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
    },
)

gym.register(
    id="MBRL-Velocity-Rough-Unitree-A1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
    },
)

gym.register(
    id="MBRL-Velocity-Rough-Unitree-A1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
    },
)
