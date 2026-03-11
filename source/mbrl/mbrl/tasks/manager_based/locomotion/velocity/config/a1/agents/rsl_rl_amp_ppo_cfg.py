# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from mbrl.rl.rsl_rl.rl_cfg import RslRlAmpPpoAlgorithmCfg

@configclass
class UnitreeA1RoughAMPPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = "AMPOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_a1_rough_amp_ppo"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlAmpPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    amp = {
        "num_preload_transition": 20000,
        "observation_dim": 43,
        "reward_coef": 1.0,
        "discr_hidden_dims": [128, 64],
        "task_reward_lerp": 0.5,
    }


@configclass
class UnitreeA1FlatAMPPPORunnerCfg(UnitreeA1RoughAMPPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.max_iterations = 5000
        self.experiment_name = "unitree_a1_flat_amp_ppo"
