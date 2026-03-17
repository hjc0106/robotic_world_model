# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from mbrl.rl.rsl_rl.rl_cfg import RslRlAmpPpoAlgorithmCfg, RslRlPpoActorCriticWmpCfg

ACTION_DIM = 12
# proprioception ang_vel（3）、gravity（3）、command（3）、dof_pos（12）、dof_vel（12）
PROP_DIM = 33
FOOT_HEIGHT_DIM = 187
FORWARD_HEIGHT_SCAN_DIM = 525
# friction（1）、restitution（1）、base_mass（1）、com_pos（3）、gains（24）、contact_force（12）、contact_flag（8）、line_vel（3）
PRIVILEGED_DIM = 53
NUM_OBS = PRIVILEGED_DIM + PROP_DIM + ACTION_DIM + FOOT_HEIGHT_DIM
# ang_vel（3）、gravity（3）、dof_pos（12）、dof_vel（12)、action（12）
HISTORY_DIM_PER_STEP = NUM_OBS - PRIVILEGED_DIM - FOOT_HEIGHT_DIM - 3
HISTORY_INTERVAL = 5
UPDATE_INTERVAL = 5


@configclass
class UnitreeA1RoughWMPPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = "WMPOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_a1_rough_wmp_ppo"
    policy = RslRlPpoActorCriticWmpCfg(
        class_name="ActorCriticWMP",
        encoder_hidden_dims=[256, 128],
        wm_encoder_hidden_dims = [64, 64],
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
        fixed_std=False,
        latent_dim = 32 + 3, 
        height_dim = FOOT_HEIGHT_DIM,
        privileged_dim = PRIVILEGED_DIM,
        history_dim_per_step = HISTORY_DIM_PER_STEP,
        history_interval = HISTORY_INTERVAL,
        wm_feature_dim = 512,
        wm_latent_dim = 32,
    )
    algorithm = RslRlAmpPpoAlgorithmCfg(
        class_name="WMPPPO",
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
        "num_preload_transition": 2000000,
        "observation_dim": 43,
        "reward_coef": 0.5 * 0.02,
        "discr_hidden_dims": [1024, 512],
        "task_reward_lerp": 0.3,
    }
    depth_predictor = {
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "training_interval": 10,
        "training_iters": 1000,
        "batch_size": 1024,
        "loss_scale": 100,
    }
    base = {
        "env": {
            "camera_sampler": "None",
            "camera_sampler_params": {
                "camera_num_envs": 512,
            },
            "update_interval": UPDATE_INTERVAL,
            "num_actions": ACTION_DIM,
            "prop_dim": PROP_DIM,
            "num_obs": NUM_OBS,
            "privileged_dim": PRIVILEGED_DIM,
            "height_dim": FOOT_HEIGHT_DIM,
            "forward_height_dim": FORWARD_HEIGHT_SCAN_DIM,
            "history_dim_per_step": HISTORY_DIM_PER_STEP,
            "resized": (64, 64),
            "wm_feature_dim": 512,
            "use_camera": True
        },
        "world_model": {
            "config_file": "source/rsl_rl_rwm/rsl_rl/modules/wm_configs/configs.yaml",
            "train_start_steps": 50,
            "train_steps_per_iter": 10,
            "batch_size": 16,
            "batch_length": 64,
        }
    }


@configclass
class UnitreeA1FlatAMPPPORunnerCfg(UnitreeA1RoughWMPPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.max_iterations = 5000
        self.experiment_name = "unitree_a1_flat_amp_ppo"
