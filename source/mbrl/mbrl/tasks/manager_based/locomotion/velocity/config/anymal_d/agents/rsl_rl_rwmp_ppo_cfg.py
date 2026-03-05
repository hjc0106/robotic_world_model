# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from sympy import AccumBounds
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg import AnymalDFlatPPORunnerCfg
from mbrl.rl.rsl_rl import (
    RslRlRwmpSystemDynamicsCfg,
    RslRlNormalizerCfg,
    RslRlMbrlImaginationCfg,
    RslRlRwmpPpoAlgorithmCfg,
    RslRlPpoActorCriticWmpCfg
)

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
class AnymalDFlatRWMPPPOPretrainRunnerCfg(AnymalDFlatPPORunnerCfg):
    class_name: str = "RWMPOnPolicyRunner"
    
    base = {
        "env": {
            "update_interval": UPDATE_INTERVAL,
            "num_actions": ACTION_DIM,
            "prop_dim": PROP_DIM,
            "privileged_dim": PRIVILEGED_DIM,
            "height_dim": FOOT_HEIGHT_DIM,
            "forward_height_dim": FORWARD_HEIGHT_SCAN_DIM,
            "history_dim_per_step": HISTORY_DIM_PER_STEP,
            "resized": (64, 64),
            "wm_feature_dim": 512,
        },
        "world_model": {
            "train_start_steps": 1000,
            "train_steps_per_iter": 10,
            "batch_size": 16,
            "batch_length": 64,
        }
    }

    policy = RslRlPpoActorCriticWmpCfg(
        class_name="ActorCriticWMP",
        encoder_hidden_dims=[256, 128],
        wm_encoder_hidden_dims = [64, 64],
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

    system_dynamics = RslRlRwmpSystemDynamicsCfg(
        ensemble_size=1,
        history_horizon=32,
        encoder_config = {
            "shapes": {
                "image": (64, 64, 1),
                "prop": (PROP_DIM,),
            },
            "mlp_keys": ".*",
            "cnn_keys": "image",
            "act": 'SiLU',
            "norm": True,
            "cnn_depth": 32,
            "kernel_size": 4,
            "minres": 4,
            "mlp_layers": 5,
            "mlp_units": 1024,
            "symlog_inputs": True,
            "use_camera": True,
        },
        head_config = {
            "grad_heads": ["decoder", "reward"],
            "loss_scale": {
                "reward": 0.0,
                "image": 1.0
            },
            "decoder": {
                "feat_size": 32*32 + 512,
                "shapes": {
                    "image": (64, 64, 1),
                    "prop": (PROP_DIM,),
                },
                "mlp_keys": ".*",
                "cnn_keys": "image",
                "act": 'SiLU',
                "norm": True,
                "cnn_depth": 32,
                "kernel_size": 4,
                "minres": 4,
                "mlp_layers": 5,
                "mlp_units": 1024,
                "cnn_sigmoid": False,
                "image_dist": "mse",
                "vector_dist": "symlog_mse",
                "outscale": 1.0,
                "use_camera": True,
            },
            "reward": {
                "inp_dim": 32*32 + 512,
                "shape": (255,),
                "layers": 2,
                "units": 512,
                "act": 'SiLU',
                "norm": True,
                "dist": "symlog_disc",
                "std": 1.0,
                "min_std": 0.1,
                "max_std": 1.0,
                "absmax": None,
                "temp": 0.1,
                "unimix_ratio": 0.01,
                "outscale": 0.0,
                "symlog_inputs": False,
                "name": "Reward",
            }
        },
        dynamics_config = {
            'type': 'rssm',
            "stoch": 32,
            "deter": 512,
            "hidden": 512,
            "rec_depth": 1,
            "discrete": 32,
            "act": 'SiLU',
            "norm": True,
            "mean_act": 'none',
            "std_act": 'sigmoid2',
            "min_std": 0.1,
            "unimix_ratio": 0.01,
            "initial": 'learned',
            "num_actions": ACTION_DIM * UPDATE_INTERVAL,
            "embed": 5120,  # 4096+1024
            "kl_free": 1.0,
            "dyn_scale": 0.5,
            "rep_scale": 0.1
        },
        forecast_config = {
            "type": "rnn",
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 256,
            "state_mean_shape": [128],
            "state_logstd_shape": [128],
            "extension_shape": [128],
            "contact_shape": [128],
            "termination_shape": [128],
        },
        freeze_auxiliary=False,
    )
    imagination = RslRlMbrlImaginationCfg(
        num_envs=0,
        num_steps_per_env=0,
        max_episode_length=0,
        command_resample_interval_range=None,
        uncertainty_penalty_weight=-0.0,
        state_normalizer=RslRlNormalizerCfg(
            mean=[
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, -1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                -2.0, -2.0, 2.0, 2.0, -6.0, 8.0, -6.0, 8.0, 12.0, -12.0, 12.0, -12.0,
                ],
            std=[
                0.5, 0.5, 0.1,
                0.3, 0.3, 0.5,
                0.02, 0.02, 0.04,
                0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
                1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5,
                15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
                ],
        ),
        action_normalizer=RslRlNormalizerCfg(
            mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
    )
    algorithm = RslRlRwmpPpoAlgorithmCfg(
        vel_predict_coef=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        policy_learning_rate=1.0e-3,
        system_dynamics_learning_rate=1.0e-3,
        system_dynamics_weight_decay=0.0,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        system_dynamics_forecast_horizon=8,
        system_dynamics_loss_weights={"state": 1.0, "sequence": 1.0, "bound": 1.0, "kl": 0.1, "extension": 1.0, "contact": 1.0, "termination": 1.0},
        system_dynamics_num_mini_batches=20,
        system_dynamics_mini_batch_size=5000,
        system_dynamics_replay_buffer_size=1000,
        system_dynamics_num_eval_trajectories=100,
        system_dynamics_len_eval_trajectory=400,
        system_dynamics_eval_traj_noise_scale=[0.1, 0.2, 0.4, 0.5, 0.8],
        # world model
        wm_learning_rate=1.0e-3,
        wm_weight_decay=0.0,
        wm_opt_eps=1e-8,
        wm_opt_type="adam",
        wm_use_amp=False,
        wm_max_grad_norm=1000,
    )

    depth_predictor = {
        "training_interval": 10,
        "training_iters": 1000,
        "batch_size": 1024,
        "loss_scale": 100,
        "model": {
            "forward_heightmap_dim": 525,
            "prop_dim": 33,
            "depth_image_dims": [64, 64],
            "encoder_hidden_dims": [256, 128],
            "depth": 32,
            "act": "ELU",
            "norm": True,
            "kernel_size": 4,
            "minres": 4,
            "outscale": 1.0,
            "cnn_sigmoid": False,
        },
        "optimizer": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
        }

    }
    run_name = "pretrain"
    load_system_dynamics = False
    system_dynamics_load_path = None
    system_dynamics_warmup_iterations = 0
    system_dynamics_num_visualizations = 4
    system_dynamics_state_idx_dict = {
        r"$v$\n$[m/s]$": [0, 1, 2],
        r"$\omega$\n$[rad/s]$": [3, 4, 5],
        r"$g$\n$[1]$": [6, 7, 8],
        r"$q$\n$[rad]$": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        r"$\dot{q}$\n$[rad/s]$": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        r"$\tau$\n$[Nm]$": [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    }
    pca_obs_buf_size = 10000

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 100000
        self.experiment_name = "anymal_d_flat_rwmp"
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [512, 256, 128]

@configclass
class AnymalDFlatRWMPPPOFinetuneRunnerCfg(AnymalDFlatRWMPPPOPretrainRunnerCfg):
    resume = True
    load_run = "2025-11-04_09-59-00"
    load_system_dynamics = True
    system_dynamics_load_path = "logs/rsl_rl/anymal_d_flat/2025-11-04_14-31-20_pretrain_rnn/model_5000.pt"
    system_dynamics_warmup_iterations = 500
    run_name = "finetune"
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # override imagination
        self.imagination.num_envs = 8192
        self.imagination.num_steps_per_env = 24
        self.imagination.max_episode_length = 256
        self.imagination.command_resample_interval_range = [100, 120]
        self.imagination.uncertainty_penalty_weight = -0.0


@configclass
class AnymalDFlatRWMPPPOVisualizeRunnerCfg(AnymalDFlatRWMPPPOPretrainRunnerCfg):
    resume = True
    load_system_dynamics = True
    system_dynamics_load_path = "logs/rsl_rl/anymal_d_flat/2025-11-04_14-31-20_pretrain_rnn/model_5000.pt"
    run_name = "visualize"
