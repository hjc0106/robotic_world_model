# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlSystemDynamicsCfg:
    """Configuration for the system dynamics networks."""
    
    ensemble_size: int = MISSING
    """The ensemble size of the system dynamics network."""

    history_horizon: int = MISSING
    """The prediction horizon of the system dynamics network."""

    architecture_config: dict = MISSING
    """The architecture configuration of the system dynamics network."""
    
    freeze_auxiliary: bool = MISSING
    """Whether to freeze the auxiliary networks."""


@configclass
class RslRlNormalizerCfg:
    """Configuration for the normalizer."""

    mean: list[float] = MISSING
    """The mean of the normalizer."""

    std: list[float] = MISSING
    """The std of the normalizer."""


@configclass
class RslRlMbrlImaginationCfg:
    """Configuration for the imagination."""
    
    num_envs: int = MISSING
    """The number of environments for the imagination."""
    
    num_steps_per_env: int = MISSING
    """The number of steps for the imagination."""
    
    max_episode_length: int = MISSING
    """The maximum episode length for the imagination."""

    command_resample_interval_range: list[float] | None = MISSING
    """The resample interval range for the command."""
    
    uncertainty_penalty_weight: float = MISSING
    """The weight for the uncertainty penalty."""
    
    state_normalizer: RslRlNormalizerCfg = MISSING
    """The normalizer for the state."""
    
    action_normalizer: RslRlNormalizerCfg = MISSING
    """The normalizer for the action."""


@configclass
class RslRlMbrlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "MBPOPPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    policy_learning_rate: float = MISSING
    """The learning rate for the policy."""

    system_dynamics_learning_rate: float = MISSING
    """The learning rate for the system dynamics."""
    
    system_dynamics_weight_decay: float = MISSING
    """The weight decay for the system dynamics."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""
    
    system_dynamics_forecast_horizon: int = MISSING
    """The forecast horizon for the system dynamics."""
    
    system_dynamics_loss_weights: dict[str, float] = MISSING
    """The loss weights for the system dynamics."""
    
    system_dynamics_num_mini_batches: int = MISSING
    """The number of mini-batches for the system dynamics."""
    
    system_dynamics_mini_batch_size: int = MISSING
    """The mini-batch size for the system dynamics."""
    
    system_dynamics_replay_buffer_size: int = MISSING
    """The replay buffer size for the system dynamics."""
    
    system_dynamics_num_eval_trajectories: int = MISSING
    """The number of evaluation trajectories for the system dynamics."""
    
    system_dynamics_len_eval_trajectory: int = MISSING
    """The length of the evaluation trajectory for the system dynamics."""
    
    system_dynamics_eval_traj_noise_scale: list[float] = MISSING
    """The noise scale for the evaluation trajectory for the system dynamics."""

@configclass
class RslRlPpoActorCriticWmpCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCriticWMP"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation for the policy. Default is False."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the encoder network."""

    wm_encoder_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the world model encoder network."""

    history_interval: int = MISSING
    """The interval of the history."""

    fixed_std: bool = MISSING
    """Whether to use a fixed standard deviation."""

    latent_dim: int = MISSING
    """The dimension of the latent vector."""

    height_dim: int = MISSING
    """The dimension of the height vector."""
    
    privileged_dim: int = MISSING
    """The dimension of the privileged vector."""

    history_dim_per_step: int = MISSING
    """The dimension of the history vector."""

    wm_feature_dim: int = MISSING
    """The dimension of the world model feature vector."""
    
    wm_latent_dim: int = MISSING
    """The dimension of the world model latent vector."""

@configclass
class RslRlRwmpSystemDynamicsCfg:
    """Configuration for the RWMP system dynamics network."""

    ensemble_size: int = MISSING
    """The ensemble size of the system dynamics network."""

    history_horizon: int = MISSING
    """The prediction horizon of the system dynamics network."""

    encoder_config: dict = MISSING
    """The encoder configuration of the system dynamics network."""

    head_config: dict = MISSING
    """The head configuration of the system dynamics network."""

    dynamics_config: dict = MISSING
    """The dynamics configuration of the system dynamics network."""

    forecast_config: dict = MISSING
    """The forecast configuration of the system dynamics network."""

    freeze_auxiliary: bool = MISSING
    """Whether to freeze the auxiliary networks."""

@configclass
class RslRlRwmpPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "RWMPPPO"
    """The algorithm class name. Default is PPO."""

    vel_predict_coef: float = MISSING
    """The coefficient for the linear velocity predict loss."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    policy_learning_rate: float = MISSING
    """The learning rate for the policy."""

    system_dynamics_learning_rate: float = MISSING
    """The learning rate for the MLP-based system dynamics (state/auxiliary prediction)."""

    system_dynamics_weight_decay: float = MISSING
    """The weight decay for the MLP-based system dynamics."""

    wm_learning_rate: float | None = None
    """The learning rate for the RSSM world model (encoder + dynamics + heads).
    Defaults to system_dynamics_learning_rate when None."""

    wm_weight_decay: float | None = None
    """The weight decay for the RSSM world model. Defaults to system_dynamics_weight_decay when None."""

    wm_opt_eps: float = 1e-4
    """The epsilon for the RSSM world model optimizer."""

    wm_opt_type: str = "adam"
    """The optimizer type for the RSSM world model. One of: adam, adamax, sgd, momentum."""

    wm_use_amp: bool = False
    """Whether to use automatic mixed precision for the RSSM world model optimizer."""

    wm_max_grad_norm: float = 1.0
    """The maximum gradient norm for the RSSM world model optimizer."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""
    
    system_dynamics_forecast_horizon: int = MISSING
    """The forecast horizon for the system dynamics."""
    
    system_dynamics_loss_weights: dict[str, float] = MISSING
    """The loss weights for the system dynamics."""
    
    system_dynamics_num_mini_batches: int = MISSING
    """The number of mini-batches for the system dynamics."""
    
    system_dynamics_mini_batch_size: int = MISSING
    """The mini-batch size for the system dynamics."""
    
    system_dynamics_replay_buffer_size: int = MISSING
    """The replay buffer size for the system dynamics."""
    
    system_dynamics_num_eval_trajectories: int = MISSING
    """The number of evaluation trajectories for the system dynamics."""
    
    system_dynamics_len_eval_trajectory: int = MISSING
    """The length of the evaluation trajectory for the system dynamics."""
    
    system_dynamics_eval_traj_noise_scale: list[float] = MISSING
    """The noise scale for the evaluation trajectory for the system dynamics."""
