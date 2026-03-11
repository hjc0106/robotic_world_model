# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from typing import Any
from collections.abc import Sequence
from isaaclab.utils.math import quat_apply
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from rsl_rl.datasets.motion_loader import MotionLoader

@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,  # [N, 12]
    dof_velocities: torch.Tensor,  # [N, 12]
    root_positions: torch.Tensor,  # [N, 3]
    root_rotations: torch.Tensor,  # [N, 4]
    key_body_positions: torch.Tensor,  # [N, 4, 3]
    progress: torch.Tensor,  # [N, 1]
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,  # [N, 12]
            dof_velocities,  # [N, 12]
            root_positions[:, 2:3],  # trunk height [N, 1]
            quaternion_to_tangent_and_normal(root_rotations),  # [N, 6]
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),  # [N, 15]
            progress,  # [N, 1]
        ),
        dim=-1,
    )
    return obs

class UnitreeA1ManagerBasedAMPRLEnv(ManagerBasedRLEnv):


    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.num_actions = cfg.num_actions
        self.reset_strategy = cfg.reset_strategy
        self.robot = self.scene["robot"]
        self._motion_loader = MotionLoader(motion_file=self.cfg.amp_motion_files, expert_mode=False, device=self.device)
        
        key_body_names = [
            "FR_foot",
            "FL_foot",
            "RR_foot",
            "RL_foot",
        ]
        # get the body ids of the feet
        self.foot_body_ids = torch.tensor(
            [self.robot.data.body_names.index(name) for name in key_body_names],
            device=self.device,
        )
        
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # Register AMP-observation indices with the motion loader so that
        # feed_forward_generator() can produce observations in the same format
        # as get_amp_observation().
        # self._motion_loader.configure_amp(
        #     dof_indexes=self.motion_dof_indexes,
        #     ref_body_index=self.motion_ref_body_index,
        #     key_body_indexes=self.motion_key_body_indexes,
        #     time_between_frames=self.physics_dt,
        # )


        self.amp_term_state_buffer = torch.zeros(
            (self.num_envs, self.cfg.amp_observation_space), device=self.device
        )

    # reset strategies
    def _reset_strategy_default(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        robot = self.scene["robot"]
        root_state = robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = robot.data.default_joint_pos[env_ids].clone()
        joint_vel = robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: Sequence[int], start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = len(env_ids)
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # get root transforms (trunk)
        motion_torso_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.05  # lift slightly to avoid ground collision
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        return root_state, dof_pos, dof_vel

    def get_amp_observation(self):      
        
        # joint_pos / joint_vel
        joint_pos = self.robot.data.joint_pos          # (num_envs, num_joints)
        joint_vel = self.robot.data.joint_vel          # (num_envs, num_joints)
        
        # foot position: directly from the physical engine, no FK
        foot_pos_w = self.robot.data.body_pos_w[:, self.foot_body_ids, :]  # (num_envs, num_feet, 3)
        root_pos_w = self.robot.data.root_pos_w                            # (num_envs, 3)
        # convert to body coordinate system (relative to base)
        foot_pos_local = (foot_pos_w - root_pos_w.unsqueeze(1)).reshape(self.num_envs, -1)
        
        # base velocity (in body coordinate system)
        base_lin_vel = self.robot.data.root_lin_vel_b  # (num_envs, 3)
        base_ang_vel = self.robot.data.root_ang_vel_b  # (num_envs, 3)
        
        # height
        z_pos = self.robot.data.root_pos_w[:, 2:3]    # (num_envs, 1)
        
        return torch.cat((joint_pos, foot_pos_local, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)

    """
    Helper functions.
    """
    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # record amp term state
        self.amp_term_state_buffer[env_ids] = self.get_amp_observation()[env_ids].to(self.device)
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # setup for amp
        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

    # override reset
    # def reset(
    #     self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    # ) -> tuple[VecEnvObs, dict]:
    #     """Resets the specified environments and returns observations.

    #     This function calls the :meth:`_reset_idx` function to reset the specified environments.
    #     However, certain operations, such as procedural terrain generation, that happened during initialization
    #     are not repeated.

    #     Args:
    #         seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
    #         env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
    #         options: Additional information to specify how the environment is reset. Defaults to None.

    #             Note:
    #                 This argument is used for compatibility with Gymnasium environment definition.

    #     Returns:
    #         A tuple containing the observations and extras.
    #     """
    #     if env_ids is None:
    #         env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

    #     # trigger recorder terms for pre-reset calls
    #     self.recorder_manager.record_pre_reset(env_ids)

    #     # set the seed
    #     if seed is not None:
    #         self.seed(seed)

    #     # reset state of scene
    #     self._reset_idx(env_ids)

    #     # update articulation kinematics
    #     self.scene.write_data_to_sim()
    #     self.sim.forward()
    #     # if sensors are added to the scene, make sure we render to reflect changes in reset
    #     if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
    #         for _ in range(self.cfg.num_rerenders_on_reset):
    #             self.sim.render()

    #     # trigger recorder terms for post-reset calls
    #     self.recorder_manager.record_post_reset(env_ids)

    #     # compute observations
    #     self.obs_buf = self.observation_manager.compute(update_history=True)

    #     if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
    #         while SimulationManager.assets_loading():
    #             self.sim.render()

    #     # return observations
    #     return self.obs_buf, self.extras


    def prepare_imagination(self):
        self.imagination_common_step_counter = 0
        self.system_dynamics_model_ids = torch.randint(0, self.system_dynamics.ensemble_size, (1, self.num_imagination_envs, 1), device=self.device)
        self._init_imagination_reward_buffer()
        self._init_intervals()
        self._init_additional_imagination_attributes()
        self._init_imagination_command()
        self.last_obs = TensorDict({"policy": torch.zeros(self.num_imagination_envs, self.observation_manager.group_obs_dim["policy"][0])}, batch_size=[self.num_imagination_envs], device=self.device)
        self.imagination_extras = {}
        self._reset_imagination_idx(torch.arange(self.num_imagination_envs, device=self.device))


    def _reset_imagination_idx(self, env_ids):
        self.imagination_extras["log"] = dict()
        self.system_dynamics.reset_partial(env_ids)
        self.system_dynamics_model_ids[:, env_ids, :] = torch.randint(0, self.system_dynamics.ensemble_size, (1, len(env_ids), 1), device=self.device)
        info = self._reset_imagination_reward_buffer(env_ids)
        self.imagination_extras["log"].update(info)
        self._reset_intervals(env_ids)
        self._reset_additional_imagination_attributes(env_ids)
        self.last_obs["policy"][env_ids] = 0.0


    def _init_imagination_command(self):
        for name, term in self.command_manager._terms.items():
            setattr(self, name, term.sample_command(self.num_imagination_envs))


    def _reset_imagination_command(self, env_ids):
        for name, term in self.command_manager._terms.items():
            getattr(self, name)[env_ids] = term.sample_command(len(env_ids))

    
    def _init_imagination_reward_buffer(self):
        self.imagination_episode_length_buf = torch.zeros(self.num_imagination_envs, device=self.device, dtype=torch.long)
        self.imagination_episode_sums = {
            term: torch.zeros(
                self.num_imagination_envs,
                device=self.device
                ) for term in self.reward_term_names
            }
        self.imagination_episode_sums["uncertainty"] = torch.zeros(self.num_imagination_envs, device=self.device)
        self.imagination_reward_per_step = {
            term: torch.zeros(
                self.num_imagination_envs,
                device=self.device
                ) for term in self.reward_term_names
            }
    
    
    def _reset_imagination_reward_buffer(self, env_ids):
        extras = {}
        self.imagination_episode_length_buf[env_ids] = 0
        for term in self.imagination_episode_sums.keys():
            episodic_sum_avg = torch.mean(self.imagination_episode_sums[term][env_ids])
            extras[term] = episodic_sum_avg / (self.max_imagination_episode_length * self.step_dt)
            self.imagination_episode_sums[term][env_ids] = 0.0
        return extras


    def _init_intervals(self):
        if self.imagination_command_resample_interval_range is None:
            return
        else:
            self.imagination_command_resample_intervals = torch.randint(self.imagination_command_resample_interval_range[0], self.imagination_command_resample_interval_range[1], (self.num_imagination_envs,), device=self.device)


    def _reset_intervals(self, env_ids):
        if self.imagination_command_resample_interval_range is None:
            return
        else:
            self.imagination_command_resample_intervals[env_ids] = torch.randint(self.imagination_command_resample_interval_range[0], self.imagination_command_resample_interval_range[1], (len(env_ids),), device=self.device)

    def imagination_step(self, rollout_action, state_history, action_history):
        rollout_action_normalized = self.imagination_action_normalizer(rollout_action)
        action_history = torch.cat([action_history[:, 1:], rollout_action_normalized.unsqueeze(1)], dim=1)
        imagination_states, aleatoric_uncertainty, self.epistemic_uncertainty, extensions, contacts, terminations = self.system_dynamics.forward(state_history, action_history, self.system_dynamics_model_ids)
        imagination_states_denormalized = self.imagination_state_normalizer.inverse(imagination_states)
        parsed_imagination_states = self._parse_imagination_states(imagination_states_denormalized)
        parsed_extensions = self._parse_extensions(extensions)
        parsed_contacts = self._parse_contacts(contacts)
        self.termination_flags = self._parse_terminations(terminations)
        self._compute_imagination_reward_terms(parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts)
        rewards, dones, extras = self._post_imagination_step()
        command_ids = self._process_command_env_ids()
        self._reset_imagination_command(command_ids)
        state_history = torch.cat([state_history[:, 1:], imagination_states.unsqueeze(1)], dim=1)
        return self.last_obs, rewards, dones, extras, state_history, action_history, self.epistemic_uncertainty
    
    
    def _post_imagination_step(self):      
        self.imagination_episode_length_buf += 1
        self.imagination_common_step_counter += 1
        rewards = torch.zeros(self.num_imagination_envs, dtype=torch.float, device=self.device)
        for term in self.imagination_episode_sums.keys():
            if term == "uncertainty":
                rewards += self.uncertainty_penalty_weight * self.epistemic_uncertainty * self.step_dt
                self.imagination_episode_sums[term] += self.uncertainty_penalty_weight * self.epistemic_uncertainty * self.step_dt
            else:
                term_cfg = self.reward_manager.get_term_cfg(term)
                term_value = self.imagination_reward_per_step[term]
                rewards += term_cfg.weight * term_value * self.step_dt
                self.imagination_episode_sums[term] += term_cfg.weight * term_value * self.step_dt
        
        terminated = self.termination_flags if self.termination_flags is not None else torch.zeros(self.num_imagination_envs, dtype=torch.bool, device=self.device)
        time_outs = self.imagination_episode_length_buf >= self.max_imagination_episode_length
        dones = (terminated | time_outs).to(dtype=torch.long)

        reset_env_ids = (terminated | time_outs).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_imagination_idx(reset_env_ids)
        self.imagination_extras["time_outs"] = time_outs
        return rewards, dones, self.imagination_extras


    def _process_command_env_ids(self):
        if self.imagination_command_resample_interval_range is None:
            return torch.empty(0, dtype=torch.int, device=self.device)
        else:
            return (self.imagination_episode_length_buf % self.imagination_command_resample_intervals == 0).nonzero(as_tuple=False).squeeze(-1)


    def _init_additional_attributes(self):
        raise NotImplementedError


    def _init_additional_imagination_attributes(self):
        raise NotImplementedError


    def _reset_additional_imagination_attributes(self, env_ids):
        raise NotImplementedError


    def get_imagination_observation(self, state_history, action_history, observation_noise=True):
        raise NotImplementedError
    

    def _parse_imagination_states(self, imagination_states_denormalized):
        raise NotImplementedError

    
    def _parse_extensions(self, extensions):
        raise NotImplementedError


    def _parse_contacts(self, contacts):
        raise NotImplementedError


    def _parse_terminations(self, terminations):
        raise NotImplementedError


    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        raise NotImplementedError
