# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint for WMP (World Model Policy) RL agent from RSL-RL.

Migrated from WMP project play features:
- Policy export as JIT
- Frame recording to disk
- Camera following robot
- State/reward logging
- Terrain type selection for rough envs
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play WMP checkpoint with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# WMP play-specific arguments (migrated from WMP project)
parser.add_argument(
    "--export_policy",
    action="store_true",
    default=False,
    help="Export policy as JIT script for deployment.",
)
parser.add_argument(
    "--record_frames",
    action="store_true",
    default=False,
    help="Save viewport frames to disk during play.",
)
parser.add_argument(
    "--move_camera",
    action="store_true",
    default=False,
    help="Follow the robot with the camera during play.",
)
parser.add_argument(
    "--terrain",
    type=str,
    default=None,
    choices=["slope", "stair", "gap", "climb", "tilt", "crawl"],
    help="Terrain type for rough envs (slope, stair, gap, climb, tilt, crawl).",
)
parser.add_argument(
    "--log_states",
    action="store_true",
    default=False,
    help="Log robot states during play (for debugging).",
)
parser.add_argument(
    "--stop_state_log",
    type=int,
    default=100,
    help="Number of steps before stopping state logging.",
)
parser.add_argument(
    "--stop_rew_log",
    type=int,
    default=None,
    help="Number of steps before printing average episode rewards. Default: max_episode_length+1.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
import mbrl.tasks  # noqa: F401
from rsl_rl.runners import AMPOnPolicyRunner, WMPOnPolicyRunner

# Ensure play_logger can be imported when run from project root
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from play_logger import PlayLogger

# Terrain mapping: WMP names -> Isaac Lab terrain generator sub_terrain names
TERRAIN_MAP = {
    "slope": "hf_pyramid_slope",
    "stair": "pyramid_stairs",
    "gap": "gap",
    "climb": "climb",
    "tilt": "tilt",
    "crawl": "crawl",
}


def _apply_terrain_override(env_cfg, terrain_name: str):
    """Override terrain proportions to use a single terrain type for play."""
    if not hasattr(env_cfg.scene, "terrain") or env_cfg.scene.terrain is None:
        return
    tg = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if tg is None or not hasattr(tg, "sub_terrains"):
        return
    sub_name = TERRAIN_MAP.get(terrain_name)
    if sub_name not in tg.sub_terrains:
        return
    for name in tg.sub_terrains:
        tg.sub_terrains[name].proportion = 1.0 if name == sub_name else 0.0


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with WMP RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # terrain override for rough envs
    if args_cli.terrain is not None:
        _apply_terrain_override(env_cfg, args_cli.terrain)
        print(f"[INFO] Terrain set to: {args_cli.terrain}")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "AMPOnPolicyRunner":
        runner = AMPOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "WMPOnPolicyRunner":
        runner = WMPOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # export policy as JIT (migrated from WMP)
    if args_cli.export_policy:
        export_dir = os.path.join(log_dir, "exported", "policies")
        os.makedirs(export_dir, exist_ok=True)
        try:
            # WMP policy has act_inference(obs, history, wm_feature) - export actor only for now
            export_policy_as_jit(runner.alg.actor_critic, export_dir)
            print(f"[INFO] Exported policy as JIT to: {export_dir}")
        except Exception as e:
            print(f"[WARN] Policy export failed (WMP policy may need custom export): {e}")

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()

    dt = env.unwrapped.step_dt
    device = env.unwrapped.device
    from mbrl.tasks.manager_based.locomotion.velocity.config.a1.agents.rsl_rl_wmp_ppo_cfg import \
        NUM_OBS, PRIVILEGED_DIM, PROP_DIM, FOOT_HEIGHT_DIM, HISTORY_INTERVAL, UPDATE_INTERVAL
    trajectory_history = torch.zeros(size=(env.num_envs, HISTORY_INTERVAL, NUM_OBS -
                                           PRIVILEGED_DIM - FOOT_HEIGHT_DIM - 3), device = device)
    obs_without_command = torch.concat((obs["policy"][:, PRIVILEGED_DIM:PRIVILEGED_DIM + 6],
                                        obs["policy"][:, PRIVILEGED_DIM + 9:-FOOT_HEIGHT_DIM]), dim=1)
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

    world_model = runner._world_model.to(device)
    wm_latent = wm_action = None
    wm_is_first = torch.ones(env.num_envs, device=device)
    wm_action_history = torch.zeros(size=(env.num_envs, UPDATE_INTERVAL, env.num_actions),
                                    device=device)
    wm_obs = {
        "prop": obs["policy"][:, PRIVILEGED_DIM: PRIVILEGED_DIM + PROP_DIM],
        "is_first": wm_is_first,
        "image": obs["camera"],
    }

    wm_feature = torch.zeros((env.num_envs, runner.base_cfg["env"]["wm_feature_dim"]), device=device)

    # Logger and play options (migrated from WMP)
    logger = PlayLogger(dt) if args_cli.log_states else None
    robot_index = 0
    joint_index = 1
    stop_state_log = args_cli.stop_state_log
    stop_rew_log = args_cli.stop_rew_log
    if stop_rew_log is None and hasattr(env.unwrapped, "max_episode_length"):
        stop_rew_log = int(env.unwrapped.max_episode_length) + 1
    elif stop_rew_log is None:
        stop_rew_log = 2000
    img_idx = 0
    frames_dir = None
    if args_cli.record_frames:
        frames_dir = os.path.join(log_dir, "exported", "frames")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"[INFO] Recording frames to: {frames_dir}")

    total_reward = 0
    not_dones = torch.ones((env.num_envs,), device=device)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if ((env.unwrapped.common_step_counter + 1) % UPDATE_INTERVAL == 0):
                wm_embed = world_model.encoder(wm_obs)
                wm_latent, _ = world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=True)
                wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
                wm_is_first[:] = 0

            # agent stepping
            history = trajectory_history.flatten(1).to(device)
            actions = policy(obs, history, wm_feature)
            # env stepping
            obs, rewards, dones, extras = env.step(actions)

            not_dones *= (~dones)
            total_reward += torch.mean(rewards * not_dones)

            wm_action_history = torch.concat(
                (wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
            wm_obs = {
                "prop": obs["policy"][:, PRIVILEGED_DIM: PRIVILEGED_DIM + PROP_DIM],
                "is_first": wm_is_first,
                "image": obs["camera"],
            }
            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
            if (len(reset_env_ids) > 0):
                wm_action_history[reset_env_ids, :] = 0
                wm_is_first[reset_env_ids] = 1

            wm_action = wm_action_history.flatten(1)

            trajectory_history[reset_env_ids] = 0
            obs_without_command = torch.concat((obs["policy"][:, PRIVILEGED_DIM:PRIVILEGED_DIM + 6],
                                                obs["policy"][:, PRIVILEGED_DIM + 9:-FOOT_HEIGHT_DIM]),
                                            dim=1)
            trajectory_history = torch.concat(
                (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        # Record frames (migrated from WMP)
        if args_cli.record_frames and frames_dir and timestep % 2 == 0:
            try:
                from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
                vp_api = get_active_viewport()
                if vp_api is not None:
                    filename = os.path.join(frames_dir, f"{img_idx:06d}.png")
                    capture_viewport_to_file(vp_api, filename)
                    img_idx += 1
            except Exception as e:
                if timestep == 0:
                    print(f"[WARN] Frame capture not available: {e}")

        # Move camera to follow robot (migrated from WMP)
        if args_cli.move_camera and hasattr(env.unwrapped, "scene"):
            try:
                robot = env.unwrapped.scene["robot"]
                root_pos = robot.data.root_pos_w
                if root_pos is not None and root_pos.shape[0] > robot_index:
                    lookat = root_pos[robot_index, :3].detach().cpu().numpy()
                    camera_pos = lookat + np.array([0.0, 1.0, 0.0])
                    sim = getattr(env.unwrapped, "sim", None)
                    if sim is not None and hasattr(sim, "set_camera_view"):
                        sim.set_camera_view(eye=camera_pos.tolist(), target=lookat.tolist())
            except Exception as e:
                if timestep == 0:
                    print(f"[WARN] Camera follow not available: {e}")

        # State logging (migrated from WMP)
        if logger is not None and timestep < stop_state_log and hasattr(env.unwrapped, "scene"):
            try:
                robot = env.unwrapped.scene["robot"]
                action_scale = getattr(env.unwrapped.cfg, "actions", None)
                if action_scale is not None and hasattr(action_scale, "scale"):
                    action_scale = action_scale.scale
                else:
                    action_scale = 1.0
                feet_indices = getattr(env.unwrapped, "foot_body_ids", None)
                if feet_indices is None and hasattr(robot.data, "body_names"):
                    feet_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
                    feet_indices = [robot.data.body_names.index(n) for n in feet_names if n in robot.data.body_names]
                contact_z = []
                if hasattr(env.unwrapped.scene, "sensors"):
                    try:
                        cf = env.unwrapped.scene.sensors["contact_forces"]
                        if cf is not None and hasattr(cf, "data"):
                            net = getattr(cf.data, "net_forces_w", None)
                            if net is not None and net.dim() >= 3:
                                # Shape (num_envs, num_bodies, 3); take z for first 4 bodies (feet)
                                n_feet = min(4, net.shape[1])
                                contact_z = net[robot_index, :n_feet, 2].cpu().numpy().tolist()
                    except (KeyError, TypeError, IndexError):
                        pass
                cmd = obs["policy"][robot_index, PRIVILEGED_DIM + 6 : PRIVILEGED_DIM + 9]
                logger.log_states({
                    "dof_pos_target": (actions[robot_index, joint_index].item() * action_scale),
                    "dof_pos": robot.data.joint_pos[robot_index, joint_index].item(),
                    "dof_vel": robot.data.joint_vel[robot_index, joint_index].item(),
                    "dof_torque": robot.data.applied_torque[robot_index, joint_index].item() if hasattr(robot.data, "applied_torque") else 0.0,
                    "command_x": cmd[0].item() if cmd.shape[0] > 0 else 0.0,
                    "command_y": cmd[1].item() if cmd.shape[0] > 1 else 0.0,
                    "command_yaw": cmd[2].item() if cmd.shape[0] > 2 else 0.0,
                    "base_vel_x": robot.data.root_lin_vel_b[robot_index, 0].item(),
                    "base_vel_y": robot.data.root_lin_vel_b[robot_index, 1].item(),
                    "base_vel_z": robot.data.root_lin_vel_b[robot_index, 2].item(),
                    "base_vel_yaw": robot.data.root_ang_vel_b[robot_index, 2].item(),
                    "contact_forces_z": contact_z if contact_z else [0.0],
                })
            except Exception as e:
                if timestep == 0:
                    print(f"[WARN] State logging failed: {e}")

        # Reward logging (migrated from WMP)
        if logger is not None and 0 < timestep < stop_rew_log:
            num_episodes = int(dones.sum().item()) if hasattr(dones, "sum") else 0
            if num_episodes > 0 and extras:
                if "log" in extras:
                    logger.log_rewards(extras["log"], num_episodes)
                elif "episode" in extras:
                    logger.log_rewards(extras["episode"], num_episodes)
        if logger is not None and timestep == stop_rew_log:
            logger.print_rewards()

        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    print('total reward:', total_reward)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
