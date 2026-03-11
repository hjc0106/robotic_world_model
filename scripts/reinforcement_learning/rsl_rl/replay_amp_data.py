# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025 Linden
# SPDX-License-Identifier: BSD 3-Clause

"""
Motion Replayer for A1 Quadruped Robot

This script replays motion data for the Unitree A1 robot in Isaac Sim.
The motion file must be generated first by running json2npz.py.

Usage:
    python motion_replayer.py [options]

Options:
    --motion MOTION_FILE    Motion data file to replay (default: a1_motions.npz)
    --device DEVICE         Device to run simulation on (default: cuda:0)

Example:
    python motion_replayer.py --motion a1_motions.npz
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

from isaaclab.app import AppLauncher

# Command line arguments
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--motion", type=str, default="a1_motions.npz")
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from rsl_rl.datasets.motion_loader import MotionLoader

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from mbrl.assets.unitree import UNITREE_A1_CFG

# Load motion data and get dt
motion = MotionLoader(args_cli.motion, device=args_cli.device)
motion.resample(0.005, kind="linear")
num_frames = motion.num_frames
print(f"motion.dt: {motion.dt}")
print(f"motion.num_frames: {motion.num_frames}")

# Find the index for the root body ('trunk' for A1 after merge_fixed_joints)
ROOT_BODY = "base"
try:
    print(f"Searching for '{ROOT_BODY}' in body names: {motion.body_names}")
    root_idx = motion.body_names.index(ROOT_BODY)
    print(f"Found root body '{ROOT_BODY}' at index: {root_idx}")
except (ValueError, AttributeError):
    print(f"\nError: Could not find '{ROOT_BODY}' in the motion file's body_names.")
    print(f"Available body names: {motion.body_names}")
    print("Please regenerate the motion file using json2npz.py.")
    simulation_app.close()
    sys.exit(1)

# Configure simulation with dt matching motion.dt
sim_cfg = sim_utils.SimulationCfg(
    dt=motion.dt,
    device=args_cli.device,
    gravity=(0.0, 0.0, -9.81),
    render_interval=1,
    enable_scene_query_support=True,
    use_fabric=True,
    physx=sim_utils.PhysxCfg(
        solver_type=1,
        min_position_iteration_count=8,
        max_position_iteration_count=8,
        min_velocity_iteration_count=4,
        max_velocity_iteration_count=4,
        enable_ccd=True,
        enable_stabilization=True,
        bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.04,
        friction_correlation_distance=0.025,
    ),
)
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.3])

# Configure scene with A1 robot
scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
scene_cfg.robot = UNITREE_A1_CFG.replace(prim_path="/World/Robot")
scene = InteractiveScene(scene_cfg)

# Add DomeLight for illumination
light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
light_cfg.func("/World/Light", light_cfg)

# Add ground plane
ground_cfg = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1))
ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, 0.0))

# Reset simulation
sim.reset()
scene.reset()

robot = scene["robot"]
# Align joint order: motion file uses [FL, FR, RL, RR], robot URDF uses [FR, FL, RR, RL]
motion_dof_indices = motion.get_dof_index(robot.joint_names)

print("\nStarting A1 motion replay loop...")
print("Tip: Close window or press Ctrl+C to exit")

try:
    while simulation_app.is_running():
        for i in range(num_frames):
            if not simulation_app.is_running():
                break

            # Get current frame's joint and root states (re-ordered to robot joint order)
            joint_pos = motion.dof_positions[i, motion_dof_indices].unsqueeze(0)
            joint_vel = motion.dof_velocities[i, motion_dof_indices].unsqueeze(0)
            root_pos = motion.body_positions[i, root_idx].unsqueeze(0)
            root_rot = motion.body_rotations[i, root_idx].unsqueeze(0)
            root_vel = motion.body_linear_velocities[i, root_idx].unsqueeze(0)
            root_ang_vel = motion.body_angular_velocities[i, root_idx].unsqueeze(0)
            root_state = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)

            # Write to simulation
            env_idx = torch.tensor([0], device=args_cli.device)
            robot.write_root_link_pose_to_sim(root_state[:, :7], env_idx)
            robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_idx)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_idx)

            # Step simulation
            scene.update(dt=sim.get_physics_dt())
            scene.write_data_to_sim()
            sim.step(render=True)

except KeyboardInterrupt:
    print("\nProgram interrupted by user")

finally:
    simulation_app.close()
