#!/usr/bin/env python3
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""Convert AMP_for_hardware JSON motion files to MotionLoader's .npz format.

The input JSON files come from AMP_for_hardware/datasets/mocap_motions/.
Each JSON frame contains 61 values:
  [0:3]   root_pos (x, y, z) in world frame
  [3:7]   root_rot (x, y, z, w) in xyzw format
  [7:19]  joint_pos in PyBullet order [FR, FL, RR, RL] x 3 joints
  [19:31] toe_pos_local in PyBullet order [FR, FL, RR, RL] x 3, in base frame
  [31:34] root_linear_velocity (x, y, z)
  [34:37] root_angular_velocity (x, y, z)
  [37:49] joint_vel in PyBullet order [FR, FL, RR, RL] x 3 joints
  [49:61] toe_vel_local in PyBullet order [FR, FL, RR, RL] x 3, in base frame

Usage:
    python json2npz.py                          # converts all files in mocap_motions/
    python json2npz.py --input pace0.txt trot0.txt --output a1_motions.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert a single quaternion from (x,y,z,w) to IsaacLab (w,x,y,z)."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def _quat_xyzw_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def convert_json_to_npz(json_files: list[str], output_file: str) -> None:
    """Convert one or more AMP JSON motion files into a single .npz.

    The output body order is: trunk, FL_calf, FR_calf, RL_calf, RR_calf.
    The output joint order (after reordering PyBullet→Isaac) is:
      FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
      RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf.
    """
    # Joint names stored in the .npz (FL, FR, RL, RR order after reordering)
    dof_names = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]
    # Body names must match IsaacLab's A1 body names after merge_fixed_joints=True:
    #   "base"    : root body (trunk merged into base, base is the surviving name)
    #   "FR_foot" etc. : foot bodies kept by activate_contact_sensors=True
    # The order matches the toe_pos_local layout (FL, FR, RL, RR after Isaac reordering).
    body_names = ["base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    B = len(body_names)

    all_traj_ids: list[int] = []
    all_traj_weights: list[float] = []
    all_dof_pos: list[np.ndarray] = []
    all_dof_vel: list[np.ndarray] = []
    all_body_pos: list[np.ndarray] = []
    all_body_rot: list[np.ndarray] = []
    all_body_lin_vel: list[np.ndarray] = []
    all_body_ang_vel: list[np.ndarray] = []
    fps: int = 0

    for idx, json_file in enumerate(json_files):
        all_traj_ids.append(idx)
        with open(json_file) as f:
            data = json.load(f)
        all_traj_weights.append(data["MotionWeight"])
        frame_duration: float = data["FrameDuration"]
        if fps == 0:
            fps = round(1.0 / frame_duration)

        frames = np.array(data["Frames"], dtype=np.float64)  # (N, 61)
        N = frames.shape[0]
        if frames.shape[1] != 61:
            raise ValueError(f"Expected 61 values per frame, got {frames.shape[1]} in {json_file}")

        # ── Root state ──────────────────────────────────────────────────────
        root_pos = frames[:, 0:3]       # (N, 3)  world frame
        root_rot_xyzw = frames[:, 3:7]  # (N, 4)  xyzw convention

        # ── Reorder joints: PyBullet [FR, FL, RR, RL] → Isaac [FL, FR, RL, RR] ─
        jp_fr = frames[:, 7:10]
        jp_fl = frames[:, 10:13]
        jp_rr = frames[:, 13:16]
        jp_rl = frames[:, 16:19]
        dof_pos = np.hstack([jp_fl, jp_fr, jp_rl, jp_rr]).astype(np.float32)  # (N,12)

        # ── Toe positions in base frame (same reordering) ───────────────────
        tp_fr = frames[:, 19:22]
        tp_fl = frames[:, 22:25]
        tp_rr = frames[:, 25:28]
        tp_rl = frames[:, 28:31]
        # FL, FR, RL, RR matches body_names[1:] order
        toe_pos_local = np.hstack([tp_fl, tp_fr, tp_rl, tp_rr])  # (N, 12)

        # ── Root velocities in world frame ──────────────────────────────────
        lin_vel = frames[:, 31:34]  # (N, 3)
        ang_vel = frames[:, 34:37]  # (N, 3)

        # ── Reorder joint velocities ────────────────────────────────────────
        jv_fr = frames[:, 37:40]
        jv_fl = frames[:, 40:43]
        jv_rr = frames[:, 43:46]
        jv_rl = frames[:, 46:49]
        dof_vel = np.hstack([jv_fl, jv_fr, jv_rl, jv_rr]).astype(np.float32)  # (N,12)

        # ── Toe velocities in base frame (same reordering) ──────────────────
        tv_fr = frames[:, 49:52]
        tv_fl = frames[:, 52:55]
        tv_rr = frames[:, 55:58]
        tv_rl = frames[:, 58:61]
        toe_vel_local = np.hstack([tv_fl, tv_fr, tv_rl, tv_rr])  # (N, 12)

        # ── Build body data arrays ───────────────────────────────────────────
        body_pos = np.zeros((N, B, 3), dtype=np.float32)
        body_rot = np.zeros((N, B, 4), dtype=np.float32)
        body_lin_vel = np.zeros((N, B, 3), dtype=np.float32)
        body_ang_vel = np.zeros((N, B, 3), dtype=np.float32)

        # trunk (body 0)
        body_pos[:, 0] = root_pos.astype(np.float32)
        # Convert quaternion xyzw → wxyz for IsaacLab
        body_rot[:, 0, 0] = root_rot_xyzw[:, 3]  # w
        body_rot[:, 0, 1] = root_rot_xyzw[:, 0]  # x
        body_rot[:, 0, 2] = root_rot_xyzw[:, 1]  # y
        body_rot[:, 0, 3] = root_rot_xyzw[:, 2]  # z
        body_lin_vel[:, 0] = lin_vel.astype(np.float32)
        body_ang_vel[:, 0] = ang_vel.astype(np.float32)

        # 4 feet (bodies 1-4): FL, FR, RL, RR
        for i in range(4):
            toe_local = toe_pos_local[:, i * 3 : i * 3 + 3]      # (N, 3) base frame
            toe_vel_loc = toe_vel_local[:, i * 3 : i * 3 + 3]    # (N, 3) base frame

            for n in range(N):
                R = _quat_xyzw_to_rot_matrix(root_rot_xyzw[n])
                toe_offset = R @ toe_local[n]
                body_pos[n, i + 1] = (root_pos[n] + toe_offset).astype(np.float32)
                # foot orientation approximated as trunk orientation
                body_rot[n, i + 1] = body_rot[n, 0]
                # foot velocity: v_trunk + ω × r  (r = toe_offset in world frame)
                foot_lin_vel = lin_vel[n] + np.cross(ang_vel[n], toe_offset)
                body_lin_vel[n, i + 1] = foot_lin_vel.astype(np.float32)
                body_ang_vel[n, i + 1] = ang_vel[n].astype(np.float32)

        all_dof_pos.append(dof_pos)
        all_dof_vel.append(dof_vel)
        all_body_pos.append(body_pos)
        all_body_rot.append(body_rot)
        all_body_lin_vel.append(body_lin_vel)
        all_body_ang_vel.append(body_ang_vel)

        print(f"Loaded {json_file}: {N} frames @ {fps} FPS ({N / fps:.2f} s)")

    # ── Concatenate all clips ────────────────────────────────────────────────
    dof_pos_all = np.concatenate(all_dof_pos, axis=0)
    dof_vel_all = np.concatenate(all_dof_vel, axis=0)
    body_pos_all = np.concatenate(all_body_pos, axis=0)
    body_rot_all = np.concatenate(all_body_rot, axis=0)
    body_lin_vel_all = np.concatenate(all_body_lin_vel, axis=0)
    body_ang_vel_all = np.concatenate(all_body_ang_vel, axis=0)

    np.savez(
        output_file,
        fps=np.int64(fps),
        traj_ids=np.array(all_traj_ids, dtype=np.int64),
        traj_weights=np.array(all_traj_weights, dtype=np.float32),
        dof_names=np.array(dof_names, dtype=np.str_),
        body_names=np.array(body_names, dtype=np.str_),
        dof_positions=dof_pos_all,
        dof_velocities=dof_vel_all,
        body_positions=body_pos_all,
        body_rotations=body_rot_all,
        body_linear_velocities=body_lin_vel_all,
        body_angular_velocities=body_ang_vel_all,
    )
    total_frames = dof_pos_all.shape[0]
    print(f"\nSaved: {output_file}")
    print(f"  Frames: {total_frames} @ {fps} FPS ({total_frames / fps:.2f} s total)")
    print(f"  DOFs:   {len(dof_names)}")
    print(f"  Bodies: {len(body_names)}")


def _default_input_files(script_dir: str) -> list[str]:
    """Find all .txt motion files in the adjacent mocap_motions/ directory."""
    mocap_dir = os.path.join(script_dir, "mocap_motions_a1_canter")
    if not os.path.isdir(mocap_dir):
        # Fallback: look for the AMP_for_hardware datasets inside the repo
        candidate = os.path.join(
            script_dir, "..", "..", "..", "..", "..", "AMP_for_hardware", "datasets", "mocap_motions"
        )
        candidate = os.path.normpath(candidate)
        if os.path.isdir(candidate):
            mocap_dir = candidate
        else:
            raise FileNotFoundError(
                f"Could not find mocap_motions directory. "
                f"Expected at {mocap_dir} or {candidate}. "
                "Please copy the JSON files from AMP_for_hardware/datasets/mocap_motions/ "
                "to a1_amp/motions/mocap_motions/."
            )
    files = sorted(f for f in os.listdir(mocap_dir) if f.endswith(".txt"))
    if not files:
        raise FileNotFoundError(f"No .txt motion files found in {mocap_dir}")
    return [os.path.join(mocap_dir, f) for f in files]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert AMP JSON motion files to .npz format")
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="Input JSON .txt motion file(s). Defaults to all files in mocap_motions/.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz file path. Defaults to a1_motions.npz in the script's directory.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_files = args.input if args.input is not None else _default_input_files(script_dir)
    output_file = args.output if args.output is not None else os.path.join(script_dir, "a1_motions.npz")

    convert_json_to_npz(input_files, output_file)
