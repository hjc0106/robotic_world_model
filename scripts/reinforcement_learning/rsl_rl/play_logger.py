# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Logger utility for play scripts - logs states and rewards during policy evaluation."""

from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np


class PlayLogger:
    """Logger for state and reward logging during play evaluation."""

    def __init__(self, dt: float):
        """Initialize the play logger.

        Args:
            dt: Simulation timestep in seconds.
        """
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key: str, value):
        """Log a single state value."""
        self.state_log[key].append(value)

    def log_states(self, state_dict: dict):
        """Log multiple state values from a dictionary."""
        for key, value in state_dict.items():
            self.log_state(key, value)

    def log_rewards(self, reward_dict: dict, num_episodes: int):
        """Log reward values from completed episodes.

        Args:
            reward_dict: Dictionary of reward terms (keys containing 'rew').
            num_episodes: Number of episodes that completed.
        """
        for key, value in reward_dict.items():
            if "rew" in key:
                if hasattr(value, "item"):
                    self.rew_log[key].append(value.item() * num_episodes)
                else:
                    self.rew_log[key].append(float(value) * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        """Clear all logged data."""
        self.state_log.clear()
        self.rew_log.clear()
        self.num_episodes = 0

    def plot_states(self, save_path: str | None = None):
        """Generate and optionally save state plots.

        Args:
            save_path: If provided, save the figure to this path instead of displaying.
        """
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(12, 10))
        log = self.state_log

        # Get time axis from first available log
        time = None
        for value in log.values():
            if value:
                time = np.linspace(0, len(value) * self.dt, len(value))
                break

        if time is None:
            plt.close(fig)
            return

        # Plot joint targets and measured positions
        ax = axs[1, 0]
        if log.get("dof_pos"):
            ax.plot(time, log["dof_pos"], label="measured")
        if log.get("dof_pos_target"):
            ax.plot(time, log["dof_pos_target"], label="target")
        ax.set(xlabel="time [s]", ylabel="Position [rad]", title="DOF Position")
        ax.legend()

        # Plot joint velocity
        ax = axs[1, 1]
        if log.get("dof_vel"):
            ax.plot(time, log["dof_vel"], label="measured")
        if log.get("dof_vel_target"):
            ax.plot(time, log["dof_vel_target"], label="target")
        ax.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title="Joint Velocity")
        ax.legend()

        # Plot base vel x
        ax = axs[0, 0]
        if log.get("base_vel_x"):
            ax.plot(time, log["base_vel_x"], label="measured")
        if log.get("command_x"):
            ax.plot(time, log["command_x"], label="commanded")
        ax.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity x")
        ax.legend()

        # Plot base vel y
        ax = axs[0, 1]
        if log.get("base_vel_y"):
            ax.plot(time, log["base_vel_y"], label="measured")
        if log.get("command_y"):
            ax.plot(time, log["command_y"], label="commanded")
        ax.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity y")
        ax.legend()

        # Plot base vel yaw
        ax = axs[0, 2]
        if log.get("base_vel_yaw"):
            ax.plot(time, log["base_vel_yaw"], label="measured")
        if log.get("command_yaw"):
            ax.plot(time, log["command_yaw"], label="commanded")
        ax.set(xlabel="time [s]", ylabel="base ang vel [rad/s]", title="Base velocity yaw")
        ax.legend()

        # Plot base vel z
        ax = axs[1, 2]
        if log.get("base_vel_z"):
            ax.plot(time, log["base_vel_z"], label="measured")
        ax.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity z")
        ax.legend()

        # Plot contact forces
        ax = axs[2, 0]
        if log.get("contact_forces_z"):
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                ax.plot(time, forces[:, i], label=f"force {i}")
        ax.set(xlabel="time [s]", ylabel="Forces z [N]", title="Vertical Contact forces")
        ax.legend()

        # Plot torque/vel curves
        ax = axs[2, 1]
        if log.get("dof_vel") and log.get("dof_torque"):
            ax.plot(log["dof_vel"], log["dof_torque"], "x", label="measured")
        ax.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        ax.legend()

        # Plot torques
        ax = axs[2, 2]
        if log.get("dof_torque"):
            ax.plot(time, log["dof_torque"], label="measured")
        ax.set(xlabel="time [s]", ylabel="Joint Torque [Nm]", title="Torque")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[INFO] Saved state plot to {save_path}")

        plt.close(fig)

    def print_rewards(self):
        """Print average rewards per second to console."""
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            if values and self.num_episodes > 0:
                mean = np.sum(np.array(values)) / self.num_episodes
                print(f"  - {key}: {mean:.4f}")
        print(f"Total number of episodes: {self.num_episodes}")
