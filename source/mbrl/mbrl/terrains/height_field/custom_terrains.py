"""Custom height-field terrain generators ported from the WMP project.

Adapted from Isaac Gym API (``isaacgym.terrain_utils``) to the IsaacLab
``@height_field_to_mesh`` height-field contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from .custom_terrains_cfg import HfClimbTerrainCfg


@height_field_to_mesh
def climb_terrain(difficulty: float, cfg: HfClimbTerrainCfg) -> np.ndarray:
    """Generate a terrain with two raised wall strips that the robot must step over.

    The terrain has two horizontal barriers spanning the full width of the tile.
    The barrier height scales linearly with ``difficulty``.  The first wall is
    placed at ``cfg.first_wall_x`` metres from the tile start and the second at
    ``cfg.second_wall_x`` metres.  The wall thickness along the x-axis is
    ``cfg.step_width`` with a small random perturbation of up to 0.2 m.

    Inspired by the ``climb_terrain`` helper in the WMP project
    (``legged_gym/utils/terrain.py``).

    Args:
        difficulty: Terrain difficulty in [0, 1].
        cfg: Configuration for the climb terrain.

    Returns:
        2-D int16 height array of shape (width_pixels, length_pixels).
        Heights are in units of ``cfg.vertical_scale`` metres.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    # wall height interpolated by difficulty
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    step_height_int = int(step_height / cfg.vertical_scale)

    # wall thickness with random variation (mirrors WMP: 1.0 + 0.2 * random)
    wall_thickness = cfg.step_width + 0.2 * np.random.random()
    wall_pixels = int(wall_thickness / cfg.horizontal_scale)

    # first wall
    x1 = int(cfg.first_wall_x / cfg.horizontal_scale)
    x2 = min(x1 + wall_pixels, width_pixels)
    heights[x1:x2, :] = step_height_int

    # second wall
    x3 = int(cfg.second_wall_x / cfg.horizontal_scale)
    x4 = min(x3 + wall_pixels, width_pixels)
    heights[x3:x4, :] = step_height_int

    return heights
