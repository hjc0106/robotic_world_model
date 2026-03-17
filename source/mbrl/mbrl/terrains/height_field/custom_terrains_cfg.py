"""Configuration classes for custom height-field terrains."""

from __future__ import annotations

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.utils import configclass

from . import custom_terrains


@configclass
class HfClimbTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a terrain with two raised wall strips.

    The robot must step over two horizontal barriers placed at
    :attr:`first_wall_x` and :attr:`second_wall_x` along the x-axis.
    Wall height scales from ``step_height_range[0]`` (easy) to
    ``step_height_range[1]`` (hard) with difficulty.

    Ported from WMP's ``climb_terrain`` helper.
    """

    function = custom_terrains.climb_terrain

    step_height_range: tuple[float, float] = (0.0, 0.6)
    """Minimum and maximum height of the wall barriers (in m)."""

    step_width: float = 1.0
    """Base thickness of each wall along the x-axis (in m).

    A random offset of up to 0.2 m is added at generation time.
    """

    first_wall_x: float = 1.0
    """X-position of the leading edge of the first wall from the tile start (in m)."""

    second_wall_x: float = 6.0
    """X-position of the leading edge of the second wall from the tile start (in m)."""

    border_width: float = 0.0
    """Border width around the terrain (in m). Defaults to 0.0."""
