"""Configuration classes for custom trimesh terrains."""

from __future__ import annotations

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from . import custom_terrains


@configclass
class MeshTiltTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a narrow-corridor tilt terrain.

    Four box walls are placed on both sides of a central walkway of width
    :attr:`tilt_width_range`.  The corridor narrows with increasing difficulty,
    challenging the robot's lateral balance.

    Ported from WMP's tilt-terrain block.
    """

    function = custom_terrains.tilt_terrain

    tilt_width_range: tuple[float, float] = (0.28, 0.32)
    """Minimum and maximum corridor width (in m).

    ``tilt_width_range[1]`` is used at difficulty 0 (widest / easiest) and
    ``tilt_width_range[0]`` at difficulty 1 (narrowest / hardest).
    """


@configclass
class MeshCrawlTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a crawl terrain with overhead horizontal bars.

    Two bars span the full tile width.  The clearance height (bottom of bar)
    decreases with difficulty, forcing the robot to crouch lower.

    Ported from WMP's crawl-terrain block.
    """

    function = custom_terrains.crawl_terrain

    crawl_height_range: tuple[float, float] = (0.20, 0.35)
    """Minimum and maximum clearance height below the bars (in m).

    ``crawl_height_range[1]`` is used at difficulty 0 (most clearance / easiest)
    and ``crawl_height_range[0]`` at difficulty 1 (least clearance / hardest).
    """
