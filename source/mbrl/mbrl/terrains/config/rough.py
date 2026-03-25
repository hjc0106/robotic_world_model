"""Configuration for custom terrains.

Terrain type set and proportions are adapted from the WMP project
(``legged_gym/envs/a1/a1_amp_config.py``) and ported to IsaacLab's
declarative ``TerrainGeneratorCfg`` API.

WMP original proportions (10 types):
    [wave=0.0, slope=0.05, stairs_up=0.15, stairs_down=0.15,
     discrete=0.0, gap=0.25, climb=0.25, tilt=0.05, crawl=0.05, rough_flat=0.05]

Active types after mapping (wave and discrete kept at 0, merged into 8 entries):
    stairs_up=0.15, stairs_down=0.15, slope=0.05,
    gap=0.25, climb=0.25, tilt=0.05, crawl=0.05, rough_flat=0.05
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from mbrl.terrains.height_field import HfClimbTerrainCfg
from mbrl.terrains.mesh import MeshCrawlTerrainCfg, MeshTiltTerrainCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # --- stairs (WMP proportion: 0.15 each) ---
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # --- slope (WMP proportion: 0.05) ---
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        # --- gap terrain (WMP proportion: 0.25) ---
        # IsaacLab's built-in gap terrain: a central platform surrounded by
        # a gap, matching WMP's gap_terrain concept.
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.25,
            gap_width_range=(0.1, 0.6),  # (0.1, 1.0)
            platform_width=4.0,
        ),
        # --- climb terrain (WMP proportion: 0.25) ---
        # Two raised wall strips that the robot must step over.
        "climb": HfClimbTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.4),  # (0.0, 0.6)
            step_width=1.0,
            first_wall_x=1.0,
            second_wall_x=6.0,
            border_width=0.0,
        ),
        # --- tilt terrain (WMP proportion: 0.05) ---
        # Narrow central corridor flanked by box walls.
        "tilt": MeshTiltTerrainCfg(
            proportion=0.05,
            tilt_width_range=(0.28, 0.32),
        ),
        # --- crawl terrain (WMP proportion: 0.05) ---
        # Two overhead horizontal bars at low clearance.
        "crawl": MeshCrawlTerrainCfg(
            proportion=0.05,
            crawl_height_range=(0.20, 0.35),
        ),
        # --- rough flat (WMP proportion: 0.05) ---
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.05,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)
"""Rough terrains configuration adapted from the WMP project."""
