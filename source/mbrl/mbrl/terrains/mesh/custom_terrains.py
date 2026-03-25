"""Custom trimesh terrain generators ported from the WMP project.

Adapted from Isaac Gym API (``isaacgym.terrain_utils`` + ``legged_gym.utils.trimesh``)
to the IsaacLab trimesh terrain contract: each generator returns
``(list[trimesh.Trimesh], np.ndarray)`` where the array is the terrain origin
in local tile coordinates.

Terrain tile spans ``(0, 0, 0)`` to ``(cfg.size[0], cfg.size[1], 0)`` in local space.
The origin is placed at the tile centre ``(size[0]/2, size[1]/2, z)`` so that
IsaacLab can translate each tile to its grid cell automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from .custom_terrains_cfg import MeshCrawlTerrainCfg, MeshTiltTerrainCfg

# thickness of the flat ground slab placed under every tile (in m)
_GROUND_THICKNESS = 1.0


def tilt_terrain(
    difficulty: float, cfg: MeshTiltTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a narrow corridor flanked by box walls.

    Four box obstacles are placed symmetrically on either side of a central
    walkway of width ``tilt_width``.  Two boxes are located roughly 2 m ahead
    of the tile centre and two roughly 2 m behind.  The walkway narrows as
    ``difficulty`` increases, making balance harder.

    Layout (top view)::

        +--[wall_L_back ]---[corridor]---[wall_R_back ]--+
        |                                                 |
        +--[wall_L_front]---[corridor]---[wall_R_front]--+

    Ported from WMP's tilt-terrain block (``legged_gym/utils/terrain.py``).

    Args:
        difficulty: Terrain difficulty in [0, 1].
        cfg: Configuration for the tilt terrain.

    Returns:
        A tuple of (mesh list, origin array).
    """
    size_x, size_y = cfg.size

    # corridor width: narrower at higher difficulty
    tilt_width = cfg.tilt_width_range[1] - difficulty * (cfg.tilt_width_range[1] - cfg.tilt_width_range[0])

    # wall dimensions
    wall_y = (size_y - tilt_width) / 2.0
    box_x = 0.4 + 0.4 * np.random.random()
    box_z = 1.0  # box height above ground

    # x-positions of the two box pairs (front and back of centre)
    front_x = size_x / 2.0 + 2.0
    back_x = size_x / 2.0 - 2.0

    # y-centres of left and right walls
    left_y = wall_y / 2.0
    right_y = size_y - wall_y / 2.0

    meshes: list[trimesh.Trimesh] = []

    # flat ground slab
    ground_pos = (size_x / 2.0, size_y / 2.0, -_GROUND_THICKNESS / 2.0)
    ground = trimesh.creation.box(
        extents=(size_x, size_y, _GROUND_THICKNESS),
        transform=trimesh.transformations.translation_matrix(ground_pos),
    )
    meshes.append(ground)

    # 4 wall boxes (front/back × left/right)
    for x_center in [front_x, back_x]:
        for y_center in [left_y, right_y]:
            pos = (x_center, y_center, box_z / 2.0)
            box = trimesh.creation.box(
                extents=(box_x, wall_y, box_z),
                transform=trimesh.transformations.translation_matrix(pos),
            )
            meshes.append(box)

    origin = np.array([size_x / 2.0, size_y / 2.0, 0.0])
    return meshes, origin


def crawl_terrain(
    difficulty: float, cfg: MeshCrawlTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with two horizontal overhead bars to crawl under.

    Two bars span the full width of the tile.  Their bottom edge is at
    ``crawl_height`` above the ground, which decreases with difficulty
    (lower clearance = harder).  The bars are placed one-third and
    two-thirds along the tile length.

    Ported from WMP's crawl-terrain block (``legged_gym/utils/terrain.py``).

    Args:
        difficulty: Terrain difficulty in [0, 1].
        cfg: Configuration for the crawl terrain.

    Returns:
        A tuple of (mesh list, origin array).
    """
    size_x, size_y = cfg.size

    # clearance height: lower at higher difficulty
    crawl_height = cfg.crawl_height_range[1] - difficulty * (
        cfg.crawl_height_range[1] - cfg.crawl_height_range[0]
    )

    # bar dimensions
    bar_x = 0.2 + 0.2 * np.random.random()  # bar thickness along x
    bar_z = 1.0  # bar height (extends upward from crawl_height)

    # x-positions of the two bars
    front_x = size_x / 2.0 + 2.0
    back_x = size_x / 2.0 - 2.0

    meshes: list[trimesh.Trimesh] = []

    # flat ground slab
    ground_pos = (size_x / 2.0, size_y / 2.0, -_GROUND_THICKNESS / 2.0)
    ground = trimesh.creation.box(
        extents=(size_x, size_y, _GROUND_THICKNESS),
        transform=trimesh.transformations.translation_matrix(ground_pos),
    )
    meshes.append(ground)

    # 2 horizontal bars
    for x_center in [front_x, back_x]:
        pos = (x_center, size_y / 2.0, crawl_height + bar_z / 2.0)
        bar = trimesh.creation.box(
            extents=(bar_x, size_y, bar_z),
            transform=trimesh.transformations.translation_matrix(pos),
        )
        meshes.append(bar)

    origin = np.array([size_x / 2.0, size_y / 2.0, 0.0])
    return meshes, origin
