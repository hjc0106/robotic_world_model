"""Terrain configurations and generators for the MBRL project."""

from .config import ROUGH_TERRAINS_CFG
from .height_field import HfClimbTerrainCfg, climb_terrain
from .mesh import MeshCrawlTerrainCfg, MeshTiltTerrainCfg, crawl_terrain, tilt_terrain
