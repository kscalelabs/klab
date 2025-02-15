"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,               # G1 sets this True
    size=(10.0, 10.0),
    border_width=20.0,
    border_height=1.0,            # G1 uses border_height=1.0
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    difficulty_range=(0.0, 1.0),   # G1 has this range

    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.2,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
            holes=False,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.02, 0.1),
            noise_step=0.02,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    }
)
