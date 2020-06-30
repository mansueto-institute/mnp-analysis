from tobler import area_weighted
from tobler.area_weighted import area_tables, area_interpolate
import geopandas as gpd 
import numpy as np 
from typing import Tuple

from . import raster_tools

# (1) Map each building to a block

def join_block_building(block_gdf: gpd.GeoDataFrame,
	                    buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

	buildings_gdf = gpd.sjoin(buildings_gdf, block_gdf,
		                      how='left', op='intersects')
	return buildings_gdf

def load_blocks_buildings(block_path: str, 
	                      building_path: str,
	                      merge_bldgs=False) -> Tuple[gpd.GeoDataFrame]:
	
	buildings_gdf = gpd.read_file(building_path)
	blocks_gdf = gpd.read_file(block_path)

	if merge_bldgs:
		buildings_gdf = join_block_building(blocks_gdf, buildings_gdf)
	return blocks_gdf, buildings_gdf


def load_landscan_geojson(landscan_path: str) -> gpd.GeoDataFrame:

	pop_ls = gpd.read_file(landscan_path)
	gt0 = pop_ls['data'] > 0
	pop_ls['data'] = pop_ls['data'] * gt0

	return pop_ls 


# (2) Do allocation










