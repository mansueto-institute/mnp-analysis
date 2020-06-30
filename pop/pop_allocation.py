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
def allocate_population(buildings_gdf: gpd.GeoDataFrame,
                        population_gdf: gpd.GeoDataFrame,
                        pop_variable: str,
                        allocate_total=True) -> gpd.GeoDataFrame:
    
    buildings_with_pop = raster_tools.simple_area_interpolate(population_gdf, 
                                                 buildings_gdf, 
                                                 extensive_variables=[pop_variable], 
                                                 allocate_total=allocate_total)

    buildings_gdf[pop_variable] = buildings_with_pop[pop_variable]
    return buildings_gdf


def fast_allocate_population(buildings_gdf: gpd.GeoDataFrame,
                             population_gdf: gpd.GeoDataFrame,
                             pop_variable: str) -> gpd.GeoDataFrame:
    
    # Map each building to the pop geom it is in
    for k in ['index_right', 'index_left']:
        for df in [buildings_gdf, population_gdf]:
            if k in df.columns:
                df.drop(columns=[k], inplace=True)

    population_gdf['pop_id'] = np.arange(population_gdf.shape[0])
    buildings_gdf['bldg_id'] = np.arange(buildings_gdf.shape[0])
    geo = gpd.sjoin(buildings_gdf, population_gdf,
                              how='left', op='intersects')

    # Numerator is the buildings area
    geo['num_area'] = geo.geometry.area

    # Denom is the area of all buildings in that pop_id
    geo_by_pop_id = geo[['pop_id', 'num_area']].groupby('pop_id').sum()
    geo_by_pop_id.rename(columns={'num_area': 'den_area'}, inplace=True)
    geo_by_pop_id.reset_index(inplace=True)
    
    # Merge the denom and generate the factor
    geo = geo.merge(geo_by_pop_id, on='pop_id', how='left')
    geo['alloc_factor'] = geo['num_area'] / geo['den_area']
    geo['bldg_pop'] = geo['alloc_factor'] * geo[pop_variable]

    return geo





