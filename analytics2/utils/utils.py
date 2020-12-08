import geopandas as gpd 
import pandas as pd 
from shapely.wkt import loads
from typing import Tuple

'''
FILE DESCRIPTION:
General purpose utilities across
various applications
'''

def join_block_building(block_gdf: gpd.GeoDataFrame,
                        buildings_gdf: gpd.GeoDataFrame,
                        ) -> gpd.GeoDataFrame:

    buildings_gdf = gpd.sjoin(buildings_gdf, block_gdf,
                              how='left', op='intersects')
    return buildings_gdf

def load_blocks_buildings(block_path: str, 
                          building_path: str,
                          merge_bldgs=False,
                          ) -> Tuple[gpd.GeoDataFrame]:
    
    buildings_gdf = gpd.read_file(building_path)
    blocks_gdf = gpd.read_file(block_path)

    if merge_bldgs:
        buildings_gdf = join_block_building(blocks_gdf, buildings_gdf)
    return blocks_gdf, buildings_gdf


def load_csv_to_geo(csv_path: str, 
	                add_file_col: bool = False,
	                crs: str = "EPSG:4326",
	                ) -> gpd.GeoDataFrame:
    '''
    Loads the csv and returns a GeoDataFrame
    '''
    df = pd.read_csv(csv_path, usecols=['block_id', 'geometry'])
    df['geometry'] = df['geometry'].apply(loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.crs = crs
    gdf['geometry'].crs = crs
    return gdf 
