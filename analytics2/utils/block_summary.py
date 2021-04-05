import numpy as np 
import geopandas as gpd 
import pandas as pd 
from shapely.wkt import loads
from typing import Tuple, Union, List
from pathlib import Path 
import argparse
from pygeos import GEOSException
import time

#from . import utils, block_stats
import utils
import block_stats
from raster_tools import extract_aoi_data_from_raster, allocate_population

'''
TO-DO:
  - make_summary function should also merge in complexity and reblocking data
  - make_summary function should save out final and intermediate outputs, as desired
  - maybe add a command line interface for most basic functionality?
  - test on data on Midway
'''


def load_gadm_file(gadm_dir: str) -> gpd.GeoDataFrame:
    """
    Loads in a GADM gep df from file, sorts the index and returns
    just the index and geometry. 
    """
    sort_fn = lambda s: -int(s.stem.split("_")[-1])
    gadm_dir = Path(gadm_dir)
    shp_files = [p for p in gadm_dir.iterdir() if p.suffix == '.shp']
    shp_files.sort(key=sort_fn)
    gdf = gpd.read_file(str(shp_files[0]))
    n = abs(sort_fn(shp_files[0]))
    s = 'GID_{}'.format(n)
    gdf.rename(columns={s: 'gadm'}, inplace=True)
    gdf = gdf[['gadm', 'geometry']]
    return gdf 


def get_gadm_list(aoi_gdf: gpd.GeoDataFrame, 
                  gadm_dir: str,
                  ) -> List[str]:
    '''
    Given an AoI defined in a dataframe and a directory 
    to the GADM folder, gets the GADMs that intersect with
    the AoI and thus are relevant when constructing datasets
    that cover the AoI
    '''

    gadm_gdf = load_gadm_file(gadm_dir)
    aoi_geom = aoi_gdf.unary_union
    gadm_gdf['i_aoi'] = gadm_gdf.intersects(aoi_geom)
    gadm_inter = gadm_gdf.loc[gadm_gdf['i_aoi']==True][['gadm', 'geometry']]
    
    return list(gadm_inter['gadm'].values)


def get_geoms_intersecting_aoi(aoi_gdf: gpd.GeoDataFrame,
                               target_geoms_dir: str,
                               gadm_list: List[str] = None,
                               ) -> gpd.GeoDataFrame:
    '''
    Given some AoI geometry and a directory of target geometry files,
    will create a geodataframe of all observations within the set
    of target geometry files that intersect with the AoI.
    NOTE: if gadm_list is not None, then will only search the files
          matching 
    '''

    aoi_geom = aoi_gdf.unary_union
    target_geoms_dir = Path(target_geoms_dir)
    possible_files = list(target_geoms_dir.iterdir())
    if gadm_list is None:
        target_files = possible_files
    else:
        # Do flexible matching of gadm with files in directory
        gadm0 = gadm_list[0]
        i = 0
        f_name = possible_files[i].name 
        while gadm0 not in f_name:
            i += 1
            f_name = possible_files[i].name 
        target_files = []

        # We have discovered the relationship btwn gadm and filename
        for gadm in gadm_list:
            gadm_fname = f_name.replace(gadm0, gadm)
            gadm_path = target_geoms_dir / gadm_fname
            target_files.append(gadm_path)
            #assert gadm_path.is_file(), "ERROR: matched {} to {} but file does not exist".format(gadm, gadm_path)

    aoi_selection = None
    # Now assemble the geometries
    for i, geom_path in enumerate(target_files):
        if geom_path.suffix == ".geojson":
            gdf = gpd.read_file(str(geom_path))
        else:
            gdf = utils.load_csv_to_geo(str(geom_path))
        try:
            gdf = gdf[gdf['geometry'].intersects(aoi_geom)]
        except GEOSException:
            gdf.geometry = gdf.geometry.buffer(0)
            gdf = gdf[gdf['geometry'].intersects(aoi_geom)]
        if gdf.shape[0] > 0:
            if aoi_selection is None:
                aoi_selection = gdf 
            else:
                aoi_selection = aoi_selection.append(gdf)
    return aoi_selection


def make_summary(aoi_path: str,
                 landscan_path: str,
                 buildings_dir: str,
                 blocks_dir: str,
                 gadm_dir: str,
                 summary_out_path: str,
                 ):
    """
    Creates a summary of the given area using its population (?), blocks,
    and boundary lines. 
    """
    # (1) Allocate Landscan
    aoi_path = Path(aoi_path)
    if aoi_path.suffix == '.csv':
        aoi_gdf = utils.load_csv_to_geo(aoi_path)
    else:
        aoi_gdf = gpd.read_file(str(aoi_path))
    gadm_list = get_gadm_list(aoi_gdf, gadm_dir)
    print("GADM list: {}".format(gadm_list))

    _, aoi_ls = extract_aoi_data_from_raster(aoi_gdf, landscan_path, save_geojson=False, save_tif=False)
    aoi_ls_bldgs = get_geoms_intersecting_aoi(aoi_ls, buildings_dir, gadm_list)

    bldg_pop_alloc = allocate_population(aoi_ls_bldgs, aoi_ls, 'pop')

    #return aoi_ls_bldgs, bldg_pop_alloc

    # (2) Now assemble the other data
    aoi_blocks = get_geoms_intersecting_aoi(aoi_gdf, blocks_dir, gadm_list)
    if 'block_id' not in aoi_blocks.columns:
        aoi_blocks['block_id'] = aoi_blocks.index.tolist()
    aoi_bldg_summary = block_stats.make_aoi_summary(bldg_pop_alloc, aoi_blocks)
    block_cols = [x for x in aoi_bldg_summary.columns if "block" in x]
    aoi_block_stats = aoi_bldg_summary[block_cols].drop_duplicates()
    aoi_block_summary = aoi_blocks.merge(aoi_block_stats, how='left', on='block_id')

    # (3) Save
    summary_out_path = Path(summary_out_path)
    fname = summary_out_path.stem
    outdir = summary_out_path.parent
    outdir.mkdir(exist_ok=True, parents=True)

    aoi_block_summary.to_file(str(summary_out_path), driver='GeoJSON')
    print("Saved to: {}".format(str(summary_out_path)))
    
    block_bldgs_out_path = outdir / (fname + "-bldgs.geojson")
    aoi_bldg_summary.to_file(str(block_bldgs_out_path), driver='GeoJSON')
    print("Saved to: {}".format(str(aoi_bldg_summary)))

    return aoi_block_summary, aoi_bldg_summary


if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser(description='Make block-level and building-level summary for Area of Interest')
    parser.add_argument('--aoi_path', required=True, type=str, help='Path to geometry which defines AoI')
    parser.add_argument('--landscan_path', required=True, type=str, help='Path to Landscan tif file')
    parser.add_argument('--buildings_dir', required=True, type=str, help='Dir to buildings geomtries')
    parser.add_argument('--blocks_dir', required=True, type=str, help='Dir to blocks geometries')
    parser.add_argument('--gadm_dir', required=True, type=str, help='Dir to GAMD geometries')
    parser.add_argument('--summary_out_path', required=True, type=str, help='Path to save block summary')

    args = parser.parse_args()
    make_summary(**vars(args))
    t1 = time.time()
    print(f"block summary took {t1-t0}")
