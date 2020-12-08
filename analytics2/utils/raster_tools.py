import numpy as np 
import rasterio 
import rasterio.features
from pathlib import Path 
import geopandas as gpd 
import pandas as pd 
from shapely.geometry import MultiPolygon, Point, Polygon, MultiLineString, LineString
from rasterio.crs import CRS 
from rasterio import features 
from shapely.wkt import loads
import geopandas as gdf 
from typing import Union, List
import affine 
import numpy.ma as ma 
from tobler import area_weighted 
from tobler.area_weighted import area_tables, area_interpolate
from tobler.area_weighted import area_tables, area_interpolate
from tobler.util.util import _check_crs, _nan_check, _check_presence_of_crs
import pandas as pd

from . import utils

# Roots
_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "data"

# Type aliases for readability
ShapelyGeom = Union[MultiPolygon, Polygon, MultiLineString, LineString]


def load_raster_selection(raster_io: rasterio.io.DatasetReader,
                          geom_list: List[ShapelyGeom],
                          ) -> np.ndarray:
    '''
    rasterio allows loading of subselection of a tiff file, so 
    given a list of geometries and an open raster DatasetReader,
    loads in the values within the geom_list.
    This allows for loading of small selections from otherwise 
        huge tiff's
    '''

    if not isinstance(geom_list, List):
        geom_list = [geom_list]

    # Find the window around the geom_list
    window = rasterio.features.geometry_window(raster_io, geom_list)
    transform = raster_io.window_transform(window)

    # Perform the windowed read
    sub_data = raster_io.read(window=window)
    return sub_data, transform

def save_np_as_geotiff(np_array: np.ndarray, 
                       transform: affine.Affine,
                       out_file: Path,
                       crs: CRS = CRS.from_epsg(4326),
                       ) -> None:
    '''
    Given a numpy array of data, and transform and crs defining
    the coordinate system, save as a geotiff
    '''

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    c, h, w = np_array.shape

    new_file = rasterio.open(
        out_file,
        'w',
        driver='GTiff',
        height=h,
        width=w,
        count=c,
        dtype=np_array.dtype,
        crs=crs,
        transform=transform)

    #print("shape = {}".format(np_array.shape))
    new_file.write(np_array)
    print("Saved GeoTiff at: {}".format(out_file))


def raster_to_geodataframe(raster_data: np.ndarray,
                           transform: affine.Affine,
                           window: rasterio.windows.Window,
                           raster_io: rasterio.io.DatasetReader,
                           ) -> gpd.GeoDataFrame:
    
    if raster_data.ndim == 3:
        assert raster_data.shape[0] == 1, "ERROR - can't handle multichannel yet"
        raster_data = raster_data[0]

    # To convert when our raster_data is from a windowed
    # read we need to go from the windowed raster_data 
    # coords -> orig dataset coords -> x,y spatial
    data_h, data_w = raster_data.shape 
    all_geoms = []
    data = []
    for h in range(data_h):
        for w in range(data_w):
            # Convert to the coord sys of the raster_io dataset
            abs_h = h + window.row_off 
            abs_w = w + window.col_off 

            # And use its xy method now
            # abs_x1, abs_y1 = raster_io.xy(abs_h, abs_w)
            # abs_x0, abs_y0 = raster_io.xy(abs_h-1, abs_w-1)
            p_ul = Point(raster_io.xy(abs_h, abs_w, offset='ul'))
            p_lr = Point(raster_io.xy(abs_h, abs_w, offset='lr'))
            geom = MultiPoint([p_ul, p_lr]).envelope
            all_geoms.append(geom)
            data.append(raster_data[h,w])
    return gpd.GeoDataFrame.from_dict({'geometry': all_geoms,
                                       'pop': data
                                       })


def extract_aoi_data_from_raster(geometry_path: str, 
                                 raster_path: str, 
                                 out_path: str, 
                                 save_geojson: bool = True, 
                                 save_tif: bool = True,
                                 ) -> None:
    '''
    Extracts the relevant data from a larger raster tiff, per the geometries
    in geometry_path file and saves out.

    Inputs:
        - geometry_path (str) geojson file of vector geometries
        - raster_path (str) tiff file of raster data
        - out_path (str) path to save extracted data
    '''
    tif_path = Path(out_path)
    geojson_path = tif_path.with_suffix('.geojson')

    raster_io = rasterio.open(raster_path)
    if geometry_path.suffix == ".csv":
        aoi_geoms = utils.load_csv_to_geo(geometry_path)
    else:
        aoi_geoms = gpd.read_file(geometry_path)

    raster_data_selection, transform = load_raster_selection(raster_io, 
                                                    aoi_geoms['geometry'])
    
    if save_tif:
        save_np_as_geotiff(raster_data_selection, transform, str(tif_path))
    if save_geojson:
        gdf_data = raster_to_geodataframe(raster_data_selection, transform)
        gdf_data.to_file(geojson_path, driver='GeoJSON')
        print("Saved GeoJSON at: {}".format(geojson_path))

def fix_invalid_polygons(geom: Union[Polygon, MultiPolygon]) -> Polygon:
    '''
    Fix self-intersection polygons
    '''
    if geom.is_valid:
        return geom 
    else:
        return geom.buffer(0)


def simple_area_interpolate(
    source_df,
    target_df,
    extensive_variables=None,
    intensive_variables=None,
    tables=None,
    allocate_total=True,
    ):
    """
 
    """

    SU, UT = area_tables(source_df, target_df)

    den = source_df["geometry"].area.values
    if allocate_total:
        den = SU.sum(axis=1)
    den = den + (den == 0)
    weights = np.dot(np.diag(1 / den), SU)

    dfs = []
    extensive = []
    if extensive_variables:
        for variable in extensive_variables:
            vals = _nan_check(source_df, variable)
            estimates = np.dot(np.diag(vals), weights)
            estimates = np.dot(estimates, UT)
            estimates = estimates.sum(axis=0)
            extensive.append(estimates)
    extensive = np.array(extensive)
    extensive = pd.DataFrame(extensive.T, columns=extensive_variables)

    ST = np.dot(SU, UT)
    area = ST.sum(axis=0)
    den = np.diag(1.0 / (area + (area == 0)))
    weights = np.dot(ST, den)
    intensive = []
    if intensive_variables:
        for variable in intensive_variables:
            vals = _nan_check(source_df, variable)
            vals.shape = (len(vals), 1)
            est = (vals * weights).sum(axis=0)
            intensive.append(est)
    intensive = np.array(intensive)
    intensive = pd.DataFrame(intensive.T, columns=intensive_variables)

    if extensive_variables:
        dfs.append(extensive)
    if intensive_variables:
        dfs.append(intensive)

    df = pd.concat(dfs, axis=0)

    data = {}
    for c in extensive:
        data[c] = df[c].values
    data['geometry'] = target_df['geometry'].values

    df = gpd.GeoDataFrame(data)
    return df

def allocate_population(buildings_gdf: gpd.GeoDataFrame,
                        population_gdf: gpd.GeoDataFrame,
                        pop_variable: str,
                        ) -> gpd.GeoDataFrame:
    
    for k in ['index_right', 'index_left']:
        for df in [buildings_gdf, population_gdf]:
            if k in df.columns:
                df.drop(columns=[k], inplace=True)

    # Map each building to the pop geom it is in
    in_cols = list(buildings_gdf.columns)
    population_gdf['pop_id'] = np.arange(population_gdf.shape[0])
    buildings_gdf['bldg_id'] = np.arange(buildings_gdf.shape[0])
    geo = gpd.sjoin(buildings_gdf, population_gdf,
                    how='left', op='intersects')[['bldg_id', 'geometry', 'pop_id', 'data']]
    
    # Handle building geoms intersecting multiple pop geoms
    # Split building into each piece, allocate, then sum by bldg_id
    geo = geo.join(population_gdf[['pop_id', 'geometry']], on='pop_id', how='left', rsuffix='_pop')
    geo['obs_count'] = 1
    bldg_count = geo[['bldg_id', 'obs_count']].groupby('bldg_id').sum()
    bldg_count.reset_index(inplace=True)
    geo.drop(columns=['obs_count'], inplace=True)
    geo = geo.merge(bldg_count, on='bldg_id', how='left')

    fn = lambda s: s['geometry'].intersection(s['geometry_pop'])
    geo['unique_geom'] = geo.apply(fn, axis=1)
    geo.set_geometry('unique_geom', inplace=True)

    # Numerator is the buildings area
    geo['num_area'] = geo.geometry.area

    # Denom is the area of all buildings in that pop_id
    geo_by_pop_id = geo[['pop_id', 'num_area']].groupby('pop_id').sum()
    geo_by_pop_id.rename(columns={'num_area': 'den_area'}, inplace=True)
    geo_by_pop_id.reset_index(inplace=True)
    
    # Merge the denom and generate the factor and allocation pop
    geo = geo.merge(geo_by_pop_id, on='pop_id', how='left')
    geo['alloc_factor'] = geo['num_area'] / geo['den_area']
    geo['bldg_pop'] = geo['alloc_factor'] * geo[pop_variable]

    # Because we split the bldg geoms w > 1 intersection, sum
    # up by bldg_id to reassemble
    bldg_pop = geo[['bldg_pop', 'bldg_id']].groupby('bldg_id').sum().reset_index()
    assert bldg_pop.shape[0] == buildings_gdf.shape[0], "ERROR - bldg count IN = {} but bldg count OUT = {}".format(buildings_gdf.shape[0], bldg_pop.shape[0] )
    in_cols.append('bldg_id')
    buildings_gdf = buildings_gdf[in_cols].merge(bldg_pop, on='bldg_id', how='left')

    return buildings_gdf


if __name__ == "__main__":
    
    freetown_ls = Path("../data/Freetown/Freetown_landscan.tif")
    freetown_fb = Path("../data/Freetown/Freetown_facebook.tif")

    # extract_aoi_data_from_raster(blocks_path, ls_path, freetown_ls)
    # extract_aoi_data_from_raster(blocks_path, fb_path, freetown_fb)


    blocks = gdf.read_file(blocks_path)

    # (1) Landscan and Facebook pop apply to blocks, via block area
    pop_ls = gpd.read_file(freetown_ls.with_suffix('.geojson'))
    gt0 = pop_ls['data'] > 1
    pop_ls['data'] = pop_ls['data'] * gt0
    ls_blocks_est = area_interpolate(pop_ls, blocks, extensive_variables=['data'])

    pop_fb = gpd.read_file(freetown_fb.with_suffix('.geojson'))
    pop_fb['geometry'] = pop_fb['geometry'].apply(fix_invalid_polygons)
    fb_blocks_est = area_interpolate(pop_fb, blocks, extensive_variables=['data'])

    blocks_diff = ls_blocks_est['data'] - fb_blocks_est['data']
    gdf_diff = gpd.GeoDataFrame({'geometry':blocks['geometry'], 'difference': blocks_diff})

