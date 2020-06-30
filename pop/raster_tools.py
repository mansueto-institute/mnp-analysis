import numpy as np 
import rasterio 
import rasterio.features
from pathlib import Path 
import geopandas as gpd 
import pandas as pd 
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, LineString
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

# Roots
_ROOT = Path(__file__).resolve().parent.parent
DATA = _ROOT / "data"

# Type aliases for readability
ShapelyGeom = Union[MultiPolygon, Polygon, MultiLineString, LineString]

# Temporary paths
ls_path = Path("/home/cooper/Documents/chicago_urban/mnp/prclz-proto/data/LandScan_Global_2018/raw_tif/ls_2018.tif")
fb_path = Path("/home/cooper/Documents/chicago_urban/mnp/prclz-proto/data/Facebook_pop/population_sle/population_sle_2019-07-01.tif")

freetown_data = DATA / "Freetown"

blocks_path = freetown_data / "freetown_blocks_SLE.4.2.1_1.geojson"
#blocks = gdf.read_file(freetown_data / "freetown_blocks_SLE.4.2.1_1.geojson")
#

def load_raster_selection(raster_io: rasterio.io.DatasetReader,
                          geom_list: List[ShapelyGeom]) -> np.ndarray:
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
                       crs: CRS = CRS.from_epsg(4326)) -> None:
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
                           transform: affine.Affine) -> gpd.GeoDataFrame:
    '''
    Given an array of raster data and the transform, creates the correponsding
    vector reprseentation of the data
    ''' 

    geom_types = {
        'Polygon': Polygon
    }
    geom_type_multi = {
        'Polygon': MultiPolygon
    }

    if raster_data.dtype == np.float64:
        raster_data = raster_data.astype(np.float32)

    shape_gen = list(features.shapes(raster_data, transform=transform))
    gdf_data = {'geometry':[], 'data':[]}
    for geom_dict, val in shape_gen:
        geom_type = geom_types[geom_dict['type']]
        multi_type = geom_type_multi[geom_dict['type']]

        coords = geom_dict['coordinates']
        if len(coords) <= 1:
            geom = geom_type(*coords)
        else:
            # This can be done better
            geom = multi_type([geom_type(coords[i]) for i in range(len(coords))])
        
        gdf_data['geometry'].append(geom)
        gdf_data['data'].append(val)
    
    return gpd.GeoDataFrame(gdf_data)

def extract_aoi_data_from_raster(geometry_path: str, raster_path: str, 
                            out_path:str, save_geojson=True, save_tif=True) -> None:
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
    freetown_geoms = gpd.read_file(geometry_path)

    raster_data_selection, transform = load_raster_selection(raster_io, 
                                                    freetown_geoms['geometry'])
    
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

