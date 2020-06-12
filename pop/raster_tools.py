import numpy as np 
import rasterio 
import rasterio.features
from pathlib import Path 
import geopandas as gpd 
import pandas as pd 
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, LineString
from rasterio.crs import CRS 
from shapely.wkt import loads
import geopandas as gdf 
from typing import Union, List
import affine 

# Roots
_ROOT = Path(__file__).resolve().parent.parent
DATA = _ROOT / "data"

# Type aliases for readability
ShapelyGeom = Union[MultiPolygon, Polygon, MultiLineString, LineString]

# Temporary paths
ls_path = Path("/home/cooper/Documents/chicago_urban/mnp/prclz-proto/data/LandScan_Global_2018/raw_tif/ls_2018.tif")

freetown_data = DATA / "Freetown"

blocks_path = freetown_data / "freetown_blocks_SLE.4.2.1_1.geojson"
blocks = gdf.read_file(freetown_data / "freetown_blocks_SLE.4.2.1_1.geojson")
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

	print("shape = {}".format(np_array.shape))
	new_file.write(np_array)
	# for i in range(c):
	# 	new_file.write(np_array, c)
	print("Saved GeoTiff at: {}".format(out_file))


def gen_freetown_LandScan(blocks_path: str, landscan_path: str) -> None:
	'''
	Save out the raster tiff sub-selection that is
	the Freetown population
	'''

	raster_io = rasterio.open(landscan_path)
	freetown_geoms = gpd.read_file(blocks_path)

	freetown_pop, transform = load_raster_selection(raster_io, 
		                                            freetown_geoms['geometry'])
	out_file = "../data/Freetown/Freetown_landscan.tiff"
	save_np_as_geotiff(freetown_pop, transform, out_file)

gen_freetown_LandScan(blocks_path, ls_path)


