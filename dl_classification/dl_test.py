import cv2
from shapely.wkt import loads
import geopandas as gpd 
import pandas as pd
from pathlib import Path 
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
import numpy as np

from geopy import distance 
from scipy.ndimage.morphology import distance_transform_edt

from typing import List, Callable, Tuple

import torch
import torch.nn as nn 

"""
Utilities to convert a vector block representation
to a raster/segmentation map

TO-DO:
 - each image needs to be divisible by how much the model downsamples
   but we can't just resize so we need to crop to the desired size
"""

def csv_to_geo(csv_path, add_file_col=False) -> gpd.GeoDataFrame:
    '''
    Given a path to a block.csv file, returns as a GeoDataFrame
    '''

    df = pd.read_csv(csv_path, usecols=['block_id', 'geometry'])

    # Block id should unique identify each block
    assert df['block_id'].is_unique, "Loading {} but block_id is not unique".format(csv_path)

    df.rename(columns={"geometry":"block_geom"}, inplace=True)
    df['block_geom'] = df['block_geom'].apply(loads)

    if add_file_col:
        f = csv_path.split("/")[-1]
        df['gadm_code'] = f.replace("blocks_", "").replace(".csv", "")

    return gpd.GeoDataFrame(df, geometry='block_geom')

def get_block_buildings_files(country_code, gadm):

    blocks_root = Path("../data/blocks/Africa/")
    buildings_root = Path("../data/buildings/Africa/")
    complexity_root = Path("../data/complexity/Africa/")

    blocks_path = blocks_root / country_code 
    buildings_path = buildings_root / country_code 
    complexity_path = complexity_root / country_code

    block_file = "blocks_{}.csv".format(gadm)
    building_file = "buildings_{}.geojson".format(gadm)
    complexity_file = "complexity_{}.csv".format(gadm)

    block_file_path = blocks_path / block_file
    building_file_path = buildings_path / building_file
    complexity_file_path = complexity_path / complexity_file

    blocks_gdf = csv_to_geo(str(block_file_path))
    buildings_gdf = gpd.read_file(str(building_file_path))
    complexity_df = pd.read_csv(complexity_file_path, usecols=['block_id', 'complexity'])

    return blocks_gdf, buildings_gdf, complexity_df


def add_buildings_to_blocks(blocks_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame):

    bldgs = gpd.sjoin(buildings_gdf, blocks_gdf, how='left', op='intersects')
    bldgs = bldgs[['geometry', 'block_id']]
    bldgs = bldgs.groupby('block_id').agg(list)
    bldgs.rename(columns={'geometry': 'bldg_geom_list'}, inplace=True)
    bldgs.reset_index(inplace=True)
    bldgs = bldgs.merge(blocks_gdf, on='block_id')

    return bldgs

def create_raster_repr(block_geom: Polygon, building_list: List[Polygon], 
                       pixel_res_m=5, img_min_side=None, buffer_amt=0.001,
                       dt_block=False, dt_road=False) -> np.array:
    '''
    Creates a raster representation of the block
    '''

    # (1) Block segmap
    block_mask, to_pixel_coords_fn = block_to_mask(block_geom, pixel_res_m, 
                                                  img_min_side, buffer_amt)

    # (1a) fill in some regions of the block mask
    inv = (block_mask==0).astype(np.float32)
    cv2.floodFill(inv, None, (0,0), 0)
    block_mask = block_mask + inv 

    # (2) Building segmap
    building_mask = np.zeros_like(block_mask)

    for building_geom in building_list:
        points = building_geom.exterior.coords
        building_mask = draw_shape_in_img(points, building_mask, 
                          to_pixel_coords_fn)
    building_mask = np.flipud(building_mask)

    # (3) Road segmap
    #road_lines = LineString(block_geom.exterior.coords)
    road_mask = np.zeros_like(block_mask)
    road_mask = draw_shape_in_img(block_geom.exterior.coords, 
                                 road_mask, to_pixel_coords_fn, fill=False)
    road_mask = np.flipud(road_mask)

    # (4) Apply transforms and masking
    if dt_block:
        block_mask = binary_mask_to_dst(block_mask, invert=False)
    if dt_road:
        road_mask = binary_mask_to_dst(road_mask)

    return block_mask, building_mask, road_mask

def view_masks(m0, m1, m2):

    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(m0)
    ax[1].imshow(m1)
    ax[2].imshow(m2)
    return fig, ax 

def create_coord_to_pixel_map(x0, y0, x1, y1, H, W):

    delta_x = (x1-x0)/H 
    delta_y = (y1-y0)/W

    def to_pixel_coords(x, y):

        x = (x-x0) / delta_x
        y = (y-y0) / delta_y
        return int(x), int(y)
    return to_pixel_coords

def block_to_mask(block_geom, pixel_res_m=5, img_min_side=None, buffer_amt=0):
    '''
    Converts block polygon to mask which denotes
    the region inside the block

    '''

    rect = (block_geom.buffer(buffer_amt)).envelope
    rect_coords = list(rect.exterior.coords)
    y0,x0 = rect_coords[0]
    y1,x1 = rect_coords[2]

    # Determine size based on specified per-meter resoultion
    if pixel_res_m is not None:
        w = distance.distance((y1, x0), (y0, x0)).meters
        h = distance.distance((y1, x0), (y1, x1)).meters

        W = int(w / pixel_res_m)
        H = int(h / pixel_res_m)

    # Can also determine through image size
    if img_min_side is not None:
        w = np.abs(y1-y0)
        h = np.abs(x1-x0)

        if h > w:
            ratio = h/w
            W = img_min_side
            H = int(W*ratio)
        else:
            ratio = w/h
            H = img_min_side
            W = int(H*ratio)

    img_size = (H, W)

    # Create the map from geom points -> pixels
    to_pixel_coords_fn = create_coord_to_pixel_map(x0, y0, x1, y1, H, W)

    rect_pixels = [to_pixel_coords_fn(p[1], p[0]) for p in rect_coords]

    img = np.zeros(img_size)
    mask = draw_shape_in_img(block_geom.exterior.coords, img, to_pixel_coords_fn)
    mask = np.flipud(mask)

    return mask, to_pixel_coords_fn

def draw_shape_in_img(point_list: List[Tuple], img: np.array, 
                      to_pixel_coords_fn: Callable, fill=True) -> np.array:
    '''
    Given some point_list from a Shapely geometry, draws that shape
    into a raster np.array
    '''

    shape_pixels = [to_pixel_coords_fn(p[1], p[0]) for p in point_list]
    
    tmp = []
    for p in shape_pixels:
        tmp.append((p[1], p[0]))
    shape_pixels = tmp 

    shape_pixels = np.array(shape_pixels, np.int32)
    shape_pixels = shape_pixels.reshape((-1,1,2))

    if fill:
        mask = cv2.fillPoly(img, [shape_pixels], 1)
    else:
        mask = cv2.polylines(img, [shape_pixels], thickness=3, isClosed=True, color=1)

    return mask


def binary_mask_to_dst(binary_mask: np.array, invert=True) -> np.array:
    '''
    Converts a 0-1 binary mask to a distance representation
    '''

    if invert:
        binary_mask = (binary_mask==0).astype(np.float32)
    dst_mask = distance_transform_edt(binary_mask) 
    return dst_mask



class BlocksDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 blocks_gdf: gpd.GeoDataFrame,
                 buildings_gdf: gpd.GeoDataFrame,
                 complexity_df: pd.DataFrame,
                 buffer_amt=0.0001, 
                 pixel_res_m=2, 
                 apply_ln=True):
        super(BlocksDataset, self).__init__()

        self.pixel_res_m = pixel_res_m 
        self.buffer_amt = buffer_amt

        #blocks_gdf, buildings_gdf, complexity_df = get_block_buildings_files(country_code, gadm)
        
        self.bldgs = add_buildings_to_blocks(blocks_gdf, buildings_gdf)
        self.complexity_df = complexity_df
        self.complexity_df.set_index('block_id', inplace=True)
        self.apply_ln = apply_ln

        # del blocks_gdf
        # del buildings_gdf
    @staticmethod
    def from_params(country_code: str, 
                    gadm: str, 
                    buffer_amt=0.0001, 
                    pixel_res_m=2, 
                    apply_ln=True):

        self.country_code = country_code
        self.gadm = gadm 
        blocks_gdf, buildings_gdf, complexity_df = get_block_buildings_files(country_code, gadm)

        return BlocksDataset(blocks_gdf, buildings_gdf, complexity_df, 
                             buffer_amt, pixel_res_m, apply_ln)


    @staticmethod
    def from_paths(blocks_path: str, 
                   buildings_path: str, 
                   complexity_path: str,
                   buffer_amt=0.0001, 
                   pixel_res_m=2, 
                   apply_ln=True):

        blocks_gdf = csv_to_geo(str(blocks_path))
        buildings_gdf = gpd.read_file(buildings_path)
        complexity_df = pd.read_csv(complexity_path, usecols=['block_id', 'complexity'])

        return BlocksDataset(blocks_gdf, buildings_gdf, complexity_df, 
                             buffer_amt, pixel_res_m, apply_ln)

    def __len__(self):
        return len(self.bldgs) 


    def __getitem__(self, idx):
        block_geom = self.bldgs.iloc[idx]['block_geom']
        building_list = self.bldgs.iloc[idx]['bldg_geom_list']
        pixel_res_m = self.pixel_res_m
        buffer_amt = self.buffer_amt

        block_mask, building_mask, road_mask = create_raster_repr(block_geom, 
                                           building_list, buffer_amt=buffer_amt, 
                                           dt_road=True, pixel_res_m=pixel_res_m, dt_block=True) 

        block_id = self.bldgs.iloc[idx]['block_id']
        complexity = self.complexity_df.loc[block_id]['complexity']

        # Now some final processing
        block_mask = torch.log(torch.from_numpy(block_mask))
        road_mask = torch.log(torch.from_numpy(road_mask))
        building_mask = torch.log(torch.from_numpy(building_mask.copy()))

        data = {'block_mask': block_mask, 
                'building_mask': building_mask,
                'road_mask': road_mask,
                }

        return data, complexity

class ConvBlock(nn.Module):

    def __init__(self, ch_in: int, ch_out: int, k=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(ch_in, ch_out, k, padding=padding, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x 

class ComplexityFeatureExtractor(nn.Module):

    def __init__(self, input_ch: int, inter_ch: List[int]):
        super(ComplexityFeatureExtractor, self).__init__()

        self.chs = inter_ch
        self.chs.insert(0, input_ch)
        self.layer_count = len(self.chs) - 1

        # Convolutional block
        for i in range(self.layer_count):
            ch_in = self.chs[i]
            ch_out = self.chs[i+1]
            self.add_module(f'conv{i}', ConvBlock(ch_in, ch_out, 3))
        
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        '''
        Input is a n-channel image representing an urban area
        '''      

        for i in range(self.layer_count):
            conv = self.getattr(f'conv{i}')
            x = self.pool(conv(x))
        return x 


class ComplexityClassifier(nn.Module):

    def __init__(self, input_res: int, fc_chs: List[int]):

        self.input_shape = (input_res, input_res)
        self.adt_pool = nn.AdaptiveMaxPool2d(output_size=self.input_shape)
        self.fc_chs = fc_chs
        self.fc_chs.insert(0, input_res*input_res)
        self.layer_count = len(self.fc_chs) - 1

        # Classifier
        for i in range(self.layer_count):
            fc_in = self.fc_chs[i]
            fc_out = self.fc_chs[i+1]
            self.add_module(f'fc{i}', nn.Linear(fc_in, fc_out))

        self.act = nn.ReLU()

        # Need to add the final classier
        # Make this a regression problem? Or classification?
        # Probably just a classification problem


# country_code = 'SLE'
# gadm = 'SLE.4.2.1_1'
# dataset = BlocksDataset(country_code, gadm)

input_ch = 3
inter_ch = [32, 64, 128, 256, 512, 512]




