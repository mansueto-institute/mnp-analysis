import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
import shapely.wkt

def csv_to_geo(csv_path: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    df['geometry'] = df['geometry'].apply(shapely.wkt.loads)
    return gpd.GeoDataFrame(df)

def find(pt: Point, gdf: gpd.GeoDataFrame):
    return gdf[gdf.contains(pt)][[_ for _ in gdf.columns if _.startswith("GID")]]

def augment_buildings(blocks, buildings):
    block_aggregation = gpd.sjoin(blocks, buildings, how="right", op="intersects")
    block_aggregation = block_aggregation[pd.notnull(block_aggregation["index_left"])].groupby("index_left")["geometry"].agg(list)
    block_buildings   = blocks.join(block_aggregation, rsuffix="_buildings") 
    block_buildings["geometry_buildings"] = block_buildings["geometry_buildings"].map(lambda l: MultiPolygon([_ for _ in l if type(_) == Polygon]))
    return block_buildings

def main():
    # nairobi
    print(find(Point(36.821531, -1.288391), gpd.read_file("GADM/KEN/gadm36_KEN_3.shp")))
    # 989   KEN  KEN.30_1  KEN.30.16_1  KEN.30.16.2_1
    nairobi = augment_buildings(csv_to_geo("complexity/Africa/KEN/complexity_KEN.30.16.2_1.csv"), gpd.read_file("buildings/Africa/KEN/buildings_KEN.30.16.2_1.geojson"))
    nairobi.to_csv("~/augmented_nairobi.csv")

    # petare 
    print(find(Point(-66.798421, 10.47581), gpd.read_file("GADM/VEN/gadm36_VEN_2.shp")))
    # 181   VEN  VEN.16_1  VEN.16.3_1
    petare = augment_buildings(csv_to_geo("complexity/South-America/VEN/complexity_VEN.16.3_1.csv"), gpd.read_file("buildings/South-America/VEN/buildings_VEN.16.3_1.geojson"))
    petare.to_csv("~/augmented_petare.csv")

    # kathmandu 
    print(find(Point(85.328364, 27.709111), gpd.read_file("GADM/NPL/gadm36_NPL_4.shp")))
    # 97   NPL  NPL.1_1  NPL.1.1_1  NPL.1.1.3_1  NPL.1.1.3.31_1
    kathmandu = augment_buildings(csv_to_geo("complexity/Asia/NPL/complexity_NPL.1.1.3.31_1.csv"), gpd.read_file("buildings/Asia/NPL/buildings_NPL.1.1.3.31_1.geojson"))
    kathmandu.to_csv("~/augmented_kathmandu.csv")

    # mexico city (nezahualcóyotl)
    print(find(Point(-98.988952, 19.391044), gpd.read_file("GADM/MEX/gadm36_MEX_2.shp")))
    # 693   MEX  MEX.15_1  MEX.15.62_1
    cdmx = augment_buildings(csv_to_geo("complexity/North-America/MEX/complexity_MEX.15.62_1.csv"), gpd.read_file("buildings/North-America/MEX/buildings_MEX.15.62_1.geojson"))
    cdmx.to_csv("~/augmented_cdmx.csv")