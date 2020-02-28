import glob
import sys
from typing import List

import geopandas as gpd
import pandas as pd
import prclz.utils
import shapely
import tqdm


def read_complexity(k_path: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(k_path)
    df["geometry"] = df['geometry'].map(shapely.wkt.loads)
    return gpd.GeoDataFrame(df, geometry='geometry')

def intersect(slums, k):
    try: 
        return gpd.sjoin(slums, k, op="intersects")[k.columns]
    except Exception as e:
        print(e)
        return gpd.GeoDataFrame()

def search(slums: gpd.GeoDataFrame, k_root: str) -> List[gpd.GeoDataFrame]:
    return pd.concat([intersect(slums, read_complexity(k)) for k in tqdm.tqdm(glob.glob(k_root))]).pipe(gpd.GeoDataFrame)

if __name__ == "__main__":
    slums_path, k_root = sys.argv[1:3]

    slums = gpd.read_file(slums_path)
    slums.geometry = slums['section_C/C2_Boundary'].map(prclz.utils.parse_ona_text)
    slums = slums[slums.geometry.is_valid]

    intersected = search(slums, k_root)
    intersected