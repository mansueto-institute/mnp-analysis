#!python3

from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely.wkt

from prclz.complexity import get_complexity, get_weak_dual_sequence


def read_data(path: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    df["geometry"]             = df["geometry"].apply(shapely.wkt.loads)
    df["geometry_buildings"]   = df["geometry_buildings"].apply(shapely.wkt.loads)
    df["centroids_multipoint"] = df["centroids_multipoint"].apply(shapely.wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf = gdf.drop(columns = [col for col in gdf.columns if col.startswith("Unnamed")])
    return gdf 

if __name__ == "__main__":
    filepath = Path(__file__).parent/"augmented_cdmx10.csv"
    gdf = read_data(filepath)
    gdf["new_calculated_k"] = [
        get_complexity(get_weak_dual_sequence(block, centroids))
        for (block, centroids) in
        gdf[["geometry", "centroids_multipoint"]].itertuples(index=False)
    ]
    