from pathlib import Path

import geopandas as gpd
import mapclassify
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shapely.wkt
import shapely.geometry
from typing import Sequence

def is_feasible(block: shapely.geometry.Polygon, buildings: Sequence[shapely.geometry.Polygon]) -> bool:
    min_area_rects = (list(p.minimum_rotated_rectangle.exterior.coords) for p in buildings)
    return block.length > sum(min(shapely.geometry.LineString(rectangle[i:i+2]).length for i in range(len(rectangle)-1)) for rectangle in min_area_rects)

def analyze_feasibility(blocks: gpd.GeoDataFrame, complexity: gpd.GeoDataFrame): 
    blocks["area"] = blocks.area
    blocks["feasible"] = [is_feasible(block, buildings) for (_, block, buildings) in blocks[["block_geom", "geometry_buildings"]].itertuples()]
    
    fig, ax = plt.subplots(1, 1)
    blocks.plot("feasible", legend = True, ax = ax, categorical = True, cmap = "RdYlGn")
    plt.title("Feasiblity of Block Building Alignment (Freetown, SL)")
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()

def csv_to_geo(csv_path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    df.rename(columns={"geometry":"block_geom"}, inplace=True)
    df['block_geom'] = df['block_geom'].apply(shapely.wkt.loads)
    return gpd.GeoDataFrame(df, geometry='block_geom')

def main():
    blocks_path     = Path("../data/blocks_dgbuildings_SLE.4.2.1_1.csv")
    complexity_path = Path("../data/complexity_dg_SLE.4.2.1_1.csv")

    blocks = csv_to_geo(blocks_path)
    blocks["geometry_buildings"] = blocks["geometry_buildings"].apply(shapely.wkt.loads)
    complexity = csv_to_geo(complexity_path)


if __name__ == "__main__":
    main()
