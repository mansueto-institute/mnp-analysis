from pathlib import Path

import geopandas as gpd
import mapclassify
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shapely.wkt
from matplotlib.colors import LinearSegmentedColormap

mnp_palette = LinearSegmentedColormap.from_list("mnp", [
    (0, "#0571b0"), (2/19.0, "#f7f7f7"), (5/19.0, "#f4a582"), (19/19.0, "#cc0022")])

def visualize_complexity(dg: gpd.GeoDataFrame, osm: gpd.GeoDataFrame):
    # get k range 
    max_k = max(dg.complexity.max(), osm.complexity.max())
    min_k = 0
    
    # visualize complexity distribution
    fig, ax = plt.subplots(1, 2, sharey='row', sharex='col')
    bins = range(0, max_k + 1, 1)
    dg.complexity.hist( ax = ax[0], bins = bins)
    osm.complexity.hist(ax = ax[1], bins = bins)
    ax[0].set_title("DigitalGlobe")
    ax[1].set_title("OpenStreetMap")
    fig.suptitle("Histogram of $k$-values for Freetown, SL")
    plt.semilogy()
    # plt.savefig("k_histogram_sl.png", dpi=300, bbox_inches="tight")
    plt.show()

    # complexity choropleth 
    fig, ax = plt.subplots(1, 2, sharey='row', sharex='col')
    plot_opts = { 
        "column"   : "complexity",
        "cmap"     : mnp_palette,
        # "scheme"   : "natural_breaks",
        "scheme"   : "User_Defined",
        "edgecolor": "grey",
        "linewidth": 0.1,
        "legend"   : True,
        "categorical" : False,
        "markersize" : 0,
        "classification_kwds" : {"bins" : [6, 15, 31, 76, max_k]} # numbers taken from natural_breaks
    }
    dg.plot( ax=ax[0],  **plot_opts)
    ax[0].get_xaxis().set_visible(False)
    ax[0].axes.get_yaxis().set_visible(False)

    # dummy polygons to make choropleth color scheme consistent 
    osmd = gpd.GeoDataFrame(pd.concat([pd.DataFrame([ 
        ["d0", shapely.geometry.MultiPoint(osm.centroid).centroid, 0,     [], 0], 
        ["d1", shapely.geometry.MultiPoint(osm.centroid).centroid, 2,     [], 0], 
        ["d2", shapely.geometry.MultiPoint(osm.centroid).centroid, 5,     [], 0], 
        ["d3", shapely.geometry.MultiPoint(osm.centroid).centroid, 19,    [], 0], 
        ["d4", shapely.geometry.MultiPoint(osm.centroid).centroid, max_k, [], 0] 
    ], columns = osm.columns), osm.copy()]).reset_index(), geometry = "block_geom")

    osmd.plot(ax=ax[1], **plot_opts)
    ax[1].get_xaxis().set_visible(False)
    ax[1].axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(left = 0.02, bottom = 0.02, right = 0.98, top = 0.98, wspace = 0.10)
    ax[0].set_title("DigitalGlobe")
    ax[1].set_title("OpenStreetMap")
    fig.suptitle("Complexity Choropleth for Freetown, SL")
    # plt.savefig("k_choropleth_sl.png", dpi=300, bbox_inches="tight")
    plt.show()

    # get block-by-block diff of k values
    diff = dg.join(osm, rsuffix="_osm")
    diff["delta_k"] = diff["complexity"] - diff["complexity_osm"]
    diff.plot("delta_k", legend=True, cmap="PRGn", scheme = "natural_breaks", missing_kwds={"color": "lightgrey"})
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.title(r'Block-by-block Difference in Complexity Measure ($k_{DG} - k_{OSM}$)')
    plt.subplots_adjust(left = 0.02, bottom = 0.02, right = 0.98, top = 0.88, wspace = 0.10)
    # plt.savefig("delta_k_choropleth_sl.png", dpi=300, bbox_inches="tight")
    plt.show()

def visualize_building_counts(dg: gpd.GeoDataFrame, osm: gpd.GeoDataFrame):
    # get n range 
    max_n = max(dg.num_buildings.max(), osm.num_buildings.max())
    min_n = 0
    
    # visualize num_buildings distribution
    fig, ax = plt.subplots(1, 2, sharey='row', sharex='col')
    bins = range(0, max_n + 1, 5000)
    dg.num_buildings.hist( ax = ax[0], bins = bins)
    osm.num_buildings.hist(ax = ax[1], bins = bins)
    ax[0].set_title("DigitalGlobe")
    ax[1].set_title("OpenStreetMap")
    fig.suptitle("Histogram of Per-Block Building Counts for Freetown, SL")
    plt.semilogy()
    # plt.savefig("n_histogram_sl.png", dpi=300, bbox_inches="tight")
    plt.show()

    # num_buildings choropleth 
    fig, ax = plt.subplots(1, 2, sharey='row', sharex='col')
    plot_opts = { 
        "column"   : "num_buildings",
        "cmap"     : "PiYG",
        # "scheme"   : "natural_breaks",
        "scheme"   : "User_Defined",
        "edgecolor": "grey",
        "linewidth": 0.1,
        "legend"   : True,
        "categorical" : False,
        "markersize" : 0,
        "classification_kwds" : {"bins" : [4270, 16922, 47521, 122413, 160779, max_n]} # numbers taken from natural_breaks
    }

    # dummy polygons to make choropleth color scheme consistent 
    dgd = gpd.GeoDataFrame(pd.concat([pd.DataFrame([ 
        ["d0", shapely.geometry.MultiPoint(osm.centroid).centroid, 0, [], 35], 
    ], columns = dg.columns), dg.copy()]).reset_index(), geometry = "block_geom")

    dgd.plot(ax=ax[0],  **plot_opts)
    ax[0].get_xaxis().set_visible(False)
    ax[0].axes.get_yaxis().set_visible(False)

    # osmd.plot(ax=ax[1], **plot_opts)
    osm.plot(ax=ax[1], **plot_opts)
    ax[1].get_xaxis().set_visible(False)
    ax[1].axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(left = 0.02, bottom = 0.02, right = 0.98, top = 0.98, wspace = 0.10)
    ax[0].set_title("DigitalGlobe")
    ax[1].set_title("OpenStreetMap")
    fig.suptitle("Building Count Choropleth for Freetown, SL")
    # plt.savefig("n_choropleth_sl.png", dpi=300, bbox_inches="tight")
    plt.show()

    # get block-by-block diff of k values
    diff = dg.join(osm, rsuffix="_osm")
    diff["delta_n"] = diff["num_buildings"] - diff["num_buildings_osm"]
    diff.plot("delta_n", legend=True, cmap="PRGn", scheme = "natural_breaks", missing_kwds={"color": "lightgrey"})
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.title(r'Block-by-block Difference in Building Count ($n_{DG} - n_{OSM}$)')
    plt.subplots_adjust(left = 0.02, bottom = 0.02, right = 0.98, top = 0.88, wspace = 0.10)
    # plt.savefig("delta_n_choropleth_sl.png", dpi=300, bbox_inches="tight")
    plt.show()

def csv_to_geo(csv_path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    df.rename(columns={"geometry":"block_geom"}, inplace=True)
    df['block_geom'] = df['block_geom'].apply(shapely.wkt.loads)
    df['num_buildings'] = df['centroids_multipoint'].apply(len)
    return gpd.GeoDataFrame(df, geometry='block_geom')

def main():
    osmpath = Path("../data/complexity_osm_SLE.4.2.1_1.csv")
    dgpath  = Path("../data/complexity_dg_SLE.4.2.1_1.csv") 
    osm     = csv_to_geo(osmpath)
    dg      = csv_to_geo(dgpath)

    visualize_complexity(dg, osm)
    visualize_building_counts(dg, osm)

if __name__ == "__main__":
    main()
