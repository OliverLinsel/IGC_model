import geopandas as gpd
from shapely import LineString
from shapely.ops import unary_union, linemerge
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import pickle

import utils
import data
crs = data.crs
nuts = data.nuts1

rail_de = gpd.read_file("C:\\Landwehr\\GIT\\Data\\binnenschiff_bundeswasserstrassen.gpkg")

network = unary_union(rail_de.geometry)
merged_lines = linemerge(network)

# Alle Knotenpunkte
nodes = network.boundary

nodes_gdf = gpd.GeoDataFrame(geometry=list(nodes.geoms), crs=rail_de.crs)

import geopandas as gpd

if merged_lines.geom_type == "LineString":
    merged_gdf = gpd.GeoDataFrame(
        geometry=[merged_lines], crs=rail_de.crs)
else:
    merged_gdf = gpd.GeoDataFrame(
        geometry=list(merged_lines.geoms), crs=rail_de.crs)
    
with open("C:\\Landwehr\\GIT\\Data\\waterways.pkl", "wb") as f:
    pickle.dump(merged_gdf, f)