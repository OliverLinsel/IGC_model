##### trading_transport_module #####
#orginated by OL 22.07.2026
#%%
### This is the script to use methods developed from the network module to determine the transport costs between any given combination of input countries and different transport commodities ###

import time
import os
import sys
import pandas as pd
import geopandas as gpd
import searoute as sr
from shapely.geometry import Point, Polygon, LineString, MultiPoint
import numpy as np
from scipy.spatial import Delaunay
from itertools import combinations
import networkx as nx
from tqdm import tqdm
from shapely import wkt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors as pc

START = time.perf_counter()

print('Execute in Directory:')
print(os.getcwd() + "\n")

### This part is to guarantee execution in normal and in debug mode to cleanly call scripts from neighbouring directories
def _find_project_root(start_dir, required_siblings=("econ_module", "relationship_module", "general_module")):
    candidate = os.path.abspath(start_dir)
    for _ in range(6):
        if all(os.path.isdir(os.path.join(candidate, s)) for s in required_siblings):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    raise RuntimeError(f"Could not locate IGC_model project root (expected {required_siblings}) starting from {start_dir}")

try:
    _start_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _start_dir = os.getcwd()

project_root = _find_project_root(_start_dir)
this_dir = os.path.join(project_root, "econ_module")

for sibling in ("relationship_module", "general_module"):
    sibling_dir = os.path.join(project_root, sibling)
    if sibling_dir not in sys.path:
        sys.path.insert(0, sibling_dir)

from model_settings import get_settings

case_study = get_settings(parameter="case_study")

default_ylim = (-90,90) #Welt
default_xlim = (-180,180) #Welt

default_epsg_1 = "EPSG:4326"
default_epsg_2 = "EPSG:6933"

pipeline_loss_fixMin = 0.005 # 0.5%
pipeline_loss_log = 100 #(log base)
pipeline_elongation_factor = 1.3

COORD_PRECISION = 6  # ~0.1m precision at the equator — adjust if your data needs coarser/finer

# Define the paths
try:
    data_path = os.path.join(this_dir, "..", "data_module", "Data", "scenario_run_data", case_study)
    econ_data_path = os.path.join(this_dir, "data", case_study)
    output_path = os.path.join(this_dir, "output")
except:
    data_path = os.path.join(this_dir, "..", "data_module", "Data", "scenario_run_data", case_study)
    output_path = os.path.join(this_dir, "output")
    econ_data_path = os.path.join(this_dir, "data", case_study)

#### Set global parameters ####
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

### read world dataset
world_gdf = gpd.read_file(os.path.join(data_path, "geoscope", "world_eu_bz.gpkg"))
world_gdf = world_gdf.set_crs(default_epsg_1)
# world_gdf = world_gdf.to_crs(default_epsg_2)
### read bathymetry
bathymetry_gdf = gpd.read_file(os.path.join(data_path, "geoscope", "bm_world_3000m.gpkg"))
bathymetry_gdf = bathymetry_gdf.set_crs(default_epsg_1)
# bathymetry_gdf = bathymetry_gdf.to_crs(default_epsg_2)
# bathymetry_gdf = bathymetry_gdf.dissolve(by="DN").reset_index(drop=True)

### read nodes ###
nodes_df = pd.read_excel(os.path.join(econ_data_path, "transport_cost_assumptions.xlsx"), sheet_name='pipeline_data')
nodes_points = nodes_df.apply(lambda row: Point(round(row.lon, COORD_PRECISION), round(row.lat, COORD_PRECISION)), axis=1) #axis=1 macht, dass es von Reihe zu Reihe geht und nicht von Spalte zu Spalte - first lon then lat
nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=nodes_points) 
#crosscheck with the regions in the market model dataset and filter
regions_df = pd.read_excel(os.path.join(this_dir, "data", case_study, "region_tables.xlsx"), sheet_name='prices')
regions_list = regions_df["region"].to_list()
nodes_gdf = nodes_gdf[nodes_gdf["name"].isin(regions_list)].reset_index(drop=True)
nodes_gdf = nodes_gdf.set_crs(default_epsg_1)
# nodes_gdf = nodes_gdf.to_crs(default_epsg_2)

### read terminals ###
terminals_df = pd.read_excel(os.path.join(econ_data_path, "transport_cost_assumptions.xlsx"), sheet_name='shipping_data')
terminals_points = terminals_df.apply(lambda row: Point(round(row.lon, COORD_PRECISION), round(row.lat, COORD_PRECISION)), axis=1) #axis=1 macht, dass es von Reihe zu Reihe geht und nicht von Spalte zu Spalte - first lon then lat
terminals_gdf = gpd.GeoDataFrame(terminals_df, geometry=terminals_points) 
#crosscheck with the regions in the market model dataset and filter
regions_df = pd.read_excel(os.path.join(this_dir, "data", case_study, "region_tables.xlsx"), sheet_name='prices')
regions_list = regions_df["region"].to_list()
terminals_gdf = terminals_gdf[terminals_gdf["node"].isin(regions_list)].reset_index(drop=True)
terminals_gdf = terminals_gdf.set_crs(default_epsg_1)
# terminals_gdf = terminals_gdf.to_crs(default_epsg_2)

#%%

### create connection lines by combining the terminal_gdf geometry with the nodes_gdf geometry to a Line geometry
connections_df = nodes_gdf[["name", "geometry", "commodity", "WACC"]].merge(terminals_gdf[["terminal_name", "commodity", "name", "node", "geometry", "alternative"]], left_on="name", right_on="node", how="right")
connections_df["geometry"] = connections_df.apply(lambda row: LineString([row["geometry_x"], row["geometry_y"]]), axis=1)
connections_df = connections_df.drop(columns=["geometry_x", "geometry_y", "name_x"])
connections_df = connections_df.rename(columns={"commodity_x": "commodity_in", "commodity_y": "commodity_out"})
connections_gdf = gpd.GeoDataFrame(connections_df, geometry="geometry")

connections_gdf = connections_gdf.set_crs(default_epsg_2)
connections_gdf["length"] = connections_gdf.length / 1000
connections_gdf = connections_gdf.to_crs(default_epsg_1)

connections_gdf["l_cost_factor"] = 0.0048  # €/MWhkm EUH2BB
connections_gdf["transport_cost_MWh"] = connections_gdf["l_cost_factor"] * connections_gdf["length"]
connections_gdf["efficiency"] = 1 - (pipeline_loss_fixMin + (np.log(connections_gdf["length"]) / np.log(pipeline_loss_log) * connections_gdf["length"] / connections_gdf["length"].mean()) / 100)

# create pipeline triangulation
tri_work_points = nodes_gdf["geometry"].copy()
existing_coords = {(pt.x, pt.y) for pt in tri_work_points}
connection_points = [
    Point(x, y)
    for geom in terminals_gdf.geometry
    for x, y in geom.coords
    if (x, y) not in existing_coords
]
tri_work_points = pd.concat(
    [tri_work_points, gpd.GeoSeries(connection_points, crs=nodes_gdf.crs)],
    ignore_index=True,
)

tri_work_array = np.array([[pp.x, pp.y] for pp in tri_work_points])

plexos_nodes_triangulation = Delaunay(tri_work_array)

# --- extract individual triangle edges (vectorized, deduplicated) ---
simplices = plexos_nodes_triangulation.simplices  # shape (n_triangles, 3)

# build all 3 edges per triangle as vertex-index pairs: (0,1), (1,2), (0,2)
edge_idx = np.vstack([
    simplices[:, [0, 1]],
    simplices[:, [1, 2]],
    simplices[:, [0, 2]],
])  # shape (3*n_triangles, 2)

# normalize direction so shared edges (A,B) and (B,A) become identical rows
edge_idx = np.sort(edge_idx, axis=1)

# drop duplicate edges (each internal triangulation edge is shared by 2 triangles)
edge_idx = np.unique(edge_idx, axis=0)

points = plexos_nodes_triangulation.points
edge_lines = [LineString([points[i], points[j]]) for i, j in edge_idx]

pipelines_gdf = gpd.GeoDataFrame(geometry=edge_lines, crs=default_epsg_1)
pipelines_gdf = pipelines_gdf.to_crs(default_epsg_2)
pipelines_gdf["length"] = pipelines_gdf.length / 1000 #* pipeline_elongation_factor
pipelines_gdf = pipelines_gdf.to_crs(default_epsg_1)

# drop pipelines that intersect the deepsea bathymetry polygons
pipelines_gdf = pipelines_gdf.sjoin(bathymetry_gdf[["geometry"]], how="left", predicate="intersects")
pipelines_gdf = pipelines_gdf[pipelines_gdf["index_right"].isna()].drop(columns=["index_right"])

pipeline_data_df = pd.read_excel(os.path.join(econ_data_path, "transport_cost_assumptions.xlsx"), sheet_name='pipeline_data')
nodes_points = pipeline_data_df.apply(lambda row: Point(round(row.lon, COORD_PRECISION), round(row.lat, COORD_PRECISION)), axis=1) #axis=1 macht, dass es von Reihe zu Reihe geht und nicht von Spalte zu Spalte - first lon then lat
pipeline_data_gdf = gpd.GeoDataFrame(pipeline_data_df, geometry=nodes_points) 

### calculate the cost factors for invest, fom and vom for pipelines ###
## annuity factor
pipeline_data_gdf["annuity_factor"] = (pipeline_data_gdf["WACC"] * (1 + pipeline_data_gdf["WACC"]) ** pipeline_data_gdf["lifetime"]) / ((1 + pipeline_data_gdf["WACC"]) ** pipeline_data_gdf["lifetime"] - 1)
## invest cost [€/MWh/km]
pipeline_data_gdf["invest_cost_MWh"] = pipeline_data_gdf["invest_cost"] * pipeline_data_gdf["annuity_factor"] / pipeline_data_gdf["capacity"] / pipeline_data_gdf["full_load_hours"]
## fom cost [€/MWh/km]
pipeline_data_gdf["fom_cost_MWh"] = pipeline_data_gdf["invest_cost_MWh"] * pipeline_data_gdf["fom_cost_MWh"] * (0.5 + 0.5 * pipeline_data_gdf["regional_factor"])
## vom cost [€/MWh/km]
pipeline_data_gdf["vom_cost_MWh"] = pipeline_data_gdf["electricity_per_energy_transported"] * pipeline_data_gdf["cost_el"]
## total cost [€/MWh/km]
pipeline_data_gdf["total_cost_MWh"] = pipeline_data_gdf["invest_cost_MWh"] + pipeline_data_gdf["fom_cost_MWh"] + pipeline_data_gdf["vom_cost_MWh"]

# sjoin on start and end geometries of the pipelines to assign the relevant paramters of the from and to region.
pipelines_gdf["start_point"] = pipelines_gdf.geometry.apply(lambda g: Point(g.coords[0]))
pipelines_gdf["end_point"]   = pipelines_gdf.geometry.apply(lambda g: Point(g.coords[-1]))

# # --- 2. Parameters to pull from the region data ---
pipeline_data_gdf_select = pd.DataFrame()
pipeline_data_gdf_select = pipeline_data_gdf[["name", "alternative", "commodity", "total_cost_MWh", "geometry"]].copy()

#%%
# --- Harmonize terminal columns to match region columns for joint merge ---
terminals_gdf_select = terminals_gdf[["name", "alternative", "commodity", "geometry"]].copy()
terminals_gdf_select["selector"] = terminals_gdf_select["name"].str.split("_").str[-1]
terminals_gdf_select["selector"] = terminals_gdf_select["selector"].str.split("-").str[:2].str.join("-")
# merge pipeline_data_gdf["total_cost_MWh"] via name onto terminals_gdf_select
terminals_gdf_select = pd.merge(terminals_gdf_select, pipeline_data_gdf_select[["name", "total_cost_MWh"]], left_on="selector", right_on="name", how="left")
terminals_gdf_select = terminals_gdf_select.drop(columns=["selector", "name_y"])
terminals_gdf_select = terminals_gdf_select.rename(columns={"name_x": "name"})

# --- Combine regions + terminals into one lookup source ---
combined_select = pd.concat([pipeline_data_gdf_select, terminals_gdf_select], ignore_index=True)

# --- Check for duplicate coordinates before building the lookup ---
combined_select["coords"] = combined_select.geometry.apply(
    lambda p: (round(p.x, COORD_PRECISION), round(p.y, COORD_PRECISION))
)
dupes = combined_select[combined_select.duplicated(subset="coords", keep=False)]
print(f"Duplicate coordinate rows: {len(dupes)}")
if len(dupes) > 0:
    print(dupes[["name", "coords"]])

# --- Build the lookup (drop geometry, index by coords) ---
region_lookup = combined_select.drop(columns="geometry").set_index("coords")

# --- Re-run the merge exactly as before ---
pipelines_gdf["start_coords"] = pipelines_gdf["start_point"].apply(
    lambda p: (round(p.x, COORD_PRECISION), round(p.y, COORD_PRECISION))
)
pipelines_gdf["end_coords"] = pipelines_gdf["end_point"].apply(
    lambda p: (round(p.x, COORD_PRECISION), round(p.y, COORD_PRECISION))
)

pipeline_model_gdf = (
    pipelines_gdf
    .merge(region_lookup.add_suffix("_from"), left_on="start_coords", right_index=True, how="left")
    .merge(region_lookup.add_suffix("_to"),   left_on="end_coords",   right_index=True, how="left")
)

# --- Recheck for remaining unmatched rows ---
print(pipeline_model_gdf[["name_from", "name_to"]].isna().sum())

##cleaning up the pipeline gdf for better usability
pipeline_model_gdf["name"] = pipeline_model_gdf["name_from"] + "_" + pipeline_model_gdf["name_to"]
pipeline_model_gdf["total_cost_MWh_km"] = (pipeline_model_gdf["total_cost_MWh_from"] + pipeline_model_gdf["total_cost_MWh_to"]) / 2 # calculate average transport cost
pipeline_model_gdf["total_cost_MWh"] = pipeline_model_gdf["total_cost_MWh_km"] * pipeline_model_gdf["length"]
## hardcoding some parameters - needs to be cleaned up if scenarios or commodities are being diversified
pipeline_model_gdf["alternative"] = "Base"
pipeline_model_gdf["commodity"] = "h2"
pipeline_model_gdf["efficiency"] = 1 - (pipeline_loss_fixMin + (np.log(pipeline_model_gdf["length"]) / np.log(pipeline_loss_log) * pipeline_model_gdf["length"] / pipeline_model_gdf["length"].mean()) / 100)
pipeline_model_gdf = pipeline_model_gdf[["name", "commodity", "alternative", "length", "total_cost_MWh", "efficiency", "geometry"]].copy()

pipelines_gdf = pipeline_model_gdf[~pipeline_model_gdf["name"].str.contains("terminal")]
terminal_connections_gdf = pipeline_model_gdf[pipeline_model_gdf["name"].str.contains("terminal")]

### pick up the terminals again and preprocess the cost and efficiencies for terminal to terminal info
terminals_gdf = terminals_gdf.rename(columns={"name": "commodity_terminal_name"})
terminals_gdf = terminals_gdf.merge(pipeline_data_gdf[["name", "WACC", "regional_factor", "cost_el"]], left_on="node", right_on="name", how="left")

terminals_gdf["capacity"] = 1000 # dummy value - replace!
#%%

### Calculate the cost factors for invest, fom and vom for conversion ###
terminals_gdf["conversion_annuity_factor"] = (terminals_gdf["WACC"] * (1 + terminals_gdf["WACC"]) ** terminals_gdf["conversion_lifetime"]) / ((1 + terminals_gdf["WACC"]) ** terminals_gdf["conversion_lifetime"] - 1)
## invest cost [€/MWh/km]
terminals_gdf["conversion_invest_cost_MWh"] = terminals_gdf["conversion_invest_cost"] * terminals_gdf["conversion_annuity_factor"]  / terminals_gdf["full_load_hours"] / terminals_gdf["capacity"]
## fom cost [€/MWh/km]
terminals_gdf["conversion_fom_cost_MWh"] = terminals_gdf["conversion_invest_cost_MWh"] * terminals_gdf["conversion_fom_capex_factor"] * (0.5 + 0.5 * terminals_gdf["regional_factor"])
## vom cost [€/MWh/km]
terminals_gdf["conversion_vom_cost_MWh"] = terminals_gdf["electricity_per_energy_converted"] * terminals_gdf["cost_el"]
## total cost [€/MWh/km]
terminals_gdf["conversion_total_cost_MWh"] = terminals_gdf["conversion_invest_cost_MWh"] + terminals_gdf["conversion_fom_cost_MWh"] + terminals_gdf["conversion_vom_cost_MWh"]

### Calculate the cost factors for invest, fom and vom for terminal ###
terminals_gdf["terminal_annuity_factor"] = (terminals_gdf["WACC"] * (1 + terminals_gdf["WACC"]) ** terminals_gdf["terminal_lifetime"]) / ((1 + terminals_gdf["WACC"]) ** terminals_gdf["terminal_lifetime"] - 1)
## invest cost [€/MWh/km]
terminals_gdf["terminal_invest_cost_MWh"] = terminals_gdf["terminal_invest_cost"] * terminals_gdf["terminal_annuity_factor"]  / terminals_gdf["full_load_hours"] / terminals_gdf["capacity"]
## fom cost [€/MWh/km]
terminals_gdf["terminal_fom_cost_MWh"] = terminals_gdf["terminal_invest_cost_MWh"] * terminals_gdf["terminal_fom_capex_factor"] * (0.5 + 0.5 * terminals_gdf["regional_factor"])
## vom cost [€/MWh/km]
terminals_gdf["terminal_vom_cost_MWh"] = terminals_gdf["electricity_per_energy_loaded"] * terminals_gdf["cost_el"]
## total cost [€/MWh/km]
terminals_gdf["terminal_total_cost_MWh"] = terminals_gdf["terminal_invest_cost_MWh"] + terminals_gdf["terminal_fom_cost_MWh"] + terminals_gdf["terminal_vom_cost_MWh"]

### Calculate the cost factors for invest, fom and vom for shipping ###
terminals_gdf["shipping_annuity_factor"] = (terminals_gdf["WACC"] * (1 + terminals_gdf["WACC"]) ** terminals_gdf["shipping_lifetime"]) / ((1 + terminals_gdf["WACC"]) ** terminals_gdf["shipping_lifetime"] - 1)
## invest cost [€/MWh/km]
terminals_gdf["shipping_invest_cost_MWh"] = terminals_gdf["shipping_invest_cost"] * terminals_gdf["shipping_annuity_factor"]  / terminals_gdf["full_load_hours"] / terminals_gdf["capacity"]
## fom cost [€/MWh/km]
terminals_gdf["shipping_fom_cost_MWh"] = terminals_gdf["shipping_invest_cost_MWh"] * terminals_gdf["shipping_fom_capex_factor"] * (0.5 + 0.5 * terminals_gdf["regional_factor"])
## vom cost [€/MWh/km]
terminals_gdf["shipping_vom_cost_MWh_km"] = terminals_gdf["fuel_consumption_km"]
## total cost [€/MWh/km]
terminals_gdf["shipping_total_cost_MWh"] = terminals_gdf["shipping_invest_cost_MWh"] + terminals_gdf["shipping_fom_cost_MWh"]

### Calculate the cost factors for invest, fom and vom for reconversion ###
terminals_gdf["reconversion_annuity_factor"] = (terminals_gdf["WACC"] * (1 + terminals_gdf["WACC"]) ** terminals_gdf["reconversion_lifetime"]) / ((1 + terminals_gdf["WACC"]) ** terminals_gdf["reconversion_lifetime"] - 1)
## invest cost [€/MWh/km]
terminals_gdf["reconversion_invest_cost_MWh"] = terminals_gdf["reconversion_invest_cost"] * terminals_gdf["reconversion_annuity_factor"]  / terminals_gdf["full_load_hours"] / terminals_gdf["capacity"]
## fom cost [€/MWh/km]
terminals_gdf["reconversion_fom_cost_MWh"] = terminals_gdf["reconversion_invest_cost_MWh"] * terminals_gdf["reconversion_fom_capex_factor"] * (0.5 + 0.5 * terminals_gdf["regional_factor"])
## vom cost [€/MWh/km]
terminals_gdf["reconversion_vom_cost_MWh"] = terminals_gdf["electricity_per_energy_reconverted"] * terminals_gdf["cost_el"]
## total cost [€/MWh/km]
terminals_gdf["reconversion_total_cost_MWh"] = terminals_gdf["reconversion_invest_cost_MWh"] + terminals_gdf["reconversion_fom_cost_MWh"] + terminals_gdf["reconversion_vom_cost_MWh"]

terminals_gdf["total_cost_MWh"] = terminals_gdf["conversion_total_cost_MWh"] + terminals_gdf["terminal_total_cost_MWh"] + terminals_gdf["reconversion_total_cost_MWh"]
terminals_gdf["total_cost_MWh_km"] = terminals_gdf["shipping_vom_cost_MWh_km"]

terminals_gdf["efficiency"] = (terminals_gdf["conversion_efficiency_substantial"] * terminals_gdf["conversion_efficiency_energetic"]
                                * terminals_gdf["terminal_efficiency"]
                                * terminals_gdf["shipping_efficiency_flat"]
                                * terminals_gdf["reconversion_efficiency_substantial"] * terminals_gdf["reconversion_efficiency_energetic"])
terminals_gdf["efficiency_km"] = terminals_gdf["shipping_efficiency_km"]
#%%

def calculate_searoute_by_coords(p_origin, p_destination):
    """Route calculation based on raw coordinates, decoupled from terminal naming."""
    route = sr.searoute(p_origin, p_destination, speed_knot=13, append_orig_dest=True)
    sr_geo = LineString(route["geometry"]["coordinates"])
    return route["properties"]["duration_hours"], route["properties"]["length"], route["geometry"]["coordinates"], sr_geo


# --- step 1: generate all origin-destination combinations at the commodity_terminal_name level ---
names = terminals_gdf["commodity_terminal_name"].unique().tolist()
combinations_names = np.array(list(combinations(names, 2)))

df_sr = pd.DataFrame(columns=["origin", "destination"])
df_sr["origin"] = combinations_names[:, 0]
df_sr["destination"] = combinations_names[:, 1]

# --- step 2: attach coordinates + node + commodity for both origin and destination ---
lookup = terminals_gdf.set_index("commodity_terminal_name")[["node", "commodity", "lat", "lon"]]

df_sr = df_sr.merge(lookup.add_prefix("origin_"), left_on="origin", right_index=True)
df_sr = df_sr.merge(lookup.add_prefix("destination_"), left_on="destination", right_index=True)

# --- step 3 (optional but recommended): only keep same-commodity pairs ---
df_sr = df_sr[df_sr["origin_commodity"] == df_sr["destination_commodity"]].reset_index(drop=True)

# --- step 4: deduplicate route calculation by physical node pair, since the geometry
df_sr["node_pair"] = df_sr.apply(
    lambda r: tuple(sorted([r["origin_node"], r["destination_node"]])), axis=1
)

unique_node_pairs = df_sr.drop_duplicates(subset="node_pair")[
    ["node_pair", "origin_lon", "origin_lat", "destination_lon", "destination_lat"]
].reset_index(drop=True)

route_cache = {}
for _, row in unique_node_pairs.iterrows():
    p_origin = [row["origin_lon"], row["origin_lat"]]
    p_destination = [row["destination_lon"], row["destination_lat"]]
    duration, length, coord, sr_geo_out = calculate_searoute_by_coords(p_origin, p_destination)
    route_cache[row["node_pair"]] = {
        "duration_hours": duration, "length": length, "geometry": sr_geo_out,
    }

# --- step 5: broadcast the cached route results back onto every commodity pair ---
df_sr["duration_hours"] = df_sr["node_pair"].map(lambda np_: route_cache[np_]["duration_hours"])
df_sr["length"] = df_sr["node_pair"].map(lambda np_: route_cache[np_]["length"])
geo_list = df_sr["node_pair"].map(lambda np_: route_cache[np_]["geometry"]).tolist()

df_sr = df_sr.drop(columns=["node_pair"])
df_sr = gpd.GeoDataFrame(df_sr, geometry=geo_list, crs="EPSG:4326")
#%%

shipping_model_gdf = (
    df_sr
    .merge(terminals_gdf[["commodity_terminal_name", "total_cost_MWh", "total_cost_MWh_km", "efficiency", "efficiency_km"]].add_suffix("_from"), left_on="origin", right_on="commodity_terminal_name_from", how="left")
    .merge(terminals_gdf[["commodity_terminal_name", "total_cost_MWh", "total_cost_MWh_km"]].add_suffix("_to"), left_on="destination", right_on="commodity_terminal_name_to", how="left")
)
#%%

shipping_model_gdf["name"] = shipping_model_gdf["origin"] + "_" + shipping_model_gdf["destination"]
## calculate average costs
shipping_model_gdf["total_cost_MWh"] = (shipping_model_gdf["total_cost_MWh_from"] + shipping_model_gdf["total_cost_MWh_to"]) / 2 + (shipping_model_gdf["total_cost_MWh_km_from"] + shipping_model_gdf["total_cost_MWh_km_to"]) / 2
shipping_model_gdf["total_cost_MWh_km"] = (shipping_model_gdf["total_cost_MWh_km_from"] + shipping_model_gdf["total_cost_MWh_km_to"]) / 2
### add the length dependend cost element ###
shipping_model_gdf["total_cost_MWh"] = shipping_model_gdf["total_cost_MWh"] + shipping_model_gdf["total_cost_MWh_km"] * shipping_model_gdf["length"]
## calculate the efficiency
shipping_model_gdf["efficiency"] = shipping_model_gdf["efficiency_from"] * (1 - shipping_model_gdf["efficiency_km_from"] * shipping_model_gdf["length"])

shipping_gdf = shipping_model_gdf[["name", "length", "total_cost_MWh", "efficiency", "geometry"]]
#%%
print(pipelines_gdf.crs, bathymetry_gdf.crs, nodes_gdf.crs, world_gdf.crs, connections_gdf.crs, shipping_gdf.crs)

### Combine geometries with cost information to routing template
combined_gdf = pd.concat(
    [pipelines_gdf[["name", "geometry", "length", "total_cost_MWh", "efficiency"]], terminal_connections_gdf[["name", "geometry", "length", "total_cost_MWh", "efficiency"]], shipping_gdf[["name", "geometry", "length", "total_cost_MWh", "efficiency"]]],
    ignore_index=True
)
# Convert the concatenated DataFrame back to a GeoDataFrame
combined_gdf = gpd.GeoDataFrame(combined_gdf, geometry='geometry')

# calculate shadow price
combined_gdf["total_cost_MWh"] = combined_gdf["total_cost_MWh"] / combined_gdf["efficiency"]
combined_gdf.to_excel(os.path.join(this_dir, "output", "model", "combined_transport_elements.xlsx"), index=False)
print("Saved combined gdf with total_cost in €/MWh to file in " + str(os.path.join(this_dir, "output", "model")) + "\n")

print("End of the preprocessing" + "\n")

#%%
### Define routing functions ###

def create_network_graph(routing_template_gdf, sources_gdf, precision=6):
    W = nx.Graph()

    def key(pt):
        return (round(pt[0], precision), round(pt[1], precision))

    routing_template_gdf = routing_template_gdf.sort_values(by='length', ignore_index=True)

    for idx, row in routing_template_gdf.iterrows():
        geom = row.geometry
        n_segments = len(list(geom.coords)) - 1 if geom.geom_type == 'LineString' else None

        if geom.geom_type == 'LineString':
            coords = [key(c) for c in geom.coords]
            # n-th root so that multiplying the segment efficiencies back together
            # reproduces the original edge's efficiency
            seg_efficiency = row["efficiency"] ** (1 / n_segments)
            for i in range(len(coords) - 1):
                W.add_edge(
                    coords[i], coords[i+1],
                    length=row["length"],
                    cost=row["total_cost_MWh"] / n_segments,
                    weight=row["total_cost_MWh"] / n_segments,
                    efficiency=seg_efficiency,
                )

        elif geom.geom_type == 'MultiLineString':
            for part in geom.geoms:
                coords = [key(c) for c in part.coords]
                n_seg = len(coords) - 1
                seg_efficiency = row["efficiency"] ** (1 / n_seg)
                for i in range(len(coords) - 1):
                    W.add_edge(
                        coords[i], coords[i+1],
                        cost=row["total_cost_MWh"] / n_seg,
                        weight=row["total_cost_MWh"] / n_seg,
                        efficiency=seg_efficiency,
                    )

    # Add nodes to the graph from the sources data gdf
    for idx, row in sources_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'Point':
            longitude, latitude = geom.x, geom.y
            node_id = (longitude, latitude)
            W.add_node(
                node_id,
                ets_id=row['name'],
                facility_name=row['geometry'],
                operator_name=row['alternative'],
                tco2_2023=row['commodity'],
                x=longitude,
                y=latitude,
            )
    return W

def dijkstra_connect_sources(W, filtered_sources_df):
    # define TEMP file path to avoid having to calculate the paths again even if nothing changed
    file_path = os.path.join(this_dir, "output", "model", "paths.csv")

    # Check if the graph is connected
    if not nx.is_connected(W):
        print("Warning: The graph is not connected. This might affect path finding.")
        connected_components = list(nx.connected_components(W))
        print(f"Number of connected components: {len(connected_components)}")
        for i, component in enumerate(connected_components):
            print(f"Connected component {i + 1}: {component}")

    # --- load existing progress, if any ---
    if os.path.exists(file_path):
        paths_df = pd.read_csv(file_path)
        print(f"Loaded existing progress: {len(paths_df)} pairs already computed.")
        # build a set of (source, sink) pairs already done, for fast lookup
        done_pairs = set(zip(paths_df["source"], paths_df["sink"]))
    else:
        paths_df = pd.DataFrame(columns=["source", "sink", "cheapest_path", "path_cost", "path_geometry"])
        done_pairs = set()

    num_sources = len(filtered_sources_df)
    new_rows = []

    for i in tqdm(range(num_sources), desc="Processing sources"):
        source_name = filtered_sources_df["name"].iloc[i]
        print(f"Currently processing {source_name}")

        for j in tqdm(range(i + 1, num_sources), desc="Processing sinks", leave=False):
            source = filtered_sources_df.iloc[i]
            sink = filtered_sources_df.iloc[j]

            # --- skip pairs that were already computed in a previous run ---
            if (source["name"], sink["name"]) in done_pairs:
                continue

            x_start, y_start = source.geometry.x, source.geometry.y
            x_end, y_end = sink.geometry.x, sink.geometry.y

            start_point = Point(x_start, y_start)
            end_point = Point(x_end, y_end)

            start_node = min(W.nodes(), key=lambda node: start_point.distance(Point(node[0], node[1])))
            end_node = min(W.nodes(), key=lambda node: end_point.distance(Point(node[0], node[1])))

            try:
                shortest_path = nx.dijkstra_path(W, start_node, end_node, weight="weight")
                total_cost = nx.dijkstra_path_length(W, start_node, end_node, weight="weight")
            except nx.NetworkXNoPath:
                shortest_path = []
                total_cost = None
                print(f"No path found between the nodes {source['name']} and {sink['name']}")

            if shortest_path:
                path_coords = [(node[0], node[1]) for node in shortest_path]
                path_geometry = LineString(path_coords)

                # cumulative efficiency = product of each hop's efficiency along the path
                path_efficiency = 1.0
                for u, v in zip(shortest_path[:-1], shortest_path[1:]):
                    path_efficiency *= W[u][v]["efficiency"]
            else:
                path_geometry = None
                path_efficiency = None

            new_rows.append({
                'source': source['name'],
                'sink': sink['name'],
                'cheapest_path': shortest_path,
                'path_cost': total_cost,
                'path_geometry': path_geometry,
                'path_efficiency': path_efficiency,
            })

            # --- periodically flush progress to disk so a crash/interrupt doesn't lose work ---
            if len(new_rows) >= 50:  # adjust batch size to taste
                paths_df = pd.concat([paths_df, pd.DataFrame(new_rows)], ignore_index=True)
                paths_df.to_csv(file_path, index=False)
                new_rows = []

    # flush any remaining rows after the loop finishes
    if new_rows:
        paths_df = pd.concat([paths_df, pd.DataFrame(new_rows)], ignore_index=True)
        paths_df.to_csv(file_path, index=False)

    # Convert the paths_df to a GeoDataFrame
    paths_gdf = gpd.GeoDataFrame(
        paths_df,
        geometry=paths_df["path_geometry"].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x),
        crs=filtered_sources_df.crs,
    )

    print("DataFrame saved to file.")
    return paths_gdf

### read paths if they exist
print("Check if paths.csv already exists")
print("If yes: load from file and do not recalculate")
print("If no: calculate paths between each country pair" + "\n")

file_path = os.path.join(this_dir, "output", "model", "paths.csv")

if os.path.exists(file_path):
    paths_df = pd.read_csv(file_path)
    paths_df["path_geometry"] = paths_df["path_geometry"].apply(
        lambda x: wkt.loads(x) if isinstance(x, str) else None
    )
    paths_gdf = gpd.GeoDataFrame(paths_df, geometry="path_geometry", crs=default_epsg_1)
    paths_gdf["length"] = paths_gdf.length
    print("paths.csv found and succesfully loaded from file." + "\n")
else:
    paths_gdf = pd.DataFrame()
    print("No existing paths file found - Creating new network graph" + "\n")
    G = nx.Graph()  # Create an empty graph
    G = create_network_graph(combined_gdf, nodes_gdf)
    print("Finished creating network graph G: " + str(G) + "\n")
    # Filter the sources dataframe
    nodes_gdf_filtered = nodes_gdf.copy()
    # nodes_gdf_filtered = nodes_gdf.head(5).reset_index(drop=True) # In case you need to test the script with a limited dataframe
    print("Execute pathing for each nodes pair on network graph G: " + "\n")
    paths_gdf = dijkstra_connect_sources(G, nodes_gdf_filtered)

### Visualising the transport routes ###

def plot_potential_transport_system(path_or_raw):
    # Assuming nodes_gdf, terminals_gdf, and connections_gdf are your GeoDataFrames
    ax = world_gdf.plot(color='yellowgreen', figsize=(20, 20), alpha=0.8)
    # Plot nodes
    nodes_gdf.plot(ax=ax, color='darkblue', markersize=10, label='Nodes')
    # Plot terminals
    terminals_gdf.plot(ax=ax, color='red', markersize=10, label='Terminals')
    # Plot connections
    # connections_gdf.plot(ax=ax, color='orange', linewidth=1, label='Connections')
    # plot pipelines
    # pipelines_gdf.plot(ax=ax, color='red', linewidth=1, label='Pipelines')
    # pipelines_gdf.plot(ax=ax, column='length', linewidth=1, cmap='magma_r', label='Pipelines')
    # shipping_gdf.plot(ax=ax, column='length', linewidth=2, cmap='rainbow', label='Shipping Routes')
    bathymetry_gdf.plot(ax=ax, color="lightblue", alpha=0.5)
    if path_or_raw == "path":
        print("Print optimized routes" + "\n")
        paths_gdf.plot(ax=ax, column='path_cost', linewidth=2, cmap='rainbow', label='Shortest Routes', alpha=0.5, legend=True, legend_kwds={'label': "Path cost (€/MWh)", 'shrink': 0.4, 'aspect': 30})
    elif path_or_raw == "raw":
        print("Print combined routes" + "\n")
        combined_gdf.plot(ax=ax, column='total_cost_MWh', linewidth=2, cmap='rainbow', label='Combined Routes', legend=True, legend_kwds={'label': "Path cost (€/MWh)", 'shrink': 0.4, 'aspect': 30})
    else: print("argument uncler")

    # Add legend and title
    ax.legend()
    ax.set_title('Nodes, Terminals, and Connections')

    plt.xlim(default_xlim)
    plt.ylim(default_ylim)
    if path_or_raw == "path":
        print("Saving optimized routes figure" + "\n")
        plt.savefig(os.path.join(output_path, "figures", "optimized_paths_map.png"), dpi=300, bbox_inches="tight")
    elif path_or_raw == "raw":
        print("Saving combined routes figure" + "\n")
        plt.savefig(os.path.join(output_path, "figures", "potential_transport_network.png"), dpi=300, bbox_inches="tight")
    else: print("No plot saved")
    
    # Show the plot
    return plt.show()

def plot_routed_paths(bathymetry_gdf=bathymetry_gdf, nodes_gdf=nodes_gdf, terminals_gdf=terminals_gdf, paths_gdf=paths_gdf, output_path=output_path,
    projection_type="natural earth", selected_countries=None):

    # --- optional pre-filtering of the input data to the selected countries ---
    if selected_countries is not None:
        nodes_gdf = nodes_gdf[nodes_gdf["name"].isin(selected_countries)].copy()

        if "node" in terminals_gdf.columns:
            terminals_gdf = terminals_gdf[terminals_gdf["node"].isin(selected_countries)].copy()

        if "source" in paths_gdf.columns and "sink" in paths_gdf.columns:
            paths_gdf = paths_gdf[
                paths_gdf["source"].isin(selected_countries) | paths_gdf["sink"].isin(selected_countries)
            ].copy()

    fig = go.Figure()

    # --- bathymetry (deep sea polygons) ---
    def add_polygon_traces(gdf, fig, color="lightblue", opacity=0.4, name="Bathymetry"):
        first = True
        trace_indices = []
        for geom in gdf.geometry:
            polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                lon, lat = poly.exterior.xy
                fig.add_trace(go.Scattergeo(
                    lon=list(lon),
                    lat=list(lat),
                    mode="lines",
                    fill="toself",
                    fillcolor=color,
                    opacity=opacity,
                    line=dict(width=0, color=color),
                    showlegend=first,
                    name=name,
                    legendgroup=name,
                    hoverinfo="skip",
                ))
                trace_indices.append(len(fig.data) - 1)
                first = False
        return trace_indices

    bathymetry_trace_idx = add_polygon_traces(bathymetry_gdf, fig, color="lightblue", opacity=0.4, name="Bathymetry")

    # --- nodes ---
    fig.add_trace(go.Scattergeo(
        lon=nodes_gdf.geometry.x,
        lat=nodes_gdf.geometry.y,
        mode="markers",
        marker=dict(size=6, color="darkblue"),
        name="Nodes",
        text=nodes_gdf["name"] if "name" in nodes_gdf.columns else None,
        hoverinfo="text",
    ))
    nodes_trace_idx = len(fig.data) - 1

    # --- terminals ---
    fig.add_trace(go.Scattergeo(
        lon=terminals_gdf.geometry.x,
        lat=terminals_gdf.geometry.y,
        mode="markers",
        marker=dict(size=5, color="red"),
        name="Terminals",
        text=terminals_gdf["node"] if "node" in terminals_gdf.columns else None,
        hoverinfo="text",
    ))
    terminals_trace_idx = len(fig.data) - 1

    # --- combined_gdf / paths_gdf colored by a numeric column ---
    def add_colored_lines(gdf, fig, column, cmap="rainbow", name="Routes", width=2, opacity=0.5):
        values = gdf[column].values
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        norm_values = (values - vmin) / (vmax - vmin + 1e-12)
        colors = pc.sample_colorscale(cmap, norm_values.tolist())

        has_source_sink = "source" in gdf.columns and "sink" in gdf.columns
        trace_pairs = []

        for row_idx, (geom, color) in enumerate(zip(gdf.geometry, colors)):
            source = gdf["source"].iloc[row_idx] if has_source_sink else None
            sink = gdf["sink"].iloc[row_idx] if has_source_sink else None

            lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
            for line in lines:
                lon, lat = line.xy
                fig.add_trace(go.Scattergeo(
                    lon=list(lon),
                    lat=list(lat),
                    mode="lines",
                    line=dict(width=width, color=color),
                    showlegend=False,
                    legendgroup=name,
                    hoverinfo="skip",
                    opacity=opacity,
                ))
                trace_pairs.append((len(fig.data) - 1, source, sink))

        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None],
            mode="markers",
            marker=dict(
                size=0.3,
                color=[vmin, vmax],
                colorscale=cmap,
                cmin=vmin, cmax=vmax,
                colorbar=dict(title="Transport costs [€/MWh]", len=0.5, thickness=15),
                showscale=True,
            ),
            name=name,
            legendgroup=name,
            showlegend=True,
            hoverinfo="skip",
        ))
        colorbar_trace_idx = len(fig.data) - 1

        return trace_pairs, colorbar_trace_idx

    route_trace_pairs, colorbar_trace_idx = add_colored_lines(
        paths_gdf, fig, column="path_cost", cmap="rainbow", name="Shortest Routes", width=2, opacity=0.5
    )

    # --- always-visible traces (base map elements, never hidden by filtering) ---
    always_visible_idx = set(bathymetry_trace_idx + [nodes_trace_idx, terminals_trace_idx, colorbar_trace_idx])

    n_traces = len(fig.data)

    def visibility_for_node(selected_node):
        vis = [False] * n_traces
        for idx in always_visible_idx:
            vis[idx] = True
        if selected_node is None:
            for idx, source, sink in route_trace_pairs:
                vis[idx] = True
        else:
            for idx, source, sink in route_trace_pairs:
                if source == selected_node or sink == selected_node:
                    vis[idx] = True
        return vis

    node_names = sorted(set(nodes_gdf["name"].dropna().unique()))

    buttons = [
        dict(label="Show all routes", method="update", args=[{"visible": visibility_for_node(None)}])
    ]
    buttons += [
        dict(label=node, method="update", args=[{"visible": visibility_for_node(node)}])
        for node in node_names
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
                showactive=True,
                pad=dict(t=0, r=0),
            )
        ]
    )

    # --- layout: flat, projected 2D map (was orthographic globe) ---
    fig.update_geos(
        projection_type=projection_type,
        showland=True,
        landcolor="yellowgreen",
        showocean=True,
        oceancolor="lightcyan",
        showcountries=True,
        countrycolor="white",
        showcoastlines=True,
        coastlinecolor="white",
    )

    fig.update_layout(
        title_text="Nodes, Terminals, and Connections — select a node to filter routes",
        xaxis_title="Relationship Factor Magnitude (rfm)",
        yaxis_title="Volume [MWh]",
        barmode="stack",
        height=900,
        width=1500,
        # height=1500,
        # width=1500,
        font=dict(family="Times New Roman", color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            tickfont=dict(color="black", size=12),
            title=dict(standoff=15, font=dict(size=14))
        ),
        yaxis=dict(
            tickfont=dict(color="black", size=12),
            title=dict(standoff=15, font=dict(size=14))
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.05,
            xanchor="center",
            x=0.7,
            itemsizing="constant",
            font=dict(size=15),
            title=dict(font=dict(size=15))
        )
    )

    fig.write_image(os.path.join(output_path, "figures", "network_map_2d.png"), scale=4)
    fig.write_html(os.path.join(output_path, "figures", "network_map_2d.html"), include_plotlyjs='cdn')
    return fig, fig.show()

plot_potential_transport_system(path_or_raw = "raw")
plot_potential_transport_system(path_or_raw = "path")
# selected_countries=["EU-DEU", "AS-CHN"]
# plot_routed_paths(selected_countries=selected_countries, projection_type="natural earth") #choose from projection:type either "orthographic" or "natural earth", mercator

STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')
#%%