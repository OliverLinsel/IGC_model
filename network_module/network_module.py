##### network_module #####
#orginated by OL 10.03.2026
#%%
#This is the network module created for the IGC.NRW research project. It is subdivided into four submodules: Determining the geoscope, Creating the path template, creating the sources and creating the sinks.
#Additionally there is a visualization script to check the created data and the resulting network. The module is designed to be flexible and adaptable to different scenarios and data inputs, while also being efficient and scalable for larger datasets.

import time
import os
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.spatial import cKDTree
import numpy as np
from networkx.algorithms.approximation import steiner_tree
from shapely.geometry import LineString, Point, MultiLineString
from tqdm import tqdm
from pyproj import CRS
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from setup import get_system_path, get_ram_usage
from shapely import wkt

#Define module parameters:
retrofit_cost_factor = 0.8
default_simplify_tolerance = 1000 #in meters, tolerance for simplifying the routing 

#define case study
case_study = "igc_nrw"
case_study = "h2bb"

START = time.perf_counter() 

print('Execute in Directory:')
print(os.getcwd() + "\n")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

#Define the paths
data_path = os.path.join(script_dir, "..", "data_module", "Data")
# data_path = get_system_path(data_path)
output_path = os.path.join(script_dir, "..", "output")

# Define the path to the TEMP subdirectory
temp_dir = os.path.join(script_dir, "TEMP")
os.makedirs(temp_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Create the directory if it doesn't exist
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

# Define the scenarios
scenarios = {
    "Base, S1, and E": ["Base", "S1", "E"],
    "Base only": ["Base"],
    "Base and S1": ["Base", "S1"]
}

#get geoscope
from geoscope_module import get_geoscope

work_geoscope_gdf_high_res, work_geoscope_gdf_agg = get_geoscope(data_path, case_study)
print("The aggregated dataframe looks like this: " + str(work_geoscope_gdf_agg) + "\n")
print("The aggregated dataframe looks like this: " + str(work_geoscope_gdf_agg) + "\n")

#create the routing template
from routing_template_module import get_routing_template, visualize_routing_template

routing_template_gdf, retrofit_routes_gdf, potential_routes_gdf = get_routing_template(data_path, case_study)
print(str(routing_template_gdf) + "\n")

#get sources
from sources_module import get_sources, visualize_sources
sources_data_gdf = get_sources(data_path, case_study)
sources_data_gdf = gpd.clip(sources_data_gdf, work_geoscope_gdf_agg)

#connect multiple emitters at the same location to be one - may need rework in the sources_module 

# Dissolve by longitude and latitude
sources_data_gdf = sources_data_gdf.dissolve(by=['longitude', 'latitude'], aggfunc={'longitude': 'first', 'latitude': 'first',
                                                                                    "ets_id":"first", "facility_name":'first', "operator_name":"first",
                                                                                    "tco2_2023":"sum", "nace_21":"first", "NACE Rev. 2.1 Code":"first",
                                                                                    "Sector":"first", "scenario":"first", "color":"first"
                                                                                    }).reset_index(drop=True)

print(str(sources_data_gdf) + "\n")

# #get sinks
# from sinks_module import get_sinks

# test_sinks_o = get_sinks(data_path, case_study)
# print(str(test_sinks_o) + "\n")

#Cut routing template to geoscope
routing_template_op_gdf = gpd.clip(routing_template_gdf, work_geoscope_gdf_agg)
routing_template_op_gdf = routing_template_op_gdf.explode(index_parts=False).reset_index(drop=True)
visualize_routing_template(routing_template_op_gdf)

# clean the network in scope
def clean_grid_template(gdf):
    # Compute the touch matrix
    touch_matrix = gdf.geometry.apply(lambda x: gdf.geometry.touches(x))

    # Build a graph from the touch matrix
    G = nx.from_pandas_adjacency(touch_matrix)

    # Find connected components
    components = list(nx.connected_components(G))

    # Initialize lists to store geometries and component IDs
    geometries = []
    component_ids = []

    # Iterate over each connected component
    for component_id, component in enumerate(components):
        # Get the indices of the lines in this component
        indices = list(component)
        # Get the geometries
        component_geoms = gdf.loc[indices, 'geometry'].tolist()

        # Ensure all geometries are LineString objects
        line_strings = []
        for geom in component_geoms:
            if isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    line_strings.append(line)
            else:
                line_strings.append(geom)

        # Create a MultiLineString from the LineString objects
        multi_line = MultiLineString(line_strings)
        geometries.append(multi_line)
        component_ids.append(component_id)

    # Create a DataFrame from the geometries and component IDs
    data = {'geometry': geometries, 'component_id': component_ids}
    df = pd.DataFrame(data)

    # Create the clean_gdf GeoDataFrame
    clean_gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=gdf.crs)
    clean_gdf["length"] = clean_gdf.geometry.length

    clean_gdf = clean_gdf[["geometry", "component_id", "length"]]
    # drop all but the longest one
    clean_gdf = clean_gdf.sort_values("length", ascending=False).iloc[0:1].drop(columns=["component_id","length"])
    return clean_gdf

clean_gdf = clean_grid_template(routing_template_op_gdf)
clean_gdf = clean_gdf.explode(index_parts=False).reset_index(drop=True)
visualize_routing_template(clean_gdf)

##### create connections from the prepared sources to the clean template and return the connections as well as the combined gdf ####

# Function to find the nearest point on a LineString
def nearest_point_on_line(line, point):
    # Convert the LineString to a linear ring to use the project method
    line_coords = list(line.coords)
    if line_coords[-1] == line_coords[0]:
        line_coords = line_coords[:-1]
    line_ring = LineString(line_coords)
    # Project the point onto the line
    distance = line_ring.project(point)
    # Interpolate the coordinate
    nearest_point = line_ring.interpolate(distance)
    return nearest_point

def connect_nodes_to_template(sources_data_op_gdf, routing_template_op_gdf):
    # Create a list to store the new connections
    new_connections = []

    file_path = os.path.join(temp_dir, 'connections.csv')
    print("The file path for the network connections is: " + str(file_path) + "\n")
    print("The paths already exist: " + str(os.path.exists(file_path)) + "\n")

    # Check if the file already exists
    if os.path.exists(file_path):
        # Read and transform the DataFrame from the file
        connections_df = pd.read_csv(file_path)
        # Ensure the 'geometry' column is properly converted to geometry objects and Geodataframe
        connections_df['geometry'] = connections_df['geometry'].apply(wkt.loads)
        connections_gdf = gpd.GeoDataFrame(connections_df, geometry='geometry')
        connections_gdf = connections_gdf.set_crs(routing_template_op_gdf.crs)
        connections_gdf = connections_gdf[["geometry"]]
    else:
    # Iterate over each point in the sources_data_op_gdf
        for idx, row in tqdm(sources_data_op_gdf.iterrows(), total=len(sources_data_op_gdf), desc="Connecting sources to network"):
            point = row['geometry']
            connected = False

            # Check if the point intersects with any LineString in the network
            for _, line_row in routing_template_op_gdf.iterrows():
                line = line_row['geometry']
                if point.intersects(line):
                    # If the point is on the network, add it to the connections
                    new_connections.append(point)
                    connected = True
                    break

            if not connected:
                # Find the nearest LineString
                distances = routing_template_op_gdf['geometry'].apply(lambda geom: point.distance(geom))
                nearest_line_idx = distances.idxmin()
                nearest_line = routing_template_op_gdf.loc[nearest_line_idx, 'geometry']

                # Find the nearest point on the LineString
                nearest_point = nearest_point_on_line(nearest_line, point)

                # Create a new LineString connecting the point to the nearest point on the LineString
                new_line = LineString([point, nearest_point])
                new_connections.append(new_line)
                # Create a new GeoDataFrame with the connections
                print("Jetzt sind wir hier: " + "\n")
                print(new_connections)
                connections_gdf = gpd.GeoDataFrame(geometry=new_connections)
                connections_gdf = connections_gdf.set_crs(routing_template_op_gdf.crs)
                connections_gdf = connections_gdf.rename(columns={"new_connections": "geometry"})
                connections_gdf = connections_gdf[["geometry"]]
                print("Wir sind jetzt hier: " + str(connections_gdf) + "\n")

        connections_gdf.to_csv(os.path.join(temp_dir, 'connections.csv'))

    routing_template_con_gdf = pd.concat([routing_template_op_gdf, connections_gdf], ignore_index=True)
    routing_template_con_gdf = routing_template_con_gdf.dissolve().explode(index_parts=False).reset_index(drop=True)
    return connections_gdf, routing_template_con_gdf

connections_gdf, routing_template_op_gdf = connect_nodes_to_template(sources_data_gdf, clean_gdf)

visualize_routing_template(connections_gdf)
visualize_routing_template(routing_template_op_gdf)

def calculate_length(gdf):
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set.")

    original_crs = gdf.crs

    # Decide on a projected CRS
    crs_obj = CRS.from_user_input(original_crs)
    if crs_obj.is_projected:
        proj_crs = crs_obj
    else:
        # Fallback: compute UTM zone from dataset centroid
        geodetic = crs_obj.get_geodetic_crs() or CRS.from_epsg(4326)
        centroid = gdf.to_crs(geodetic).unary_union.centroid
        lon, lat = centroid.x, centroid.y
        zone = int((lon + 180) // 6) + 1
        epsg = (32600 if lat >= 0 else 32700) + zone
        proj_crs = CRS.from_epsg(epsg)

    # Project, compute length, and project back
    gdf_proj = gdf.to_crs(proj_crs)
    lengths = gdf_proj.geometry.apply(lambda geom: geom.length if geom is not None else np.nan)
    gdf_proj["length"] = lengths

    # Return back in original CRS with the new column
    gdf_out = gdf_proj.to_crs(original_crs)
    return gdf_out

routing_template_op_gdf = calculate_length(routing_template_op_gdf).drop(columns=["index"], errors="ignore")

def plot_network_and_sources_nodes(W, sources_data_op_out_gdf, routing_template_op_gdf):
    # Set the font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # Loop through each scenario option
    for scenario_name, scenario_list in scenarios.items():
        # Filter the DataFrame for the scenario condition
        filtered_data = sources_data_op_out_gdf[sources_data_op_out_gdf["scenario"].isin(scenario_list)]

        # Set the figure size
        plt.figure(figsize=(12, 12))  # Adjust the size as needed

        # Draw the graph
        pos = {node: node for node in W.nodes()}
        nx.draw(W, pos, with_labels=False, node_size=0, node_color="black")

        # Ensure that the "color" column contains only valid color values
        valid_colors = filtered_data["color"].dropna().unique()
        color_map = {color: color for color in valid_colors}

        # Plot the data
        ax = filtered_data.plot(
            ax=plt.gca(),
            color=filtered_data["color"].map(color_map),  # Use the "color" column for coloring
            markersize=filtered_data["tco2_2023"]/2000,
            legend=False,  # Disable the automatic legend
            zorder=2,
            alpha=0.8
        )

        # Create a legend mapping from the "Sector" column
        sector_colors = filtered_data.groupby("Sector")["color"].first()
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=sector,
                                markerfacecolor=color, markersize=10)
                        for sector, color in sector_colors.items()]
        
        # test plot for shortest 
        # paths_gdf = paths_gdf.to_crs(epsg="3035")
        # paths_gdf.plot(ax=ax, color='red', linewidth=2, label='Paths')
        
        # Add title and custom legend
        plt.title(f"Routing Template with Industry Sources - {scenario_name}")

        # Add the custom legend to the bottom right
        plt.legend(handles=legend_elements, title="Sector", loc="lower right")

        # Get the bounds of the routing_template_op_gdf
        minx, miny, maxx, maxy = routing_template_op_gdf.total_bounds

        # Set the plot limits to the bounds of the routing_template_op_gdf
        # plt.xlim(minx, maxx)
        # plt.ylim(miny, maxy)

        # Save the plot as a PNG file in the figures subfolder
        plt.savefig(os.path.join(figures_dir, f"Routing_Template_with_Industry_Sources_{scenario_name.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
        plt.ioff()
        plt.draw()
    return

# define the network
def create_network_graph(routing_template_gdf, sources_gdf):
    W = nx.Graph()  # Create an empty graph

    # Sort the GeoDataFrame by length
    routing_template_gdf = routing_template_gdf.sort_values(by='length', ignore_index=True)

    # Add edges to the graph from the routing template gdf
    for idx, row in routing_template_gdf.iterrows():
        geom = row.geometry

        if geom.geom_type == 'LineString':
            # Get the coordinates of the LineString
            coords = list(geom.coords)
            # Add edges between consecutive coordinates
            for i in range(len(coords) - 1):
                W.add_edge(coords[i], coords[i+1], length=row["length"], weight=geom.length / len(coords))  # Distribute the length equally

        elif geom.geom_type == 'MultiLineString':
            # Iterate over each LineString in the MultiLineString
            for part in geom:
                coords = list(part.coords)
                # Add edges between consecutive coordinates
                for i in range(len(coords) - 1):
                    W.add_edge(coords[i], coords[i+1], weight=part.length / len(coords))  # Distribute the length equally

    # Add nodes to the graph from the sources data gdf
    for idx, row in sources_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'Point':
            # Extract longitude and latitude from the Point geometry
            longitude, latitude = geom.x, geom.y
            # Create a tuple to represent the node (you can use any unique identifier)
            node_id = (longitude, latitude)
            # Add the node to the graph with attributes
            W.add_node(
                node_id,
                ets_id=row['ets_id'],
                facility_name=row['facility_name'],
                operator_name=row['operator_name'],
                x=longitude,
                y=latitude,
                tco2_2023=row['tco2_2023'],
                nace_21=row['nace_21'],
                geometry=geom,
                NACE_Rev_2_1_Code=row['NACE Rev. 2.1 Code'],
                Sector=row['Sector'],
                scenario=row['scenario'],
                color=row['color']
            )
    return W

G = nx.Graph()  # Create an empty graph
print("Fear the routing template: " + "\n")
print(routing_template_op_gdf)
print(sources_data_gdf)
G = create_network_graph(routing_template_op_gdf, sources_data_gdf)
plot_network_and_sources_nodes(G, sources_data_gdf, routing_template_op_gdf)

def dijkstra_connect_sources(W, filtered_sources_df):
    # define TEMP file path to avoid having to calculate the paths again even if nothing changed
    file_path = os.path.join(temp_dir, 'paths_gdf.csv')
    print("The file path for the paths_gdf is: " + str(file_path) + "\n")
    print("The paths already exist: " + str(os.path.exists(file_path)) + "\n")

    # Check if the graph is connected
    if not nx.is_connected(W):
        print("Warning: The graph is not connected. This might affect path finding.")

        # Identify connected components
        connected_components = list(nx.connected_components(W))
        print(f"Number of connected components: {len(connected_components)}")

        for i, component in enumerate(connected_components):
            print(f"Connected component {i + 1}: {component}")

    # Check if the file already exists
    if os.path.exists(file_path):
        # Read the DataFrame from the file
        paths_df = pd.read_csv(file_path)
        # Convert WKT strings back to geometry objects
        paths_df['path_geometry'] = paths_df['path_geometry'].apply(lambda x: wkt.loads(x) if isinstance(x, str) and x != 'None' else None)
        paths_df = paths_df.rename(columns={'path_geometry': 'geometry'})
        paths_gdf = gpd.GeoDataFrame(paths_df, geometry='geometry')
        paths_gdf = paths_gdf.set_crs(routing_template_op_gdf.crs)
        paths_gdf = calculate_length(paths_gdf).drop(columns=["index"], errors="ignore")
    else:
        # Assuming filtered_sources_df is your GeoDataFrame and  is your NetworkX graph
        paths_df = pd.DataFrame()

        # Get the number of sources
        num_sources = len(filtered_sources_df)

        # Loop through all connections between the source data points with a progress bar
        for i in tqdm(range(num_sources), desc="Processing sources"):
            # Monitor RAM usage
            get_ram_usage(i)
            for j in tqdm(range(i + 1, num_sources), desc="Processing sinks", leave=False):
                source = filtered_sources_df.iloc[i]
                sink = filtered_sources_df.iloc[j]

                x_start, y_start = source.geometry.x, source.geometry.y
                x_end, y_end = sink.geometry.x, sink.geometry.y

                # Create Point objects for the start and end coordinates
                start_point = Point(x_start, y_start)
                end_point = Point(x_end, y_end)

                # Find the nearest node in the graph to the start and end points
                start_node = min(W.nodes(), key=lambda node: start_point.distance(Point(node[0], node[1])))
                # print("Start node: " + str(start_node) + "\n")
                end_node = min(W.nodes(), key=lambda node: end_point.distance(Point(node[0], node[1])))
                # print("End node: " + str(end_node) + "\n")

                # Print debugging information
                # print(f"Source: {source['facility_name']} ({x_start}, {y_start}) -> Node: {start_node}")
                # print(f"Sink: {sink['facility_name']} ({x_end}, {y_end}) -> Node: {end_node}")
                try:
                    shortest_path = nx.dijkstra_path(W, start_node, end_node)
                    # print(f"Shortest path found: {shortest_path}")
                except nx.NetworkXNoPath:
                    shortest_path = []
                    print("No path found between the nodes.")

                # Convert the shortest path to a LineString geometry
                if shortest_path:
                    # Get the coordinates of the nodes in the shortest path
                    path_coords = [(node[0], node[1]) for node in shortest_path]
                    path_geometry = LineString(path_coords)
                else:
                    path_geometry = None

                # Create a new row for the shortest path and append it to the DataFrame
                new_row = pd.DataFrame({
                    'source': [source['facility_name']],
                    'sink': [sink['facility_name']],
                    'shortest_path': [shortest_path],
                    'path_geometry': [path_geometry]
                })
                paths_df = pd.concat([paths_df, new_row], ignore_index=True)
        
        # Convert the paths_df to a GeoDataFrame
        paths_gdf = gpd.GeoDataFrame(paths_df, geometry='path_geometry', crs=filtered_sources_df.crs)

        # Save the DataFrame to the file - disable for every change in input, scenario or filter
        # paths_gdf.to_csv(file_path, index=False)
        print("DataFrame saved to file.")
    return paths_gdf

# Filter the sources dataframe
# filtered_sources_df = sources_data_gdf.copy() 
filtered_sources_df = sources_data_gdf[sources_data_gdf["scenario"] != "E"]
# filtered_sources_df = filtered_sources_df.head(5).reset_index(drop=True)

paths_gdf = dijkstra_connect_sources(G, filtered_sources_df)
visualize_routing_template(paths_gdf)

def plot_mst(W, sources_data_op_out_gdf, routing_template_op_gdf, figures_dir):
    # Set the font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # Create the MST
    mst = nx.minimum_spanning_tree(W)

    # Set the figure size
    plt.figure(figsize=(30, 30))  # Increased the size

    # Draw the graph
    pos = {node: node for node in W.nodes()}
    nx.draw(W, pos, with_labels=False, node_size=0, node_color="black")

    # Draw the MST edges
    mst_edges = mst.edges()
    mst_edge_list = [(u, v) for u, v in mst_edges]
    nx.draw_networkx_edges(W, pos, edgelist=mst_edge_list, edge_color='red', width=4)

    # Ensure that the "color" column contains only valid color values
    valid_colors = sources_data_op_out_gdf["color"].dropna().unique()
    color_map = {color: color for color in valid_colors}

    # Plot the data
    ax = sources_data_op_out_gdf.plot(
        ax=plt.gca(),
        color=sources_data_op_out_gdf["color"].map(color_map),  # Use the "color" column for coloring
        markersize=sources_data_op_out_gdf["tco2_2023"]/1000,  # Adjusted markersize
        legend=False,  # Disable the automatic legend
        zorder=2,
        alpha=0.8
    )

    # Create a legend mapping from the "Sector" column
    sector_colors = sources_data_op_out_gdf.groupby("Sector")["color"].first()
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=sector,
                            markerfacecolor=color, markersize=20)
                     for sector, color in sector_colors.items()]

    # Add title and custom legend
    plt.title("Minimum Spanning Tree with Industry Sources", fontsize=16)  # Increased title font size

    # Add the custom legend to the bottom right
    plt.legend(handles=legend_elements, title="Sector", loc="lower right", fontsize=12)  # Increased legend font size

    # Get the bounds of the routing_template_op_gdf
    minx, miny, maxx, maxy = routing_template_op_gdf.total_bounds

    # Set the plot limits to the bounds of the routing_template_op_gdf
    # plt.xlim(minx, maxx)
    # plt.ylim(miny, maxy)

    # Save the plot as a PNG file in the figures subfolder
    file_path = os.path.join(figures_dir, f"Minimum_Spanning_Tree_with_Industry_Sources_{scenario_name.replace(' ', '_')}.png")
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.ioff()
    plt.draw()
    return

def plot_steiner_tree(W, sources_data_op_out_gdf, routing_template_op_gdf, figures_dir):
    # Set the font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # Get the list of source nodes
    source_nodes = [(row.geometry.x, row.geometry.y) for idx, row in sources_data_op_out_gdf.iterrows()]

    # Create the Steiner Tree
    steiner_tree = nx.approximation.steiner_tree(W, source_nodes)

    # Set the figure size
    plt.figure(figsize=(12, 12))  # Increased the size

    # Draw the graph
    pos = {node: node for node in W.nodes()}
    nx.draw(W, pos, with_labels=False, node_size=0, node_color="black")

    # Draw the Steiner Tree edges
    steiner_edges = steiner_tree.edges()
    steiner_edge_list = [(u, v) for u, v in steiner_edges]
    nx.draw_networkx_edges(W, pos, edgelist=steiner_edge_list, edge_color='yellowgreen', width=6)

    # Ensure that the "color" column contains only valid color values
    valid_colors = sources_data_op_out_gdf["color"].dropna().unique()
    color_map = {color: color for color in valid_colors}

    # Plot the data
    ax = sources_data_op_out_gdf.plot(
        ax=plt.gca(),
        color=sources_data_op_out_gdf["color"].map(color_map),  # Use the "color" column for coloring
        markersize=sources_data_op_out_gdf["tco2_2023"]/2000,  # Adjusted markersize
        legend=False,  # Disable the automatic legend
        zorder=2,
        alpha=0.8
    )

    # Create a legend mapping from the "Sector" column
    sector_colors = sources_data_op_out_gdf.groupby("Sector")["color"].first()
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=sector,
                            markerfacecolor=color, markersize=10)
                     for sector, color in sector_colors.items()]

    # Add title and custom legend
    plt.title("Steiner Tree with Industry Sources", fontsize=16)  # Increased title font size

    # Add the custom legend to the bottom right
    plt.legend(handles=legend_elements, title="Sector", loc="lower right", fontsize=12)  # Increased legend font size

    # Get the bounds of the routing_template_op_gdf
    minx, miny, maxx, maxy = routing_template_op_gdf.total_bounds

    # Set the plot limits to the bounds of the routing_template_op_gdf
    # plt.xlim(minx, maxx)
    # plt.ylim(miny, maxy)

    # Save the plot as a PNG file in the figures subfolder
    file_path = os.path.join(figures_dir, f"Steiner_Tree_with_Industry_Sources_{scenario_name.replace(' ', '_')}.png")
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.ioff()
    plt.draw()
    return

#reduce routing template to Dijkstra Paths
routing_template_op_gdf = paths_gdf.dissolve().explode(index_parts=False).reset_index(drop=True)

# Loop through each scenario option
for scenario_name, scenario_list in scenarios.items():
    G = nx.Graph()
    # Filter the DataFrame for the scenario condition
    filtered_sources_df = sources_data_gdf[sources_data_gdf["scenario"].isin(scenario_list)]

    # Create the network graph for the filtered sources
    G = create_network_graph(routing_template_op_gdf, filtered_sources_df)

    # Plot the MST for the current scenario
    plot_mst(G, filtered_sources_df, routing_template_op_gdf, figures_dir)

    # Plot the Steiner Tree for the current scenario
    plot_steiner_tree(G, filtered_sources_df, routing_template_op_gdf, figures_dir)

STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')