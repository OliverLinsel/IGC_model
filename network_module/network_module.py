##### network_module #####
#orginated by OL 10.03.2026

#This is the network module created for the IGC.NRW research project. It is subdivided into four submodules: Determining the geoscope, Creating the path template, creating the sources and creating the sinks.
#Additionally there is a visualization script to check the created data and the resulting network. The module is designed to be flexible and adaptable to different scenarios and data inputs, while also being efficient and scalable for larger datasets.

import time
import os
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.spatial import cKDTree
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import steiner_tree

#Define module parameters:
retrofit_cost_factor = 0.8
simplify_tolerance = 1000 #in meters, tolerance for simplifying the routing 

#define case study
case_study = "igc_nrw"
# case_study = "h2bb"

START = time.perf_counter() 

print('Execute in Directory:')
print(os.getcwd() + "\n")

#Define the paths
data_path = r".\data_module\Data"
output_path = r"output"

#get geoscope
from geoscope_module import get_geoscope

work_geoscope_gdf_high_res, work_geoscope_gdf_agg = get_geoscope(data_path, case_study)
print("The aggregated dataframe looks like this: " + str(work_geoscope_gdf_agg) + "\n")

#create the routing template
from routing_template_module import get_routing_template

routing_template_gdf = get_routing_template(data_path, case_study)
print(str(routing_template_gdf) + "\n")

#get sources
from sources_module import get_sources
sources_data_gdf = get_sources(data_path, case_study)
print(str(sources_data_gdf) + "\n")

#get sinks
from sinks_module import get_sinks

test_sinks_o = get_sinks(data_path, case_study)
print(str(test_sinks_o) + "\n")

# define the network
def define_network(work_geoscope_gdf_agg, routing_template_gdf, sources_data_gdf):
    #routing algorithm
    #connect all the industry points with oneanother using dijkstra's algorithm and networkx

    #create a graph from the routing template gdf
    G = nx.Graph()

    #add edges to the graph from the routing template gdf
    for idx, row in routing_template_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            for i in range(len(coords) - 1):
                G.add_edge(coords[i], coords[i+1], weight=geom.length)
        elif geom.geom_type == 'MultiLineString':
            for part in geom:
                coords = list(part.coords)
                for i in range(len(coords) - 1):
                    G.add_edge(coords[i], coords[i+1], weight=part.length)

    #get the nearest point in the routing template gdf for each industry point and add an edge to the graph if it is not already connected
    for idx, row in sources_data_gdf.iterrows():
        industry_point = row.geometry
        nearest_point = None
        min_distance = float('inf')
        for idx2, row2 in routing_template_gdf.iterrows():
            template_line = row2.geometry
            if template_line.geom_type == 'LineString':
                distance = industry_point.distance(template_line)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = template_line.interpolate(template_line.project(industry_point))
        if nearest_point is not None and not G.has_edge((industry_point.x, industry_point.y), (nearest_point.x, nearest_point.y)):
            G.add_edge((industry_point.x, industry_point.y), (nearest_point.x, nearest_point.y), weight=min_distance)
    return G

G = define_network(work_geoscope_gdf_agg, routing_template_gdf, sources_data_gdf)
print("The graph has " + str(G.number_of_nodes()) + " nodes and " + str(G.number_of_edges()) + " edges." + "\n")
#plot the graph G
pos = {node: node for node in G.nodes()}
nx.draw(G, pos, node_size=10)

def steiner_tree(G, sources, weight='weight'):
    #create a complete graph from the terminals
    complete_graph = nx.complete_graph(len(terminals))
    for i in range(len(terminals)):
        for j in range(i+1, len(terminals)):
            source = terminals[i]
            target = terminals[j]
            try:
                path_length = nx.dijkstra_path_length(G, source, target, weight=weight)
                complete_graph.add_edge(i, j, weight=path_length)
            except nx.NetworkXNoPath:
                complete_graph.add_edge(i, j, weight=float('inf'))

    #get the minimum spanning tree of the complete graph
    mst = nx.minimum_spanning_tree(complete_graph)

    #create a subgraph of G that contains only the edges in the minimum spanning tree
    steiner_tree_edges = []
    for edge in mst.edges(data=True):
        i, j, data = edge
        if data['weight'] < float('inf'):
            source = terminals[i]
            target = terminals[j]
            try:
                path = nx.dijkstra_path(G, source, target, weight=weight)
                steiner_tree_edges.extend([(path[k], path[k+1]) for k in range(len(path)-1)])
            except nx.NetworkXNoPath:
                continue

    steiner_tree_subgraph = G.edge_subgraph(steiner_tree_edges)

    return steiner_tree_subgraph

STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')