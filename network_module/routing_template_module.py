##### routing_template_module #####
#orginated by OL 10.03.2026

#This is the network module created for the IGC.NRW research project. It is subdivided into four submodules: Determining the geoscope, Creating the path template, creating the sources and creating the sinks.
#Additionally there is a visualization script to check the created data and the resulting network. The module is designed to be flexible and adaptable to different scenarios and data inputs, while also being efficient and scalable for larger datasets.

import time
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from setup import get_system_path

#delete after development
case_study = "igc_nrw"
case_study = "h2bb"
data_path = r".\data_module\Data"
output_path = r"output"
epsg="3035"
default_simplify_tolerance = 1000
#end of delete after development

START = time.perf_counter() 

print('Execute in Directory:')
print(os.getcwd() + "\n")

def read_all_existing_pipelines(data_path=os.path.join("data_module", "Data"), case_study="igc_nrw"):
    geoscope_dir = get_system_path(data_path, "scenario_run_data", case_study, "routing_template", "potential_retrofit_pipelines")
    #loop for all files that are geopackages, read them as gdf and concatenate them to one gdf
    potential_retrofit_routes_gdf = gpd.GeoDataFrame()
    for file in os.listdir(geoscope_dir):
        if file.endswith('.gpkg'):
            temp_gdf = gpd.read_file(os.path.join(geoscope_dir, file))
            #to crs
            temp_gdf = temp_gdf.to_crs(epsg=epsg)
            potential_retrofit_routes_gdf = pd.concat([potential_retrofit_routes_gdf, temp_gdf], ignore_index=True)
        #if there are no geopackage files, raise an error
    if potential_retrofit_routes_gdf.empty:
        raise FileNotFoundError(f"No .gpkg files found in {geoscope_dir}")
    return potential_retrofit_routes_gdf

def read_all_potential_grids(data_path=os.path.join("data_module", "Data"), case_study="igc_nrw"):
    geoscope_dir = get_system_path(data_path, "scenario_run_data", case_study, "routing_template", "potential_routes")
    #loop for all files that are geopackages, read them as gdf and concatenate them to one gdf
    potential_routes_gdf = gpd.GeoDataFrame()
    for file in os.listdir(geoscope_dir):
        if file.endswith('.gpkg'):
            temp_gdf = gpd.read_file(os.path.join(geoscope_dir, file))
            #to crs
            temp_gdf = temp_gdf.to_crs(epsg=epsg)
            potential_routes_gdf = pd.concat([potential_routes_gdf, temp_gdf], ignore_index=True)
        #if there are no geopackage files, raise an error
    if potential_routes_gdf.empty:
        raise FileNotFoundError(f"No .gpkg files found in {geoscope_dir}")
    return potential_routes_gdf

def combine_and_process_routes(potential_retrofit_routes_gdf, potential_routes_gdf):
    geometries_df = pd.concat([potential_retrofit_routes_gdf[["geometry"]], potential_routes_gdf[["geometry"]]], ignore_index=True)
    geometries_gdf = geometries_df.set_geometry("geometry")
    geometries_gdf = geometries_gdf.set_crs(epsg=epsg).reset_index(drop=True)
    #dissolve all geometries to one geometry to create a routing template that covers the entire geoscope
    routing_template_gdf = geometries_gdf.dissolve()
    print(str(routing_template_gdf) + "\n")
    #simplify the routing template geometry to reduce the complexity of the network
    routing_template_gdf["geometry"] = routing_template_gdf["geometry"].simplify(tolerance=default_simplify_tolerance)
    return routing_template_gdf

def visualize_routing_template(routing_template_gdf, case_study, version):
    routing_template_gdf['index'] = routing_template_gdf.index
    routing_template_gdf.plot(column='index', cmap="viridis")
    plt.draw()
    plt.savefig(os.path.join("network_module", "figures", f"{case_study}_{version}_routing_template_plot.png"), dpi=300)
    return

def get_routing_template(data_path=r".\data_module\Data", case_study="igc_nrw"):
    #read all existing pipelines and potential grids, visualize them and return them as a gdf
    potential_retrofit_routes_gdf = read_all_existing_pipelines(data_path, case_study)
    potential_routes_gdf = read_all_potential_grids(data_path, case_study)
    routing_template_gdf = combine_and_process_routes(potential_retrofit_routes_gdf, potential_routes_gdf)
    return routing_template_gdf, potential_retrofit_routes_gdf, potential_routes_gdf

routing_template_gdf, retrofit_routes_gdf, potential_routes_gdf = get_routing_template(data_path, case_study)
visualize_routing_template(routing_template_gdf, case_study, 1)
visualize_routing_template(retrofit_routes_gdf, case_study, 2)
visualize_routing_template(potential_routes_gdf, case_study, 3)

STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')