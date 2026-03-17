##### network_module #####
#orginated by OL 10.03.2026

#This is the network module created for the IGC.NRW research project. It is subdivided into four submodules: Determining the geoscope, Creating the path template, creating the sources and creating the sinks.
#Additionally there is a visualization script to check the created data and the resulting network. The module is designed to be flexible and adaptable to different scenarios and data inputs, while also being efficient and scalable for larger datasets.

import time
import os
import pandas as pd
import geopandas as gpd
# import plotly.express as px
import matplotlib.pyplot as plt

START = time.perf_counter() 

print('Execute in Directory:')
print(os.getcwd() + "\n")

#read gpkg file with geo scope data

def get_geoscope(data_path=r".\data_module\Data", case_study="igc_nrw"):
    geoscope_dir = os.path.join(data_path, "scenario_run_data", str(case_study), "geoscope")
    gpkg_file = next((f for f in os.listdir(geoscope_dir) if f.endswith('.gpkg')), None)
    if gpkg_file is None:
        raise FileNotFoundError(f"No .gpkg file found in {geoscope_dir}")
    geoscope_gdf_high_res = gpd.read_file(os.path.join(geoscope_dir, gpkg_file))
    #unite all geometries to one geometry to determine the maximum area scope
    geoscope_gdf_agg = geoscope_gdf_high_res.dissolve()
    #only keep geometry column in the aggregated geoscope gdf
    geoscope_gdf_agg = geoscope_gdf_agg[["geometry"]]
    return geoscope_gdf_high_res, geoscope_gdf_agg

def visualize_geoscope(geoscope_gdf):
    geoscope_gdf['index'] = geoscope_gdf.index
    geoscope_gdf.plot(column='index', cmap="viridis")
    plt.show()
    return

work_geoscope_gdf_high_res, work_geoscope_gdf_agg = get_geoscope()
print("The disaggregated dataframe looks like this: " + str(work_geoscope_gdf_high_res) + "\n")
print("The aggregated dataframe looks like this: " + str(work_geoscope_gdf_agg) + "\n")

visualize_geoscope(work_geoscope_gdf_high_res)
visualize_geoscope(work_geoscope_gdf_agg)

STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')