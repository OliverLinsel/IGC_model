##### geoscope_template_module #####
#orginated by OL 10.03.2026
#%%
#This is the network module created for the IGC.NRW research project. It is subdivided into four submodules: Determining the geoscope, Creating the path template, creating the sources and creating the sinks.
#Additionally there is a visualization script to check the created data and the resulting network. The module is designed to be flexible and adaptable to different scenarios and data inputs, while also being efficient and scalable for larger datasets.

import time
import os
import pandas as pd
import geopandas as gpd
# import plotly.express as px
import matplotlib.pyplot as plt
from setup import get_system_path

epsg="3035"
case_study = "igc_nrw"
case_study = "h2bb"
START = time.perf_counter() 

print('Execute in Directory:')
print(os.getcwd() + "\n")

#read gpkg file with geo scope data
def get_geoscope(data_path=os.path.join("data_module", "Data"), case_study=case_study):
    geoscope_dir = get_system_path(data_path, "scenario_run_data", case_study, "geoscope")
    gpkg_file = next((f for f in os.listdir(geoscope_dir) if f.endswith('.gpkg')), None)
    if gpkg_file is None:
        raise FileNotFoundError(f"No .gpkg file found in {geoscope_dir}")
    geoscope_gdf_high_res = gpd.read_file(os.path.join(geoscope_dir, gpkg_file))
    #to crs
    geoscope_gdf_high_res = geoscope_gdf_high_res.to_crs(epsg=epsg)
    invalid_geometries = geoscope_gdf_high_res[~geoscope_gdf_high_res.is_valid]
    if not invalid_geometries.empty:
        print(f"Found {len(invalid_geometries)} invalid geometries. Dropping them..." + "\n")
        print(invalid_geometries)
        geoscope_gdf_high_res = geoscope_gdf_high_res.drop(invalid_geometries.index)
    #unite all geometries to one geometry to determine the maximum area scope
    geoscope_gdf_agg = geoscope_gdf_high_res.dissolve()
    #only keep geometry column in the aggregated geoscope gdf
    geoscope_gdf_agg = geoscope_gdf_agg[["geometry"]].set_crs(epsg=epsg)
    return geoscope_gdf_high_res, geoscope_gdf_agg

def visualize_geoscope(geoscope_gdf, case_study = "igc_nrw"):
    geoscope_gdf['index'] = geoscope_gdf.index
    geoscope_gdf.plot(column='index', cmap="viridis")
    plt.ioff()
    plt.draw()
    plt.savefig(os.path.join("network_module", "figures", f"{case_study}_geoscope_geoplot.png"), dpi=300)
    # geoscope_gdf.to_clipboard()
    return

work_geoscope_gdf_high_res, work_geoscope_gdf_agg = get_geoscope()
print("The disaggregated dataframe looks like this: " + str(work_geoscope_gdf_high_res) + "\n")
print("The aggregated dataframe looks like this: " + str(work_geoscope_gdf_agg) + "\n")

visualize_geoscope(work_geoscope_gdf_high_res, case_study)
visualize_geoscope(work_geoscope_gdf_agg, case_study)

STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')

#%%