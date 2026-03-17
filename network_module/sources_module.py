##### network_module #####
#orginated by OL 10.03.2026

#This is the network module created for the IGC.NRW research project. It is subdivided into four submodules: Determining the geoscope, Creating the path template, creating the sources and creating the sinks.
#Additionally there is a visualization script to check the created data and the resulting network. The module is designed to be flexible and adaptable to different scenarios and data inputs, while also being efficient and scalable for larger datasets.

import time
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

#delete after development
case_study = "igc_nrw"
data_path = r".\data_module\Data"
output_path = r"output"
#end of delete after development

START = time.perf_counter() 

print('Execute in Directory:')
print(os.getcwd() + "\n")

def get_sources_from_spreadsheet(emissions_dir):
    sources_df = pd.read_excel(os.path.join(emissions_dir, next((f for f in os.listdir(emissions_dir) if f.endswith('.xlsx')), None)))
    #use columns longitude (lon, y) & latitude (lat, x) to create Point geometries, send error message if there are no such 
    if not any(col in sources_df.columns for col in ["Longitude", 'longitude', 'lon', 'y']):
        raise ValueError(f"No longitude column found in spreadsheet in {emissions_dir}")
    if not any(col in sources_df.columns for col in ["Latitude", 'latitude', 'lat', 'x']):
        raise ValueError(f"No latitude column found in spreadsheet in {emissions_dir}")
    lon_col = next(col for col in sources_df.columns if col in ["Longitude", 'longitude', 'lon', 'y'])
    lat_col = next(col for col in sources_df.columns if col in ["Latitude", 'latitude', 'lat', 'x'])
    sources_gdf = gpd.GeoDataFrame(sources_df, geometry=gpd.points_from_xy(sources_df[lon_col], sources_df[lat_col]))
    return sources_gdf

def get_sources_from_geospatial_data(emissions_dir):
    sources_gdf = gpd.read_file(os.path.join(emissions_dir, next((f for f in os.listdir(emissions_dir) if f.endswith('.gpkg')), None)))
    return sources_gdf

def get_sources_from_database(database_connection_info):
    #this function is a placeholder for future development, as the current data for the IGC.NRW project is not stored in a database. The implementation of this function will depend on the specific database being used and the structure of the data within that database.
    return

def assign_sources_to_scenarios(sources_data):
    return sources_data

def visualize_sources(sources_data):
    #plot the source locations with diamter corresponding to the "EMISSIONS_2023" or "tco2_2023" column and color group by "Final Main Activity Type Name"
    size_col = "tco2_2023"
    color_col = "nace_21"
    sources_data.plot(column=color_col, markersize=sources_data[size_col]/10000, legend=True, figsize=(10,10))
    #increase plot size and reduce legend size
    plt.show()
    return

def get_sources(data_path=r".\data_module\Data", case_study="igc_nrw"):
    emissions_dir = os.path.join(data_path, "scenario_run_data", str(case_study), "emissions")
    #test if there is a spreadsheet or a geospatial data file in the emissions directory and call the corresponding function to get the sources. If there are both, prioritize the geospatial data file. If there are none, raise an error.
    spreadsheet_file = next((f for f in os.listdir(emissions_dir) if f.endswith('.xlsx')), None)
    geospatial_file = next((f for f in os.listdir(emissions_dir) if f.endswith('.gpkg')), None)
    database_file = None #placeholder for future development
    if database_file is not None:
        sources_data_gdf = get_sources_from_database(database_file)
    elif geospatial_file is not None:
        sources_data_gdf = get_sources_from_geospatial_data(emissions_dir)
    elif spreadsheet_file is not None:
        sources_data_gdf = get_sources_from_spreadsheet(emissions_dir)
    else:
        raise FileNotFoundError(f"No spreadsheet or geospatial data file found in {emissions_dir}")
    return sources_data_gdf

test_sources_o = get_sources()
print(str(test_sources_o) + "\n")
visualize_sources(test_sources_o)

STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')