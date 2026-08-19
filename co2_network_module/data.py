import pickle
from geopy.distance import geodesic
import shapely
from shapely import Point
import pandas as pd
import geopandas as gpd
import utils

###############################################################################
###                     Scenario settings                                   ###
###############################################################################

scenario_name = 'Base only'
#scenario = 'Base only'
scenarios = {"Base, S1, and E": ["Base", "S1", "E"],
             "Base and S1": ["Base", "S1"],
             "Base only": ["Base"],}
scenario = scenarios[scenario_name]

#years = [2030, 2035, 2040, 2045]
years = [2030, 2034, 2038, 2042, 2046, 2050]
no_years = len(years)
r = 0.05

timesteps = [0]
snapshots = pd.MultiIndex.from_product([years, timesteps],names=["period", "snapshot"])
storage_costs = pd.Series(15000, index=snapshots, name='storage_costs')

co2_price = pd.Series([100000, 120000, 140000, 160000, 180000, 200000], index=snapshots, name='co2_price')
###############################################################################
###                     Transport costs                                     ###
###############################################################################
def crf(n):
    return r * (1 + r)**n / ((1 + r)**n - 1)
annuity = crf(20)
#CAPEX
#per distance
capex_pipeline_dis     = 100000
capex_pipeline_off_dis = 171000
#per capacity
capex_pipeline_cap = 1000 * annuity    
capex_pipeline_off_cap = 1710 * annuity  
capex_ship_cap     = 9 * annuity    #times distance
capex_truck_cap    = 4 * annuity   #times distance

#OPEX
#fixed
opex_truck_fix    = 44300
#variable
opex_pipeline_var = 8 #times distance
opex_pipeline_off_var = 13.68 #times distance
opex_truck_var    = 480 #times distance
opex_ship_var     = 60 #times distance

capture_cost = 75000
eps = 1e-4


time_period = 8760
include_storages = False
###
carrier = 'CO2'
crs = 25832
lng_cap = 1.1
Grids = ['_P', '_T', '_W']
Grids2 = ['', '_P', '_T', '_W']

###############################################################################
###                              NUTS                                       ###
###############################################################################

nuts1 = utils.load_geodataframe('C:\\Landwehr\\GIT\\Data\\NUTS_RG_01M_2024_3035.gpkg', 'geometry', 'gpd', '', crs)
nrw_nuts_gdf = utils.load_geodataframe('C:\\Landwehr\\GIT\\Data\\NUTS3_NRW_2024_4326.gpkg', 'geometry', 'gpd', '', crs)
nrw_nuts_gdf = nrw_nuts_gdf.drop(columns=['LEVL_CODE', 'CNTR_CODE', 'NAME_LATN', 'NUTS_NAME', 'MOUNT_TYPE', 'URBN_TYPE', 'COAST_TYPE'])
nrw_nuts_gdf = nrw_nuts_gdf.set_index('NUTS_ID')
nrw_nuts_gdf['location'] = nrw_nuts_gdf['geometry'].centroid


###############################################################################
###                          CO2 demand                                     ###
###############################################################################
demand_gdf = utils.load_geodataframe('C:\\Landwehr\\GIT\\Data\\CO2_demand.xlsx', 'geometry', 'xlsx', crs, crs)
demand_gdf = demand_gdf.dropna()
demand_gdf = demand_gdf.rename(columns={"geometry": "location"})
demand_gdf = demand_gdf.rename(columns={"Unnamed: 0": "name"})
demand_gdf['name'] = demand_gdf['name'].apply(lambda x: 'SC-' + x)
demand_gdf = demand_gdf.set_index('name')
demand_gdf = gpd.GeoDataFrame(demand_gdf, geometry='location')
#demand_gdf = demand_gdf.set_crs(epsg=4326)
#demand_gdf = demand_gdf.to_crs(epsg=crs)
###############################################################################
###                          Emissions                                      ###
###############################################################################

emissions_gdf = utils.load_geodataframe('C:\\Landwehr\\GIT\\Data\\emissions.xlsx', 'geometry', 'xlsx', 4326, crs)
emissions_gdf = emissions_gdf[emissions_gdf['Scenario'].isin(scenario)]
emissions_gdf = emissions_gdf.drop(columns=['geometry','operator_name','Scenario', 'geometry'])
emissions_gdf = emissions_gdf.reset_index()
emissions_gdf = emissions_gdf.drop(columns=['index'])
for i in emissions_gdf.index:
    longitude = emissions_gdf.loc[i, 'longitude']
    latitude = emissions_gdf.loc[i, 'latitude']
    emissions_gdf.loc[i, 'location'] = Point(longitude, latitude)
emissions_gdf = gpd.GeoDataFrame(emissions_gdf, geometry='location')
emissions_gdf = emissions_gdf.set_crs(epsg=4326)
emissions_gdf = emissions_gdf.to_crs(epsg=crs)

emissions_gdf['Similar_location'] = None

for i in emissions_gdf.index:
    min_distance = 9e9
    point_i = emissions_gdf.loc[i, 'location']
    name_i = emissions_gdf.loc[i, 'facility_name']
    nace_i = emissions_gdf.loc[i, 'nace_21']
    if emissions_gdf.loc[i, 'Similar_location'] != None:
        continue
    for j in emissions_gdf.index:
        if i == j:
            continue
        point_j = emissions_gdf.loc[j, 'location']
        name_j = emissions_gdf.loc[j, 'facility_name']
        nace_j = emissions_gdf.loc[j, 'nace_21']
        distance = point_i.distance(point_j) / 1e3
        if distance < min_distance:
            if nace_i == nace_j:
                min_distance = distance
                similar_location = name_j
                similar_j = j
                #print(name_i, name_j)
    if min_distance < 5:
        emissions_gdf.loc[i, 'Similar_location'] = similar_location 
        emissions_gdf.loc[similar_j, 'Similar_location'] = similar_location 
    else:
        emissions_gdf.loc[i, 'Similar_location'] = name_i 
       
emissions_loc_gdf = emissions_gdf.groupby(by='Similar_location')
emissions_loc_gdf = emissions_loc_gdf.agg({'tco2_2023': 'sum'})

emissions_gdf = emissions_gdf.set_index('facility_name')
for facility in emissions_loc_gdf.index:
    for category in ['location', 'CCS Cost']:#['NUTS_ID', 'location', 'CCS Cost']:
        nuts_id = emissions_gdf.loc[facility, category]
        if isinstance(nuts_id, pd.Series):
            nuts_id = nuts_id.iloc[0] # Representation by first entry
        emissions_loc_gdf.loc[facility, category] = nuts_id

emissions_loc_gdf = gpd.GeoDataFrame(emissions_loc_gdf, geometry='location')
emissions_gdf = emissions_gdf.set_crs(epsg=crs)
emissions_loc_gdf = emissions_loc_gdf.reset_index()
for i in emissions_loc_gdf.index:
    name_i = 'Source-' + str(i)
    emissions_loc_gdf.loc[i, 'name'] = name_i
emissions_loc_gdf = emissions_loc_gdf.drop(columns=['Similar_location'])
emissions_loc_gdf['tco2_2023'] /=  1000
emissions_loc_gdf = emissions_loc_gdf.rename(columns={"tco2_2023": "ktco2_2023"})

emissions_gdf['tco2_2023'] /=  1000
emissions_gdf = emissions_gdf.rename(columns={"tco2_2023": "ktco2_2023"})

sum_emissions = emissions_loc_gdf['ktco2_2023'].sum()

###############################################################################
###                            Terminals                                    ###
###############################################################################

terminals_gdf = gpd.GeoDataFrame(columns=['A', 'location', 'B', 'C', 'D'])
terminals_gdf.loc['Terminal-WIL'] = [None, Point(8.109756, 53.64209), None, None, None]
terminals_gdf.loc['Terminal-ROT'] = [None, Point(4.250000, 51.90000), None, None, None]
terminals_gdf.loc['Terminal-LUB'] = [None, Point(13.65900, 54.14700), None, None, None]
terminals_gdf.loc['Terminal-STA'] = [None, Point(9.504460, 53.64853), None, None, None]
terminals_gdf = terminals_gdf.set_geometry('location')
terminals_gdf = terminals_gdf.set_crs(epsg=4326)
terminals_gdf = terminals_gdf.to_crs(epsg=crs)
terminals_gdf = gpd.GeoDataFrame(terminals_gdf, geometry='location')

    
storages_gdf = gpd.GeoDataFrame(columns=['A', 'location', 'B', 'C', 'D'])
storages_gdf.loc['Aurora-Resevoir'] = [None, Point(3.604167, 60.516667), None, None, None]
storages_gdf = storages_gdf.set_geometry('location')
storages_gdf = storages_gdf.set_crs(epsg=4326)
storages_gdf = storages_gdf.to_crs(epsg=crs)
storages_gdf = gpd.GeoDataFrame(storages_gdf, geometry='location')

###############################################################################
###                            Ports                                        ###
###############################################################################

ports_gdf = utils.load_geodataframe('C:\\Landwehr\\GIT\\Data\\binnenschiff_haefen.gpkg', 'geometry', 'gpd', '', crs)
ports_gdf['Keep'] = True

while 1:
    min_distance_ij = 9e9
    ports_gdf = ports_gdf[ports_gdf['Keep']==True]
    ports_gdf = ports_gdf.reset_index()
    ports_gdf = ports_gdf.drop(columns=['index'])
    for i in ports_gdf.index:
        point_i = ports_gdf.loc[i, 'geometry']
        for j in ports_gdf.index:
            if i == j:
                continue
            point_j = ports_gdf.loc[j, 'geometry']
            distance_ij = point_i.distance(point_j) / 1e3
            if distance_ij < min_distance_ij:
                min_distance_ij = distance_ij
                next_j = j
    if min_distance_ij < 50:
        ports_gdf.loc[next_j, 'Keep'] = False
    else:
        break

ports_gdf = ports_gdf.drop(columns=['address', 'modalität'])
ports_gdf = ports_gdf.rename(columns={"geometry": "location"})

ports_gdf = ports_gdf.set_geometry('location')
ports_gdf = ports_gdf.to_crs(epsg=crs)
ports_gdf = gpd.GeoDataFrame(ports_gdf, geometry='location')
###############################################################################
###                     tbd                                                 ###
###############################################################################

assert type(nrw_nuts_gdf) == type(gpd.GeoDataFrame())
assert type(emissions_loc_gdf) == type(gpd.GeoDataFrame())
assert type(terminals_gdf) == type(gpd.GeoDataFrame())