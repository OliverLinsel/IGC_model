import pandas as pd
import geopandas as gpd
from shapely import Point
import utils
crs = 25832#4326

ccs_cost = {'19.1' : 50000,
            '19.2' : 145000,
            '23.51': 125000,
            '23.52': 75000,
            '24.1': 112500,
            '20.11': 30000}

def f_ccs_cost(x):
    try:
        cost = ccs_cost[x]
    except KeyError:
        cost = 75000
    return cost

def find_nace(x):
    try:
        scenario = nace_df.loc[str(x), 'scenario']
    except KeyError:
        scenario = 'NotFound'
    return scenario

#Load Data
nace_df = pd.read_excel('C:\\Landwehr\\GIT\\Data\\nace_definitions.xlsx')
loc_df = pd.read_excel('C:\\Landwehr\\GIT\\Data\\EU_ETS_DE_Geo.xlsx')
#nrw_nuts_gdf = gpd.read_file('C:\\Landwehr\\GIT\\Data\\NUTS3_NRW_2024_4326.gpkg', layer=None)

#Process Data
nace_df["NACE Rev. 2.1 Code"] = nace_df["NACE Rev. 2.1 Code"].apply(lambda x: str(x))
nace_df = nace_df.set_index("NACE Rev. 2.1 Code")
loc_df.index = loc_df['MS-ID']
#nrw_nuts_gdf = nrw_nuts_gdf.to_crs(epsg=crs)
#nrw_nuts_gdf = nrw_nuts_gdf.drop(columns=['LEVL_CODE', 'CNTR_CODE', 'NAME_LATN', 'NUTS_NAME', 'MOUNT_TYPE', 'URBN_TYPE', 'COAST_TYPE'])
#nrw_nuts_gdf = nrw_nuts_gdf.set_index('NUTS_ID')
#nrw_nuts_gdf['location'] = nrw_nuts_gdf['geometry'].centroid

emissions_gdf = gpd.read_file("C:\\Landwehr\\GIT\\Data\\CO2_Point_Locations.gpkg")
  

emissions_gdf['Scenario'] = emissions_gdf['nace_21'].apply(lambda x: find_nace(x))

geometry_ps = [Point([(lon, lat)]) for lon, lat in zip(emissions_gdf['longitude'], emissions_gdf['latitude'])]

emissions_gdf = gpd.GeoDataFrame(emissions_gdf, geometry=geometry_ps, crs=4326)
emissions_gdf = emissions_gdf.to_crs(crs)

#nuts1 = utils.load_geodataframe('C:\\Landwehr\\GIT\\Data\\NUTS_RG_01M_2024_3035.gpkg', 'geometry', 'gpd', '', crs)
#nuts1 = nuts1.drop(columns=['LEVL_CODE', 'CNTR_CODE', 'NAME_LATN', 'NUTS_NAME', 'MOUNT_TYPE', 'URBN_TYPE', 'COAST_TYPE'])
#nuts1 = nuts1.set_index('NUTS_ID')
#nuts1['location'] = nuts1['geometry'].centroid

emissions_loc_gdf = emissions_gdf#gpd.sjoin(nuts1, emissions_gdf, how="inner", predicate='intersects')

emissions_loc_gdf['CCS Cost'] = emissions_loc_gdf['nace_21'].apply(lambda x: f_ccs_cost(x))

emissions_loc_gdf.to_excel("C:\\Landwehr\\GIT\\Data\\emissions.xlsx")
