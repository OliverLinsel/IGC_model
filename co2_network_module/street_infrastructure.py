import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import pickle
from shapely.geometry import LineString

import utils
import data
crs = data.crs
nuts = data.nuts1

###############################################################################
###                             Streets                                     ###
###############################################################################

streets_nrw_gdf = utils.load_geodataframe('C:\\Landwehr\\GIT\\Data\\strasse_autobahnen.gpkg', 'location', 'gpd', 4326, crs)
nrw = nuts[nuts['NUTS_ID'].str.contains('DEA')]
nrw = nrw.to_crs(crs)

streets_nrw_gdf = streets_nrw_gdf.reset_index()
streets_nrw_gdf = streets_nrw_gdf[['id', 'geometry']]
streets_nrw_gdf.columns = ['name', 'geometry']

streets_netherlands_raw = gpd.read_file('C:\\Landwehr\\GIT\\Data\\Wegvakken.gpkg')
street_names = streets_netherlands_raw[streets_netherlands_raw['ROUTELTR']=='A']

street_names = street_names['STT_NAAM']
street_names = street_names.drop_duplicates()

streets_netherlands = pd.DataFrame()

def add_geometry(geometry, point_1, point_2):
    if point_1 not in geometry_i:
        geometry.append(point_1)
        end_km_j0 = end_km_j1
    if point_2 not in geometry_i:
        geometry.append(point_2)
        end_km_j0 = end_km_j1
    return geometry

def add_to_gdf(geometry, df, street):
    if len(geometry) > 1:
        line_i = LineString(geometry)
        idx = len(df)
        df.loc[idx, 'name'] = street
        df.loc[idx, 'geometry'] = line_i
    return df

print("Filter all relevant dutch streets")
for street_i in tqdm(street_names):
    street_netherlands_i = streets_netherlands_raw[streets_netherlands_raw['STT_NAAM']==street_i]
    street_netherlands_i = street_netherlands_i.sort_values(by='BEGINKM', ascending=True)
    street_netherlands_i = street_netherlands_i[street_netherlands_i['POS_TV_WOL']=='R']
    street_netherlands_i = street_netherlands_i[street_netherlands_i['BST_CODE']=='HR']
    wegnummer_i = street_netherlands_i['WEGNUMMER']
    for wegnummer_ij in wegnummer_i:
        street_netherlands_ij = street_netherlands_i[street_netherlands_i['WEGNUMMER']==wegnummer_ij]
    #if 'Rijksweg' not in street_i:
    #    continue
        geometry_i = []
        i = 0
        for j in street_netherlands_ij.index:
            line_j = street_netherlands_ij.loc[j, 'geometry']
            #beginn_km_j1 = street_netherlands_ij.loc[j, 'BEGINKM']
            end_km_j1 = street_netherlands_ij.loc[j, 'EINDKM']
            point_1 = line_j.coords[0]
            point_2 = line_j.coords[1]
            if i > 0:
                if end_km_j1 - end_km_j0 > 10:
                    streets_netherlands = add_to_gdf(geometry_i, streets_netherlands, street_i)
                    geometry_i = []
                    geometry_i = add_geometry(geometry_i, point_1, point_2)
                    i = 0
                    continue
            geometry_i = add_geometry(geometry_i, point_1, point_2)
            end_km_j0 = end_km_j1
            i += 1
        streets_netherlands = add_to_gdf(geometry_i, streets_netherlands, street_i)

streets_netherlands = streets_netherlands.drop_duplicates(subset='geometry')
          
streets_netherlands.columns = streets_nrw_gdf.columns
streets_netherlands = gpd.GeoDataFrame(streets_netherlands)
streets_netherlands = streets_netherlands.set_crs(streets_netherlands_raw.crs)
streets_netherlands = streets_netherlands.to_crs(data.crs)
streets_netherlands = streets_netherlands.set_geometry('geometry')

streets_nrw_gdf = pd.concat([streets_nrw_gdf, streets_netherlands], ignore_index=True)

with open("C:\\Landwehr\\GIT\\Data\\streets_nrw_gdf.pkl", "wb") as f:
    pickle.dump(streets_nrw_gdf, f)
    
#####

#streets_nrw_gdf = utils.load_geodataframe('Data/schiene_schienennetzdb_2D.gpkg', 'location', 'gpd', 4326, crs)