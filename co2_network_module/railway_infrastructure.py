import geopandas as gpd
from shapely import LineString
from tqdm import tqdm
import pandas as pd

import utils
import data
crs = data.crs
nuts = data.nuts1

rail_de = gpd.read_file("C:\\Landwehr\\GIT\\Data\\schiene_schienennetzdb_2D.gpkg")
rail_de = rail_de[rail_de['bahnnutzun'].str.contains('Gz')]
rail_de = rail_de[rail_de['bahnart'].str.contains('Hauptbahn')]

rail_nl = gpd.read_file("C:\\Landwehr\\GIT\\Data\\RailTransportNetwork.gml", layer="RailwayLink")

rail_de.plot()
rail_nl.plot()
raise Error

gdf_raw = gdf.copy(deep=True)

nrw = nuts[nuts['NUTS_ID'].str.contains('DEA')]
nrw = nrw.to_crs(crs)

gdf = gdf.reset_index()
gdf = gdf[['id', 'geometry', 'strecke_ku']]
gdf.columns = ['name', 'geometry', 'strecke_ku']

track_names = gdf['strecke_ku']
track_names = track_names.drop_duplicates()

railways = pd.DataFrame()

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
for street_i in tqdm(track_names):
    street_netherlands_i = gdf_raw[gdf_raw['strecke_ku']==street_i]
    street_netherlands_i = street_netherlands_i.sort_values(by='von_km_i', ascending=True)
    wegnummer_i = street_netherlands_i['strecke_nr']
    for wegnummer_ij in wegnummer_i:
        street_netherlands_ij = street_netherlands_i[street_netherlands_i['strecke_nr']==wegnummer_ij]
        geometry_i = []
        i = 0
        for j in street_netherlands_ij.index:
            line_j = street_netherlands_ij.loc[j, 'geometry']
            end_km_j1 = street_netherlands_ij.loc[j, 'bis_km_i']
            point_1 = line_j.coords[0]
            point_2 = line_j.coords[1]
            if i > 0:
                if end_km_j1 - end_km_j0 > 10:
                    railways = add_to_gdf(geometry_i, railways, street_i)
                    geometry_i = []
                    geometry_i = add_geometry(geometry_i, point_1, point_2)
                    i = 0
                    continue
            geometry_i = add_geometry(geometry_i, point_1, point_2)
            end_km_j0 = end_km_j1
            i += 1
        streets_netherlands = add_to_gdf(geometry_i, railways, street_i)

railways = railways.drop_duplicates(subset='geometry')
          
#streets_netherlands.columns = streets_nrw_gdf.columns
railways = gpd.GeoDataFrame(railways)
railways = railways.set_crs(gdf_raw.crs)
railways = railways.to_crs(data.crs)
railways = railways.set_geometry('geometry')

#streets_nrw_gdf = pd.concat([streets_nrw_gdf, streets_netherlands], ignore_index=True)