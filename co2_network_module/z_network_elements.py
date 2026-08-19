crs = 4326
include_storages = False
co2_price = 200
import data
nrw_nuts_gdf = data.nrw_nuts_gdf
storages_gdf = data.storages_gdf
import pickle
import geopandas as gpd
import pandas as pd
from geopy.distance import geodesic
from shapely import Point, MultiPoint

import utils

#for i in storages_gdf.index:
 #   min_distance = 9e9
 #   loc_storage = storages_gdf.loc[i, 'location']
 #   for NUTS_j in nrw_nuts_gdf.index:
 #       loc_NUT = nrw_nuts_gdf.loc[NUTS_j, 'location']
 #       distance_ij = geodesic((loc_storage.x, loc_storage.y), (loc_NUT.x, loc_NUT.y)).kilometers
 #       if distance_ij < min_distance:
 #           storages_gdf.loc[i, 'distance'] = distance_ij
 #           min_distance = distance_ij
#storages_gdf = storages_gdf[storages_gdf['distance']<100]

storages_gdf = gpd.GeoDataFrame(storages_gdf, geometry='location', crs=crs)

if not include_storages: 
    storages_gdf = storages_gdf[storages_gdf.index.str.contains('Terminal')]
    

###Set crs
storages_gdf = utils.change_crs(storages_gdf, 4326)
