import geopandas as gpd
from shapely import wkt
import pandas as pd
import pickle

def change_crs(gdf, target_crs):
    if isinstance(gdf.index, gpd.GeoSeries):
        gdf.index= gdf.index.to_crs(epsg=target_crs)
    for c in gdf.columns:
        if isinstance(gdf[c], gpd.GeoSeries):
            gdf[c] = gdf[c].to_crs(epsg=target_crs)
    return gdf

def load_geodataframe(path, geometry, file_type, from_crs, crs):
    if file_type == "gpd":
        gdf = gpd.read_file(path)
    elif file_type == "xlsx":
        df = pd.read_excel(path)
        df[geometry] = df[geometry].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        gdf = gdf.set_crs(epsg=from_crs)
    elif file_type == "pickle":
        with open(path, "rb") as file:
            gdf = pickle.load(file)
        if not isinstance(gdf, gpd.GeoDataFrame):
            gdf = gpd.GeoDataFrame(gdf, geometry="location")
        gdf = gdf.set_crs(epsg=from_crs)
    else:
        raise ValueError("Unsupported file type")
    gdf = gdf.to_crs(epsg=25832)
    return gdf

def filter_nuts(gdf, nuts_id):
    gdf = gdf.reset_index()
    gdf = gdf.set_index('NUTS_ID')
    keep = False
    cntr_code = gdf.loc[nuts_id, 'CNTR_CODE']
    levl_code = gdf.loc[nuts_id, 'LEVL_CODE']
    if cntr_code == 'DE':
        if levl_code == 2:
            #if 'DE9' in nuts_id: #Niedersachsen
            keep = True
        elif levl_code == 3:
            if 'DEA' in nuts_id: # NRW
                keep = True
    elif cntr_code in ['NL']:
        if levl_code == 2:
            keep = True
    return keep