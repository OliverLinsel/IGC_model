import utils
import data
import numpy as np
crs = data.crs
from tqdm import tqdm
import pickle
import pandas as pd
from shapely.geometry import Point, MultiPoint, LineString
from shapely.ops import nearest_points, split
import geopandas as gpd
import matplotlib.pyplot as plt
import data
nuts = data.nuts1
nrw_nuts_gdf = data.nrw_nuts_gdf

nuts['Keep'] = nuts['NUTS_ID'].apply(lambda x: utils.filter_nuts(nuts, x))
region = nuts[nuts['Keep']==True]

def add_link(df, point_i, next_j, min_distance_ij):
    link_ij =LineString([point_i, next_j])
    link_ji =  LineString([next_j, point_i])
    if len(df) == 0:
        pass
    elif link_ij in df['Link_ij'].values or link_ji in df['Link_ij'].values:
        return df
    idx = len(simple_grid_df)
    df.loc[idx, 'Point i'] = point_i
    df.loc[idx, 'Point j'] = next_j
    df.loc[idx, 'Distance_ij'] = min_distance_ij
    df.loc[idx, 'Link_ij'] = link_ij
    return df

def find_distance(i, j):
    distance_ij = i.distance(j) / 1e3
    return distance_ij

def remove_duplicates(df):
    Points = df['Point i'].tolist() + df['Point j'].tolist()
    for point_i in tqdm(Points):
        for point_j in Points:
            df_start_i = df[df['Point i'] == point_i]
            df_start_ij = df_start_i[df_start_i['Point j'] == point_j]  
            
            df_start_j = df[df['Point i'] == point_j]
            df_start_ji = df_start_j[df_start_j['Point j'] == point_i]  
            
            if len(df_start_ij) > 1 or len(df_start_ji) > 1:
                raise Error
    return df

def f_distance(df):
    points = df['point_ij'].values
    coords = np.array([(p.x, p.y) for p in points])
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
    distance_df = pd.DataFrame(dist_matrix)
    distance_df /= 1e3
    distance_df.columns = points
    distance_df.index = points   
    return distance_df

def covers_nrw(point):
    covers = nrw.covers(point).any()
    min_distance = nrw.distance(point).min() / 1e3
    return covers, min_distance

def covers_region(point, gdf):
    cov = False
    for geo in gdf['geometry']:
        cov_i = geo.covers(point)#.any()
        if cov_i == True:
            cov = True
    return cov

def find_loose_start(link, df):
    start_i = Point(link.coords[0])   
    df_start_i = df[df['Point i'] == start_i]
    df_start_j = df[df['Point j'] == start_i]   
    if (len(df_start_i) + len(df_start_j)) <=1:
        return start_i
    else:
        return None
    
def find_loose_end(link, df):
    end_i = Point(link.coords[-1])      
    df_end_i = df[df['Point i'] == end_i]
    df_end_j = df[df['Point j'] == end_i]
    if (len(df_end_i) + len(df_end_j)) <= 1:
        return end_i
    else:
        return None
    
def add_connection(df, point_i, next_j, min_distance_ij):
    link_ij =LineString([point_i, next_j])
    link_ji =  LineString([next_j, point_i])
    if len(df) == 0:
        pass
    elif link_ij in df['Link_ij'].values or link_ji in df['Link_ij'].values:
        return df
    #Add new link
    df.loc[link_ij, 'Point i'] = point_i
    df.loc[link_ij, 'Point j'] = next_j
    df.loc[link_ij, 'Distance_ij'] = min_distance_ij
    df.loc[link_ij, 'Link_ij'] = LineString([point_i, next_j])
    df.loc[link_ij, 'Loose_end'] = False
    return df

def erase_corners(df):
    df['Used'] = True
    Points = df['Point i'].tolist() + df['Point j'].tolist()
    for point_i in tqdm(Points):  
        df_start_i = df[df['Point i'] == point_i]
        df_start_j = df[df['Point j'] == point_i]   
        if (len(df_start_i) + len(df_start_j)) == 2:
            
            if len(df_start_i) == 1:
                link_i = df_start_i.index[0]
                link_j = df_start_j.index[0]
                point_i_new = df_start_i['Point j'].iloc[0]
                point_j_new = df_start_j['Point i'].iloc[0]
            elif len(df_start_i) == 2:
                link_i = df_start_i.index[0]
                link_j = df_start_i.index[1]
                point_i_new = df_start_i['Point j'].iloc[0]
                point_j_new = df_start_i['Point j'].iloc[1]
            elif len(df_start_j) == 2:
                link_i = df_start_j.index[0]
                link_j = df_start_j.index[1]
                point_i_new = df_start_j['Point i'].iloc[0]
                point_j_new = df_start_j['Point i'].iloc[1]
            df.loc[link_i, 'Used'] = False
            df.loc[link_j, 'Used'] = False
            
            distance_i = df.loc[link_i, 'Distance_ij']
            distance_j = df.loc[link_j, 'Distance_ij']
            distance_ij = distance_i + distance_j
            
            new_link_ij = LineString([point_i_new, point_j_new])
            new_link_ji = LineString([point_j_new, point_i_new])
            
            try:
                distance_old = df.loc[new_link_ij, 'Distance_ij']
                if distance_old < distance_ij:
                    continue
                else:
                    pass
            except KeyError:
                pass
            try:
                distance_old = df.loc[new_link_ji, 'Distance_ij']
                if distance_old < distance_ij:
                    continue
                else:
                    pass
            except KeyError:
                pass
          
            df.loc[new_link_ij, 'Used'] = True
            
            df.loc[new_link_ij, 'Point i'] = point_i_new
            df.loc[new_link_ij, 'Point j'] = point_j_new
            df.loc[new_link_ij, 'Distance_ij'] = distance_ij
            df.loc[new_link_ij, 'Link_ij'] = new_link_ij
        df = df[df['Used']==True]
    return df

def delete_loose_ends(df, delete_distance):
    df['Delete'] = False
    df['Link_ij_index'] = df['Link_ij']
    df = df.set_index('Link_ij_index')
    
    for link_i in tqdm(df.index):
        start_i = Point(link_i.coords[0])
        end_i = Point(link_i.coords[-1])
        
        df_start_i = df[df['Point i'] == start_i]
        df_start_j = df[df['Point j'] == start_i]
        
        df_end_i = df[df['Point i'] == end_i]
        df_end_j = df[df['Point j'] == end_i]
        
        if df.loc[link_i, 'Distance_ij']> delete_distance:
            if (len(df_start_i) + len(df_start_j)) <= 1 and (len(df_end_i) + len(df_end_j)) <= 1:
                df.loc[link_i, 'Delete'] = True
        else:
            if (len(df_start_i) + len(df_start_j)) <= 1 or (len(df_end_i) + len(df_end_j)) <= 1:
                df.loc[link_i, 'Delete'] = True
        
    df = df[df['Delete']==False]   
    return df

def connect_loose_ends(df, mode, connection_distance):
    
    df['Keep'] = True  
    df['Loose_start'] = df['Link_ij'].apply(lambda x: find_loose_start(x, df))
    df['Loose_end'] = df['Link_ij'].apply(lambda x: find_loose_end(x, df))
       
    Loose_points = pd.DataFrame(df['Loose_start'].tolist() + df['Loose_end'].tolist(), columns=['point_ij'])
    Loose_points = Loose_points.dropna()
    Loose_points = Loose_points.drop_duplicates()
    distance_lp_df = f_distance(Loose_points)
    for loose_point_i in Loose_points['point_ij']:
        min_distance_ij = 9e9
        next_point_j = None
        if mode == 'point':
            for loose_point_j in Loose_points['point_ij']:
                if loose_point_i == loose_point_j:
                    continue
                distance_ij = distance_lp_df.loc[loose_point_i, loose_point_j]
                if distance_ij < min_distance_ij:
                    min_distance_ij = distance_ij
                    next_point_j = loose_point_j
            if min_distance_ij < connection_distance:
                df = add_connection(df, loose_point_i, next_point_j, min_distance_ij)
        if mode == 'link':
            for link_j, point_ji, point_jj in zip(df['Link_ij'],df['Point i'],df['Point j']):
                if loose_point_i == point_ji or loose_point_i == point_jj:
                    continue
                distance_ij = loose_point_i.distance(link_j) / 1e3
                if distance_ij < min_distance_ij:
                    min_distance_ij = distance_ij
                    next_link_j = link_j
            if min_distance_ij < connection_distance:
                next_point_j = next_link_j.interpolate(next_link_j.project(loose_point_i))
                df = add_connection(df, loose_point_i, next_point_j, min_distance_ij)
    return df

nrw = nuts[nuts['NUTS_ID'].str.contains('DEA')]
nrw = nrw.to_crs(crs)

##############
    
mode = 'Street'

#with open("C:\\Landwehr\\GIT\\Data\\streets_nrw_gdf.pkl", "rb") as f:
#    streets_nrw_gdf = pickle.load(f)
#    c = 'C4'
streets_nrw_gdf = utils.load_geodataframe('C:\\Landwehr\\GIT\\Data\\strasse_autobahnen.gpkg', 'location', 'gpd', 4326, crs)
c = 'C4'

###############################################################################
fig, axes = plt.subplots(2, 2, figsize=[10,12], dpi=500)
fig.suptitle('Infrastructure', size=30)
ax_i = axes[0,0] 
ax_i.set_facecolor((0.8, 0.9, 1))
nuts.plot(ax=ax_i, facecolor='darkgrey', edgecolor='lightgrey', linewidth=0.5)
region.plot(ax=ax_i, facecolor='white', edgecolor='lightgrey', linewidth=0.5)
streets_nrw_gdf.plot(ax=ax_i, facecolor='none', edgecolor=c, linewidth=1)
ax_i.set_xlim(155000, 925000)
ax_i.set_ylim(5225000, 6100000)
###############################################################################
 
if mode == 'Water':
    with open("C:\\Landwehr\\GIT\\Data\\waterways.pkl", "rb") as f:
        streets_nrw_gdf = pickle.load(f)
        c = 'C0'

###############################################################################
ax_i = axes[1,0] 
ax_i.set_facecolor((0.8, 0.9, 1))
nuts.plot(ax=ax_i, facecolor='darkgrey', edgecolor='lightgrey', linewidth=0.5)
region.plot(ax=ax_i, facecolor='white', edgecolor='lightgrey', linewidth=0.5)
streets_nrw_gdf.plot(ax=ax_i, facecolor='none', edgecolor=c, linewidth=1)
ax_i.set_xlim(155000, 925000)
ax_i.set_ylim(5225000, 6100000)
###############################################################################

##raise Error
sindex = streets_nrw_gdf.sindex
intersections_df = pd.DataFrame()

print("Calculate intersections")
for i in tqdm(streets_nrw_gdf.index):
    line_i = streets_nrw_gdf.loc[i, 'geometry']
    Intersections_i = []
 
    possible_matches_index = list(sindex.intersection(line_i.bounds))

    for j in possible_matches_index:
        if i == j:
            continue
        line_j = streets_nrw_gdf.loc[j, 'geometry']

        intersect_ij = line_i.intersects(line_j)

        if intersect_ij == True:
            intersection_ij = line_i.intersection(line_j)

            if intersection_ij.geom_type == "MultiPoint":
                for point_ij in intersection_ij.geoms:
                    index = len(intersections_df)
                    intersections_df.loc[index, 'line_i'] = line_i
                    intersections_df.loc[index, 'line_j'] = line_j
                    intersections_df.loc[index, 'point_ij'] = point_ij

            elif intersection_ij.geom_type == "MultiLineString" or intersection_ij.geom_type == "LineString":
                continue

            else:
                index = len(intersections_df)
                intersections_df.loc[index, 'line_i'] = line_i
                intersections_df.loc[index, 'line_j'] = line_j
                intersections_df.loc[index, 'point_ij'] = intersection_ij              
                
for i in tqdm(streets_nrw_gdf.index):
    line_i = streets_nrw_gdf.loc[i, 'geometry']
    start_i = line_i.coords[0]
    end_i = line_i.coords[-1]

    index = len(intersections_df)
    intersections_df.loc[index, 'line_i'] = line_i
    intersections_df.loc[index, 'line_j'] = line_i
    intersections_df.loc[index, 'point_ij'] = Point(start_i)
    
    index = len(intersections_df)
    intersections_df.loc[index, 'line_i'] = line_i
    intersections_df.loc[index, 'line_j'] = line_i
    intersections_df.loc[index, 'point_ij'] = Point(end_i)
                
intersections = intersections_df['point_ij']
intersections = intersections.drop_duplicates()

intersections_df = pd.DataFrame(intersections)
intersections_df['point_ij_index'] = intersections_df['point_ij']
intersections_df = intersections_df.set_index('point_ij_index')
 
covers_df = intersections_df.copy(deep=True) 
covers_df['Covers'] = covers_df['point_ij'].apply(lambda x: covers_nrw(x)[0])
covers_df['Min Distance'] = covers_df['point_ij'].apply(lambda x: covers_nrw(x)[1])

#if mode != 'Water':
#intersections_df['Covers'] = intersections_df['point_ij'].apply(lambda x: covers_region(x, region))
#intersections_df = intersections_df[intersections_df['Covers']==True]

print("Count number of close points")

if mode == 'Water':
    radius_0, radius_1 = 15, 15
elif mode == 'Street':
    radius_0, radius_1 = 10, 20

next_points_df = intersections_df.copy(deep=True)
next_points_df['Used'] = True

def no_next_points(p, distance_df, covers_df, r0, r1):
    r = r0
    if not covers_df.loc[p, 'Covers']:
        r = r1
    np_i = distance_df[distance_df[p]<r]
    np_i = len(np_i)
    #next_points_df.loc[p, 'Number of NPs'] = np_i
    return np_i#next_points_df

next_points_df['point_i'] = next_points_df.index
i = 1
while i:
    next_points_df = next_points_df[next_points_df['Used']==True]
    
    if len(next_points_df) == 0:
        break
    
    distance_df = f_distance(next_points_df)  
    next_points_df['Number of NPs'] = next_points_df['point_i'].apply(lambda x: no_next_points(x, distance_df, covers_df, radius_0, radius_1))
    #for point_i in tqdm(next_points_df.index):

    max_nps = max(next_points_df['Number of NPs'].values)
    print(max_nps)
    if max_nps == 1:
        for point_j in next_points_df.index:
           intersections_df.loc[point_j, 'Used'] = True 
           intersections_df.loc[point_j, 'Replaced by'] = point_j
        i = 0
    else:
        next_points_df = next_points_df.sort_values(by='Number of NPs', ascending=False)
        for j in range(len(next_points_df)):
            most_nps = next_points_df.index[j]
            if covers_df.loc[most_nps, 'Covers']:
                break
            if covers_df.loc[most_nps, 'Min Distance'] > radius_1:
                print(covers_df.loc[most_nps, 'Min Distance'])
                break
            else:
                most_nps = next_points_df.index[0]
        radius = radius_0
        if not covers_df.loc[most_nps, 'Covers']:
            radius = radius_1
        np_i = distance_df[distance_df[most_nps]<radius]
        replace = np_i.index.to_list()
        next_points_df.loc[replace, 'Used'] = False
        intersections_df.loc[replace, 'Used'] = False
        intersections_df.loc[most_nps, 'Used'] = True
        intersections_df.loc[replace, 'Replaced by'] = most_nps
 

intersections_df = intersections_df.reset_index()
intersections_df['point_ij_index'] = intersections_df['point_ij']
intersections_df = intersections_df.set_index('point_ij_index')
intersections_gdf = intersections_df.set_geometry('point_ij')
intersections_gdf['point_ij'] = intersections_gdf['point_ij'].drop_duplicates()

simple_grid_df = pd.DataFrame(columns=['Point i', 'Point j', 'Distance_ij', 'Link_ij'])
simple_grid_df['Link_ij'] = simple_grid_df['Link_ij'].drop_duplicates()

############################
# Generate the new grid 🔥#
############################

tol = 1#m
Points = intersections_gdf.index.to_list()#[intersections_gdf['Used']== True]


print("Create the new grid")
for line_l in tqdm(streets_nrw_gdf['geometry']):

    for point_i in Points:#possible_points:
        next_pos_j, next_neg_j = None, None
        distance_il = point_i.distance(line_l) / 1e3
        if distance_il > tol:
            continue
        
        #possible_points = [p for p in Points if p != point_i and point_i.distance(p) < (1.1*radius_1 * 1e3)]
        
        min_pos_distance_ij = 9e9
        min_neg_distance_ij = -9e9
        for point_j in Points:
            #if point_i == point_j:
            #    continue
            distance_jl = point_j.distance(line_l) / 1e3
            if distance_jl > tol:
                continue

            # From hereon both points are on the line
            if intersections_gdf.loc[point_i, 'Used'] == False:
                point_i_used = intersections_gdf.loc[point_i, 'Replaced by']
            else:
                point_i_used = point_i

            if intersections_gdf.loc[point_j, 'Used'] == False:
                point_j_used = intersections_gdf.loc[point_j, 'Replaced by']
            else:
                point_j_used = point_j
                
            if point_i_used == point_j_used:
                continue

            distance_i = point_i.distance(point_i_used) /1e3
            distance_j = point_j.distance(point_j_used) /1e3            
            
            d1 = line_l.project(point_i)
            d2 = line_l.project(point_j)

            distance_ij = (d1 - d2) / 1e3
            if distance_ij > 0:
                distance_ij += distance_i
                distance_ij += distance_j
                if distance_ij < min_pos_distance_ij:
                    min_pos_distance_ij = distance_ij
                    next_pos_j = point_j_used
            else:
                distance_ij -= distance_i
                distance_ij -= distance_j
                if distance_ij > min_neg_distance_ij:
                    min_neg_distance_ij = distance_ij
                    next_neg_j = point_j_used

        ####
        if min_pos_distance_ij != 9e9 and next_pos_j is not None:
            simple_grid_df = add_link(simple_grid_df, point_i_used, next_pos_j, min_pos_distance_ij)
        if min_neg_distance_ij != -9e9 and next_neg_j is not None:
            simple_grid_df = add_link(simple_grid_df, point_i_used, next_neg_j, abs(min_neg_distance_ij))
          
path = 'C:\\Landwehr\\GIT\\Data\\simple_grid_'
path += mode
path += '.pkl'
with open(path, "wb") as f:
    pickle.dump(simple_grid_df, f)
    
with open(path, "rb") as f:
    simple_grid_df = pickle.load(f)

simple_grid_df['Link_ij_index'] = simple_grid_df['Link_ij']
simple_grid_df = simple_grid_df.set_index('Link_ij_index')
simple_grid_df = simple_grid_df[simple_grid_df['Distance_ij']>0]

##################################
# Connect loose ends to the grid #
##################################
print("Simplify grid")

simple_grid_df['Link_ij_index'] = simple_grid_df['Link_ij']
simple_grid_df = simple_grid_df.set_index('Link_ij_index')   

################################
# Check for unneccesary breaks #
################################

################################
# Process remaining loose ends #
################################

###############################
#  #
###############################

if mode == 'Water':
    for i in range(2):
        simple_grid_df = erase_corners(simple_grid_df)
        simple_grid_df = simple_grid_df[simple_grid_df['Distance_ij']!=0]
    simple_grid_df = delete_loose_ends(simple_grid_df, 25)
    simple_grid_df = simple_grid_df[simple_grid_df['Distance_ij']!=0]
    
    for i in range(2):
        simple_grid_df = erase_corners(simple_grid_df)
        simple_grid_df = simple_grid_df[simple_grid_df['Distance_ij']!=0]
    
elif mode == 'Street':
    for i in range(2):
        simple_grid_df = erase_corners(simple_grid_df)
        simple_grid_df = simple_grid_df[simple_grid_df['Distance_ij']!=0]
        
    simple_grid_df = connect_loose_ends(simple_grid_df, 'point', 25)
    simple_grid_df = simple_grid_df[simple_grid_df['Distance_ij']!=0]
    
    for i in range(2):
        simple_grid_df = erase_corners(simple_grid_df)
        simple_grid_df = simple_grid_df[simple_grid_df['Distance_ij']!=0]
    
    simple_grid_df = delete_loose_ends(simple_grid_df, 20)
    simple_grid_df = simple_grid_df[simple_grid_df['Distance_ij']!=0]
   
simple_grid_gdf = simple_grid_df.set_geometry('Link_ij')

with open(path, "wb") as f:
    pickle.dump(simple_grid_gdf, f)

###############################
#  Plot the results           #
###############################

with open("C:\\Landwehr\\GIT\\Data\\simple_grid_Street.pkl", "rb") as f:
    simple_grid_street = pickle.load(f)

ax_i = axes[0,1] 
ax_i.set_facecolor((0.8, 0.9, 1))
nuts.plot(ax=ax_i, facecolor='darkgrey', edgecolor='lightgrey', linewidth=0.5)
region.plot(ax=ax_i, facecolor='white', edgecolor='lightgrey', linewidth=0.5)
simple_grid_street.plot(ax=ax_i, facecolor='none', edgecolor='C4', linewidth=1)
ax_i.set_xlim(155000, 925000)
ax_i.set_ylim(5225000, 6100000)

with open("C:\\Landwehr\\GIT\\Data\\simple_grid_Water.pkl", "rb") as f:
    simple_grid_water = pickle.load(f)

ax_i = axes[1,1] 
ax_i.set_facecolor((0.8, 0.9, 1))
nuts.plot(ax=ax_i, facecolor='darkgrey', edgecolor='lightgrey', linewidth=0.5)
region.plot(ax=ax_i, facecolor='white', edgecolor='lightgrey', linewidth=0.5)
simple_grid_water.plot(ax=ax_i, facecolor='none', edgecolor='C0', linewidth=1)
ax_i.set_xlim(155000, 925000)
ax_i.set_ylim(5225000, 6100000)

for ax in axes.flat:
    ax.label_outer()
    
plt.tight_layout()
plt.show()