import pypsa
from shapely.ops import triangulate, nearest_points
from shapely import Point, MultiPoint, wkt, LineString
import pandas as pd
import pickle
from tqdm import tqdm
import data
terminals_gdf = data.terminals_gdf
storages_gdf = data.storages_gdf
ports_gdf = data.ports_gdf
nuts1 = data.nuts1
scenario = data.scenario_name
import utils
import numpy as np

    
ppln_code = '_P'
trck_code = '_T'
wtr_code = '_W'
demand = True

nrw = pypsa.Network()
nrw.investment_periods = data.years
nrw.set_snapshots(data.snapshots)
nrw.snapshot_weightings = pd.Series(1,index=nrw.snapshots)
base = data.years[0]
nrw.investment_period_weightings = pd.Series({y: 1 / (1 + data.r) ** (y - base) for y in data.years})
eps = 1e-3
nrw.add("Carrier", data.carrier)

region_nrw = data.nrw_nuts_gdf
region = nuts1.copy(deep=True)
region['Keep'] = region['NUTS_ID'].apply(lambda x: utils.filter_nuts(region, x))
region = region[region['Keep']==True]
region = region.drop(columns=['Keep', 'LEVL_CODE', 'CNTR_CODE', 'NAME_LATN', 'NUTS_NAME', 'MOUNT_TYPE', 'URBN_TYPE', 'COAST_TYPE'])
region = region.set_index('NUTS_ID')
region['location'] = region['geometry'].centroid

def network(nrw):
    
    with open("C:\\Landwehr\\GIT\\Data\\simple_grid_Street.pkl", "rb") as f:
        network_street = pickle.load(f)
        network_street = network_street.rename(columns={"Point i": "From","Point j": "To", "Distance_ij": "Distance", "Link_ij": "line"})

    with open("C:\\Landwehr\\GIT\\Data\\simple_grid_Water.pkl", "rb") as f:
        network_water = pickle.load(f)
        network_water = network_water.rename(columns={"Point i": "From","Point j": "To", "Distance_ij": "Distance", "Link_ij": "line"})  


    def move_source(network, loc_source, geo):
        new_loc = loc_source
        min_distance = 9e9
        if not geo.covers(loc_source).any():
            for index in network.index:
                line_i = network.loc[index, 'line']
                start_i = Point(line_i.coords[0])
                end_i = Point(line_i.coords[0])
                distance_start_source = start_i.distance(loc_source)
                distance_end_source = end_i.distance(loc_source)
                if distance_start_source < min_distance and distance_start_source < 25:
                    min_distance = distance_start_source
                    new_loc = start_i
                if distance_end_source < min_distance and distance_end_source < 25:
                    min_distance = distance_end_source
                    new_loc = end_i
        else:
            for index in network.index:
                line_i = network.loc[index, 'line']
                start_i = Point(line_i.coords[0])
                end_i = Point(line_i.coords[0])
                distance_start_source = start_i.distance(loc_source)
                distance_end_source = end_i.distance(loc_source)
                if distance_start_source < min_distance and distance_start_source < 5:
                    min_distance = distance_start_source
                    new_loc = start_i
                if distance_end_source < min_distance and distance_end_source < 5:
                    min_distance = distance_end_source
                    new_loc = end_i
        return new_loc
    
    geo = region_nrw['geometry']
    emissions_gdf = data.emissions_loc_gdf
    emissions_gdf['new_location'] = emissions_gdf['location'].apply(lambda x: move_source(network_street, x, geo))
    emissions_gdf['location'] = emissions_gdf['new_location']
    emissions_gdf = emissions_gdf.drop(columns=['new_location'])
    
    demand_gdf = data.demand_gdf
    demand_gdf['new_location'] = demand_gdf['location'].apply(lambda x: move_source(network_street, x, geo))
    demand_gdf['location'] = demand_gdf['new_location']
    demand_gdf = demand_gdf.drop(columns=['new_location'])
    
    def format_network(df, network_type):
        Nodes = pd.DataFrame(df['From'].tolist() + df['To'].tolist())
        Nodes = Nodes.drop_duplicates()
        Nodes['New_name'] = None
        Nodes['index'] = Nodes[0]
        Nodes = Nodes.set_index('index')
        Nodes = Nodes.drop(columns=[0])
        for i in range(len(Nodes)):
            node = Nodes.index[i]
            new_name = format(i, "02X") + '-' + network_type
            #new_name = format(i, "02X") + '|co2'
            Nodes.loc[node, 'New_name'] = new_name
        df['From'] = df['From'].apply(lambda x: Nodes.loc[x,'New_name'])
        df['To'] = df['To'].apply(lambda x: Nodes.loc[x,'New_name'])
        return df
     
    def add(gdf, names, new_connection):
        start_name, end_name = names
        new = len(gdf)
        gdf.loc[new, 'line'] = new_connection
        gdf.loc[new, 'From'] = start_name
        gdf.loc[new, 'To'] = end_name
        gdf.loc[new, 'Distance'] = new_connection.length / 1e3
        gdf.loc[new, 'Split'] = False
        return gdf

    def next_link(p, network):
        min_distance_pl = 9e9
        for l in network.index:
            if network.loc[l, 'Number'] != None:
                continue
            if 'Source' in network.loc[l, 'From']:
                continue
            if 'Source' in network.loc[l, 'To']:
                continue
            if 'Terminal' in network.loc[l, 'From']:
                continue
            if 'Terminal' in network.loc[l, 'To']:
                continue
            link = network.loc[l, 'line']
            p, p_l = nearest_points(p, link)
            distance_pl = p.distance(p_l) / 1e3  # km   
            if distance_pl < min_distance_pl:
                min_distance_pl = distance_pl
                next_link = link
                next_point = p_l
        return next_link, next_point
    
    def split_line(network, next_line, next_point):
        
        relevant_idx = network.index[network['line'] == next_line]    
        new_rows = []  # neue Zeilen sammeln
        
        for l in relevant_idx:
            row = network.loc[l]
            line_l = row['line']
            
            length_l = line_l.length / 1e3
            from_l, to_l = line_l.coords[0], line_l.coords[1]
            from_l, to_l = Point(from_l), Point(to_l)
            
            name_from_l = row['From']
            name_to_l = row['To']
            
            new_line_0 = LineString([from_l, next_point])
            length_0 = new_line_0.length / 1e3
            frac_0 = length_0 / length_l
            
            if frac_0 != 0 and frac_0 != 1:
                
                network_i = network[(network['From'] == name_from_l) &(network['To'] == name_to_l)]
                
                new_rows.append({'line': new_line_0,'Keep': True,
                    'From': name_from_l,'To': name_to_l,
                    'Number': len(network_i) - 1,'Distance': row['Distance'] * frac_0})
                
                network.loc[l, 'Keep'] = False
    
        if new_rows:
            network = pd.concat([network, pd.DataFrame(new_rows)], ignore_index=True)
        
        return network
    
    
    def include_new_points(network, gdf):
        network['Number'] = None
        for p in tqdm(gdf.index):  
            results = gdf['location'].apply(lambda x: next_link(x, network))
            gdf[['next_line', 'next_point']] = list(results)
            
            network = network.reset_index()
            network = network.drop(columns=['index'])  
            
            next_line = gdf.loc[p, 'next_line']
            next_point = gdf.loc[p, 'next_point']
            network = split_line(network, next_line, next_point)
        return network
       
    
    def update_labels(network):
        for from_i in network['From'].drop_duplicates().values:
            network_i = network[network['From']==from_i]
            for to_i in network_i['To'].drop_duplicates().values:
                network_ij = network_i[network_i['To']==to_i]
                network_ij = network_ij.sort_values(by='Distance', ascending=True)           
                count = -1
                stop = len(network_ij)-1
                distance_t0 = 0
                end_0 = None
                for index in network_ij.index:
                    distance_t1 = network_ij.loc[index, 'Distance']
                    line_ij = network_ij.loc[index, 'line']
                    end_1 = Point(line_ij.coords[1])               
                    
                    if end_0 == end_1:
                        stop -= 1
                        network.loc[index, 'Keep'] = False
                        continue
                
                    if count >= 0 and count < stop:
                        line_ij_new = LineString([end_0,end_1])
                        network.loc[index, 'line'] = line_ij_new
                        new_name = f"{from_i}({count}){to_i}"
                        network.loc[index, 'From'] = new_name
                    count += 1
                    if count < stop:
                        new_name = f"{from_i}({count}){to_i}"
                        network.loc[index, 'To'] = new_name
                    
                    network.loc[index, 'Distance'] = distance_t1 - distance_t0              
                    network.loc[index, 'Keep'] = True
                    
                    distance_t0 = distance_t1
                    end_0 = end_1
        return network
          
    def next_point(p, network):
        min_distance_lp = 9e9
        for l in network.index:
            line_l = network.loc[l, 'line']
            start_name = network.loc[l, 'From']
            end_name = network.loc[l, 'To']
            if 'Source' in start_name or 'Source' in end_name:
                continue
            if 'Terminal' in start_name or 'Terminal' in end_name:
                continue
            start_l = line_l.coords[0]
            end_l = line_l.coords[1]
            start_l, end_l = Point(start_l), Point(end_l)
            distance_lp_start = start_l.distance(p) / 1e3
            if distance_lp_start < min_distance_lp:
                min_distance_lp = distance_lp_start
                name_np = start_name
                location_np = start_l                
            distance_lp_end = end_l.distance(p) / 1e3
            if distance_lp_end < min_distance_lp:
                min_distance_lp = distance_lp_end
                name_np = end_name
                location_np = end_l               
        return location_np, name_np
    
    def points2grid(network, gdf):   
        results = gdf['location'].apply(lambda x: next_point(x, network))
        gdf[['location_NP', 'name_NP']] = list(results)        
        network = network.reset_index()
        network = network.drop(columns=['index'])       
        for storage_i in tqdm(gdf.index):
            end_name_p = gdf.loc[storage_i, 'name_NP']
            start_name_p = gdf.loc[storage_i, 'name']           
            start_p = gdf.loc[storage_i, 'location']
            end_p = gdf.loc[storage_i, 'location_NP']         
            new = len(network)+1
            network.loc[new, 'From'] = start_name_p
            network.loc[new, 'To'] = end_name_p           
            new_connection = LineString([start_p, end_p])           
            network.loc[new, 'line'] = new_connection
            network.loc[new, 'Distance'] = new_connection.length / 1e3
            network.loc[new, 'Split'] = False  
            
        return network
    
    ports_gdf['name'] = ports_gdf['binnenhafen'] 
    terminals_gdf['name'] = terminals_gdf.index.tolist()   
    demand_gdf['name'] = demand_gdf.index.tolist()   
    network_street['Split'] = False
    
    'Street'
    network_street = format_network(network_street, 'S')
    network_street = network_street.reset_index()
    network_street = network_street.drop(columns=['Link_ij_index'])  
    #Sources
    network_street = include_new_points(network_street, emissions_gdf) 
    #Demand
    network_street = include_new_points(network_street, demand_gdf) 
    #Ports
    network_street = include_new_points(network_street, ports_gdf)  
    #Terminals
    network_street = include_new_points(network_street, terminals_gdf)  
    network_street = update_labels(network_street)      
    network_street = network_street[network_street['Keep'] == True]
    
    #Source
    network_street = points2grid(network_street, emissions_gdf)
    #Demand
    network_street = points2grid(network_street, demand_gdf)
    #Port
    network_street = points2grid(network_street, ports_gdf)
    #Terminal
    network_street = points2grid(network_street, terminals_gdf)
    network_street = network_street.reset_index()
    network_street = network_street.drop(columns=['index'])
       
    'Water'
    network_water = format_network(network_water, 'W')
    network_water = network_water.reset_index()
    network_water = network_water.drop(columns=['Link_ij_index']) 
    #Ports  
    network_water = include_new_points(network_water, ports_gdf) 
    #Terminals 
    network_water = include_new_points(network_water, terminals_gdf) 
    network_water = update_labels(network_water)      
    network_water = network_water[network_water['Keep'] == True]
    #Port
    network_water = points2grid(network_water, ports_gdf)
    #Terminal
    network_water = points2grid(network_water, terminals_gdf)
    
    ###############################################################################
    ###                       Add sequestration site                            ###
    ###############################################################################
    
    
    for terminal_t in tqdm(terminals_gdf.index):
        for storage_s in storages_gdf.index:
            start_p = terminals_gdf.loc[terminal_t, 'location']
            end_p = storages_gdf.loc[storage_s, 'location']
            
            start_name_p = terminal_t
            end_name_p = storage_s
            
            #street
            new = len(network_street)+1
            network_street.loc[new, 'From'] = start_name_p
            network_street.loc[new, 'To'] = end_name_p           
            new_connection = LineString([start_p, end_p])           
            network_street.loc[new, 'line'] = new_connection
            network_street.loc[new, 'Distance'] = new_connection.length / 1e3
            network_street.loc[new, 'Split'] = False 
            
            #water
            new = len(network_water)+1
            network_water.loc[new, 'From'] = start_name_p
            network_water.loc[new, 'To'] = end_name_p            
            new_connection = LineString([start_p, end_p])            
            network_water.loc[new, 'line'] = new_connection
            network_water.loc[new, 'Distance'] = new_connection.length / 1e3
            network_water.loc[new, 'Split'] = False 
    
    with open(f'C:\Landwehr\GIT\Data\\network_street_{scenario}.pkl', "wb") as f:
        pickle.dump(network_street, f)
    with open(f'C:\Landwehr\GIT\Data\\network_water_{scenario}.pkl', "wb") as f:
        pickle.dump(network_water, f)  
    
    with open(f'C:\Landwehr\GIT\Data\\network_street_{scenario}.pkl', "rb") as f:
        network_street = pickle.load(f)
    with open(f'C:\Landwehr\GIT\Data\\network_water_{scenario}.pkl', "rb") as f:
        network_water = pickle.load(f)
    
    ###############################################################################
    ###                                                                         ###
    ###                     Start of network construction                       ###
    ###                                                                         ###
    ###############################################################################
    print("Starting the network construction..")
    ###############################################################################
    ###                     Add busses to network                               ###
    ###############################################################################
    
    list_buses_street = network_street['From'].tolist() + network_street['To'].tolist()
    list_buses_water = network_water['From'].tolist() + network_water['To'].tolist()
    list_buses = list_buses_street + list_buses_water
    list_buses = list(dict.fromkeys(list_buses)) #Delete duplicates
    
    print("Add busses to network")
    for bus_i in tqdm(list_buses):
        for grid in data.Grids2:
            if 'Terminal' in bus_i:
                pass
            if 'Source' in bus_i and grid == wtr_code:
                continue
            if 'SC-' in bus_i and grid == wtr_code:
                continue
            elif 'Resevoir' in bus_i and grid == trck_code:
                continue
            elif '-S' in bus_i and 'STA' not in bus_i and grid == wtr_code:
                #Skip Water if Street
                continue
            elif '-W' in bus_i and '-WIL' not in bus_i and (grid == trck_code or grid == ppln_code) and 'Worringen' not in bus_i and 'Wesseling' not in bus_i:
                #print("!!") 
                continue
            #elif 'SC_' in bus_i and grid == wtr_code:
            #    #print("!!")
            #    continue
            name_i = bus_i
            if grid != '':
                #name_i += '-'
                name_i += grid
            if name_i not in nrw.buses.index:
                nrw.add('Bus', name_i, carrier=data.carrier)
                    
    nrw.add('Bus', 'Aurora-Resevoir', carrier=data.carrier)
    
    ###############################################################################
    ###                       Add sources to network                            ###
    ###############################################################################   
     
    print("Add sources of emission to network")
    emissions_loc_gdf = emissions_gdf[emissions_gdf['ktco2_2023']>0]
    
    for i in tqdm(emissions_loc_gdf.index):
        bus_i = emissions_loc_gdf.loc[i, 'name']
        name_i = 'co2|'
        name_i += emissions_loc_gdf.loc[i, 'name'] #+ '-CO2'
        name_i += '|'
        name_i += bus_i
        p_nom_i = emissions_loc_gdf.loc[i, 'ktco2_2023']   
        
        #Normal
        if '24.1' in name_i:
            nrw.add('Generator', name_i, bus=bus_i, p_nom = p_nom_i, p_min_pu=0.1, marginal_cost=eps)
        else:
            nrw.add('Generator', name_i, bus=bus_i, p_nom = p_nom_i, p_min_pu=1, marginal_cost=eps)
        
    for generator_i in tqdm(nrw.generators.index):
        for s in nrw.snapshots:
            nrw.generators_t.marginal_cost.loc[s, generator_i] = data.co2_price[s]
     
    ###############################################################################
    ###                       Add storages to network                           ###
    ###############################################################################
    
    print("Add storages to network")
    
    for i in tqdm(emissions_loc_gdf.index):
        bus_i = emissions_loc_gdf.loc[i, 'name']
        name_i = 'co2|'
        name_i += emissions_loc_gdf.loc[i, 'name']# + '-CO2_atmos'
        name_i += '|Atmos'
        p_nom_i = emissions_loc_gdf.loc[i, 'ktco2_2023'] 
        capture_cost = emissions_loc_gdf[emissions_loc_gdf['name']==bus_i]
        capture_cost = capture_cost['CCS Cost']
        capture_cost = capture_cost.iloc[0]
        nrw.add('StorageUnit', name_i, bus=bus_i, p_nom_extendable=True, p_max_pu=0, capital_cost=eps, marginal_cost=eps)
    
    nrw.add('StorageUnit', 'CO2|Sequestration|AuroraResevoir|', bus='Aurora-Resevoir',  p_nom_extendable=True, p_max_pu=0, capital_cost=eps, marginal_cost=eps)
     
    
    ###############################################################################
    ###                       Demand                                            ###
    ###############################################################################
    
    for location in demand_gdf.index:
        nrw.add("Load",f"Load_{location}",bus=location,p_set=eps)       
    
    ###############################################################################
    ###                       Add links to network                              ###
    ###############################################################################
    
    print("Add links to network") 
    
    def connections(network, list_connections):
        for i in tqdm(network.index):
                from_i = network.loc[i, 'From']
                to_i = network.loc[i, 'To']
                name_ij = from_i  + '_' + to_i
                list_connections.append(name_ij)
        return list_connections
    
    list_connections = []  
    list_connections = connections(network_street, list_connections)
    list_connections = connections(network_water, list_connections)
    
    #Add the link to the model
    def add_link(network, bus0, bus1, name, network_type, distance, bidirectional=False):
            if network_type == trck_code:
                marginal_cost = data.opex_truck_var * distance
                capital_cost = data.capex_truck_cap * distance
            elif network_type == wtr_code:
                marginal_cost = data.opex_ship_var * distance
                capital_cost = data.capex_ship_cap * distance
            elif network_type == ppln_code:
                marginal_cost = data.opex_pipeline_var * distance
                capital_cost = data.capex_pipeline_cap #Not distance dependent
            elif network_type == '_P_off':
                marginal_cost = data.opex_pipeline_off_var * distance
                capital_cost = data.capex_pipeline_off_cap
            elif network_type == 'hub':
                if trck_code in bus1:
                    marginal_cost = 0#data.opex_truck_fix
                    capital_cost = data.opex_truck_fix
                elif wtr_code in bus1:
                    marginal_cost = 0
                    capital_cost = 0
                elif ppln_code in bus1:
                    marginal_cost = 0
                    capital_cost = 0
                elif trck_code in bus0 or wtr_code in bus0 or ppln_code in bus0:
                    marginal_cost = 0
                    capital_cost = 0
                else:
                    raise Error
            else:
                raise Error
            
            #p_min_pu = -1 if bidirectional else 0
            
            network.add('Link', name, bus0=bus0, bus1=bus1, p_nom_extendable=True, p_nom_max=data.sum_emissions, 
                    p_min_pu=0, capital_cost=capital_cost, distance=distance, marginal_cost=marginal_cost)
            
            return network
        
    #Lookup if connection between two points exist
    def new_connection(network, name_i, name_j, model):
        connection_ij = network[network['From'] == name_i]
        connection_ij = connection_ij[connection_ij['To'] == name_j]
        
        connection_ji = network[network['From'] == name_j]
        connection_ji = connection_ji[connection_ji['To'] == name_i]
        
        connection_ji = connection_ji.drop_duplicates()
        connection_ij = connection_ij.drop_duplicates()
        
        if len(connection_ij) == 0 and len(connection_ji) == 0:
            pass
        else:
            if len(connection_ij) == 1:
                distance = connection_ij['Distance'].tolist()[0]
            elif len(connection_ji) == 1:
                distance = connection_ji['Distance'].tolist()[0]
            else:
                raise Error
                
            if type_i != "":   
                name_ij = bus_i + '_' + bus_j
                name_ji = bus_j + '_' + bus_i 
                if name_ij not in nrw.links.index and name_ji not in nrw.links.index:
                    if 'Aurora' in name_ij and type_i == ppln_code and distance > 0: #Offshore Pipeline
                        model = add_link(model, bus_i, bus_j, name_ij, '_P_off', distance, bidirectional=False)
                        model = add_link(model, bus_j, bus_i, name_ji, '_P_off', distance, bidirectional=False)
                    elif distance > 0: #Onshore Pipeline
                        model = add_link(model, bus_i, bus_j, name_ij, type_i, distance, bidirectional=False)
                        model = add_link(model, bus_j, bus_i, name_ji, type_i, distance, bidirectional=False)
                    #elif (type_i == wtr_code or type_i == trck_code) and distance > 0:
                    #    model = add_link(model, bus_i, bus_j, name_ij, type_i, distance, bidirectional=False)
                    #    model = add_link(model, bus_j, bus_i, name_ji, type_i, distance, bidirectional=False)
        
        return model
    
    #Add the connections between the different locations
    #Street, Water and Pipeline Network
    for bus_i in tqdm(nrw.buses.index):
        name_i = bus_i.split("_")
        
        if len(name_i) > 1: #Bus is not a hub
            name_i, type_i = name_i[0], name_i[1]
            type_i = '_' + type_i
        else: #Bus is a hub
            continue
            
        for bus_j in nrw.buses.index:
            name_j = bus_j.split("_")
            if len(name_j) > 1:
                name_j, type_j = name_j[0], name_j[1]
                type_j = '_' + type_j
            else:
                continue
              
            name_ij = name_i  + '_' + name_j
            if name_i == name_j or type_i != type_j or name_ij not in list_connections:
                continue  
            
            #Street
            nrw = new_connection(network_street, name_i, name_j, nrw)            
            #Water
            nrw = new_connection(network_water, name_i, name_j, nrw)          
    
    #Connect the buses to the grid
    for bus_i in tqdm(nrw.buses.index):
        name_i = bus_i.split("_")
        if len(name_i) == 1:
            for grid in data.Grids:
                bus_j = name_i[0] + grid
                if bus_j not in nrw.buses.index:
                    continue
                name_ij = bus_i + '_' + bus_j
                name_ji = bus_j + '_' + bus_i
                if 'Source' in bus_j and '_P' in bus_j:
                    nrw = add_link(nrw, bus_i, bus_j, name_ij, 'hub', 0, bidirectional=False)
                elif 'SC-' in bus_i:
                    nrw = add_link(nrw, bus_j, bus_i, name_ji, 'hub', 0, bidirectional=False) 
                else:
                    nrw = add_link(nrw, bus_i, bus_j, name_ij, 'hub', 0, bidirectional=False)
                    nrw = add_link(nrw, bus_j, bus_i, name_ji, 'hub', 0, bidirectional=False)                 
    
    if demand:
        for location in demand_gdf.index:
            for s, year in zip(nrw.snapshots, data.years):
                loc = 'Load_' + location
                demand_ij = demand_gdf.loc[location, year]
                nrw.loads_t.p_set.loc[s, loc] = demand_ij
            
    ###############################################################################
    ###                       Modification of costs                             ###
    ###############################################################################  
    
    #Modify the marginal costs            
    for link_ij in tqdm(nrw.links.index):
        bus0 = nrw.links.loc[link_ij, 'bus0']
        bus1 = nrw.links.loc[link_ij, 'bus1']
        distance_ij = nrw.links.loc[link_ij, 'distance']
        if ('Source' in bus0) and not (ppln_code in bus0 or trck_code in bus0) and (ppln_code in bus1 or trck_code in bus1) and distance_ij == 0:
            capture_cost = emissions_loc_gdf[emissions_loc_gdf['name']==bus0]
            capture_cost = capture_cost['CCS Cost']
            capture_cost = capture_cost.iloc[0]
            for s in nrw.snapshots:
                 #nrw.links_t.marginal_cost.loc[s, link_ij] = -data.co2_price[s]
                 nrw.links_t.marginal_cost.loc[s, link_ij] = capture_cost
                 #nrw.links_t.marginal_cost.loc[s, link_ij] += 0 #transfer cost
        elif 'Source' in bus1 and not (ppln_code in bus1 or trck_code in bus1) and (ppln_code in bus0 or trck_code in bus0) and distance_ij == 0:
            capture_cost = emissions_loc_gdf[emissions_loc_gdf['name']==bus1]
            capture_cost = capture_cost['CCS Cost']
            capture_cost = capture_cost.iloc[0]
            for s in nrw.snapshots:
                 #nrw.links_t.marginal_cost.loc[s, link_ij] = data.co2_price[s]
                 nrw.links_t.marginal_cost.loc[s, link_ij] = -capture_cost
                 #nrw.links_t.marginal_cost.loc[s, link_ij] += 0 #transfer cost
        else:
            pass
    
    #Modify the storage costs                
    for link_ij in tqdm(nrw.links.index):
        bus1 = nrw.links.loc[link_ij, 'bus1']
        distance_ij = nrw.links.loc[link_ij, 'distance']
        if 'Resevoir' in bus1 and distance_ij == 0:
            for s in nrw.snapshots:
                nrw.links_t.marginal_cost.loc[s, link_ij] = data.storage_costs[s]
        else:
            pass
    
    if demand:
        nrw.export_to_netcdf(f"Results/nrw_raw_{scenario}_d.nc")   
    else:
        nrw.export_to_netcdf(f"Results/nrw_raw_{scenario}_nd.nc") 
    
    return list_connections

if __name__ == '__main__':
    a_connections = network(nrw)
    
if demand:
    nrw = pypsa.Network(f"Results/nrw_raw_{scenario}_d.nc")
else:
    nrw = pypsa.Network(f"Results/nrw_raw_{scenario}_nd.nc")
###############################################################################
###                       Error detection                                   ###
###############################################################################