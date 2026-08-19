import pypsa
import data
from tqdm import tqdm
import constraints as nrw_constr
import pandas as pd
emissions_loc_gdf = data.emissions_loc_gdf

ppln_code = '_P'
trck_code = '_T'
wtr_code = '_W'

Costs = []
for diff in range(-50, 51, 10): 
    nrw = pypsa.Network("Results/nrw_optimized.nc")
    
    #nrw.links.p_nom = nrw.links.p_nom_opt
    nrw.links.p_nom_max = nrw.links.p_nom_opt
    links_to_drop = nrw.links[nrw.links.p_nom_opt == 0].index
    nrw.remove("Link", links_to_drop)
    nrw.determine_network_topology()
    
    snapshots = nrw.snapshots                    
       
    new_co2_price = data.co2_price.copy(deep=True)
    i = 0
    for s in snapshots:
        diff_s = diff*i
        new_co2_price[s] += diff_s
        i += 1
        
    for generator_i in tqdm(nrw.generators.index):
        for s in nrw.snapshots:
            nrw.generators_t.marginal_cost.loc[s, generator_i] = new_co2_price[s]
            #nrw.generators_t.marginal_cost.loc[s, :] = data.co2_price[s]
         
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
                 nrw.links_t.marginal_cost.loc[s, link_ij] = -new_co2_price[s]
                 nrw.links_t.marginal_cost.loc[s, link_ij] += capture_cost
        elif 'Source' in bus1 and not (ppln_code in bus1 or trck_code in bus1) and (ppln_code in bus0 or trck_code in bus0) and distance_ij == 0:
            capture_cost = emissions_loc_gdf[emissions_loc_gdf['name']==bus1]
            capture_cost = capture_cost['CCS Cost']
            capture_cost = capture_cost.iloc[0]
            for s in nrw.snapshots:
                 nrw.links_t.marginal_cost.loc[s, link_ij] = new_co2_price[s]
                 nrw.links_t.marginal_cost.loc[s, link_ij] -= capture_cost
                 
    #nrw.optimize(solver_name="gurobi", multi_investment_periods=True,
    #        solver_options={'MIPGap':0.0001}, extra_functionality=nrw_constr.constraints)
    
    nrw.optimize(solver_name="gurobi")
    
    p_t_df = nrw.links_t.p
    p_t_df = p_t_df.transpose()  
    p_t_df = p_t_df.loc[(p_t_df != 0).any(axis=1)]
    
    cost = nrw.objective
    Costs.append(cost)