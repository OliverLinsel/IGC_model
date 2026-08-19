import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pypsa
import network
import data
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
emissions_gdf = data.emissions_loc_gdf
emissions_gdf = emissions_gdf.set_crs(data.crs)
terminals = data.terminals_gdf
scenario = data.scenario_name

nrw_nuts_gdf = data.nrw_nuts_gdf
region = network.region
nuts = data.nuts1
nuts = nuts[nuts['LEVL_CODE']==1]
#import network_elements
#storages_gdf = network_elements.storages_gdf

plt.rcParams.update({"text.usetex": True,"font.family": "serif"})
Grids = data.Grids

nrw = pypsa.Network(f"Results/nrw_optimized_{scenario}.nc")

with open('C:\Landwehr\GIT\Data\\network_street.pkl', "rb") as f:
    street_network_gdf = pickle.load(f)
#street_network_gdf = network.network_street
street_network_plot_gdf = street_network_gdf.reset_index()
street_network_plot_gdf = street_network_plot_gdf.set_index('line')
street_network_plot_gdf['Geometry'] = street_network_plot_gdf.index.tolist()
street_network_plot_gdf = street_network_plot_gdf.set_geometry("Geometry")
street_network_plot_gdf = street_network_plot_gdf.set_crs(data.crs)

with open('C:\Landwehr\GIT\Data\\network_water.pkl', "rb") as f:
    water_network_gdf = pickle.load(f)
#water_network_gdf = network.network_water
water_network_plot_gdf = water_network_gdf.reset_index()
water_network_plot_gdf = water_network_plot_gdf.set_index('line')
water_network_plot_gdf['Geometry'] = water_network_plot_gdf.index.tolist()
water_network_plot_gdf = water_network_plot_gdf.set_geometry("Geometry")
water_network_plot_gdf = water_network_plot_gdf.set_crs(data.crs)

idx = nrw.snapshots
emissions = nrw.storage_units_t.p.transpose()
max_emission = 0
raise Error
for s in nrw.snapshots:
    nrw_nuts_gdf[s[0]] = 0.0
    
def find_nuts(region, p):
    for nuts_i in region.index:
        region_i = region.loc[nuts_i, 'geometry']
        distance = p.distance(region_i)
        if distance == 0:
            return nuts_i
    return None
    
for source_i in emissions.index:
    if 'Source' in source_i:
        #_, no, _ = source_i.split("-")
        _, no, _ = source_i.split("|")
        _, no = no.split("-")
        name_i = 'Source-' + str(no)
        emissions_gdf_i = emissions_gdf[emissions_gdf['name']==name_i]
        point_i = emissions_gdf_i['location'].iloc[0]
        nuts_id_i = find_nuts(region, point_i)
        if nuts_id_i == None:
            continue
        if 'DEA' not in nuts_id_i: #EMission outside of NRW
            continue
        emissions = emissions.transpose()
        for s in nrw.snapshots:
            emission_i_s = -emissions.loc[s, source_i]
            emission_i_s = np.log(1+emission_i_s)
            if emission_i_s > max_emission:
                print(emission_i_s)
                max_emission = emission_i_s
            nrw_nuts_gdf.loc[nuts_id_i, s[0]] += emission_i_s
        emissions = emissions.transpose()
nrw_nuts_gdf = nrw_nuts_gdf.fillna(0)

max_flow = max(nrw.links_t.p.values[0])
f = 0.5
factor = 2 / (max_flow**f)

def add_flow(df, s, key, flow):
    try:
        flow += abs(df.loc[s, key])
        #flow = int(flow)
        #flow = max(flow,1)
    except KeyError:
        flow += 0
    return flow

p_df = nrw.links
p_t_df = nrw.links_t.p
p_t_df = p_t_df.transpose()  
p_t_df = p_t_df.loc[(p_t_df != 0).any(axis=1)]
p_t_df = p_t_df.astype(float)

def calculate_flows(network, s):
   
    network['Flow_P'] = 0
    network['Flow_T'] = 0
    network['Flow_W'] = 0
    
    for link in tqdm(p_t_df.index):
        
        added = False
        
        link_split = link.split('_')
        if len(link_split) != 4 or link_split[1] != link_split[3] :
            #print("SKIP")
            #print(link)
            continue
        
        bus0 = link_split[0] 
        bus1 = link_split[2]
        
        network_i = network[network['From']==bus0]
        network_i = network_i[network_i['To']==bus1]
        #print(f"{bus0}  und  {bus1}")
        #print(f"{link_split[1]}")
        if len(network_i) > 0:
            link_i = network_i.index
            flow = p_t_df.loc[link, s]
            flow = int(flow)
            name = 'Flow_'
            name += link_split[1] 
            network.loc[link_i, name] += flow
            added = True
            
        network_j = network[network['To']==bus0]
        network_j = network_j[network_j['From']==bus1]
        
        if len(network_j) > 0:
            link_i = network_j.index
            flow = p_t_df.loc[link, s]
            flow = int(flow)
            name = 'Flow_'
            name += link_split[1] 
            network.loc[link_i, name] += flow
            added = True
            
        if not added:
            print("Error: ", link)
            #raise Error
        else:
            print("Correct: ", link)
              
    network_flow_pipeline = (network["Flow_P"]**f) * factor
    network_flow_truck    = (network["Flow_T"]**f) * factor
    network_flow_water    = (network["Flow_W"]**f) * factor
    
    return network_flow_pipeline, network_flow_truck, network_flow_water

for s in nrw.snapshots:
    emissions_s = emissions[s].tolist()
    emissions_s = emissions_s[:-1]
    emissions_gdf[s] = emissions_s
    emissions_gdf[s] = emissions_gdf[s].apply(lambda x: -x)
    emissions_gdf[s] = emissions_gdf[s].apply(lambda x: max(x,0))
   
    
###############################################################################
###                       Plotting of the results                           ###
###############################################################################     
   
fig, axes = plt.subplots(2, int(len(data.years)/2), figsize=[12,10.5], dpi=500)
fig.suptitle(r"NRW $CO_{2}$ grid", size=30)

custom_lines = [Line2D([0], [0], color="C4", lw=4),
                Line2D([0], [0], color="C0", lw=4),
                Line2D([0], [0], color="yellowgreen", lw=4)]

years = data.years#[2030,2035,2040,2045]
   
network = pd.concat([street_network_plot_gdf, water_network_plot_gdf])

#Germany
count = 0
for year_i in years:
    s = nrw.snapshots[count] 
    if s[0] not in years:
        continue
    #network_flow_pipeline, network_flow_truck, network_flow_water = calculate_flows(street_network_plot_gdf, water_network_plot_gdf, s)
    network_flow_pipeline, network_flow_truck, network_flow_water = calculate_flows(network, s)
    #raise Error
    row = int(count // 3)
    col = int(count % 3)

    ax_i = axes[row, col] 
    ax_i.set_facecolor((0.8, 0.9, 1))
    
    nuts.plot(ax=ax_i, facecolor='darkgrey', edgecolor='lightgrey', linewidth=0.5)
    region.plot(ax=ax_i, facecolor='white', edgecolor='lightgrey', linewidth=0.5) #network.
    nrw_nuts_gdf.plot(ax=ax_i, facecolor='white', edgecolor='black', linewidth=0.5)
    nrw_nuts_gdf.plot(s[0], cmap='Oranges', vmin=0, vmax=max_emission, ax=ax_i)

    network.plot(linewidth=network_flow_pipeline, color="yellowgreen", ax=ax_i)
    network.plot(linewidth=network_flow_truck, color="C4", ax=ax_i)
    network.plot(linewidth=network_flow_water, color="C0", ax=ax_i)
    terminals.plot(color='C1', marker = 's', ax=ax_i, zorder=2)

    #emissions_loc_gdf.plot(color='black',alpha=0.7,markersize=emissions_loc_gdf['tco2_2023'] / 2e4,ax=ax_i)
    emissions_gdf.plot(color='none', edgecolor='grey', linewidth=0.5,markersize= emissions_gdf['tco2_2023']/2e4,ax=ax_i)
    emissions_gdf.plot(color='grey',alpha=0.7,markersize= (emissions_gdf[s]/2e4),ax=ax_i)

    #storages_gdf.plot(ax=ax_i, color="C5", alpha=0.75, linewidth=3)

    ax_i.set_title(f"{year_i}", size=25)
    
    ax_i.set_xlim(155000, 925000)
    ax_i.set_ylim(5225000, 6100000)
    #ax_i.set_xlim(4, 9)
    #ax_i.set_ylim(50, 54)
    ax_i.set_xlabel('Longitude', size=20)
    ax_i.set_ylabel('Latitude', size=20)
    
    ax_i.text(0.95, 0.05, rf"$c_{{CO_2}} = {data.co2_price[s]:.0f}$ €/t",  
    transform=ax_i.transAxes, ha='right',va='bottom',fontsize=15)
    
    if row == 0 and col == 2:
        ax_i.legend(custom_lines, ['Truck', 'Ship', 'Pipeline'], loc='upper right', fontsize=15)
    
    count += 1

plt.tight_layout()

fig.canvas.draw()

ax_i = axes[1, 2]

renderer = fig.canvas.get_renderer()
bbox = ax_i.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())

fig.savefig(f"C:\Landwehr\IGC.NRW_PyPsa\Results\{scenario}_GER.png", bbox_inches=bbox)

for ax in axes.flat:
    ax.label_outer()
    
plt.tight_layout()
plt.show()

#NRW
print("- - - - - ")
fig, axes = plt.subplots(2, int(len(data.years)/2), figsize=[12,9.5], dpi=500)
fig.suptitle(r"NRW $CO_{2}$ grid", size=30)

custom_lines = [Line2D([0], [0], color="C4", lw=4),
                Line2D([0], [0], color="C0", lw=4),
                Line2D([0], [0], color="yellowgreen", lw=4)]
count = 0
for year_i in years:
    s = nrw.snapshots[count] 
    if s[0] not in years:
        continue
    #network_flow_pipeline, network_flow_truck, network_flow_water = calculate_flows(street_network_plot_gdf, water_network_plot_gdf, s)
    network_flow_pipeline, network_flow_truck, network_flow_water = calculate_flows(network, s)
    #raise Error
    row = int(count // 3)
    col = int(count % 3)

    ax_i = axes[row, col] 
    ax_i.set_facecolor((0.8, 0.9, 1))
    
    nuts.plot(ax=ax_i, facecolor='darkgrey', edgecolor='lightgrey', linewidth=0.5)
    region.plot(ax=ax_i, facecolor='white', edgecolor='lightgrey', linewidth=0.5) #network.
    nrw_nuts_gdf.plot(ax=ax_i, facecolor='white', edgecolor='black', linewidth=0.5)
    nrw_nuts_gdf.plot(s[0], cmap='Oranges', vmin=0, vmax=max_emission, ax=ax_i)

    network.plot(linewidth=network_flow_pipeline, color="yellowgreen", ax=ax_i)
    network.plot(linewidth=network_flow_truck, color="C4", ax=ax_i)
    network.plot(linewidth=network_flow_water, color="C0", ax=ax_i)
    terminals.plot(color='C1', marker = 's', ax=ax_i, zorder=2)

    #emissions_loc_gdf.plot(color='black',alpha=0.7,markersize=emissions_loc_gdf['tco2_2023'] / 2e4,ax=ax_i)
    emissions_gdf.plot(color='none', edgecolor='grey', linewidth=0.5,markersize= emissions_gdf['tco2_2023']/2e4,ax=ax_i)
    emissions_gdf.plot(color='grey',alpha=0.7,markersize= (emissions_gdf[s]/2e4),ax=ax_i)

    #storages_gdf.plot(ax=ax_i, color="C5", alpha=0.75, linewidth=3)

    ax_i.set_title(f"{year_i}", size=25)
    
    ax_i.set_xlim(270000, 540000)
    ax_i.set_ylim(5570000, 5830000)
    #ax_i.set_xlim(4, 9)
    #ax_i.set_ylim(50, 54)
    ax_i.set_xlabel('Longitude', size=20)
    ax_i.set_ylabel('Latitude', size=20)
    
    ax_i.text(0.95, 0.05, rf"$c_{{CO_2}} = {data.co2_price[s]:.0f}$ €/t",  
    transform=ax_i.transAxes, ha='right',va='bottom',fontsize=15)
    
    
    if row == 0 and col == 2:
        ax_i.legend(custom_lines, ['Truck', 'Ship', 'Pipeline'], loc='upper right', fontsize=15)
    
    count += 1
    
plt.tight_layout()

fig.canvas.draw()

ax_i = axes[1, 2]

renderer = fig.canvas.get_renderer()
bbox = ax_i.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())

fig.savefig(f"C:\Landwehr\IGC.NRW_PyPsa\Results\{scenario}_NRW.png", bbox_inches=bbox)

for ax in axes.flat:
    ax.label_outer()
    
plt.tight_layout()
plt.show()


#####
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor((0.8, 0.9, 1))

nuts.plot(ax=ax, facecolor='darkgrey', edgecolor='lightgrey', linewidth=0.5)
#nrw_nuts_gdf.plot(ax=ax, facecolor='white', edgecolor='black', linewidth=0.5)
region.plot(ax=ax, facecolor='white', edgecolor='lightgrey', linewidth=0.5)

emissions_gdf.plot(
    ax=ax,
    color='grey',
    edgecolor='grey',
    linewidth=0.5,
    markersize=emissions_gdf['tco2_2023'] / 2e4)

# Titel & Achsen
ax.set_title(f"Sources of Emission ({scenario})", size=12)

ax.set_xlim(155000, 925000)
ax.set_ylim(5225000, 6100000)
ax.set_xlabel('Longitude', size=10)
ax.set_ylabel('Latitude', size=10)

# Text unten rechts
ax.text(
    0.95, 0.05,
    rf"$c_{{CO_2}} = {data.co2_price[s]:.0f}$ €/t",
    transform=ax.transAxes,
    ha='right',
    va='bottom',
    fontsize=15)

plt.tight_layout()
plt.show()