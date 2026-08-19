import pypsa
import pandas as pd

nrw = pypsa.Network("Results/nrw_optimized.nc")

links_df = nrw.links
links_df = links_df[links_df['p_nom_opt']>0]
bus_0 = links_df['bus0']
bus_1 = links_df['bus1']
buses = pd.concat([bus_0, bus_1])
buses = buses.drop_duplicates()
Keep = []
for bus_i in buses:
    splitted = bus_i.split('_')
    keep_i = splitted[0]
    Keep.append(keep_i)
    
keep_df = pd.DataFrame(Keep)
keep_df = keep_df.drop_duplicates()
try:
    Keep = keep_df[0].tolist()
except KeyError:
    links_df = nrw.links
    links_df = links_df[links_df['p_nom_opt']==0]
    bus_0 = links_df['bus0']
    bus_1 = links_df['bus1']
    buses = pd.concat([bus_0, bus_1])
    buses = buses.drop_duplicates()
    Keep = buses.tolist()