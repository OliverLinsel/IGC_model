import gurobipy as gp
import os
import pypsa
import numpy as np

import data
#import network
#from network import nrw
import constraints as nrw_constr

os.environ['GRB_LICENSE_FILE'] = r"C:/Users/roman/gurobi.lic"

demand = False
scenario = data.scenario_name

if demand:
    nrw = pypsa.Network(f"Results/nrw_raw_{scenario}_d.nc")
else:
    nrw = pypsa.Network(f"Results/nrw_raw_{scenario}_nd.nc")
    
def main():
    
            
    nrw.optimize(
    solver_name="gurobi",
    multi_investment_periods=True,
    solver_options={'MIPGap': 0.005, 'MIPFocus': 3, 'Heuristics': 0.2, 'Cuts': 2,              
                    'Presolve': 2, 'ImproveStartTime': 600, },
    extra_functionality=nrw_constr.constraints)
    #solver_options={'MIPGap': 0.01,},
    #extra_functionality=nrw_constr.constraints)       

    path = 'Results/nrw_optimized_'
    path += scenario
    if demand:
        path += '_d'
    else:
         path += '_nd'
    path += '.nc'
    nrw.export_to_netcdf(path)


if __name__ == '__main__':
    main()