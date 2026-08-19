import data
import pandas as pd

Big_M = 1.01 * data.sum_emissions
cap_start_year = 2030
cap_end_year = 2050

def constraints(nrw, snapshots):

    m = nrw.model
    p = m.variables["Link-p"]
    p_nom = m.variables["Link-p_nom"]

    periods = list(nrw.investment_periods)
    weight = nrw.investment_period_weightings["objective"]
    snapshot_period = nrw.snapshots.get_level_values("period")

    def add_period_investment(link_ij, distance, name_prefix):
        y_vars = {}
        prev = 0
        for period in periods:
            
            y_vars[period] = m.add_variables(binary=True, name=f"Link-y_{name_prefix}_{link_ij}_{period}")
            
            frac = (cap_end_year - period) / (cap_end_year - cap_start_year)
            frac = 1 - max(0, min(0.9, frac))
            Big_M_j = Big_M * frac
            
            snaps_in_period = nrw.snapshots[snapshot_period == period]
            m.add_constraints(p.loc[snaps_in_period, link_ij] <= Big_M_j * y_vars[period],
                               name=f"cap_res_{link_ij}_{period}")
            
            build_event = y_vars[period] - prev
            
            if 'Aurora' in link_ij:
                m.objective += data.capex_pipeline_off_dis * distance * weight.loc[period] * build_event
            else:
                m.objective += data.capex_pipeline_dis * distance * weight.loc[period] * build_event
                
            prev = y_vars[period]
          
        for i in range(1, len(periods)):
            m.add_constraints(y_vars[periods[i]] >= y_vars[periods[i - 1]],
                               name=f"{name_prefix}-monotonic-{link_ij}-{periods[i]}")

        m.add_constraints(p_nom.loc[link_ij] <= Big_M * y_vars[periods[-1]],
                            name=f"{name_prefix}-pnom-link-{link_ij}")       

    def add_co2_cap(nrw, m, periods, snapshot_period):
        p_gen = m.variables["Generator-p"]
        p_store = m.variables["StorageUnit-p_store"]
        atmos_storages = nrw.storage_units.index[nrw.storage_units.index.str.contains("Atmos")]

        for period in periods:
            snaps = nrw.snapshots[snapshot_period == period]
            frac = (cap_end_year - period) / (cap_end_year - cap_start_year)
            frac = max(0, min(0.9, frac))
            total_emissions = p_gen.sel(snapshot=snaps).sum()
            vented = p_store.sel(name=atmos_storages, snapshot=snaps).sum()
            m.add_constraints(vented <= frac * total_emissions, name=f"co2-cap-{period}")

##############################################################################################################

    add_co2_cap(nrw, m, periods, snapshot_period)

    for link_ij in nrw.links.index:

        bus0 = nrw.links.loc[link_ij, 'bus0']
        bus1 = nrw.links.loc[link_ij, 'bus1']
        distance = nrw.links.loc[link_ij, "distance"]
              
        if "_P" in link_ij and '_P' in bus0 and '_P' in bus1 and distance != 0:          
            add_period_investment(link_ij, distance, "build_pipeline")
        
    for bus in nrw.generators["bus"]:
        inflow_links = nrw.links.index[nrw.links.bus1 == bus]
        outflow_links = nrw.links.index[nrw.links.bus0 == bus]

        if len(outflow_links) == 0 and len(inflow_links) == 0:
            continue

        outflow = (p.sel(name=outflow_links).sum("name") if len(outflow_links) else 0)
        inflow = (p.sel(name=inflow_links).sum("name") if len(inflow_links) else 0)

        m.add_constraints(outflow >= inflow, name=f"outflow-inflow-{bus}")