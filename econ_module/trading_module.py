### This is the trading_module intended to build a small optimization framework that combines datasets of prices and demand/supply over different time steps (years or higher resolution) to build a global bilateral trading optimization module.
#%%

import pandas as pd
import time
import os
import sys
import xarray as xr
import numpy as np
import linopy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
import math
import seaborn as sns

### This part is to guarantee execution in normal and in debug mode to cleanly call scripts from neighbouring directories
def _find_project_root(start_dir, required_siblings=("econ_module", "relationship_module", "general_module")):
    candidate = os.path.abspath(start_dir)
    for _ in range(6):
        if all(os.path.isdir(os.path.join(candidate, s)) for s in required_siblings):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    raise RuntimeError(f"Could not locate IGC_model project root (expected {required_siblings}) starting from {start_dir}")

try:
    _start_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _start_dir = os.getcwd()

project_root = _find_project_root(_start_dir)
this_dir = os.path.join(project_root, "econ_module")

for sibling in ("relationship_module", "general_module"):
    sibling_dir = os.path.join(project_root, sibling)
    if sibling_dir not in sys.path:
        sys.path.insert(0, sibling_dir)

from relationship_module import build_relationship_base, calculate_relationship_factor
from model_io import load_model_run, save_model_run, save_complete_model, load_complete_model
from model_settings import get_settings
#%%

START = time.perf_counter()
print("Execute in Directory:")
print(os.getcwd() + "\n")

script_dir = this_dir
data_path = os.path.join(script_dir, "data")
output_path = os.path.join(script_dir, "output")

### Define central parameter values ###
case_study = get_settings(parameter="case_study")
transport_costs_param = get_settings(parameter="transport_costs")
base_step_param = get_settings(parameter="base_step")
reference_region = "EU-DEU"

#%%

# Load the data
prices_df = pd.read_excel(os.path.join(data_path, case_study, "region_tables.xlsx"), sheet_name='prices')
demands_df = pd.read_excel(os.path.join(data_path, case_study, "region_tables.xlsx"), sheet_name='demands')
supply_df = pd.read_excel(os.path.join(data_path, case_study, "region_tables.xlsx"), sheet_name='supply')

def calculate_sensitivity_differences(df):
    # Initialize a list to store the results
    results = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Get the region, commodity, and scenario
        region = row['region']
        commodity = row['commodity']
        scenario = row['scenario']

        # Get all the sensitivity columns
        sensitivity_cols = [col for col in df.columns if col.startswith('ds_')]

        # Sort the sensitivity columns based on the numerical value in the column name
        # and taking into account that 'm' is negative and 'p' is positive
        sensitivity_cols.sort(key=lambda x: -int(x.split('_')[-1]) if x.split('_')[-2] == 'm' else int(x.split('_')[-1]))

        # Get the values from the sensitivity columns
        values = [row[col] for col in sensitivity_cols]

        # Calculate the differences between the sensitivities
        differences = []
        for i in range(1, len(values)):
            difference = values[i] - values[i-1]
            differences.append(difference)

        # Store the results
        results.append({
            'region': region,
            'commodity': commodity,
            'scenario': scenario,
            sensitivity_cols[0]: values[0],  # Retain the original column name for the initial value
            **{f'{col}': diff for col, diff in zip(sensitivity_cols[1:], differences)}
        })

    return pd.DataFrame(results)

supply_diff_df = calculate_sensitivity_differences(supply_df)

# Reshape the data from wide to long format
def reshape_data(df, value_name):
    df_long = df.melt(id_vars=['region', 'commodity', 'scenario'],
                      var_name='supply_step',
                      value_name=value_name)
    return df_long

prices_long = reshape_data(prices_df, 'price')
demands_long = reshape_data(demands_df, 'demand')
supply_long = reshape_data(supply_df, 'supply')
supply_diff_df_long = reshape_data(supply_diff_df, "supply_diff")

# Merge the DataFrames and store nodal information in data_1D_df
data_1D_df = pd.merge(prices_long, demands_long, on=['region', 'commodity', 'scenario', 'supply_step'])
data_1D_df = pd.merge(data_1D_df, supply_long, on=['region', 'commodity', 'scenario', 'supply_step'])
data_1D_df = pd.merge(data_1D_df, supply_diff_df_long, on=['region', 'commodity', 'scenario', 'supply_step'])

# Define a custom order for the supply_step column based on demand scenario
def get_step_order(supply_step_str):
    if supply_step_str.startswith('ds_m_'):
        # For negative demand sensitivity, extract the percentage and convert to negative
        percentage = int(supply_step_str.split('_')[2])
        return -percentage
    elif supply_step_str.startswith('ds_p_'):
        # For positive demand sensitivity, extract the percentage
        percentage = int(supply_step_str.split('_')[2])
        return percentage
    else:
        # For other cases, return a large number to place them at the end
        return float('inf')


# Create a new column with the custom order
data_1D_df['supply_step_order'] = data_1D_df['supply_step'].apply(get_step_order)

# Sort the DataFrame by the custom order
data_1D_df = data_1D_df.sort_values('supply_step_order')

# Drop the temporary column used for sorting
data_1D_df = data_1D_df.drop('supply_step_order', axis=1)

# Reset the index if needed
data_1D_df = data_1D_df.reset_index(drop=True)

def load_relationship_factors(data_path, case_study, target_max_multiplier=1.5):
    try:
        from relationship_module import build_relationship_base, calculate_relationship_factor
        base = build_relationship_base(case_study=case_study)
        relationship_df = calculate_relationship_factor(base, target_max_multiplier=target_max_multiplier)
        used_static_fallback = False

    except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
        import warnings
        warnings.warn(f"relationship_module unreachable ({type(e).__name__}: {e}) -- using static fallback.")
        relationship_df = pd.read_excel(
            os.path.join(data_path, case_study, "relationship_transport_data.xlsx"),
            sheet_name="relationship",
        )
        used_static_fallback = True

    return relationship_df, used_static_fallback

# Read relationship factors
base = build_relationship_base()
relationship_df, used_static_fallback = load_relationship_factors(data_path, case_study, target_max_multiplier=1.5)
if used_static_fallback == True: print("Used static fallback for relationships. Check if dynamic calculation is required")

transport_costs = pd.read_excel(os.path.join(data_path, case_study, "relationship_transport_data.xlsx"), sheet_name='transport_costs')
# transport_costs_long = transport_costs.melt(id_vars=['region1', "region2", 'commodity', 'scenario'],
#                     var_name='supply_step')
transport_efficiencies = pd.read_excel(os.path.join(data_path, case_study, "relationship_transport_data.xlsx"), sheet_name='transport_efficiencies')

data_2D_df = pd.merge(transport_costs, relationship_df, on=['region1', 'region2', 'scenario'], how="left")
data_2D_df = pd.merge(data_2D_df, transport_efficiencies,  on=['region1', 'region2', "commodity", 'scenario'], how="left")

#%%

# data_2D_df_reci = data_2D_df.copy()
# data_2D_df_reci = data_2D_df_reci.rename(columns={"region1":"region2", "region2":"region1"})
# pd.concat([data_2D_df, data_2D_df_reci], ignore_index=True)

print(data_1D_df.head())
print(data_2D_df.head())

# -----------------------------
# Prepare nodal datasets from dataframe to xarray
# -----------------------------

# Identify the dimension columns
dimension_columns = [col for col in data_1D_df.columns if col in ["region", "commodity", "scenario", "supply_step"]]

# Identify value columns
value_columns = [col for col in data_1D_df.columns if col not in dimension_columns]

# Initialize a dictionary to hold the lists for each dimension
dimension_lists = {}
# Create separate DataArrays for each value column
data_arrays = {}

# Create lists of unique values for each dimension column
for column in dimension_columns:
    dimension_lists[column] = data_1D_df[column].unique().tolist()

for value_column in value_columns:
    data_arrays[value_column] = xr.DataArray(
        np.full(tuple(len(lst) for lst in dimension_lists.values()), np.nan),
        dims=dimension_columns,
        coords=dimension_lists
    )

# Fill the DataArrays with values from the DataFrame
for _, row in data_1D_df.iterrows():
    for value_column in value_columns:
        data_arrays[value_column].loc[tuple(row[dim] for dim in dimension_columns)] = row[value_column]


# Combine the DataArrays into a single Dataset
data_1D = xr.Dataset(data_arrays)

# -----------------------------
# Create transport pairs
# -----------------------------

# Identify the unique regions:
regions = data_1D_df["region"].unique()

# Create a DataFrame with all possible pairs of the unique regions
pairs = pd.MultiIndex.from_product(
    [regions, regions],
    names=["from_region", "to_region"]
)
# Convert the pairs MultiIndex to a list of tuples
pairs_list = list(pairs)

# Filter pair_df to include only rows where (region1, region2) is in pairs_list
data_2D_df = data_2D_df[data_2D_df.apply(lambda row: (row["region1"], row["region2"]) in pairs_list, axis=1)]

# Create a copy of the original DataFrame for reciprocal data
data_2D_df_reci = data_2D_df.copy()

# Swap region1 and region2 for reciprocal data
data_2D_df_reci["region1"], data_2D_df_reci["region2"] = data_2D_df_reci["region2"], data_2D_df_reci["region1"].copy()

# Concatenate the original and reciprocal DataFrames
data_2D_df_combined = pd.concat([data_2D_df, data_2D_df_reci], ignore_index=True)

# Identify the dimension columns
dimension_columns_2D = [col for col in data_2D_df_combined.columns if col in ["region1", "region2", "commodity", "scenario", "supply_step"]]

# Identify value columns
value_columns_2D = [col for col in data_2D_df_combined.columns if col not in dimension_columns_2D]

# Initialize a dictionary to hold the lists for each dimension
dimension_lists_2D = {}
# Create a dictionary to hold the DataArrays for each value column
data_arrays_2D = {}

# Create lists of unique values for each dimension column
for column in dimension_columns_2D:
    dimension_lists_2D[column] = data_2D_df_combined[column].unique().tolist()

# Create DataArrays for each value column
for value_column_2D in value_columns_2D:
    data_arrays_2D[value_column_2D] = xr.DataArray(
        np.full(tuple(len(lst) for lst in dimension_lists_2D.values()), np.nan),
        dims=dimension_columns_2D,
        coords=dimension_lists_2D
    )

# Fill the DataArrays with values from the combined DataFrame
for _, row in data_2D_df_combined.iterrows():
    for value_column_2D in value_columns_2D:
        data_arrays_2D[value_column_2D].loc[tuple(row[dim] for dim in dimension_columns_2D)] = row[value_column_2D]

# Combine the DataArrays into a single Dataset
data_2D_combined = xr.Dataset(data_arrays_2D)

# Remove self-transport (optional)
data_2D_combined = data_2D_combined.where(data_2D_combined.region1 != data_2D_combined.region2, drop=True)

### Collect Dimensions ###
regions = data_1D.region.values
commodities = data_1D.commodity.values
scenarios = data_1D.scenario.values
supply_steps = data_1D.supply_step.values

### Create Region Pairs ###
all_pairs = pd.MultiIndex.from_product([regions, regions], names=["region1", "region2"])
all_pairs = pd.DataFrame(index=all_pairs).reset_index()
all_pairs = all_pairs[all_pairs.region1 != all_pairs.region2]

extra_dims = pd.MultiIndex.from_product([commodities, scenarios], names=["commodity","scenario"]) # supply_steps "supply_step"
extra_dims = pd.DataFrame(index=extra_dims).reset_index()

### Cartesian Expansion ###
all_pairs["key"] = 1
extra_dims["key"] = 1
full_pairs = all_pairs.merge(extra_dims, on="key").drop("key", axis=1)

#%%

### Merge Transport Data ###
pair_df_tmp = data_2D_df_combined.copy()
data_full = full_pairs.merge(pair_df_tmp, on=["region1", "region2", "commodity", "scenario"], how="left") # "supply_step"

### Convert to xarray ###
data_2D = data_full.set_index(["region1", "region2", "commodity", "scenario"]).to_xarray() #"supply_step"
data_2D = data_2D.reindex(region1=regions, region2=regions)

#%%

### Expand Nodal Data ###
data_1D_expanded = data_1D.expand_dims(region1=regions, region2=regions)
data_1D = data_1D.fillna(0)

# Reindex data_2D to match the supply_step coordinates of data_1D_expanded
data_2D = data_2D.reindex(supply_step=data_1D_expanded.supply_step.values)

#%%

# Now merge the datasets
data_new = xr.merge([data_1D_expanded, data_2D], join="exact")
data = data_new.fillna(0)

### Remove NaNs ###
data_1D = data_1D.fillna(0)
data_2D = data_2D.fillna({"transport_cost": 0, "counter": 0, "shared_weight":0, "alliance_index":0, "gamma":0, "vom_multiplier": 0, "transport_efficiency":0})

for v in data_2D.data_vars:
    n_missing = np.isnan(data_2D[v]).sum()
    if n_missing > 0: print(f"{v}: {n_missing} NaNs")

for v in data_1D.data_vars:
    n_missing = np.isnan(data_1D[v]).sum()
    if n_missing > 0: print(f"{v}: {n_missing} NaNs")

print("NaN check complete")
print("\nCoordinate check:")
print(data_new.region1.values)
print(data_new.region2.values)

# -----------------------------
# Create price/supply segments
# -----------------------------

segment_capacity = data_1D["supply_diff"]
segment_price = data_1D["price"]

# segment_coords = data_1D.coords
# complete_region_coords = {"region":regions,"commodity":commodities, "scenario":scenarios}
transport_coords = {"region1":regions, "region2":regions, "commodity":commodities, "scenario":scenarios, "supply_step":supply_steps}
demand_xr = (data_1D.sel(supply_step=base_step_param, drop=True))

#%%

def build_and_run_opt_model(data_1D, data_2D, demand_xr, segment_price, max_total_dependence_rel = 1, max_indiv_dependence_rel = 1, rfm = 1):
    print(f"Determine relationship factor for rfm {rfm}")
    rel_df, used_static_fallback = load_relationship_factors(data_path, case_study, target_max_multiplier=rfm)
    rel_df = rel_df[["region1", "region2", "scenario", "vom_multiplier"]]

    if not (rel_df["region1"] < rel_df["region2"]).all():
        raise ValueError("rel_df is not in canonical (region1 < region2) order -- mirroring below would be wrong.")

    mirrored = rel_df.rename(columns={"region1": "region2", "region2": "region1"})
    regions = data_2D.region1.values
    diagonal = pd.DataFrame({
        "region1": regions, "region2": regions,
        "scenario": rel_df["scenario"].iloc[0], "vom_multiplier": 1.0,
    })
    full_rel_df = pd.concat([rel_df, mirrored, diagonal], ignore_index=True)

    vom_da = (
        full_rel_df.set_index(["region1", "region2", "scenario"])["vom_multiplier"]
        .to_xarray()
        .reindex(region1=data_2D.region1, region2=data_2D.region2, scenario=data_2D.scenario)
    )

    missing = int(vom_da.isnull().sum())
    if missing:
        raise ValueError(f"{missing} region pair(s) missing from rel_df after reindexing -- check region code mismatches.")

    data_2D["vom_multiplier"] = vom_da.broadcast_like(data_2D["vom_multiplier"])

    print(f"rfm={rfm}: vom_multiplier mean={data_2D['vom_multiplier'].mean().item():.3f}, max={data_2D['vom_multiplier'].max().item():.3f}")

    # -----------------------------
    # Build the optimization model
    # -----------------------------

    ### Initialize Model ###
    model = linopy.Model()
    #identify number of individual regions
    n = len(data_1D.region.values)

    ### Variables ###
    v_supply_segment = model.add_variables(
        lower=0,
        upper=data_1D["supply_diff"],
        coords=data_1D.coords,
        dims=data_1D.dims, # region, commodity, scenario, supply_step
        name="v_supply_segment"
    )

    v_transport = model.add_variables(
    lower=0,
    coords=[
        data_2D.coords["region1"],
        data_2D.coords["region2"],
        data_2D.coords["commodity"],
        data_2D.coords["scenario"],
        data_1D.coords["supply_step"],
    ],
    dims=["region1", "region2", "commodity", "scenario", "supply_step"],
    name="v_transport"
    )

    # v_transport = model.add_variables(
    #     lower=0,
    #     coords=data_2D.coords,
    #     dims=data_2D.dims, # region1, region2, commodity, scenario
    #     name="v_transport"
    # )

    v_unmet = model.add_variables(
        lower=0,
        coords=demand_xr.coords,
        dims=demand_xr.dims, # region, commodity, scenario
        name="v_unmet"
    )

    ### Flow Accounting ###
    transport_efficiency = data_2D["transport_efficiency"]  # dims: region1, region2, commodity, scenario

    # apply route-specific efficiency to each flow *before* aggregating
    v_transport_delivered = v_transport * transport_efficiency  # dims: region1, region2, commodity, scenario, supply_step

    # imports into region i (raw, pre-loss):
    inflow = (v_transport.sum("region1").rename(region2="region"))
    # exports from region i:
    outflow = (v_transport.sum("region2").rename(region1="region"))

    exports_by_step=(
        v_transport
        .sum("region2")
        .rename(region1="region")
    )

    ### Regional Accounting ###
    regional_production = (v_supply_segment.sum(dim="supply_step"))

    # delivered imports, accounting for route-specific transport losses
    regional_import_total = (
        v_transport_delivered
        .sum("region1")
        .rename(region2="region")
        .sum(dim="supply_step")
    )   
    # regional_import_total = ((inflow * transport_efficiency).sum(dim="supply_step"))
    regional_import_indiv = (
    v_transport_delivered
    .sum(dim="supply_step")
    .rename(region2="region")
    )
    # regional_import_indiv = (v_transport.sum("supply_step").rename(region2="region"))

    # -----------------------------
    # Implement derivative conversion efficiencies
    # -----------------------------

    ### Conversion efficiencies for hydrogen and derivatives ###

    ### retain total consumption constraint ###

    ### Constraints ###
    # max_total_dependence_rel = get_settings(case_study_arg=case_study, parameter="max_total_dependence_rel")
    print("Maximum share of total imports: " + str(int(max_total_dependence_rel*100)) + " %")
    # max_indiv_dependence_rel = get_settings(case_study_arg=case_study, parameter="max_indiv_dependence_rel")
    print("Maximum share of individual imports: " + str(int(max_indiv_dependence_rel*100)) + " %")

    # Ensure that the total import share does not exceed 50%
    c_dependence_total = model.add_constraints(
        (regional_import_total / demand_xr["demand"] <=max_total_dependence_rel),
        name="c_dependence_total"
    )

    # Ensure that the import share from any single region does not exceed 10%
    c_dependence_indiv = model.add_constraints(
        (regional_import_indiv / demand_xr["demand"] <= max_indiv_dependence_rel),
        name="c_dependence_indiv"
    )

    c_export_link = model.add_constraints(
        exports_by_step
        <=
        v_supply_segment,
        name="c_export_link"
    )

    c_balance = model.add_constraints(
        (v_supply_segment
        - outflow
        + v_transport_delivered.sum("region1").rename(region2="region")).sum(dim="supply_step")
        + v_unmet
        == demand_xr["demand"],
        name="c_balance")
    
    ### Objective ###
    production_costs = (v_supply_segment * segment_price
                        ).sum() #dim= explizit definieren manchmal praktisch für Lösung

    effective_multiplier = 1 + rfm * (data_2D["vom_multiplier"] - 1)
    print(effective_multiplier)
    transport_costs = (v_transport * data_2D["transport_cost"] * effective_multiplier).sum()

    penalty_costs = (v_unmet * 100000).sum()

    ### Objective Function ###
    obj_fun = model.add_objective(
        production_costs + transport_costs + penalty_costs, sense="min")

    ### Solve ###
    model.solve(solver_name="gurobi")
    solution = model.solution

    if solution is not None:
        print("Solution found")
        print(model.objective)
        print(solution["v_supply_segment"])
        print(solution["v_transport"])
        #define model run name
        name = f"model_run_{n}n_{case_study}_{max_total_dependence_rel*100:.0f}_{max_indiv_dependence_rel*100:.0f}_rfm{rfm*100:.0f}"
        #save solution to file
        save_model_run(output_path, data_1D, data_2D, solution, name,
                    meta={"solver": "gurobi",
                            "case_study": case_study,
                            "max_total_dependence_rel": max_total_dependence_rel,
                            "max_indiv_dependence_rel": max_indiv_dependence_rel,
                            "relationship_factor_magnitude": rfm,
                            "n":n
                            })
        save_complete_model(model, output_path, name)
    else:
        print("No solution available")
    return model, solution

# model, solution = build_and_run_opt_model(data_1D, data_2D, demand_xr, segment_price, 0.75, 0.2, 1)

# total_dep_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
# indiv_dep_list = [0, 0.2, 0.4, 0.6, 0.8, 1]

# # Execute sensitivity analysis for dependency parameters
# for t_d in total_dep_list:
#     for i_d in indiv_dep_list:
#         print("Execute optimization for dependency parameters: " + str(t_d) + "_" + str(i_d))
#         model, solution = build_and_run_opt_model(data_1D, data_2D, demand_xr, segment_price, t_d, i_d)
#         print("Optimization successfull")

relationship_factor_magnitude = [1, 1.2, 1.5, 1.8, 2.0]

# Execute sensitivity analysis for dependency parameters
for rfm in relationship_factor_magnitude:
    print("Execute optimization for relationship magnitude: " + str(rfm))
    model, solution = build_and_run_opt_model(data_1D, data_2D, demand_xr, segment_price, 0.75, 0.2, rfm)
    print("Optimization successfull")

# ==========================
# Extract shadow prices
# ==========================

marginals = -model.constraints["c_balance"].dual.copy()

# flip sign only if needed
if marginals.mean() < 0:
    marginals = -marginals

print(marginals)

#%%
STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')
#%%