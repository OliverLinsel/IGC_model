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
import matplotlib.pyplot as plt, pandas as pd, seaborn as sns

START = time.perf_counter()

print('Execute in Directory:')
print(os.getcwd() + "\n")

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths
try:
    data_path = os.path.join(script_dir, "data")
    output_path = os.path.join(script_dir, "output")
except:
    data_path = os.path.join(script_dir, "data")
    output_path = os.path.join(script_dir, "output")

from model_settings import get_settings
### Define central parameter values ###
case_study = get_settings(parameter="case_study")
transport_costs_param = get_settings(parameter="transport_costs")
base_step_param = get_settings(parameter="base_step")
reference_region = "EU-DEU"

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

# Read relationship factors
relationship_df = pd.read_excel(os.path.join(data_path, case_study, "relationship_transport_data.xlsx"), sheet_name='relationship')
transport_costs = pd.read_excel(os.path.join(data_path, case_study, "relationship_transport_data.xlsx"), sheet_name='transport_costs')
transport_costs_long = transport_costs.melt(id_vars=['region1', "region2", 'commodity', 'scenario'],
                    var_name='supply_step')

data_2D_df = pd.merge(relationship_df, transport_costs_long, on=['region1', 'region2', 'scenario'])

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

extra_dims = pd.MultiIndex.from_product([commodities, scenarios, supply_steps], names=["commodity","scenario","supply_step"])
extra_dims = pd.DataFrame(index=extra_dims).reset_index()

### Cartesian Expansion ###
all_pairs["key"] = 1
extra_dims["key"] = 1
full_pairs = all_pairs.merge(extra_dims, on="key").drop("key", axis=1)

### Merge Transport Data ###
pair_df_tmp = data_2D_df_combined.copy()
data_full = full_pairs.merge(pair_df_tmp, on=["region1", "region2", "commodity", "scenario", "supply_step"], how="left")

### Fill Missing Values ###
data_full["value"] = data_full["value"].fillna(transport_costs_param)
np.random.seed(42)
data_full["value"] += np.random.uniform(0, .001, len(data_full))

### Convert to xarray ###
data_2D = data_full.set_index(["region1", "region2", "commodity", "scenario", "supply_step"]).to_xarray()
data_2D = data_2D.reindex(region1=regions, region2=regions)

### Expand Nodal Data ###
data_1D_expanded = data_1D.expand_dims(region1=regions, region2=regions)
data_1D = data_1D.fillna(0)

# Reindex data_2D to match the supply_step coordinates of data_1D_expanded
data_2D = data_2D.reindex(supply_step=data_1D_expanded.supply_step.values)

# Now merge the datasets
data_new = xr.merge([data_1D_expanded, data_2D], join="exact")
data = data_new.fillna(0)

### Remove NaNs ###
data_1D = data_1D.fillna(0)
data_2D = data_2D.fillna({"value": 0, "counter": 0, "rel_relation": 0, "vom_multiplier": 0})

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

# -----------------------------
# Build the optimization model
# -----------------------------

### Initialize Model ###
model = linopy.Model()

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
    coords=data_2D.coords,
    dims=data_2D.dims, # region1, region2, commodity, scenario
    name="v_transport"
)

v_unmet = model.add_variables(
    lower=0,
    coords=demand_xr.coords,
    dims=demand_xr.dims, # region, commodity, scenario
    name="v_unmet"
)

### Flow Accounting ###
transport_efficiency = 0.95

# imports into region i:
inflow = (v_transport.sum("region1").rename(region2="region"))
# exports from region i:
outflow = (v_transport.sum("region2").rename(region1="region"))
# acoounting the export to the supply steps
exports_by_step=(v_transport.sum("region2").rename(region1="region"))

### Regional Accounting ###
regional_production = (v_supply_segment.sum(dim="supply_step"))
regional_import_total = ((inflow * transport_efficiency).sum(dim="supply_step"))
regional_import_indiv = (v_transport.sum("supply_step").rename(region2="region"))

### Constraints ###
max_total_dependence_rel = get_settings(parameter="max_total_dependence_rel")
print("Maximum share of total imports: " + str(int(max_total_dependence_rel*100)) + " %")
max_indiv_dependence_rel = get_settings(parameter="max_indiv_dependence_rel")
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
    + inflow * transport_efficiency).sum(dim="supply_step")
    + v_unmet
    >= demand_xr["demand"],
    name="c_balance")

### Objective ###
production_costs = (v_supply_segment * segment_price
                    ).sum() #dim= explizit definieren manchmal praktisch für Lösung

transport_costs = (v_transport * data_2D["value"] * data_2D["vom_multiplier"]
                   ).sum()

penalty_costs = (v_unmet * 100000).sum()

### Objective Function ###
obj_fun = model.add_objective(
    production_costs + transport_costs + penalty_costs, sense="min")

### Solve ###
model.solve(solver_name="gurobi")
sol = model.solution

if sol is not None:
    print("Solution found")
    print(model.objective)
    print(sol["v_supply_segment"])
    print(sol["v_transport"])
else:
    print("No solution available")

# ==========================
# Extract shadow prices
# ==========================

marginals = -model.constraints["c_balance"].dual.copy()

# flip sign only if needed
if marginals.mean() < 0:
    marginals = -marginals

ds = model.solution
regions = ds.region.values
commodity = "h2"
scenario = "Base"

# Layout
n = len(regions) + 1
ncols, nrows = 3, math.ceil(n/3)
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols,6*nrows))
axes = np.array(axes).flatten()

# Colors
base_colors = {"production":"#1f77b4", "unmet":"grey", "demand":"#d62728"}
cmap = plt.get_cmap("Wistia", len(regions))
transport_colors = {r:cmap(i) for i,r in enumerate(regions)}
region_colors = {**{r:cmap(i) for i,r in enumerate(regions)}, "Total":"white"}

legend_handles = [
    Patch(color=base_colors[k], label=k) for k in base_colors.keys()
] + [Patch(color=transport_colors[r], label=f"Imports from {r}") for r in regions]

def draw_donut(ax, region, title, add_labels=True):
    production = float(ds["v_supply_segment"].sel(region=region, commodity=commodity, scenario=scenario).sum("supply_step"))
    unmet = float(ds["v_unmet"].sel(region=region, commodity=commodity, scenario=scenario))
    demand = float(demand_xr["demand"].sel(region=region, commodity=commodity, scenario=scenario))
    imports, import_colors = [], []

    for source in regions:
        if source == region: continue
        val = float(ds["v_transport"].sel(region1=source, region2=region, commodity=commodity, scenario=scenario).sum("supply_step"))
        if val > 1e-6:
            imports.append(val)
            import_colors.append(transport_colors[source])

    values = [production] + imports + [unmet, demand]
    colors = [base_colors["production"]] + import_colors + [base_colors["unmet"], base_colors["demand"]]

    ax.set_title(title, fontsize=15, bbox=dict(boxstyle="round", ec="white", fc=region_colors[title]))
    wedges, _ = ax.pie(values, radius=1, startangle=90, colors=colors, wedgeprops=dict(width=.7, edgecolor="white"))

    total = np.sum(values)
    for w, val in zip(wedges, values):
        if val/total < .05: continue
        theta = np.deg2rad((w.theta1+w.theta2)/2)
        x, y = .72*np.cos(theta), .72*np.sin(theta)
        if add_labels:
            ax.text(x, y, f"{val:.0f}", ha="center", va="center", fontsize=11)
    ax.set(aspect="equal")

# Regional plots
for i, region in enumerate(regions):
    draw_donut(axes[i], region, region)

# TOTAL plot
total_production = float(ds["v_supply_segment"].sel(commodity=commodity, scenario=scenario).sum())
total_unmet = float(ds["v_unmet"].sel(commodity=commodity, scenario=scenario).sum())
total_demand = float(demand_xr["demand"].sel(commodity=commodity, scenario=scenario).sum())
imports, import_colors = [], []

for source in regions:
    val = float(ds["v_transport"].sel(region1=source, commodity=commodity, scenario=scenario).sum())
    if val > 1e-6:
        imports.append(val)
        import_colors.append(transport_colors[source])

values = [total_production] + imports + [total_unmet, total_demand]
colors = [base_colors["production"]] + import_colors + [base_colors["unmet"], base_colors["demand"]]

ax = axes[len(regions)]
ax.set_title("Total", bbox=dict(boxstyle="round", fc="white"))
wedges, _ = ax.pie(values, radius=1, startangle=90, colors=colors, wedgeprops=dict(width=.7, edgecolor="white"))
ax.set(aspect="equal")

# Add labels to TOTAL plot
total = np.sum(values)
for w, val in zip(wedges, values):
    if val/total < .05: continue
    theta = np.deg2rad((w.theta1+w.theta2)/2)
    x, y = .72*np.cos(theta), .72*np.sin(theta)
    ax.text(x, y, f"{val:.0f}", ha="center", va="center", fontsize=11)

# Cleanup
for i in range(len(regions)+1, len(axes)):
    fig.delaxes(axes[i])

fig.legend(handles=legend_handles, loc="center right")
plt.tight_layout()
plt.subplots_adjust(right=.84)

# Save the plot as a PNG file in the figures subfolder
file_path = os.path.join(output_path, f"Production_transport_demand_balance_{case_study.replace(' ', '_')}.png")
if os.path.exists(file_path):
    os.remove(file_path)
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.ioff()

sns.set_style("whitegrid")

commodity,scenario="h2","Base"

# ----- price + demand -----

market_price=abs(
    model.constraints["c_balance"]
    .dual
    .sel(
        region=reference_region,
        commodity=commodity,
        scenario=scenario
    )
    .item()
)

demand=(
    demand_xr["demand"]
    .sel(
        region=reference_region,
        commodity=commodity,
        scenario=scenario
    )
    .item()
)

# ----- local production -----

prod=(sol["v_supply_segment"]
    .sel(region=reference_region)
    .to_dataframe("quantity")
    .reset_index()
)

prod=prod[prod.quantity>1e-6]

prod=prod.merge(
    data_1D["price"]
    .sel(region=reference_region)
    .to_dataframe("production_cost")
    .reset_index(),
    on=["commodity","scenario","supply_step"]
)

prod["source"]=reference_region
prod["transport_cost"]=0
prod["kind"]="Local"

# ----- imports -----

imp=(sol["v_transport"]
    .sel(region2=reference_region)
    .to_dataframe("quantity")
    .reset_index()
)

imp=imp[
    (imp.quantity>1e-6)
    &(imp.region1!=reference_region)
]

if len(imp):

    imp=imp.merge(
        data_1D["price"]
        .to_dataframe("production_cost")
        .reset_index()
        .rename(columns={"region":"region1"}),
        on=[
            "region1",
            "commodity",
            "scenario",
            "supply_step"
        ]
    )

    imp=imp.merge(
        data_2D["value"]
        .to_dataframe("transport_cost")
        .reset_index(),
        on=[
            "region1",
            "region2",
            "commodity",
            "scenario",
            "supply_step"
        ]
    )

    imp["source"]=imp.region1
    imp["kind"]="Import"

else:

    imp=pd.DataFrame(columns=[
        "source",
        "quantity",
        "production_cost",
        "transport_cost",
        "kind",
        "supply_step"
    ])

# ----- stack -----

cols=[
    "source",
    "quantity",
    "production_cost",
    "transport_cost",
    "kind",
    "supply_step"
]

stack=pd.concat(
    [prod[cols],imp[cols]],
    ignore_index=True
)

stack["delivered_cost"]=(
    stack.production_cost+
    stack.transport_cost
)

stack=stack.sort_values(
    "delivered_cost"
)

stack["end"]=stack.quantity.cumsum()
stack["start"]=stack.end-stack.quantity

# ----- plot -----

fig,ax=plt.subplots(figsize=(16,8))

colors={
    "Local":"tab:blue",
    "Import":"tab:orange"
}

for _,r in stack.iterrows():

    ax.bar(
        r.start,
        r.production_cost,
        width=r.quantity,
        align="edge",
        color=colors[r.kind]
    )

    if r.transport_cost>1e-6:

        ax.bar(
            r.start,
            r.transport_cost,
            width=r.quantity,
            align="edge",
            bottom=r.production_cost,
            color="red",
            alpha=.5
        )

    txt=(
        f"{r.source}"
        f"\n{r.supply_step}"
        f"\nQ={r.quantity:.1f}"
        f"\nP={r.production_cost:.1f}"
    )

    if r.transport_cost>0:
        txt += (
            f"\nT={r.transport_cost:.1f}"
        )

    txt += (
        f"\nΣ={r.delivered_cost:.1f}"
    )

    ax.text(
            r.start+r.quantity/2,
            r.delivered_cost+2,
            txt,
            fontsize=8,
            rotation=90,
            ha="center"
        )

# demand line + label

ax.axvline(
    demand,
    c="black",
    ls="--",
    lw=3
)

ax.text(
    demand,
    ax.get_ylim()[1]*0.95,
    f"Demand\n{demand:.1f}",
    rotation=90,
    va="top",
    ha="right",
    fontweight="bold"
)

# price line + label

ax.axhline(
    market_price,
    c="green",
    ls=":",
    lw=3
)

ax.text(
    stack.end.max()*0.98,
    market_price,
    f"{market_price:.1f} €/MWh",
    color="green",
    ha="right",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    bbox=dict(
        facecolor="white",
        edgecolor="green",
        alpha=.8
    )
)

# marginal supplier

mb=stack[stack.end>=demand]

if len(mb):

    mb=mb.iloc[0]

    ax.axvspan(
        mb.start,
        mb.end,
        alpha=.15,
        color="green"
    )

ax.set(
    xlabel="Quantity",
    ylabel="Delivered cost €/MWh",
    title=f"Supply composition: {reference_region}"
)

ax.legend(handles=[
    Patch(color="tab:blue",label="Local"),
    Patch(color="tab:orange",label="Import"),
    Patch(color="red",label="Transport")
])

plt.tight_layout()
# Save the plot as a PNG file in the figures subfolder
file_path = os.path.join(output_path, f"Supply_composition_{reference_region}_{case_study.replace(' ', '_')}.png")
if os.path.exists(file_path):
    os.remove(file_path)
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.ioff()
plt.show() 

#%%
STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')