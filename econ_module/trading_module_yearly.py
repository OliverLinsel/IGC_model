### This is the trading_module intended to build a small optimization framework that combines datasets of prices and demand/supply over different time steps (years or higher resolution) to build a global bilateral trading optimization module.
#%%

import pandas as pd
import time
import os
import xarray as xr
import numpy as np
import linopy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
import math

case_study = "h2bb"
# case_study = "igc_nrw"
START = time.perf_counter()

print('Execute in Directory:')
print(os.getcwd() + "\n")

# Get the directory where this script is located
script_dir = os.getcwd() #os.path.dirname(os.path.abspath(__file__))

#Define the paths
try:
    data_path = os.path.join(script_dir, "econ_module", "data")
    output_path = os.path.join(script_dir, "econ_module", "output")
except:
    data_path = os.path.join(script_dir, "econ_module", "data")
    output_path = os.path.join(script_dir, "econ_module", "output")

# Load the data
prices_df = pd.read_excel(os.path.join(data_path, case_study, "region_tables_yearly.xlsx"), sheet_name='prices')
demands_df = pd.read_excel(os.path.join(data_path, case_study, "region_tables_yearly.xlsx"), sheet_name='demands')
supply_df = pd.read_excel(os.path.join(data_path, case_study, "region_tables_yearly.xlsx"), sheet_name='supply')

# Reshape the data from wide to long format
def reshape_data(df, value_name):
    df_long = df.melt(id_vars=['region', 'commodity', 'scenario'],
                      var_name='time',
                      value_name=value_name)
    return df_long

prices_long = reshape_data(prices_df, 'price')
demands_long = reshape_data(demands_df, 'demand')
supply_long = reshape_data(supply_df, 'supply')

# Merge the DataFrames and store nodal information in data_1D_df
data_1D_df = pd.merge(prices_long, demands_long, on=['region', 'commodity', 'scenario', 'time'])
data_1D_df = pd.merge(data_1D_df, supply_long, on=['region', 'commodity', 'scenario', 'time'])

# Read relationship factors
relationship_df = pd.read_excel(os.path.join(data_path, case_study, "relationship_transport_data_yearly.xlsx"), sheet_name='relationship')
transport_costs = pd.read_excel(os.path.join(data_path, case_study, "relationship_transport_data_yearly.xlsx"), sheet_name='transport_costs')
transport_costs_long = transport_costs.melt(id_vars=['region1', "region2", 'commodity', 'scenario'],
                    var_name='time')

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
dimension_columns = [col for col in data_1D_df.columns if col in ["region", "commodity", "scenario", "time"]]

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
dimension_columns_2D = [col for col in data_2D_df_combined.columns if col in ["region1", "region2", "commodity", "scenario", "time"]]

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

### Initialize Model ###
model = linopy.Model()

### Build Transport Graph ###
regions = data_1D.region.values

### Create Region Pairs ###
all_pairs = pd.MultiIndex.from_product([regions, regions], names=["region1", "region2"])
all_pairs = pd.DataFrame(index=all_pairs).reset_index()
all_pairs = all_pairs[all_pairs.region1 != all_pairs.region2]

### Collect Dimensions ###
commodities = data_1D.commodity.values
scenarios = data_1D.scenario.values
times = data_1D.time.values

extra_dims = pd.MultiIndex.from_product([commodities, scenarios, times], names=["commodity","scenario","time"])
extra_dims = pd.DataFrame(index=extra_dims).reset_index()

### Cartesian Expansion ###
all_pairs["key"] = 1
extra_dims["key"] = 1
full_pairs = all_pairs.merge(extra_dims, on="key").drop("key", axis=1)

### Merge Transport Data ###
pair_df_tmp = data_2D_df_combined.copy()
data_full = full_pairs.merge(pair_df_tmp, on=["region1", "region2", "commodity", "scenario", "time"], how="left")

### Fill Missing Values ###
DEFAULT_TRANSPORT_COST = 30
data_full["value"] = data_full["value"].fillna(DEFAULT_TRANSPORT_COST)
np.random.seed(42)
data_full["value"] += np.random.uniform(0, .001, len(data_full))

### Convert to xarray ###
data_2D = data_full.set_index(["region1", "region2", "commodity", "scenario", "time"]).to_xarray()
data_2D = data_2D.reindex(region1=regions, region2=regions)

### Expand Nodal Data ###
data_1D_expanded = data_1D.expand_dims(region1=regions, region2=regions)
data_1D = data_1D.fillna(0)
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
# Build the optimization model
# -----------------------------

### Variables ###
v_production = model.add_variables(
    lower=0, upper=data_1D["supply"], coords=data_1D.coords, dims=data_1D.dims, name="v_production")

v_transport = model.add_variables(
    lower=0, coords=data_2D.coords, dims=data_2D.dims, name="v_transport")

v_unmet = model.add_variables(
    lower=0, coords=data_1D.coords, dims=data_1D.dims, name="v_unmet")

### Flow Accounting ###
transport_efficiency = .95
inflow = v_transport.sum(dim="region1").rename(region2="region")
outflow = v_transport.sum(dim="region2").rename(region1="region")

### Diagnostics ###
net_demand = data_1D_df["supply"].sum() - data_1D_df["demand"].sum()
if net_demand < 0: print("Warning: demand exceeds supply")
else: print(f"{net_demand:.2f} excess production")

### Constraints ###
c_production = model.add_constraints(
    v_production <= data_1D["supply"], name="c_supply")

c_balance = model.add_constraints(
    v_production - outflow + inflow*transport_efficiency + v_unmet >= data_1D["demand"], name="c_balance")

### Objective ###
production_costs = (v_production * data_1D["price"]
                    ).sum()

transport_costs = (v_transport * data_2D["value"] * data_2D["vom_multiplier"]
                   ).sum()

penalty_costs = (v_unmet * 10000).sum()

### Objective Function ###
obj_fun = model.add_objective(
    production_costs + transport_costs + penalty_costs, sense="min")

### Solve ###
model.solve(solver_name="gurobi")
sol = model.solution

if sol is not None:
    print("Solution found")
    print(model.objective)
    print(sol["v_production"])
    print(sol["v_transport"])
else:
    print("No solution available")

ds = model.solution
regions = ds.region.values
times_all = ds.time.values
times = times_all[::2]

n = len(regions) + 1
ncols = 3
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols,6*nrows))
axes = np.array(axes).flatten()

ring_width = .12
base_radius = .5

base_colors = {
    "production": "#1f77b4",
    "unmet": "grey",
    "demand": "#d62728"
}

cmap = cm.get_cmap("Wistia", len(regions))
transport_colors = {region: cmap(i) for i, region in enumerate(regions)}
region_label_colors = {region: cmap(i) for i, region in enumerate(regions)}
region_label_colors["TOTAL"] = "white"

legend_handles = [
    Patch(color=base_colors["production"], label="Production"),
    Patch(color=base_colors["unmet"], label="Unmet"),
    Patch(color=base_colors["demand"], label="Demand")
]

for r in regions:
    legend_handles.append(Patch(color=transport_colors[r], label=f"Import from {r}"))

def draw_donut(ax, region, production_series, unmet_series, demand_series, title):
    ax.set_title(title, fontsize=11, bbox=dict(boxstyle="round", ec="white", fc=region_label_colors[title]))
    for j, t in enumerate(times):
        radius = base_radius + j * ring_width
        year = str(t)
        production = float(production_series.sel(time=t))
        unmet = float(unmet_series.sel(time=t))
        demand = float(demand_series.sel(time=t))

        imports = []
        import_colors = []
        for source in regions:
            if source == region:
                continue
            val = float(ds["v_transport"].sel(region1=source, region2=region, commodity="h2", scenario="Base", time=t))
            if val > 0:
                imports.append(val)
                import_colors.append(transport_colors[source])

        values = [production] + imports + [unmet, demand]
        colors = [base_colors["production"]] + import_colors + [base_colors["unmet"], base_colors["demand"]]

        wedges, _ = ax.pie(values, radius=radius, startangle=90, colors=colors, wedgeprops=dict(width=ring_width, edgecolor="white"))
        total = np.sum(values)

        for w, val in zip(wedges, values):
            if val / total < 0.08:
                continue
            theta = (w.theta1 + w.theta2) / 2
            theta = np.deg2rad(theta)
            r_text = radius - ring_width / 2
            x = r_text * np.cos(theta)
            y = r_text * np.sin(theta)
            ax.text(x, y, f"{val:.0f}\n{year}", ha="center", va="center", fontsize=6)
    ax.set(aspect="equal")

for i, region in enumerate(regions):
    production = ds["v_production"].sel(region=region, commodity="h2", scenario="Base")
    unmet = ds["v_unmet"].sel(region=region, commodity="h2", scenario="Base")
    demand = data_1D["demand"].sel(region=region, commodity="h2", scenario="Base")
    draw_donut(axes[i], region, production, unmet, demand, region)

total_production = ds["v_production"].sel(commodity="h2", scenario="Base").sum("region")
total_unmet = ds["v_unmet"].sel(commodity="h2", scenario="Base").sum("region")
total_demand = data_1D["demand"].sel(commodity="h2", scenario="Base").sum("region")
draw_donut(axes[len(regions)], regions[0], total_production, total_unmet, total_demand, "TOTAL")

for j in range(len(regions) + 1, len(axes)):
    fig.delaxes(axes[j])

fig.legend(handles=legend_handles, loc="center right")
plt.tight_layout()
plt.subplots_adjust(right=.83)

# Save the plot as a PNG file in the figures subfolder
file_path = os.path.join(output_path, f"Yearly_production_transport_demand_balance_{case_study.replace(' ', '_')}.png")
if os.path.exists(file_path):
    os.remove(file_path)
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.ioff()
plt.show()

# Your existing code for creating the plot
ds = model.solution
regions = ds.region.values
times_all = ds.time.values
times = times_all[::2]

n = len(regions) + 1
ncols = 3
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols,6*nrows))
axes = np.array(axes).flatten()

bar_width = 1
index = np.arange(len(times))

base_colors = {
    "production": "#1f77b4",
    "unmet": "grey",
    "demand": "#d62728"
}

cmap = cm.get_cmap("Wistia", len(regions))
transport_colors = {region: cmap(i) for i, region in enumerate(regions)}
region_label_colors = {region: cmap(i) for i, region in enumerate(regions)}
region_label_colors["TOTAL"] = "white"

legend_handles = [
    Patch(color=base_colors["production"], label="Production"),
    Patch(color=base_colors["unmet"], label="Unmet"),
    Patch(color=base_colors["demand"], label="Demand")
]

for r in regions:
    legend_handles.append(Patch(color=transport_colors[r], label=f"Import from {r}"))

def draw_stacked_bar(ax, region, production_series, unmet_series, demand_series, title):
    ax.set_title(title, fontsize=11, bbox=dict(boxstyle="round", ec="white", fc=region_label_colors[title]))
    bottom = np.zeros(len(times))

    for j, t in enumerate(times):
        year = str(t)
        production = float(production_series.sel(time=t))
        unmet = float(unmet_series.sel(time=t))
        demand = float(demand_series.sel(time=t))

        imports = []
        import_colors = []
        for source in regions:
            if source == region:
                continue
            val = float(ds["v_transport"].sel(region1=source, region2=region, commodity="h2", scenario="Base", time=t))
            if val > 0:
                imports.append(val)
                import_colors.append(transport_colors[source])

        values = [production] + imports + [unmet, demand]
        colors = [base_colors["production"]] + import_colors + [base_colors["unmet"], base_colors["demand"]]

        for val, color in zip(values, colors):
            ax.bar(j, val, bottom=bottom[j], width=bar_width, color=color, edgecolor="white")
            bottom[j] += val

        total = np.sum(values)
        for k, val in enumerate(values):
            if val / total < 0.08:
                continue
            ax.text(j, bottom[j] - val / 2, f"{val:.0f}\n{year}", ha="center", va="center", fontsize=6, color="white")

    ax.set_xticks(index)
    ax.set_xticklabels(times)
    ax.set_xlabel("Time")
    ax.set_ylabel("Quantity")
    ax.set_ylim(0, np.max(bottom) * 1.1)

for i, region in enumerate(regions):
    production = ds["v_production"].sel(region=region, commodity="h2", scenario="Base")
    unmet = ds["v_unmet"].sel(region=region, commodity="h2", scenario="Base")
    demand = data_1D["demand"].sel(region=region, commodity="h2", scenario="Base")
    draw_stacked_bar(axes[i], region, production, unmet, demand, region)

total_production = ds["v_production"].sel(commodity="h2", scenario="Base").sum("region")
total_unmet = ds["v_unmet"].sel(commodity="h2", scenario="Base").sum("region")
total_demand = data_1D["demand"].sel(commodity="h2", scenario="Base").sum("region")
draw_stacked_bar(axes[len(regions)], regions[0], total_production, total_unmet, total_demand, "TOTAL")

for j in range(len(regions) + 1, len(axes)):
    fig.delaxes(axes[j])

fig.legend(handles=legend_handles, loc="center right")
plt.tight_layout()
plt.subplots_adjust(right=.83)

# Save the plot as a PNG file in the figures subfolder
file_path = os.path.join(output_path, f"Yearly_production_transport_demand_columns_{case_study.replace(' ', '_')}.png")
# Remove the file if it already exists
if os.path.exists(file_path):
    os.remove(file_path)
plt.savefig(file_path, dpi=300)  # bbox_inches='tight' is commented out
print(f"Plot saved successfully at {file_path}")
plt.ioff()

#%%
STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')