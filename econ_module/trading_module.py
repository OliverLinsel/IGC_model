### This is the trading_module intended to build a small optimization framework that combines datasets of prices and demand/supply over different time steps (years or higher resolution) to build a global bilateral trading optimization module.
#%%

import pandas as pd
import time
import os
import xarray as xr
import numpy as np
import linopy
import matplotlib.pyplot as plt

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
prices_df = pd.read_excel(os.path.join(data_path, case_study, "region_tables.xlsx"), sheet_name='prices')
demands_df = pd.read_excel(os.path.join(data_path, case_study, "region_tables.xlsx"), sheet_name='demands')
supply_df = pd.read_excel(os.path.join(data_path, case_study, "region_tables.xlsx"), sheet_name='supply')

# Reshape the data from wide to long format
def reshape_data(df, value_name):
    df_long = df.melt(id_vars=['region', 'commodity', 'scenario'],
                      var_name='time',
                      value_name=value_name)
    return df_long

prices_long = reshape_data(prices_df, 'price')
demands_long = reshape_data(demands_df, 'demand')
supply_long = reshape_data(supply_df, 'supply')

# Merge the DataFrames
merged_df = pd.merge(prices_long, demands_long, on=['region', 'commodity', 'scenario', 'time'])
merged_df = pd.merge(merged_df, supply_long, on=['region', 'commodity', 'scenario', 'time'])

# Read relationship factors
relationship_df = pd.read_excel(os.path.join(data_path, case_study, "relationship_transport_data.xlsx"), sheet_name='relationship')
transport_costs = pd.read_excel(os.path.join(data_path, case_study, "relationship_transport_data.xlsx"), sheet_name='transport_costs')
transport_costs_long = transport_costs.melt(id_vars=['region1', "region2", 'commodity', 'scenario'],
                    var_name='time')

pair_df = pd.merge(relationship_df, transport_costs_long, on=['region1', 'region2', 'scenario'])

# -----------------------------
# Prepare nodal datasets from dataframe to xarray
# -----------------------------

# Identify the dimension columns
dimension_columns = [col for col in merged_df.columns if col in ["region", "commodity", "scenario", "time"]]

# Identify value columns
value_columns = [col for col in merged_df.columns if col not in dimension_columns]

# Initialize a dictionary to hold the lists for each dimension
dimension_lists = {}
# Create separate DataArrays for each value column
data_arrays = {}

# Create lists of unique values for each dimension column
for column in dimension_columns:
    dimension_lists[column] = merged_df[column].unique().tolist()

for value_column in value_columns:
    data_arrays[value_column] = xr.DataArray(
        np.full(tuple(len(lst) for lst in dimension_lists.values()), np.nan),
        dims=dimension_columns,
        coords=dimension_lists
    )

# Fill the DataArrays with values from the DataFrame
for _, row in merged_df.iterrows():
    for value_column in value_columns:
        data_arrays[value_column].loc[tuple(row[dim] for dim in dimension_columns)] = row[value_column]


# Combine the DataArrays into a single Dataset
data_1D = xr.Dataset(data_arrays)

# -----------------------------
# Create transport pairs
# -----------------------------

# pair_df = pair_df[pair_df["time"] == 2030]

#identify the unique regions:
regions = merged_df["region"].unique()

# Create a DataFrame with all possible pairs of the unique regions
pairs = pd.MultiIndex.from_product(
    [regions, regions],
    names=["from_region", "to_region"]
)
# Convert the pairs MultiIndex to a list of tuples
pairs_list = list(pairs)

# Filter pair_df to include only rows where (region1, region2) is in pairs_list
pair_df = pair_df[pair_df.apply(lambda row: (row["region1"], row["region2"]) in pairs_list, axis=1)]

# Identify the dimension columns
dimension_columns_2D = [col for col in pair_df.columns if col in ["region1", "region2", "commodity", "scenario", "time"]]

# Identify value columns
value_columns_2D = [col for col in pair_df.columns if col not in dimension_columns_2D]

# Initialize a dictionary to hold the lists for each dimension
dimension_lists_2D = {}
# Create separate DataArrays for each value column
data_arrays_2D = {}

# Create lists of unique values for each dimension column
for column in dimension_columns_2D:
    dimension_lists_2D[column] = pair_df[column].unique().tolist()

for value_column_2D in value_columns_2D:
    data_arrays_2D[value_column_2D] = xr.DataArray(
        np.full(tuple(len(lst) for lst in dimension_lists_2D.values()), np.nan),
        dims=dimension_columns_2D,
        coords=dimension_lists_2D
    )

# Fill the DataArrays with values from the DataFrame
for _, row in pair_df.iterrows():
    for value_column_2D in value_columns_2D:
        data_arrays_2D[value_column_2D].loc[tuple(row[dim] for dim in dimension_columns_2D)] = row[value_column_2D]

# Combine the DataArrays into a single Dataset
data_2D = xr.Dataset(data_arrays_2D)

import linopy
import numpy as np
import pandas as pd

model = linopy.Model()

# -----------------------------
# Get dimensions
# -----------------------------

data_1D = data_1D.fillna(0)
data_2D = data_2D.fillna(0)

df_1D = merged_df.copy()
dims_1D = list(data_1D.dims)
df_2D = pair_df.copy()
dims_2D = list(data_2D.dims)

data_1D = data_1D.fillna(0)
data_2D = data_2D.fillna(0)

regions = data_1D["region"]
commodities = np.unique(data_1D["commodity"])
scenarios = np.unique(data_1D["scenario"])
times = np.unique(data_1D["time"])

data_2D = data_2D.sel(
    region1=[r.item() for r in data_2D.region1.values if r in regions.values],
    region2=[r.item() for r in data_2D.region2.values if r in regions.values],
)

# -----------------------------
# Variables
# -----------------------------

v_production = model.add_variables(
    lower=0,
    upper=data_1D["supply"] * 100,
    coords=data_1D.coords,
    dims=data_1D.dims,
    name="v_production"
)

v_transport = model.add_variables(
    lower=0,
    coords=data_2D.coords,
    dims=data_2D.dims,
    name="v_transport"
)

v_unmet = model.add_variables(
    lower=0,
    coords=data_1D.coords,
    dims=data_1D.dims,
    name="v_unmet"
)

# -----------------------------
# Flow terms
# -----------------------------

inflow = v_transport.sum(dim="region1")
outflow = v_transport.sum(dim="region2")

# -----------------------------
# Constraints
# -----------------------------

# Sanity check for total supply
net_demand = df_1D["supply"].sum() - df_1D["demand"].sum()
if net_demand.sum() < 0:
    print("Warning: Total demand exceeds total supply. The model might be infeasible.")
else:
    print(str(net_demand.sum()) + " TWh excess production potential")

c_supply = model.add_constraints(
    v_production <= data_1D["supply"],
    name="c_supply"
)

c_balance = model.add_constraints(
    v_production
    + inflow
    - outflow
    + v_unmet
    >= data_1D["demand"],
    name="c_balance"
)

# -----------------------------
# Cost terms
# -----------------------------

production_costs = (
    v_production * data_1D["price"]).sum()

transport_costs = (
    v_transport * data_2D["value"]).sum()

#Energy from heaven feasibility penalty
penalty_costs = (
    v_unmet * 1000).sum()

# -----------------------------
# Objective function
# -----------------------------

obj_func = model.add_objective(
    production_costs + transport_costs + penalty_costs,
    sense="min"
)

# -----------------------------
# Solve and debug
# -----------------------------

model.solve(solver_name="gurobi")

sol = model.solution

if sol is not None:
    print("Solution found")

    # example: production
    print(sol["v_production"])

    # example: transport
    print(sol["v_transport"])
else:
    print("No solution available")

import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Patch

# Create arbitrary transport values for testing
def create_arbitrary_transport_values():
    transport_values = np.random.rand(len(times), len(ds.region1), len(ds.region2)) * 10
    transport_array = xr.DataArray(
        transport_values,
        dims=["time", "region1", "region2"],
        coords={
            "time": times,
            "region1": ds.region1.values,
            "region2": ds.region2.values
        }
    )
    return transport_array

# Assuming ds is your xarray Dataset
ds = model.solution

regions = ds.region.values
times_all = ds.time.values

# every 2nd year
times = times_all[::2]

n = len(regions) + 1
ncols = 3
nrows = math.ceil(n / ncols)

# -----------------------------
# moderate figure size (KEY FIX)
fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(6 * ncols, 6 * nrows)
)

axes = np.array(axes).flatten()

ring_width = 0.12
base_radius = 0.5

colors = {
    "production": "#1f77b4",
    "transport": "#ff7f0e",
    "unmet": "#2ca02c",
    "demand": "#d62728"
}

legend_handles = [
    Patch(color=colors["production"], label="Production"),
    Patch(color=colors["transport"], label="Transport"),
    Patch(color=colors["unmet"], label="Unmet"),
    Patch(color=colors["demand"], label="Demand")
]

def draw_donut(ax, production_series, transport_series, unmet_series, demand_series, title):
    ax.set_title(title, fontsize=11)

    for j, t in enumerate(times):
        radius = base_radius + j * ring_width
        year = str(t)

        production = float(production_series.sel(time=t))
        transport = float(transport_series.sel(time=t).sum())
        unmet = float(unmet_series.sel(time=t))
        demand = float(demand_series.sel(time=t))

        wedges, _ = ax.pie(
            [production, transport, unmet, demand],
            radius=radius,
            startangle=90,
            colors=[colors["production"], colors["transport"], colors["unmet"], colors["demand"]],
            wedgeprops=dict(width=ring_width, edgecolor="white")
        )

        for w, val in zip(wedges, [production, transport, unmet, demand]):
            theta = (w.theta1 + w.theta2) / 2.0
            theta_rad = np.deg2rad(theta)

            r_text = radius - ring_width / 2
            x = r_text * np.cos(theta_rad)
            y = r_text * np.sin(theta_rad)

            ax.text(
                x, y,
                f"{val:.0f}\n({year})",
                ha="center",
                va="center",
                fontsize=7  # reduced for clarity
            )

    ax.set(aspect="equal")

# -----------------------------
# region plots
# -----------------------------
for i, region in enumerate(regions):
    production = ds["v_production"].sel(region=region, commodity="h2", scenario="Base")
    transport = ds["v_transport"].sel(scenario="Base", commodity="h2")
    # transport = create_arbitrary_transport_values()
    unmet = ds["v_unmet"].sel(region=region, commodity="h2", scenario="Base")
    demand = data_1D["demand"].sel(region=region, commodity="h2", scenario="Base")

    draw_donut(axes[i], production, transport, unmet, demand, str(region))

# -----------------------------
# TOTAL plot
# -----------------------------
total_production = ds["v_production"].sel(commodity="h2", scenario="Base").sum(dim="region")
total_transport = ds["v_transport"].sel(scenario="Base", commodity="h2").sum(dim=["region1", "region2"])
# total_transport = create_arbitrary_transport_values().sum(dim=["region1", "region2"])
total_unmet = ds["v_unmet"].sel(commodity="h2", scenario="Base").sum(dim="region")
total_demand = data_1D["demand"].sel(commodity="h2", scenario="Base").sum(dim="region")

draw_donut(axes[len(regions)], total_production, total_transport, total_unmet, total_demand, "TOTAL")

for j in range(len(regions) + 1, len(axes)):
    fig.delaxes(axes[j])

fig.legend(handles=legend_handles, loc="upper right")

plt.tight_layout()
fig.subplots_adjust(wspace=0.25, hspace=0.35)

# Save the plot as a PNG file in the figures subfolder
file_path = os.path.join(output_path, f"Production_transport_demand_balance_{case_study.replace(' ', '_')}.png")
if os.path.exists(file_path):
    os.remove(file_path)
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.ioff()
plt.show()

#%%
STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')