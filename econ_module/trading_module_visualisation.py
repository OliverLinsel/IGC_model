### This is the corresponding visualisation module for the trading module
#%%
import os
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import math
import linopy
import json
from shapely import wkt
import plotly.colors as pc
import xarray as xr
import gc

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

START = time.perf_counter()

#define paths
print('Execute in Directory:')
print(os.getcwd() + "\n")

#%%

#### Set global parameters ####
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
default_dpi = 300

# Define a custom Wistia color palette with 30 shades
wistia_colors = [
    '#FFFFFF', '#FFFDF9', '#FFF9F2', '#FFF6EB', '#FFF2E4', '#FFEFDD', '#FFECD6',
    '#FFE9CF', '#FFE6C8', '#FFE3C1', '#FFE0BA', '#FFDDB3', '#FFDAAD', '#FFD7A6',
    '#FFD4A0', '#FFD199', '#FFCE92', '#FFCB8B', '#FFC884', '#FFC57E', '#FFC277',
    '#FFBF70', '#FFBC69', '#FFB962', '#FFB65B', '#FFB354', '#FFB04D', '#FFAE46',
    '#FFAB3F', '#FFA838', '#FFA531', '#FFA22A', '#FFA023', '#FFA01C', '#FFA015',
    '#FFA00E', '#FFA007', '#FFA000', '#FFA300', '#FFA600'
]

wistia_colors = ['#FFFF00', '#FFFB00', '#FFF700', '#FFF300', '#FFEF00', '#FFEB00', '#FFE700', '#FFE300',
 '#FFDF00', '#FFDB00', '#FFD700', '#FFD300', '#FFCF00', '#FFCB00', '#FFC700', '#FFC300',
 '#FFBF00', '#FFBB00', '#FFB700', '#FFB300', '#FFAF00', '#FFAB00', '#FFA700', '#FFA300',
 '#FF9F00', '#FF9B00', '#FF9700', '#FF9300', '#FF8F00', '#FF8B00', '#FF8700', '#FF8300',
 '#FF7F00', '#FF7B00', '#FF7700', '#FF7300', '#FF6F00', '#FF6B00', '#FF6700', '#FF6300',
 '#FF5F00', '#FF5B00', '#FF5700', '#FF5300', '#FF4F00', '#FF4B00', '#FF4700', '#FF4300',
 '#FF3F00', '#FF3B00']

# Set global parameters for Plotly
template = {
    "layout": {
        "font": {
            "family": "Times New Roman",
            "color": "black"
        },
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "xaxis": {
            "tickfont": {
                "color": "black",
                "size": 12
            },
            "title": {
                "standoff": 15,
                "font": {
                    "size": 14
                }
            }
        },
        "yaxis": {
            "tickfont": {
                "color": "black",
                "size": 12
            },
            "title": {
                "standoff": 15,
                "font": {
                    "size": 14
                }
            }
        },
        "legend": {
            "font": {
                "size": 12
            }
        }
    }
}

MWH_TO_TWH = 1e6  # model results are in MWh; divide by 1e6 for TWh

def plot_supply_curves(data_1D, reference_region = "EU-DEU"):
    data_1D_df = (
        data_1D
        .sel(commodity=commodity, scenario=scenario)
        .to_dataframe()
        .reset_index()
    )

    regions = data_1D_df["region"].unique()
    non_ref_regions = [r for r in regions if r != reference_region]
    color_map = {r: mcolors.to_hex(c) for r, c in zip(non_ref_regions, plt.cm.viridis(np.linspace(0, 1, len(non_ref_regions))))}

    fig = go.Figure()

    first_step_twh_by_region = []
    total_twh_by_region = {}

    for region in regions:
        region_data_sorted = data_1D_df[data_1D_df["region"] == region].sort_values(by="demand")
        supply_diff_twh = region_data_sorted["supply_diff"] / MWH_TO_TWH
        cumulative_supply_twh = supply_diff_twh.cumsum()

        nonzero_steps = supply_diff_twh[supply_diff_twh > 0]
        if not nonzero_steps.empty:
            first_step_twh_by_region.append(nonzero_steps.min())
        total_twh_by_region[region] = cumulative_supply_twh.iloc[-1]

        if region == reference_region:
            color = "red"
            line_width = 3
        else:
            color = color_map[region]
            line_width = 2

        fig.add_trace(
            go.Scatter(
                x=cumulative_supply_twh, y=region_data_sorted["price"], mode="lines", name=region,
                line=dict(color=color, width=line_width),
                showlegend=False,
                hovertemplate=(
                    f"<b>{region}</b><br>Cumulative supply: %{{x:.2f}} TWh/a<br>"
                    "Price: %{y:.2f} €/MWh<extra></extra>"
                ),
            )
        )

        last_x = cumulative_supply_twh.iloc[-1]
        last_y = region_data_sorted["price"].iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last_x], y=[last_y],
                mode="text",
                text=[region],
                textposition="middle right",
                textfont=dict(family="Times New Roman", size=13, color=color),  # bumped from 10
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Regional supply curves — {commodity}, {scenario}",
            font=dict(family="Times New Roman", size=20, color="black"),  # bumped, was unset (default ~17)
        ),
        xaxis=dict(title=dict(text="Supply potential (TWh/a)", font=dict(family="Times New Roman", size=16, color="black")),
                    tickfont=dict(family="Times New Roman", size=16, color="black")),
        yaxis=dict(title=dict(text="Price (€/MWh)", font=dict(family="Times New Roman", size=16, color="black")),
                    tickfont=dict(family="Times New Roman", size=16, color="black")),
        showlegend=False,
        font=dict(family="Times New Roman", color="black"),
        plot_bgcolor="white", paper_bgcolor="white",
        width=1400, height=800, margin=dict(r=20),  # was r=200
    )

    x_lower = np.percentile(first_step_twh_by_region, 1)
    x_upper = max(total_twh_by_region.values()) * 1.7
    fig.update_xaxes(
        type="log",
        range=[np.log10(x_lower), np.log10(x_upper)],
        showgrid=True, gridcolor="lightgrey", gridwidth=1, zeroline=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1, zeroline=False)

    fig.write_image(os.path.join(output_path, "figures", f"supply_curves_{commodity}_{scenario}.png"), scale=4)
    # fig.show()
    plt.ioff()
    return

#### define the Herfindahl-Hirschmann Index (HHI) calculation function ####
def calculate_hhi(ds):
    ### Herfindahl-Hirschman Index (HHI) calculation for market concentration ###
    # The HHI is calculated as the sum of the squares of the market shares of each firm (or region, in this case) in the market. It ranges from 0 to 10,000, where higher values indicate higher market concentration

    # Calculate HHI for each commodity
    hhi_results = {}
    for commodity in commodities:
        market_shares = []
        total_supply = ds["v_supply_segment"].sel(commodity=commodity, scenario=scenario).sum().item()
        
        for region in regions:
            regional_supply = ds["v_supply_segment"].sel(region=region, commodity=commodity, scenario=scenario).sum().item()
            market_share = regional_supply / total_supply if total_supply > 0 else 0
            market_shares.append(market_share)
        
        hhi = sum([share ** 2 for share in market_shares]) * 10000  # Scale to 0-10,000
        hhi_results[commodity] = hhi

    # Display HHI results
    for commodity, hhi in hhi_results.items():
        if hhi < 1500:
            print(f"Herfindahl-Hirschman Index (HHI) for {commodity}: {hhi:.2f}")
            print("Market is considered to be competitive.")
        elif 1500 <= hhi < 2500:
            print(f"Herfindahl-Hirschman Index (HHI) for {commodity}: {hhi:.2f}")
            print("Market is considered to be moderately concentrated.")
        else:
            print(f"Herfindahl-Hirschman Index (HHI) for {commodity}: {hhi:.2f}")
            print("Market is considered to be highly concentrated.")
    return hhi

def plot_supply_composition(model, ds, mtd, mid, rfm, reference_region):
    # ----- price + demand -----
    market_price = abs(
        model.constraints["c_balance"]
        .dual
        .sel(region=reference_region, commodity=commodity, scenario=scenario)
        .item()
    )

    demand_xr = (data_1D.sel(supply_step=base_step_param, drop=True))
    demand = (
        demand_xr["demand"]
        .sel(region=reference_region, commodity=commodity, scenario=scenario)
        .item()
    ) / MWH_TO_TWH

    # ----- local production -----
    prod = (
        ds["v_supply_segment"]
        .sel(region=reference_region)
        .to_dataframe("quantity")
        .reset_index()
    )
    prod = prod[prod.quantity > 1e-6]

    prod = prod.merge(
        data_1D["price"]
        .sel(region=reference_region)
        .to_dataframe("production_cost")
        .reset_index(),
        on=["commodity", "scenario", "supply_step"],
    )

    prod["source"] = reference_region
    prod["transport_cost"] = 0
    prod["kind"] = "Local"

    # ----- imports -----
    imp = (
        ds["v_transport"]
        .sel(region2=reference_region)
        .to_dataframe("quantity")
        .reset_index()
    )
    imp = imp[(imp.quantity > 1e-6) & (imp.region1 != reference_region)]

    if len(imp):
        imp = imp.merge(
            data_1D["price"]
            .to_dataframe("production_cost")
            .reset_index()
            .rename(columns={"region": "region1"}),
            on=["region1", "commodity", "scenario", "supply_step"],
        )
        imp = imp.merge(
            data_2D["transport_cost"].to_dataframe("transport_cost").reset_index(),
            on=["region1", "region2", "commodity", "scenario"],
        )
        imp["source"] = imp.region1
        imp["kind"] = "Import"
    else:
        imp = pd.DataFrame(columns=[
            "source", "quantity", "production_cost", "transport_cost", "kind", "supply_step"
        ])

    # ----- stack -----
    cols = ["source", "quantity", "production_cost", "transport_cost", "kind", "supply_step"]
    stack = pd.concat([prod[cols], imp[cols]], ignore_index=True)

    stack["delivered_cost"] = stack.production_cost + stack.transport_cost
    stack["quantity"] = stack["quantity"] / MWH_TO_TWH  # MWh -> TWh, matches x-axis label
    stack = stack.sort_values("delivered_cost")
    stack["end"] = stack.quantity.cumsum()
    stack["start"] = stack.end - stack.quantity

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = {"Local": "tab:blue", "Import": "tab:orange"}
    transport_threshold = 1e-6  # single threshold used consistently below (was inconsistent: 1e-6 vs 0)

    for _, r in stack.iterrows():
        ax.bar(r.start, r.production_cost, width=r.quantity, align="edge", color=colors[r.kind])

        has_transport = r.transport_cost > transport_threshold
        if has_transport:
            ax.bar(
                r.start, r.transport_cost, width=r.quantity, align="edge",
                bottom=r.production_cost, color="red", alpha=0.5,
            )

        txt = f"{r.source}\n{r.supply_step}\nQ={r.quantity:.1f}\nP={r.production_cost:.1f}"
        if has_transport:
            txt += f"\nT={r.transport_cost:.1f}"
        txt += f"\nΣ={r.delivered_cost:.1f}"

        ax.text(
            r.start + r.quantity / 2, r.delivered_cost + 2, txt,
            fontsize=6, rotation=90, ha="center", fontfamily="Times New Roman", color="black",
        )

    # demand line + label
    ax.axvline(demand, c="black", ls="--", lw=3)
    ax.text(
        demand, ax.get_ylim()[1] * 0.95, f"Demand\n{demand:.1f}",
        rotation=90, va="top", ha="right", fontweight="bold",
        fontfamily="Times New Roman", color="black",
    )

    # price line + label
    ax.axhline(market_price, c="green", ls=":", lw=3)
    ax.text(
        stack.end.max() * 0.98, market_price, f"{market_price:.1f} €/MWh",
        color="green", ha="right", va="bottom", fontsize=11, fontweight="bold",
        fontfamily="Times New Roman",
        bbox=dict(facecolor="white", edgecolor="green", alpha=0.8),
    )

    # marginal supplier
    mb = stack[stack.end >= demand]
    if len(mb):
        mb = mb.iloc[0]
        ax.axvspan(mb.start, mb.end, alpha=0.15, color="green")

    ax.set_xlabel("Quantity [TWh]", fontfamily="Times New Roman")
    ax.set_ylabel("Delivered cost €/MWh", fontfamily="Times New Roman")
    ax.set_title(f"Supply composition: {reference_region} - Max import share: {mtd*100:.0f}% total, {mid*100:.0f}% per region)", fontfamily="Times New Roman", fontsize=15)
    ax.set_xlim(0, stack.end.max() * 1.05)

    # tick labels don't reliably inherit rcParams["font.family"] -- set explicitly
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Times New Roman")

    legend = ax.legend(handles=[
        Patch(color="tab:blue", label="Domestic Production"),
        Patch(color="tab:orange", label="Import"),
        Patch(color="red", label="Transport Costs"),
        Line2D([0], [0], color="green", lw=2, ls=":", label="Marginal Price"),
        Line2D([0], [0], color="black", lw=2, ls="--", label="Demand"),
    ], prop={"family": "Times New Roman", "size": 12})

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig(os.path.join(output_path, "figures", "supply_composition", f"supply_composition_{reference_region}_{mtd*100:.0f}_{mid*100:.0f}_rfm{rfm*100:.0f}.png"), bbox_inches="tight", dpi=default_dpi)
    plt.ioff()
    # plt.show()
    return

def plot_sankey_flows(ds, hhi_results, mtd, mid, rfm):
    # ---- config -------------------------------------------------------------
    min_flow = 0.0

    # ---- extract data ---------------------------------------------------------
    regions = ds.region.values.tolist()
    n = len(regions)

    production = {
        r: float(ds["v_supply_segment"].sel(region=r, commodity=commodity, scenario=scenario).sum("supply_step"))
        for r in regions
    }
    unmet = {
        r: float(ds["v_unmet"].sel(region=r, commodity=commodity, scenario=scenario))
        for r in regions
    }
    demand_xr = (data_1D.sel(supply_step=base_step_param, drop=True))
    demand = {
        r: float(demand_xr["demand"].sel(region=r, commodity=commodity, scenario=scenario))
        for r in regions
    }

    trade = {}
    for src in regions:
        for dst in regions:
            if src == dst:
                continue
            val = float(
                ds["v_transport"]
                .sel(region1=src, region2=dst, commodity=commodity, scenario=scenario)
                .sum("supply_step")
            )
            if val > min_flow:
                trade[(src, dst)] = val

    # ---- node layout ----------------------------------------------------------
    # index 0..n-1        : "<region> — production"  (left column)
    # index n..2n-1        : "<region> — demand"       (right column)
    # index 2n             : "Unmet demand"            (synthetic source, left)
    production_idx = {r: i for i, r in enumerate(regions)}
    demand_idx = {r: i + n for i, r in enumerate(regions)}
    unmet_node = 2 * n

    # Values converted to TWh (raw data assumed to be in units where /1e6 -> TWh)
    # and rounded to the nearest whole TWh, shown on both the production and
    # demand node labels.
    labels = (
        [f"{r} — production ({production[r] / 1e6:.0f} TWh)" for r in regions]
        + [f"{r} — demand ({demand[r] / 1e6:.0f} TWh)" for r in regions]
        + ["Unmet demand"]
    )

    # one distinct color per region, reused for its production and demand node
    cmap = plt.get_cmap("plasma", n)
    region_rgba = {r: cmap(i) for i, r in enumerate(regions)}

    def to_plotly_rgba(rgba, alpha=None):
        r, g, b, a = rgba
        a = alpha if alpha is not None else a
        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.2f})"

    node_colors = (
        [to_plotly_rgba(region_rgba[r], 0.9) for r in regions]
        + [to_plotly_rgba(region_rgba[r], 0.9) for r in regions]
        + ["rgba(150,150,150,0.9)"]
    )

    # ---- links ------------------------------------------------------------
    source, target, value, link_color = [], [], [], []

    # domestic production consumed locally
    for r in regions:
        if production[r] > min_flow:
            source.append(production_idx[r])
            target.append(demand_idx[r])
            value.append(production[r])
            link_color.append(to_plotly_rgba(region_rgba[r], 0.55))

    # cross-border trade, colored by exporting region
    for (src, dst), val in trade.items():
        source.append(production_idx[src])
        target.append(demand_idx[dst])
        value.append(val)
        link_color.append(to_plotly_rgba(region_rgba[src], 0.75))

    # unmet demand, grey
    for r in regions:
        if unmet[r] > min_flow:
            source.append(unmet_node)
            target.append(demand_idx[r])
            value.append(unmet[r])
            link_color.append("rgba(150,150,150,0.45)")

    # ---- draw ---------------------------------------------------------------
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                pad=15,
                thickness=100,
                color=node_colors,
                line=dict(width=0),
            ),
            link=dict(source=source, target=target, value=value, color=link_color),
        )
    )
    fig.update_layout(
        title=(
            f"{commodity.upper()} trade flows [TWh] — {scenario} — "
            f"{hhi:.0f} HHI — "
            f"Max import share: {mtd*100:.0f}% total, "
            f"{mid*100:.0f}% per region"
        ),
        font_size=15,
        font_family="Times New Roman",
        font_color="black",
        width=1000,
        height=2000,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    os.makedirs("output", exist_ok=True)
    # fig.write_html(os.path.join("output", f"sankey_diagram_constrained_{mtd*100:.0f}_{mid*100:.0f}.html"))
    fig.write_image(os.path.join(output_path, "figures", "sankey_flows", f"sankey_diagram_constrained_v_{n}n_{mtd*100:.0f}_{mid*100:.0f}_rfm{rfm*100:.0f}.png"), scale=4)
    # fig.show()

    fig.update_layout(
        width=2000,
        height=1000)
    os.makedirs("output", exist_ok=True)
    fig.write_image(os.path.join(output_path, "figures", "sankey_flows", f"sankey_diagram_constrained_h_{n}n_{mtd*100:.0f}_{mid*100:.0f}_rfm{rfm*100:.0f}.png"), scale=4)

    # fig.show()
    plt.ioff()
    return

def plot_sankey_flows_selected(ds, hhi_results, mtd, mid, rfm, countries_of_interest):
    # ---- config -------------------------------------------------------------
    min_flow = 0.0

    # ---- extract data over the FULL region set (unchanged) --------------------
    regions = ds.region.values.tolist()
    n = len(regions)

    production = {
        r: float(ds["v_supply_segment"].sel(region=r, commodity=commodity, scenario=scenario).sum("supply_step"))
        for r in regions
    }
    unmet = {
        r: float(ds["v_unmet"].sel(region=r, commodity=commodity, scenario=scenario))
        for r in regions
    }
    demand_xr = (data_1D.sel(supply_step=base_step_param, drop=True))
    demand = {
        r: float(demand_xr["demand"].sel(region=r, commodity=commodity, scenario=scenario))
        for r in regions
    }

    trade = {}
    for src in regions:
        for dst in regions:
            if src == dst:
                continue
            val = float(
                ds["v_transport"]
                .sel(region1=src, region2=dst, commodity=commodity, scenario=scenario)
                .sum("supply_step")
            )
            if val > min_flow:
                trade[(src, dst)] = val

    coi_set = set(countries_of_interest)

    # ---- filter links to only those touching a country of interest -------
    # domestic production->demand: only relevant if the country itself is selected
    domestic_links = [r for r in regions if production[r] > min_flow and r in coi_set]

    # cross-border trade: keep if EITHER side is a country of interest, so
    # outside trade partners connected to a selected country are retained
    trade_links = {(src, dst): val for (src, dst), val in trade.items()
                   if src in coi_set or dst in coi_set}

    # unmet demand: only relevant for selected countries
    unmet_links = [r for r in regions if unmet[r] > min_flow and r in coi_set]

    # ---- NEW: derive the node set from whichever regions actually appear ------
    # in the filtered links above (this is what pulls in outside trade partners)
    production_regions = set(domestic_links) | {src for (src, dst) in trade_links}
    demand_regions = set(domestic_links) | {dst for (src, dst) in trade_links} | set(unmet_links)

    used_regions = sorted(production_regions | demand_regions)
    n_used = len(used_regions)
    n = len(countries_of_interest)
    has_unmet = len(unmet_links) > 0

    # ---- node layout, reindexed to only the used regions -----------------------
    production_idx = {r: i for i, r in enumerate(used_regions) if r in production_regions}
    demand_idx = {r: i + n_used for i, r in enumerate(used_regions) if r in demand_regions}
    # note: to keep index math simple, we allocate a demand slot at position
    # (n_used + position in used_regions) for every used region, even if that
    # region has no demand-side link — Sankey simply won't draw an isolated node's
    # link, and the label is still meaningful to show
    demand_idx = {r: i + n_used for i, r in enumerate(used_regions)}
    production_idx = {r: i for i, r in enumerate(used_regions)}
    unmet_node = 2 * n_used

    labels = (
        [f"{r} — production ({production[r] / 1e6:.0f} TWh)" for r in used_regions]
        + [f"{r} — demand ({demand[r] / 1e6:.0f} TWh)" for r in used_regions]
    )
    if has_unmet:
        labels.append("Unmet demand")

    cmap = plt.get_cmap("plasma", n_used)
    region_rgba = {r: cmap(i) for i, r in enumerate(used_regions)}

    def to_plotly_rgba(rgba, alpha=None):
        r, g, b, a = rgba
        a = alpha if alpha is not None else a
        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.2f})"

    node_colors = (
        [to_plotly_rgba(region_rgba[r], 0.9) for r in used_regions]
        + [to_plotly_rgba(region_rgba[r], 0.9) for r in used_regions]
    )
    if has_unmet:
        node_colors.append("rgba(150,150,150,0.9)")

    # ---- links, using the filtered sets from above -----------------------------
    source, target, value, link_color = [], [], [], []

    for r in domestic_links:
        source.append(production_idx[r])
        target.append(demand_idx[r])
        value.append(production[r])
        link_color.append(to_plotly_rgba(region_rgba[r], 0.55))

    for (src, dst), val in trade_links.items():
        source.append(production_idx[src])
        target.append(demand_idx[dst])
        value.append(val)
        link_color.append(to_plotly_rgba(region_rgba[src], 0.75))

    if has_unmet:
        for r in unmet_links:
            source.append(unmet_node)
            target.append(demand_idx[r])
            value.append(unmet[r])
            link_color.append("rgba(150,150,150,0.45)")

    # ---- draw ---------------------------------------------------------------
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                pad=15,
                thickness=100,
                color=node_colors,
                line=dict(width=0),
            ),
            link=dict(source=source, target=target, value=value, color=link_color),
        )
    )
    fig.update_layout(
        title=(
            f"{commodity.upper()} trade flows [TWh] — {scenario} — "
            f"{hhi:.0f} HHI — "
            f"Max import share: {mtd*100:.0f}% total, "
            f"{mid*100:.0f}% per region — "
            f"filtered to: {', '.join(countries_of_interest)}"
        ),
        font_size=15,
        font_family="Times New Roman",
        font_color="black",
        width=1200,
        height=1200 * np.sqrt(2),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    os.makedirs(os.path.join(output_path, "figures", "sankey_flows"), exist_ok=True)
    fig.write_image(os.path.join(output_path, "figures", "sankey_flows", f"sankey_diagram_selected_v_{n}n_{mtd*100:.0f}_{mid*100:.0f}_rfm{rfm*100:.0f}.png"), scale=4)

    fig.update_layout(width=2000, height=1000)
    fig.write_image(os.path.join(output_path, "figures", "sankey_flows", f"sankey_diagram_selected_h_{n}n_{mtd*100:.0f}_{mid*100:.0f}_rfm{rfm*100:.0f}.png"), scale=4)

    plt.ioff()
    return

def plot_supply_demand_donuts(ds, mtd, mid, rfm):
    def clean(val, tol=1e-3):
        """Snap near-zero (including small negative numerical noise) to exactly 0."""
        return 0.0 if abs(val) < tol else val

    # Layout — configurable number of donuts per row
    n = len(regions) + 1
    donuts_per_row = int(math.sqrt(n))
    ncols, nrows = donuts_per_row, math.ceil(n / donuts_per_row)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
    axes = np.array(axes).flatten()

    # Commodity colors (production slice color depends on commodity)
    commodities_color_dict = {
        "h2": "turquoise",
        # add other commodities here, e.g. "elec": "#1f77b4",
    }

    # Colors
    base_colors = {"unmet": "grey", "demand": "#d62728"}
    production_color = commodities_color_dict.get(commodity, "#1f77b4")  # fallback if commodity not in dict

    cmap = plt.get_cmap("Wistia", len(regions))
    transport_colors = {r: cmap(i) for i, r in enumerate(regions)}
    region_colors = {**{r: cmap(i) for i, r in enumerate(regions)}, "Total": "white"}

    legend_handles = (
        [Patch(color=production_color, label="production")]
        + [Patch(color=base_colors[k], label=k) for k in base_colors.keys()]
        + [Patch(color=transport_colors[r], label=f"Imports from {r}") for r in regions]
    )

    SLICE_LABEL_FONTSIZE = 25  # increased from 11
    demand_xr = (data_1D.sel(supply_step=base_step_param, drop=True))

    def draw_donut(ax, region, title, add_labels=True):
        production = clean(float(ds["v_supply_segment"].sel(region=region, commodity=commodity, scenario=scenario).sum("supply_step")))
        unmet = clean(float(ds["v_unmet"].sel(region=region, commodity=commodity, scenario=scenario)))
        demand = clean(float(demand_xr["demand"].sel(region=region, commodity=commodity, scenario=scenario)))
        imports, import_colors = [], []
        for source in regions:
            if source == region:
                continue
            val = clean(float(ds["v_transport"].sel(region1=source, region2=region, commodity=commodity, scenario=scenario).sum("supply_step")))
            if val > 1e-6:
                imports.append(val)
                import_colors.append(transport_colors[source])
        values = [production] + imports + [unmet, demand]
        colors = [production_color] + import_colors + [base_colors["unmet"], base_colors["demand"]]
        ax.set_title(title, fontsize=15, fontfamily="Times New Roman",
                    bbox=dict(boxstyle="round", ec="white", fc=region_colors[title]))
        wedges, _ = ax.pie(values, radius=1.2, startangle=90, colors=colors, wedgeprops=dict(width=.8, edgecolor="white"))
        total = np.sum(values)
        for w, val in zip(wedges, values):
            if val/total < .05:
                continue
            theta = np.deg2rad((w.theta1+w.theta2)/2)
            x, y = .72*np.cos(theta), .72*np.sin(theta)
            if add_labels:
                ax.text(x, y, f"{val/1e6:.1f}", ha="center", va="center",
                        fontsize=SLICE_LABEL_FONTSIZE, fontfamily="Times New Roman")
        ax.set(aspect="equal")

    # Regional plots
    for i, region in enumerate(regions):
        draw_donut(axes[i], region, region)

    # TOTAL plot
    total_production = clean(float(ds["v_supply_segment"].sel(commodity=commodity, scenario=scenario).sum()))
    total_unmet = clean(float(ds["v_unmet"].sel(commodity=commodity, scenario=scenario).sum()))
    total_demand = clean(float(demand_xr["demand"].sel(commodity=commodity, scenario=scenario).sum()))
    imports, import_colors = [], []
    for source in regions:
        val = clean(float(ds["v_transport"].sel(region1=source, commodity=commodity, scenario=scenario).sum()))
        if val > 1e-6:
            imports.append(val)
            import_colors.append(transport_colors[source])
    values = [total_production] + imports + [total_unmet, total_demand]
    colors = [production_color] + import_colors + [base_colors["unmet"], base_colors["demand"]]

    ax = axes[len(regions)]
    ax.set_title("Total", fontfamily="Times New Roman", bbox=dict(boxstyle="round", fc="white"))
    wedges, _ = ax.pie(values, radius=1.2, startangle=90, colors=colors, wedgeprops=dict(width=.8, edgecolor="white"))
    ax.set(aspect="equal")

    # Add labels to TOTAL plot
    total = np.sum(values)
    for w, val in zip(wedges, values):
        if val/total < .05:
            continue
        theta = np.deg2rad((w.theta1+w.theta2)/2)
        x, y = .72*np.cos(theta), .72*np.sin(theta)
        ax.text(x, y, f"{val/1e6:.1f}", ha="center", va="center",
                fontsize=SLICE_LABEL_FONTSIZE, fontfamily="Times New Roman")

    # Cleanup
    for i in range(len(regions)+1, len(axes)):
        fig.delaxes(axes[i])

    # Overall figure title, including dependency variables
    fig.suptitle(
        (
            f"{commodity.upper()} trade flows [TWh] — {scenario} — "
            f"{hhi:.0f} HHI — "
            f"Max import share: {mtd*100:.0f}% total, "
            f"{mid*100:.0f}% per region"
        ),
        fontsize=40,
        fontfamily="Times New Roman",
        y=0.94,
    )

    fig.legend(handles=legend_handles, loc="center right", prop={"family": "Times New Roman", "size": 18})

    # Control spacing directly — no tight_layout, so nothing overrides these values
    plt.subplots_adjust(top=0.90, right=0.88, left=0.03, bottom=0.03, wspace=0.05, hspace=0.1)

    # Save output with dependency variables in the filename
    os.makedirs("output", exist_ok=True)
    fig.savefig(os.path.join(output_path, "figures", "supply_composition", f"donut_charts_{n}n_{mtd*100:.0f}_{mid*100:.0f}_rfm{rfm*100:.0f}.png"), dpi=100, bbox_inches="tight")

    # plt.show()
    plt.ioff()
    return

def load_all_model_runs(output_path):
    """
    Iterate through all run folders inside output_path/<model_subdir>.
    Each run folder is expected to contain a solution.nc (loaded via
    load_model_run) and a metadata.json describing the run's parameters.
 
    Returns a list of dicts:
        {
            "run_name": <folder name>,
            "metadata": <parsed metadata.json as dict>,
            "ds": <solution dataset returned by load_model_run>,
        }
    """
    runs = []
 
    for run_name in sorted(os.listdir(os.path.join(output_path, "model"))):
        run_path = os.path.join(output_path, "model", run_name)
        if not os.path.isdir(run_path):
            print("Skip")
            continue
 
        meta_path = os.path.join(run_path, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
 
        with open(meta_path, "r") as f:
            metadata = json.load(f)
 
        # load_model_run(base_path, run_name) mirrors your existing call:
        #   load_model_run(output_path, "model_run_159n_h2bb_100_10")
        # Here the runs sit directly under output/model, so output_path is
        # the base_path and run_name is passed exactly as-is. 
        print(run_name)

        runs.append({
            "run_name": run_name,
            "metadata": metadata
        })
    return runs

def analyse_results(output_path, runs):
    """
    Given a list of runs (as returned by load_all_model_runs), iterate
    through every combination of max_total_dependence_rel and
    max_indiv_dependence_rel found across the runs' metadata, calculate
    the HHI for the matching run's dataset, and collect the results
    into a single DataFrame. Also extracts the optimized total system
    cost (objective value) for each run via its full model.

    Returns a tuple of two DataFrames:
        hhi_df:  columns "max_total", "max_indiv", "HHI"
        cost_df: columns "max_total", "max_indiv", "total_system_costs"
    """

    # Initialize empty lists to store the data
    hhi_data_list = []
    cost_data_list = []

    # Iterate through each run in the runs list
    for run in runs:
        run_name = run["run_name"]
        mtd = run["metadata"]["max_total_dependence_rel"]
        mid = run["metadata"]["max_indiv_dependence_rel"]

        # Load the run solution
        data_1D, data_2D, solution, meta_data = load_model_run(output_path, run_name)

        # Calculate the HHI
        hhi = calculate_hhi(solution)

        hhi_data_list.append({
            "mtd": mtd,
            "mid": mid,
            "HHI": hhi
        })

        # --- Extract total system cost from the full model ---
        model = load_complete_model(output_path, run_name)
        total_system_costs = model.objective.value

        cost_data_list.append({
            "mtd": mtd,
            "mid": mid,
            "total_system_costs": total_system_costs
        })

    # Create DataFrames from the lists
    hhi_df = pd.DataFrame(hhi_data_list)
    cost_df = pd.DataFrame(cost_data_list)

    return hhi_df, cost_df

# def plot_hhi_sens(hhi_df, low_thresh=None, high_thresh=None):
#     """
#     Plot a 3D surface showing HHI as a function of max_total_dependence
#     and max_indiv_dependence, with demarcation lines for the standard
#     HHI concentration corridors (unconcentrated / moderate / high).

#     hhi_df must have columns "max_total", "max_indiv", "HHI".
#     low_thresh, high_thresh: override the default corridor thresholds.
#         If None, auto-detected as 1500/2500 (0-10000 scale) or
#         0.15/0.25 (0-1 scale) based on the max HHI value present.
#         The high threshold is only drawn if the surface actually
#         exceeds it somewhere.
#     """
#     # Pivot into a grid: rows = max_total, cols = max_indiv
#     pivot = hhi_df.pivot(
#         index="mtd",
#         columns="mid",
#         values="HHI",
#     ).sort_index().sort_index(axis=1)

#     x = pivot.columns.values  # max_indiv
#     y = pivot.index.values    # max_total
#     X, Y = np.meshgrid(x, y)
#     Z = pivot.values

#     zmax = np.nanmax(Z)
#     zmin = np.nanmin(Z)

#     # --- auto-detect scale and set default thresholds ---
#     if low_thresh is None or high_thresh is None:
#         if zmax <= 1.5:  # looks like a 0-1 scale
#             low_thresh, high_thresh = 0.15, 0.25
#         else:  # standard 0-10000 scale
#             low_thresh, high_thresh = 1500, 2500

#     show_high = zmax > high_thresh  # only demarcate the high corridor if data crosses it

#     fig = plt.figure(figsize=(10, 12))
#     ax = fig.add_subplot(111, projection="3d")

#     surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k",
#                             linewidth=0.3, alpha=0.7)
#     # ax.invert_xaxis()
#     # ax.invert_yaxis()

#     ax.zaxis.labelpad = 15
#     fig.subplots_adjust(right=0.85)
#     ax.set_xlabel("Max individual import dependence")
#     ax.set_ylabel("Max total import dependence")
#     ax.text2D(0.95, 0.82, "HHI", transform=ax.transAxes, ha="center", fontsize=12)
#     ax.set_title("HHI Sensitivity to Dependence Constraints", fontsize=18)

#     z_floor = zmin - 0.05 * (zmax - zmin)
#     ax.set_zlim(z_floor, zmax)

#     thresholds = [(low_thresh, "black", "Treshold HHI > 1500 for moderately concentrated market")]
#     if show_high:
#         thresholds.append((high_thresh, "black", "Treshold HHI > 2500 for highly concentrated market"))

#     legend_lines = []
#     xr = (x.min(), x.max())
#     yr = (y.min(), y.max())

#     for thresh, color, label in thresholds:
#         # floor contour: traces where the surface crosses this HHI level
#         ax.contour(X, Y, Z, levels=[thresh], zdir="z", offset=thresh,
#                    colors=color, linewidths=2, linestyles="--")

#         # wall lines: outline the threshold height on the xz/yz background panes
#         for y_edge in yr:
#             ax.plot(xr, [y_edge, y_edge], [thresh, thresh],
#                     color=color, lw=2, ls=":", alpha=1, zorder=2)
#         for x_edge in xr:
#             ax.plot([x_edge, x_edge], yr, [thresh, thresh],
#                     color=color, lw=2, ls=":", alpha=1, zorder=2)

#         legend_lines.append(
#             Line2D([0], [0], color=color, lw=2, ls="--",
#                    label=f"{label} ({thresh})")
#         )

#     ax.legend(handles=legend_lines, loc="upper left", fontsize=9)

#     plt.tight_layout()
#     # plt.show()
#     plt.ioff()
#     fig.savefig(os.path.join(output_path, "figures", f"sens_area_{n}n_{commodity}_rfm{rfm*100:.0f}.png"))
#     return fig, ax

# def plot_cost_sens(cost_df):
#     """
#     Plot a 3D surface showing total_system_costs as a function of
#     max_total_dependence and max_indiv_dependence.

#     cost_df must have columns "max_total", "max_indiv", "total_system_costs".
#     """
#     currency_unit="€"
#     raw_unit_scale=1e9

#     pivot = cost_df.pivot(
#         index="mtd",
#         columns="mid",
#         values="total_system_costs",
#     ).sort_index().sort_index(axis=1)

#     x = pivot.columns.values  # mid
#     y = pivot.index.values    # mtd
#     X, Y = np.meshgrid(x, y)
#     Z = pivot.values / raw_unit_scale 

#     fig = plt.figure(figsize=(10, 12))
#     ax = fig.add_subplot(111, projection="3d")

#     surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k",
#                             linewidth=0.3, alpha=0.8)
#     # ax.invert_xaxis()
#     # ax.invert_yaxis()

#     ax.zaxis.labelpad = 15
#     fig.subplots_adjust(right=0.85)
#     ax.set_xlabel("Max individual import dependence")
#     ax.set_ylabel("Max total import dependence")
#     ax.text2D(0.95, 0.82, f"Total system\ncosts [bn {currency_unit}]",
#               transform=ax.transAxes, ha="center", fontsize=12)
#     ax.set_title(f"Total System Cost Sensitivity to Dependence Constraints [bn {currency_unit}]",
#                  fontsize=18)

#     plt.tight_layout()
#     # plt.show()
#     plt.ioff()
#     fig.savefig(os.path.join(output_path, "figures", f"cost_surface_{n}n_{commodity}.png"))
#     return fig, ax

# def get_widia_cmap(rfm_value, rfm_sweep):
#     """Creates a Widia-based colormap that varies with rfm_value"""
#     base_color = '#4B0082'  # Change to your actual Widia color
#     rfm_min = min(rfm_sweep)
#     rfm_max = max(rfm_sweep)
#     norm_rfm = (rfm_value - rfm_min) / (rfm_max - rfm_min) if rfm_max > rfm_min else 0.5

#     light_factor = 1.0 + 0.5 * (1 - norm_rfm)
#     dark_factor = 0.7 + 0.3 * norm_rfm

#     light_color = adjust_lightness(base_color, light_factor)
#     dark_color = adjust_lightness(base_color, dark_factor)

#     return LinearSegmentedColormap.from_list('widia_seq', [light_color, dark_color])

def plot_hhi_sens_multi(hhi_dfs, rfm_values, low_thresh=None, high_thresh=None):
    """
    Plot multiple 3D surfaces showing HHI as a function of max_total_dependence,
    max_indiv_dependence, and relationship_factor_magnitude (rfm).
    """
    # Sort rfm values and select colors from wistia_colors
    sorted_rfm = sorted(rfm_values)
    n_rfm = len(sorted_rfm)
    color_indices = [int(i * 29 / max(1, n_rfm-1)) for i in range(n_rfm)]  # Map to 0-29
    surface_colors = [wistia_colors[i] for i in color_indices]

    # Find global min/max for z-axis
    all_z = []
    for hhi_df in hhi_dfs:
        pivot = hhi_df.pivot(index="mtd", columns="mid", values="HHI")
        all_z.extend(pivot.values.flatten())
    zmin, zmax = np.nanmin(all_z), np.nanmax(all_z)

    # Auto-detect thresholds
    if low_thresh is None or high_thresh is None:
        if zmax <= 1.5:  # 0-1 scale
            low_thresh, high_thresh = 0.15, 0.25
        else:  # 0-10000 scale
            low_thresh, high_thresh = 1500, 2500
    show_high = zmax > high_thresh

    fig = plt.figure(figsize=(12, 14))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each surface with its wistia_colors color
    legend_elements = []
    for i, (hhi_df, color) in enumerate(zip(hhi_dfs, surface_colors)):
        pivot = hhi_df.pivot(
            index="mtd",
            columns="mid",
            values="HHI",
        ).sort_index().sort_index(axis=1)

        x = pivot.columns.values
        y = pivot.index.values
        X, Y = np.meshgrid(x, y)
        Z = pivot.values

        # Plot surface with solid color
        surf = ax.plot_surface(X, Y, Z, color=color, edgecolor="k",
                             linewidth=0.3, alpha=0.7, zorder=10-i)

        # Add to legend
        legend_elements.append(plt.Rectangle((0,0), 1, 1, fc=color,
                                          label=f"rfm={sorted_rfm[i]:.2f}"))

    # Set labels and title
    ax.zaxis.labelpad = 15
    fig.subplots_adjust(right=0.85)
    ax.set_xlabel("Max individual import dependence")
    ax.set_ylabel("Max total import dependence")
    ax.text2D(0.95, 0.82, "HHI", transform=ax.transAxes, ha="center", fontsize=12)
    ax.set_title("HHI Sensitivity to Dependence Constraints (Multiple rfm Values)", fontsize=18)

    # Set z-axis limits
    z_floor = zmin - 0.05 * (zmax - zmin)
    ax.set_zlim(z_floor, zmax)

    # Add threshold lines
    thresholds = [(low_thresh, "black", "Threshold HHI > 1500 for moderately concentrated market")]
    if show_high:
        thresholds.append((high_thresh, "black", "Threshold HHI > 2500 for highly concentrated market"))

    xr = (x.min(), x.max())
    yr = (y.min(), y.max())

    for thresh, color, label in thresholds:
        ax.contour(X, Y, Z, levels=[thresh], zdir="z", offset=thresh,
                  colors=color, linewidths=2, linestyles="--")
        for y_edge in yr:
            ax.plot(xr, [y_edge, y_edge], [thresh, thresh],
                   color=color, lw=2, ls=":", alpha=1, zorder=2)
        for x_edge in xr:
            ax.plot([x_edge, x_edge], yr, [thresh, thresh],
                   color=color, lw=2, ls=":", alpha=1, zorder=2)

    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    plt.tight_layout()

    # Save with all rfm values in filename
    rfm_str = "_".join([f"{r*100:.0f}" for r in sorted_rfm])
    fig.savefig(os.path.join(output_path, "figures", f"hhi_multi_rfm_{rfm_str}.png"))
    plt.close('all')
    return fig, ax

def plot_cost_sens_multi(cost_dfs, rfm_values, currency_unit="€", raw_unit_scale=1e9):
    """
    Plot multiple 3D surfaces showing total_system_costs as a function of
    max_total_dependence and max_indiv_dependence for multiple rfm values.
    """
    # Sort rfm values and select colors from wistia_colors
    sorted_rfm = sorted(rfm_values)
    n_rfm = len(sorted_rfm)
    color_indices = [int(i * 29 / max(1, n_rfm-1)) for i in range(n_rfm)]
    surface_colors = [wistia_colors[i] for i in color_indices]

    # Find global min/max for z-axis
    all_z = []
    for cost_df in cost_dfs:
        pivot = cost_df.pivot(index="mtd", columns="mid", values="total_system_costs")
        all_z.extend((pivot.values/raw_unit_scale).flatten())
    zmin, zmax = np.nanmin(all_z), np.nanmax(all_z)

    fig = plt.figure(figsize=(12, 14))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each surface with its wistia_colors color
    legend_elements = []
    for i, (cost_df, color) in enumerate(zip(cost_dfs, surface_colors)):
        pivot = cost_df.pivot(
            index="mtd",
            columns="mid",
            values="total_system_costs",
        ).sort_index().sort_index(axis=1)

        x = pivot.columns.values
        y = pivot.index.values
        X, Y = np.meshgrid(x, y)
        Z = pivot.values / raw_unit_scale

        surf = ax.plot_surface(X, Y, Z, color=color, edgecolor="k",
                             linewidth=0.3, alpha=0.7, zorder=10-i)

        # Add to legend
        legend_elements.append(plt.Rectangle((0,0), 1, 1, fc=color,
                                          label=f"rfm={sorted_rfm[i]:.2f}"))

    # Set labels and title
    ax.zaxis.labelpad = 15
    fig.subplots_adjust(right=0.85)
    ax.set_xlabel("Max individual import dependence")
    ax.set_ylabel("Max total import dependence")
    ax.text2D(0.95, 0.82, f"Total system\ncosts [bn {currency_unit}]",
              transform=ax.transAxes, ha="center", fontsize=12)
    ax.set_title(f"Total System Cost Sensitivity (Multiple rfm Values) [bn {currency_unit}]",
                 fontsize=18)

    # Set z-axis limits
    z_floor = zmin - 0.05 * (zmax - zmin)
    ax.set_zlim(z_floor, zmax)

    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    plt.tight_layout()

    # Save with all rfm values in filename
    rfm_str = "_".join([f"{r*100:.0f}" for r in sorted_rfm])
    fig.savefig(os.path.join(output_path, "figures", f"cost_multi_rfm_{rfm_str}.png"))
    plt.close('all')
    return fig, ax

def _curly_brace_path(x, y0, y1, width, q=0.6):
    """
    Build an SVG path string for a vertical curly brace between y0 and y1,
    anchored at horizontal position x, bulging outward by `width`.
    """
    ym = (y0 + y1) / 2
    return (
        f"M {x},{y0} "
        f"C {x+width*q},{y0} {x+width},{y0+(ym-y0)*(1-q)} {x+width},{ym} "
        f"C {x+width},{y1-(y1-ym)*(1-q)} {x+width*q},{y1} {x},{y1}"
    )

# def plot_trade_off_curve(hhi_df, cost_df, hhi_threshold=1500):
#     """
#     Merge hhi_df and cost_df, identify the trade of curve frontier (minimizing
#     both HHI and total_system_costs), and plot it interactively with
#     Plotly. Highlights the HHI concentration threshold as a low-risk
#     "zone", marks the best option within the low-concentration zone in
#     red with a label of its dependence settings, and annotates the cost
#     premium of staying in that zone versus the global cost minimum via
#     a curly brace on the right side of the plot.

#     Returns fig.
#     """
#     df = hhi_df.merge(cost_df, on=["mtd", "mid"]).reset_index(drop=True)

#     # --- identify trade-of-efficient front points ---
#     costs = df[["total_system_costs", "HHI"]].values
#     n = costs.shape[0]
#     is_efficient = np.ones(n, dtype=bool)
#     for i in range(n):
#         dominating = (
#             np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
#         )
#         if np.any(dominating):
#             is_efficient[i] = False
#     df["lies_on_front"] = is_efficient

#     efficient = df[df["lies_on_front"]].sort_values("HHI").reset_index(drop=True)
#     efficient["cost_bn"] = efficient["total_system_costs"] / 1e9
#     df["cost_bn"] = df["total_system_costs"] / 1e9

#     # --- select best option: lowest cost among frontier points within the
#     #     low-concentration zone (HHI <= threshold); fallback to the
#     #     frontier point closest to the threshold if none qualify ---
#     within_zone = efficient[efficient["HHI"] <= hhi_threshold]
#     if not within_zone.empty:
#         best = within_zone.loc[within_zone["cost_bn"].idxmin()]
#     else:
#         best = efficient.loc[(efficient["HHI"] - hhi_threshold).abs().idxmin()]

#     global_min_cost = df["cost_bn"].min()
#     cost_premium = best["cost_bn"] - global_min_cost
#     cost_premium_pct = cost_premium / global_min_cost * 100

#     fig = go.Figure()

#     # --- transparent blue "safe zone" for HHI <= threshold ---
#     fig.add_vrect(
#         x0=df["HHI"].min() - 50, x1=hhi_threshold,
#         fillcolor="blue", opacity=0.08, line_width=0,
#         annotation_text="Low concentration zone", annotation_position="top left"
#     )

#     # --- vertical threshold line ---
#     fig.add_vline(
#         x=hhi_threshold, line_width=2, line_dash="dash", line_color="blue",
#         annotation_text=f"HHI threshold ({hhi_threshold})", annotation_position="top"
#     )

#     # --- horizontal line at the minimum optimal cost within the zone ---
#     fig.add_hline(
#         y=best["cost_bn"], line_width=2, line_dash="dash", line_color="blue",
#         annotation_text=f"Min. cost within zone ({best['cost_bn']:.2f} bn €)",
#         annotation_position="bottom left"
#     )

#     # --- smoothed trend/approximation curve: quadratic fit (trend, not
#     #     an interpolation through every point) ---
#     if len(efficient) >= 3:
#         coeffs = np.polyfit(efficient["HHI"], efficient["cost_bn"], deg=2)
#         trend = np.poly1d(coeffs)
#         x_smooth = np.linspace(efficient["HHI"].min(), efficient["HHI"].max(), 200)
#         y_smooth = trend(x_smooth)
#         fig.add_trace(go.Scatter(
#             x=x_smooth, y=y_smooth, mode="lines",
#             line=dict(color="lightgrey", dash="dot", width=2),
#             name="Trend approximation", hoverinfo="skip"
#         ))

#     # --- Trade of curve ---
#     fig.add_trace(go.Scatter(
#         x=efficient["HHI"], y=efficient["cost_bn"], mode="lines+markers",
#         line=dict(color="black", width=2),
#         marker=dict(size=8, color="black"),
#         name="Trade of curve",
#         text=efficient[["mtd", "mid"]].apply(lambda row: f"Max Total: {row['mtd']}, Max Indiv: {row['mid']}", axis=1)
#     ))

#     # --- per-point labels (disabled) ---
#     # for row in efficient.itertuples():
#     #     fig.add_annotation(
#     #         x=row.HHI, y=row.cost_bn,
#     #         text=f"({row.mtd:.1f}, {row.mid:.1f})",
#     #         showarrow=False,
#     #         yshift=36,
#     #         font=dict(family="Times New Roman", size=11, color="black")
#     #     )

#     --- dominated points (disabled) ---
#     fig.add_trace(go.Scatter(
#         x=dominated["HHI"], y=dominated["total_system_costs"] / 1e9, mode="markers",
#         marker=dict(size=6, color="lightgray"),
#         name="Dominated"
#     ))

#     # --- best option, highlighted as a larger red dot, with a clear label ---
#     fig.add_trace(go.Scatter(
#         x=[best["HHI"]], y=[best["cost_bn"]], mode="markers",
#         marker=dict(size=16, color="red", symbol="circle"),
#         name="Selected optimum",
#         text=f"Max Total: {best['mtd']}, Max Indiv: {best['mid']}"
#     ))
#     fig.add_annotation(
#         x=best["HHI"], y=best["cost_bn"],
#         text=f"{best['mtd']*100:.0f}% total dependence,<br>{best['mid']*100:.0f}% individual dependence",
#         showarrow=True, arrowhead=2, ax=60, ay=-40,
#         font=dict(family="Times New Roman", size=15, color="black"),
#         bgcolor="white", bordercolor="red", borderwidth=1
#     )

#     # --- curly brace on the right side, under the legend:
#     #     global min cost -> min cost within zone ---
#     brace_x = 1.06    # paper coords, just right of the axis
#     brace_width = 0.025
#     fig.add_shape(
#         type="path",
#         xref="paper", yref="y",
#         path=_curly_brace_path(brace_x, global_min_cost, best["cost_bn"], brace_width),
#         line=dict(color="black", width=1.5),
#     )
#     fig.add_annotation(
#         xref="paper", yref="y",
#         x=brace_x + brace_width * 1.6,
#         y=(global_min_cost + best["cost_bn"]) / 2,
#         text=f"+{cost_premium:.2f} bn €<br>(+{cost_premium_pct:.1f}%)",
#         showarrow=False,
#         textangle=-90,
#         xanchor="left",
#         font=dict(family="Times New Roman", size=15, color="black")
#     )

#     # --- global formatting, matched to your matplotlib rcParams ---
#     fig.update_layout(
#         title="Total System Costs vs. Import Concentration",
#         xaxis_title="HHI (market concentration)",
#         yaxis_title="Total system costs [bn €]",
#         template="plotly_white",
#         font=dict(family="Times New Roman", color="black", size=20),
#         xaxis=dict(title_font=dict(size=15), tickfont=dict(size=15)),
#         yaxis=dict(title_font=dict(size=15), tickfont=dict(size=15)),
#         legend=dict(font=dict(size=15), x=1.0, y=1.0, xanchor="left", yanchor="top"),
#         margin=dict(r=160),  # extra room on the right for legend + brace + label
#         width=1000, height=650
#     )

#     plt.ioff()
#     fig.write_image(os.path.join(output_path, "figures", f"trade_of_curve_{n}n_{commodity}.png"), scale=4)
#     return fig

def plot_trade_off_curve_multi(hhi_dfs, cost_dfs, rfm_values, hhi_threshold=1500):
    """
    Plot trade-off curves for multiple rfm values in a single interactive plot.

    hhi_dfs: list of DataFrames with columns ["mtd", "mid", "HHI"]
    cost_dfs: list of DataFrames with columns ["mtd", "mid", "total_system_costs"]
    rfm_values: list of rfm values corresponding to each DataFrame pair
    hhi_threshold: HHI threshold for the low-concentration zone
    """
    # Sort rfm values and select colors from wistia_colors
    sorted_rfm = sorted(rfm_values)
    n_rfm = len(sorted_rfm)
    color_indices = [int(i * 29 / max(1, n_rfm-1)) for i in range(n_rfm)]
    curve_colors = [wistia_colors[i] for i in color_indices]

    fig = go.Figure()

    # --- Plot trade-off curves for each rfm ---
    for idx, (hhi_df, cost_df, color) in enumerate(zip(hhi_dfs, cost_dfs, curve_colors)):
        df = hhi_df.merge(cost_df, on=["mtd", "mid"]).reset_index(drop=True)

        # Identify trade-off curve frontier points
        costs = df[["total_system_costs", "HHI"]].values
        n = costs.shape[0]
        is_efficient = np.ones(n, dtype=bool)
        for i in range(n):
            dominating = (
                np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
            )
            if np.any(dominating):
                is_efficient[i] = False
        df["lies_on_front"] = is_efficient

        efficient = df[df["lies_on_front"]].sort_values("HHI").reset_index(drop=True)
        efficient["cost_bn"] = efficient["total_system_costs"] / 1e9
        df["cost_bn"] = df["total_system_costs"] / 1e9

        # Select best option within low-concentration zone
        within_zone = efficient[efficient["HHI"] <= hhi_threshold]
        if not within_zone.empty:
            best = within_zone.loc[within_zone["cost_bn"].idxmin()]
        else:
            best = efficient.loc[(efficient["HHI"] - hhi_threshold).abs().idxmin()]

        # Add trade-off curve
        fig.add_trace(go.Scatter(
            x=efficient["HHI"],
            y=efficient["cost_bn"],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color),
            name=f"rfm={sorted_rfm[idx]:.2f}",
            text=efficient[["mtd", "mid"]].apply(
                lambda row: f"Max Total: {row['mtd']}, Max Indiv: {row['mid']}", axis=1
            )
        ))

        # Add best option marker (only for the first rfm to avoid clutter)
        if idx == 0:
            global_min_cost = df["cost_bn"].min()
            cost_premium = best["cost_bn"] - global_min_cost
            cost_premium_pct = cost_premium / global_min_cost * 100

            fig.add_trace(go.Scatter(
                x=[best["HHI"]],
                y=[best["cost_bn"]],
                mode="markers",
                marker=dict(size=16, color="red", symbol="circle"),
                name="Selected optimum",
                text=f"Max Total: {best['mtd']}, Max Indiv: {best['mid']}"
            ))
            fig.add_annotation(
                x=best["HHI"],
                y=best["cost_bn"],
                text=f"{best['mtd']*100:.0f}% total dependence,<br>{best['mid']*100:.0f}% individual dependence",
                showarrow=True,
                arrowhead=2,
                ax=60,
                ay=-40,
                font=dict(family="Times New Roman", size=15, color="black"),
                bgcolor="white",
                bordercolor="red",
                borderwidth=1
            )

            # Safe zone
            fig.add_vrect(
                x0=df["HHI"].min() - 50,
                x1=hhi_threshold,
                fillcolor="blue",
                opacity=0.08,
                line_width=0,
                annotation_text="Low concentration zone",
                annotation_position="top left"
            )

            # Threshold line
            fig.add_vline(
                x=hhi_threshold,
                line_width=2,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"HHI threshold ({hhi_threshold})",
                annotation_position="top"
            )

            # Horizontal line at best option
            fig.add_hline(
                y=best["cost_bn"],
                line_width=2,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Min. cost within zone ({best['cost_bn']:.2f} bn €)",
                annotation_position="bottom left"
            )

            # Curly brace and cost premium annotation
            brace_x = 1.06
            brace_width = 0.025
            fig.add_shape(
                type="path",
                xref="paper", yref="y",
                path=_curly_brace_path(brace_x, global_min_cost, best["cost_bn"], brace_width),
                line=dict(color="black", width=1.5),
            )
            fig.add_annotation(
                xref="paper", yref="y",
                x=brace_x + brace_width * 1.6,
                y=(global_min_cost + best["cost_bn"]) / 2,
                text=f"+{cost_premium:.2f} bn €<br>(+{cost_premium_pct:.1f}%)",
                showarrow=False,
                textangle=-90,
                xanchor="left",
                font=dict(family="Times New Roman", size=15, color="black")
            )

    # --- Global formatting ---
    fig.update_layout(
        title="Total System Costs vs. Import Concentration (Multiple rfm Values)",
        xaxis_title="HHI (market concentration)",
        yaxis_title="Total system costs [bn €]",
        template="plotly_white",
        font=dict(family="Times New Roman", color="black", size=20),
        xaxis=dict(title_font=dict(size=15), tickfont=dict(size=15)),
        yaxis=dict(title_font=dict(size=15), tickfont=dict(size=15)),
        legend=dict(
            font=dict(size=15),
            x=1.0,
            y=1.0,
            xanchor="left",
            yanchor="top"
        ),
        margin=dict(r=160),
        width=1000,
        height=650
    )

    plt.ioff()
    rfm_str = "_".join([f"{r*100:.0f}" for r in sorted_rfm])
    fig.write_image(os.path.join(output_path, "figures", f"trade_of_curve_multi_rfm_{rfm_str}.png"), scale=4)
    return fig

def get_transport_flows(solution):
    # squeeze singleton dims, sum over supply_step to get total flow per edge
    transport_total = (
        solution["v_transport"]
        .sum(dim="supply_step")
        .sel(commodity="h2", scenario="Base")  # or .squeeze() if truly singleton
    )

    transport_df = transport_total.to_dataframe(name="transport_amount").reset_index()
    # columns: region1, region2, transport_amount

    # drop self-loops and zero flows to keep things light
    transport_flows_df = transport_df[
        (transport_df["region1"] != transport_df["region2"]) &
        (transport_df["transport_amount"] > 1e-9)
    ]
    return transport_flows_df

def load_transport_paths(output_path, transport_flows_df, commodity="h2", scenario="Base"):
    ### read paths if they exist
    file_path = os.path.join(output_path, "model", "paths.csv")

    if os.path.exists(file_path):
        paths_df = pd.read_csv(file_path)
        paths_df["path_geometry"] = paths_df["path_geometry"].apply(
            lambda x: wkt.loads(x) if isinstance(x, str) else None
        )
        # raw coordinates from the Dijkstra step are already WGS84 lon/lat degrees,
        # so label them as such directly rather than mislabeling as a projected CRS
        paths_gdf = gpd.GeoDataFrame(paths_df, geometry="path_geometry", crs=default_epsg_1)
        paths_gdf["length"] = paths_gdf.length
        print("DataFrame loaded from file.")
    else:
        paths_gdf = pd.DataFrame()
        print("No existing paths file found.")
        return paths_gdf

    # --- filter to the relevant commodity/scenario slice before indexing ---
    flows = transport_flows_df
    if "commodity" in flows.columns:
        flows = flows[flows["commodity"] == commodity]
    if "scenario" in flows.columns:
        flows = flows[flows["scenario"] == scenario]

    dupe_count = flows.duplicated(subset=["region1", "region2"]).sum()
    if dupe_count > 0:
        print(f"Warning: {dupe_count} duplicate (region1, region2) rows after filtering — "
              f"aggregating with sum(). Check upstream if this is unexpected.")
        flows = flows.groupby(["region1", "region2"], as_index=False)["transport_amount"].sum()

    flow_lookup = flows.set_index(["region1", "region2"])["transport_amount"]

    # --- diagnostic: check key overlap before merging ---
    paths_pairs = set(zip(paths_gdf["source"], paths_gdf["sink"]))
    flow_pairs = set(zip(flows["region1"], flows["region2"]))

    direct_overlap = paths_pairs & flow_pairs
    reversed_overlap = paths_pairs & {(b, a) for a, b in flow_pairs}

    paths_countries = set(paths_gdf["source"]) | set(paths_gdf["sink"])
    flow_countries = set(flows["region1"]) | set(flows["region2"])

    print(f"[load_transport_paths diagnostic]")
    print(f"  paths_gdf pairs: {len(paths_pairs)} | transport_flows_df pairs: {len(flow_pairs)}")
    print(f"  direct overlap: {len(direct_overlap)} | reversed overlap: {len(reversed_overlap)}")
    only_in_paths = paths_countries - flow_countries
    only_in_flows = flow_countries - paths_countries
    if only_in_paths:
        print(f"  countries only in paths_gdf ({len(only_in_paths)}): {sorted(only_in_paths)[:10]}...")
    if only_in_flows:
        print(f"  countries only in transport_flows_df ({len(only_in_flows)}): {sorted(only_in_flows)[:10]}...")
    if len(direct_overlap) == 0 and len(reversed_overlap) == 0:
        print("  WARNING: zero overlap between paths_gdf and transport_flows_df keys — "
              "no flows will be matched. Check region coverage / naming above.")

    # --- build directional lookup ---
    def get_net_flow(row):
        fwd = flow_lookup.get((row["source"], row["sink"]), 0.0)
        bwd = flow_lookup.get((row["sink"], row["source"]), 0.0)
        return fwd - bwd  # positive = net flow source->sink

    paths_gdf["net_transport"] = paths_gdf.apply(get_net_flow, axis=1)
    paths_gdf["gross_transport"] = paths_gdf.apply(
        lambda r: flow_lookup.get((r["source"], r["sink"]), 0.0)
                + flow_lookup.get((r["sink"], r["source"]), 0.0),
        axis=1
    )
    return paths_gdf

def plot_transport_flows_map(paths_gdf, output_path, marginal_costs_df, mtd=0.8, mid=0.2, rfm=1,
                              projection_type="natural earth"):
    FONT_FAMILY = "Times New Roman"
    FONT_COLOR = "black"
    LABEL_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 12

    fig = go.Figure()

    # =========================================================
    # 0. Choropleth layer: marginal costs by country (bottom layer)
    # =========================================================
    mc_df = marginal_costs_df.copy()
    mc_df["iso3"] = mc_df["region"].apply(lambda x: x.split("-")[-1])

    mc_vmin, mc_vmax = mc_df["marginal_cost"].min(), mc_df["marginal_cost"].max()

    fig.add_trace(go.Choropleth(
        locations=mc_df["iso3"],
        z=mc_df["marginal_cost"],
        locationmode="ISO-3",
        colorscale="YlGn",
        marker_line_color="white",
        marker_line_width=0.5,
        zmin=mc_vmin, zmax=mc_vmax,
        # marginal cost colorbar
        colorbar=dict(
            title=dict(text="", font=dict(family=FONT_FAMILY, size=LABEL_SIZE, color=FONT_COLOR)),
            len=0.4, thickness=10,          # was len=0.5, thickness=15
            x=0.9, y=0.5, xanchor="left",
            tickvals=[mc_vmin, mc_vmax],
            ticktext=[f"{mc_vmin:.1f}", f"{mc_vmax:.1f}"],
            tickfont=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR),
    ),
        name="Marginal cost",
        showlegend=False,
    ))
    choropleth_trace_idx = len(fig.data) - 1

    # --- identify country nodes + coordinates from path_geometry endpoints ---
    def extract_node_coords(gdf):
        node_coords = {}
        for _, row in gdf.iterrows():
            geom = row["path_geometry"]
            first_line = geom.geoms[0] if geom.geom_type == "MultiLineString" else geom
            last_line = geom.geoms[-1] if geom.geom_type == "MultiLineString" else geom
            src_lon, src_lat = first_line.coords[0][0], first_line.coords[0][1]
            snk_lon, snk_lat = last_line.coords[-1][0], last_line.coords[-1][1]
            node_coords.setdefault(row["source"], (src_lon, src_lat))
            node_coords.setdefault(row["sink"], (snk_lon, snk_lat))
        return node_coords

    node_coords = extract_node_coords(paths_gdf)
    node_names_all = sorted(node_coords.keys())

    fig.add_trace(go.Scattergeo(
        lon=[node_coords[n][0] for n in node_names_all],
        lat=[node_coords[n][1] for n in node_names_all],
        mode="markers",
        marker=dict(size=6, color="darkblue"),
        name="Nodes",
        text=node_names_all,
        hoverinfo="text",
        showlegend=False,
    ))
    nodes_trace_idx = len(fig.data) - 1

    # =========================================================
    # Log-scaled route lines, colored by turquoise shade = volume
    # (direction-based red/blue coloring removed — the from/to accounting
    # behind net_transport was unreliable, so this now shows magnitude only,
    # via both line thickness and color shade, no directional claim)
    # =========================================================
    def add_volume_colored_lines(
        gdf, fig,
        amount_col="gross_transport",
        width_range=(1, 6), opacity=0.8,
        unit_divisor=1e6,  # MWh -> TWh for display only
    ):
        amounts = gdf[amount_col].values.astype(float)

        log_amounts = np.log1p(amounts)
        amin, amax = np.nanmin(log_amounts), np.nanmax(log_amounts)
        norm_amounts = (log_amounts - amin) / (amax - amin + 1e-12)

        wmin, wmax = width_range
        widths = wmin + (wmax - wmin) * norm_amounts

        turquoise_colors = pc.sample_colorscale("Teal", norm_amounts.tolist())
        turquoise_colors = pc.sample_colorscale("Rainbow", norm_amounts.tolist())

        route_trace_pairs = []
        for row_idx, (geom, w, color) in enumerate(zip(gdf.geometry, widths, turquoise_colors)):
            source = gdf["source"].iloc[row_idx]
            sink = gdf["sink"].iloc[row_idx]
            amount_twh = amounts[row_idx] / unit_divisor

            lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
            for line in lines:
                lon, lat = line.xy
                fig.add_trace(go.Scattergeo(
                    lon=list(lon), lat=list(lat),
                    mode="lines",
                    line=dict(width=w, color=color),
                    showlegend=False,
                    legendgroup="Transport volume",
                    hoverinfo="text",
                    text=f"{source} — {sink}<br>volume: {amount_twh:.3g} TWh",
                    opacity=opacity,
                ))
                route_trace_pairs.append((len(fig.data) - 1, source, sink))

        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        tick_orig = np.nanquantile(amounts, quantiles)
        tick_vals_transformed = np.log1p(tick_orig)
        tick_text = [f"{v / unit_divisor:.2g}" for v in tick_orig]

        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None],
            mode="markers",
            marker=dict(
                size=0.1,
                color=[amin, amax],
                colorscale="Rainbow", #colorscale="Teal",
                cmin=amin, cmax=amax,
                # transport volume colorbar
                colorbar=dict(
                    title=dict(text="", font=dict(family=FONT_FAMILY, size=LABEL_SIZE, color=FONT_COLOR)),
                    len=0.4, thickness=10,
                    x=0.98, y=0.5, xanchor="left",
                    tickvals=tick_vals_transformed,
                    ticktext=tick_text,
                    tickfont=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR),
                ),
                showscale=True,
            ),
            showlegend=False,
            hoverinfo="skip",
        ))
        legend_trace_idx = len(fig.data) - 1

        return route_trace_pairs, [legend_trace_idx]

    flow_gdf = paths_gdf[paths_gdf["gross_transport"] > 1e-9].copy().reset_index(drop=True)

    route_trace_pairs = []
    legend_trace_idxs = []

    if flow_gdf.empty:
        print("No routes with nonzero transport flow — nothing to plot.")
    else:
        route_trace_pairs, legend_trace_idxs = add_volume_colored_lines(flow_gdf, fig)

    # =========================================================
    # Vertical colorbar labels — now just 2: marginal cost + transport volume
    # =========================================================
    colorbar_x_positions = [0.9, 0.98]  # was [0.80, 0.88]
    colorbar_labels = ["Marginal cost (€/MWh)", "Transport volume (TWh)"]

    annotations = []
    for x_pos, label in zip(colorbar_x_positions, colorbar_labels):
        annotations.append(dict(
            text=label,
            xref="paper", yref="paper",
            x=x_pos - 0.03, y=0.5,   # was x_pos - 0.045 — labels sit closer to their bars now
            xanchor="center", yanchor="middle",
            textangle=-90,
            showarrow=False,
            font=dict(family=FONT_FAMILY, size=LABEL_SIZE - 1, color=FONT_COLOR),  # slightly smaller
        ))

    # --- dropdown menu to filter routes by country node ---
    always_visible_idx = {nodes_trace_idx, choropleth_trace_idx, *legend_trace_idxs}
    n_traces = len(fig.data)

    def visibility_for_node(selected_node):
        vis = [False] * n_traces
        for idx in always_visible_idx:
            vis[idx] = True
        if selected_node is None:
            for idx, source, sink in route_trace_pairs:
                vis[idx] = True
        else:
            for idx, source, sink in route_trace_pairs:
                if source == selected_node or sink == selected_node:
                    vis[idx] = True
        return vis

    nodes_with_flow = sorted(set(flow_gdf["source"]).union(flow_gdf["sink"])) if not flow_gdf.empty else []

    buttons = [
        dict(label="Show all routes", method="update", args=[{"visible": visibility_for_node(None)}])
    ]
    buttons += [
        dict(label=node, method="update", args=[{"visible": visibility_for_node(node)}])
        for node in nodes_with_flow
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons, direction="down",
                x=0.01, y=0.9, xanchor="left", yanchor="top",
                showactive=True, pad=dict(t=0, r=0),
                font=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR),
            )
        ],
        annotations=annotations,
    )

    # --- tighter geo domain: colorbars need much less width than 28% ---
    fig.update_geos(
        projection_type=projection_type,
        showland=True, landcolor="lightgrey",
        showocean=True, oceancolor="lightblue",
        showcountries=True, countrycolor="white",
        showcoastlines=True, coastlinecolor="white",
        domain=dict(x=[0, 0.85], y=[0, 1]),  # was 0.72 — map now uses most of the canvas
    )

    fig.update_layout(
        title=dict(
            text="Nodes, Terminals, and Connections — select a country to filter routes",
            font=dict(family=FONT_FAMILY, size=LABEL_SIZE + 2, color=FONT_COLOR),
            y=0.98, yanchor="top",
            automargin=True,
        ),
        height=700, width=1300,
        font=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR),
        showlegend=False,
        margin=dict(t=40, b=0, l=20, r=90),
    )

    fig.write_image(os.path.join(output_path, "figures", "transport_flow", f"transport_flow_map_{mtd*100:.0f}_{mid*100:.0f}_rfm{rfm*100:.0f}.png"), scale=2)
    fig.write_html(os.path.join(output_path, "figures", "transport_flow", f"transport_flow_map_{mtd*100:.0f}_{mid*100:.0f}_rfm{rfm*100:.0f}.html"), include_plotlyjs='cdn')

    plt.ioff()
    return

def marginals_to_df(marginals, commodity="h2", scenario="Base"):
    mc_df = (
        marginals
        .sel(commodity=commodity, scenario=scenario)
        .to_dataframe(name="marginal_cost")
        .reset_index()
    )
    # drop the now-redundant commodity/scenario columns (constant after .sel)
    mc_df = mc_df[["region", "marginal_cost"]]
    return mc_df

def load_market_results_in_df(runs, rfm_sweep):
    ### first we now load the respective global data from the model runs loaded in the sweep ###
    # Initialize an empty list to store the data
    data_list = []

    # Loop through each rfm in the rfm_sweep
    for rfm in rfm_sweep:
        # Loop through each country in the countries_of_interest
        for country in countries_of_interest:
            # Extract the domestic production volume (v_supply_segment)
            domestic_prod = runs[rfm]["solution"]["v_supply_segment"].sel(region=country, commodity="h2", scenario="Base").sum(dim="supply_step").values

            # Extract the demand (demand)
            demand = runs[rfm]["data_1D"]["demand"].sel(region=country, commodity="h2", scenario="Base", supply_step="ds_p_0").values

            # Extract the import volumes (v_transport) and source countries
            imports_data = runs[rfm]["solution"]["v_transport"].sel(region2=country, commodity="h2", scenario="Base")
            imports = imports_data.sum(dim="supply_step").values
            source_countries = imports_data["region1"].values

            # Loop through each source country and its corresponding import volume
            for source_country, import_volume in zip(source_countries, imports):
                if import_volume != 0:  # Only include non-zero imports
                    # Create a dictionary with the extracted data
                    data_dict = {
                        "RFM": rfm,
                        "Region": country,
                        "Source Country": source_country,
                        "Domestic Production": domestic_prod,
                        "Demand": demand,
                        "Imports": import_volume
                    }

                    # Append the dictionary to the list
                    data_list.append(data_dict)

    # Convert the list to a pandas DataFrame
    rfm_sweep_results_df = pd.DataFrame(data_list)
    
    # Display the DataFrame
    print(rfm_sweep_results_df)
    return rfm_sweep_results_df

def compute_export_decomposition(solution, data_2D, country, commodity="h2", scenario="Base"):
    """Exports FROM `country` TO each destination partner, summed over supply_step."""
    transport = solution["v_transport"].sel(region1=country, commodity=commodity, scenario=scenario)
    export_df = transport.sum(dim="supply_step").to_dataframe(name="amount").reset_index()
    export_df = export_df[(export_df["region2"] != country) & (export_df["amount"] > 1e-6)]
    return export_df.set_index("region2")["amount"]

def get_domestic_and_demand(run, country, commodity="h2", scenario="Base"):
    """Fetch a country's raw (gross) production and demand directly from the
    run's own data, independent of whether it has any import relationships."""
    data_1D = run["data_1D"]
    solution = run["solution"]

    domestic_prod = float(
        solution["v_supply_segment"].sel(region=country, commodity=commodity, scenario=scenario).sum()
    )
    demand = float(
        data_1D["demand"].sel(region=country, commodity=commodity, scenario=scenario,
                                supply_step=base_step_param)
    )
    return domestic_prod, demand

def plot_rfm_sens_selec_country(rfm_sweep_results_df, rfm_sweep_runs,
                                 commodity="h2", scenario="Base"):
    FONT_FAMILY = "Times New Roman"
    
    mtd = rfm_sweep_runs[1]["meta"]["max_total_dependence_rel"]
    mid = rfm_sweep_runs[1]["meta"]["max_indiv_dependence_rel"]

    fig = make_subplots(
        rows=1,
        cols=len(countries_of_interest),
        subplot_titles=countries_of_interest,
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    # --- import-side colors: source country -> shade of orange/wistia ---
    source_country_colors = {}
    for i, source_country in enumerate(rfm_sweep_results_df["Source Country"].unique()):
        source_country_colors[source_country] = wistia_colors[i % len(wistia_colors)]

    # --- export-side colors: destination country -> shade of teal ---
    export_destinations = set()
    export_by_country_rfm = {}
    for country in countries_of_interest:
        for rfm in rfm_sweep:
            run = rfm_sweep_runs.get(rfm)
            if run is None:
                continue
            exports = compute_export_decomposition(run["solution"], run["data_2D"], country,
                                                     commodity=commodity, scenario=scenario)
            export_by_country_rfm[(country, rfm)] = exports
            export_destinations.update(exports.index)

    export_destinations = sorted(export_destinations)
    teal_colors = pc.sample_colorscale(
        "Teal", [i / max(1, len(export_destinations) - 1) for i in range(len(export_destinations))]
    ) if export_destinations else ["teal"]
    export_dest_colors = {dest: teal_colors[i % len(teal_colors)] for i, dest in enumerate(export_destinations)}

    demand_values = {}

    for i, country in enumerate(countries_of_interest):
        country_data = rfm_sweep_results_df[rfm_sweep_results_df["Region"] == country]

        for rfm in rfm_sweep:
            run = rfm_sweep_runs.get(rfm)
            if run is None:
                continue

            raw_domestic_prod, demand = get_domestic_and_demand(run, country,
                                                                  commodity=commodity, scenario=scenario)
            demand_values[country] = demand

            rfm_data = country_data[country_data["RFM"] == rfm] if not country_data.empty else pd.DataFrame()
            imports = rfm_data.groupby("Source Country")["Imports"].sum() if not rfm_data.empty else pd.Series(dtype=float)

            exports = export_by_country_rfm.get((country, rfm), pd.Series(dtype=float))
            total_exports = float(exports.sum())
            domestic_retained = max(0.0, raw_domestic_prod - total_exports)

            fig.add_trace(
                go.Bar(
                    x=[rfm], y=[domestic_retained], name="Domestic Production",
                    marker_color="turquoise", text=["Domestic"], textposition="auto",
                    hoverinfo="text", showlegend=False,
                ),
                row=1, col=i + 1,
            )
            if not exports.empty:
                for dest_country, export_volume in exports.items():
                    fig.add_trace(
                        go.Bar(
                            x=[rfm], y=[export_volume], name=f"Export: {dest_country}",
                            marker_color=export_dest_colors.get(dest_country, teal_colors[0]),
                            text=[dest_country], textposition="auto", hoverinfo="text",
                            showlegend=False,
                        ),
                        row=1, col=i + 1,
                    )
            if not imports.empty:
                for source_country, import_volume in imports.items():
                    fig.add_trace(
                        go.Bar(
                            x=[rfm], y=[import_volume], name=source_country,
                            marker_color=source_country_colors.get(source_country, wistia_colors[0]),
                            text=[source_country], textposition="auto", hoverinfo="text",
                            showlegend=False,
                        ),
                        row=1, col=i + 1,
                    )

        demand = demand_values.get(country, 0)
        fig.add_trace(
            go.Scatter(
                x=rfm_sweep, y=[demand] * len(rfm_sweep), mode="lines",
                line=dict(color="red", width=3), name="Demand", showlegend=False,
            ),
            row=1, col=i + 1,
        )

    # --- legend: Domestic production + Domestic demand, same row as the colorbars ---
    FOOTER_Y = -0.1

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", marker=dict(size=10, color="turquoise"),
        showlegend=True, name="Domestic Production",
        legend="legend1",  # just a reference string here
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", line=dict(color="red", width=3),
        showlegend=True, name="Domestic Demand",
        legend="legend2",  # different reference string for the second legend
    ))

    fig.update_layout(
        legend1=dict(
            orientation="h", yanchor="middle", y=FOOTER_Y, xanchor="center", x=0.2,
            itemsizing="constant", font=dict(size=14),
        ),
        legend2=dict(
            orientation="h", yanchor="middle", y=FOOTER_Y, xanchor="center", x=0.4,
            itemsizing="constant", font=dict(size=14),
        ),
    )

    import_country_list = list(source_country_colors.keys())
    export_country_list = list(export_dest_colors.keys())

    if import_country_list:
        import_colorscale = pc.make_colorscale(list(source_country_colors.values()))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                size=0.1, color=[0, 1], colorscale=import_colorscale, cmin=0, cmax=1,
                showscale=True,
                colorbar=dict(
                    orientation="h", x=0.6, xanchor="center", y=FOOTER_Y, yanchor="middle",
                    len=0.16, thickness=10,
                    title=dict(text="Countries imported from", font=dict(family=FONT_FAMILY, size=14)),
                    tickvals=[],
                ),
            ),
            showlegend=False, hoverinfo="skip",
        ))

    if export_country_list:
        export_colorscale = pc.make_colorscale(list(export_dest_colors.values()))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                size=0.1, color=[0, 1], colorscale=export_colorscale, cmin=0, cmax=1,
                showscale=True,
                colorbar=dict(
                    orientation="h", x=0.8, xanchor="center", y=FOOTER_Y, yanchor="middle",
                    len=0.16, thickness=10,
                    title=dict(text="Countries exportet to", font=dict(family=FONT_FAMILY, size=14)),
                    tickvals=[],
                ),
            ),
            showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        title_text="Country-wise production, import and export for relationship factor magnitude (rfm) sensitivity",
        barmode="stack",
        bargap = 0.05,
        height=900, width=1800,
        font=dict(family=FONT_FAMILY, color="black"),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(tickfont=dict(color="black", size=12), title=dict(standoff=15, font=dict(size=14))),
        yaxis=dict(title_text="Volume [MWh]", tickfont=dict(color="black", size=12), title=dict(standoff=15, font=dict(size=14))),
        margin=dict(b=130),
    )

    for i in range(1, len(countries_of_interest) + 1):
        fig.update_xaxes(type="category", title_text="rfm", tickvals=rfm_sweep, row=1, col=i)

    os.makedirs(os.path.join(output_path, "figures"), exist_ok=True)
    fig.write_image(os.path.join(output_path, "figures", f"relationship_sensitivity_analysis_{mtd*100:.0f}_{mid*100:.0f}.png"), scale=4)
    fig.write_html(os.path.join(output_path, "figures", f"relationship_sensitivity_analysis_{mtd*100:.0f}_{mid*100:.0f}.html"), include_plotlyjs='cdn')

    return fig

def compute_country_trade_stats(solution, data_2D, rfm, commodity="h2", scenario="Base"):
    """Per-country trade stats for a single run, tagged with its rfm value.
    net_position: positive = net exporter, negative = net importer
    (i.e. exports - imports, matching the requested x-axis convention)."""
    transport = solution["v_transport"].sel(commodity=commodity, scenario=scenario)
    transport_by_route = transport.sum(dim="supply_step")

    self_loop_mask = xr.DataArray(
        data_2D.region1.values[:, None] == data_2D.region2.values[None, :],
        dims=["region1", "region2"],
        coords={"region1": data_2D.region1, "region2": data_2D.region2},
    )
    flows_df = transport_by_route.where(~self_loop_mask).to_dataframe(name="amount").reset_index()
    flows_df = flows_df.dropna(subset=["amount"])
    flows_df = flows_df[flows_df["amount"] > 1e-6]

    records = []
    for country in data_2D.region1.values:
        exports = flows_df[flows_df["region1"] == country]
        imports = flows_df[flows_df["region2"] == country]

        partners = set(exports["region2"]).union(set(imports["region1"]))
        n_relations = len(partners)

        total_exports = float(exports["amount"].sum())
        total_imports = float(imports["amount"].sum())
        net_position = total_exports - total_imports  # positive = net exporter, negative = net importer

        records.append({
            "rfm": rfm,
            "country": country,
            "n_relations": n_relations,
            "total_traded_quantity": total_exports + total_imports,
            "net_position": net_position,
        })

    return pd.DataFrame(records)

def compute_country_trade_stats_all_rfm(rfm_sweep_runs, commodity="h2", scenario="Base"):
    """Runs compute_country_trade_stats for every rfm in the sweep and concatenates."""
    all_stats = []
    for rfm, run in rfm_sweep_runs.items():
        stats = compute_country_trade_stats(run["solution"], run["data_2D"], rfm,
                                              commodity=commodity, scenario=scenario)
        all_stats.append(stats)
    mtd = rfm_sweep_runs[1]["meta"]["max_total_dependence_rel"]
    mid = rfm_sweep_runs[1]["meta"]["max_indiv_dependence_rel"]
    return pd.concat(all_stats, ignore_index=True), mtd, mid


def plot_country_trade_scatter(country_stats_df, output_path, mtd, mid, log_x):
    """Plotly scatter, all rfm sensitivities in one plot.
    x = net trade volume (exports - imports; positive = net exporter, negative = net importer)
    y = number of active trade relations
    color = rfm sensitivity run (sequential turquoise, darker = LOWER rfm)
    size = |net trade volume| (diameter)
    labels = country name shown once per country, at its highest-rfm point only
    log_x = symmetric log transform on the x-axis (handles negative values correctly,
            since a plain log scale can't). Set to False to revert to a linear x-axis.
    """
    FONT_FAMILY = "Times New Roman"
    FONT_COLOR = "black"
    LABEL_SIZE = 14
    TICK_SIZE = 12

    df = country_stats_df[country_stats_df["total_traded_quantity"] > 0].copy()

    df["abs_net_position"] = df["net_position"].abs()
    size_min, size_max = 6, 60
    max_abs = df["abs_net_position"].max()
    df["marker_size"] = size_min if max_abs == 0 else (
        size_min + (size_max - size_min) * (df["abs_net_position"] / max_abs)
    )

    rfm_values = sorted(df["rfm"].unique())

    # sequential turquoise shades: DARK -> light as rfm increases (reversed from before)
    turquoise_scale = ["#5fb8a8", "#45c9b1", "#2ea88f", "#17a589", "#0e6655", "#073b31"]
    n = len(rfm_values)
    if n == 1:
        color_map = {rfm_values[0]: turquoise_scale[0]}
    else:
        idxs = np.linspace(0, len(turquoise_scale) - 1, n)
        color_map = {rfm: turquoise_scale[int(round(i))] for rfm, i in zip(rfm_values, idxs)}

    max_rfm = max(rfm_values)

    # --- symmetric log transform for the x-axis, since values can be negative ---
    if log_x:
        df["x_plot"] = np.sign(df["net_position"]) * np.log10(1 + df["net_position"].abs())
    else:
        df["x_plot"] = df["net_position"]

    fig = go.Figure()
    for rfm in rfm_values:
        sub = df[df["rfm"] == rfm]

        fig.add_trace(go.Scatter(
            x=sub["x_plot"],
            y=sub["n_relations"],
            mode="markers+text",
            text=sub["country"],
            textposition="top center",
            textfont=dict(family=FONT_FAMILY, size=8, color=color_map[rfm]),
            marker=dict(size=sub["marker_size"], color=color_map[rfm], opacity=0.6,
                         line=dict(width=1, color=color_map[rfm])),
            name=f"rfm = {rfm}",
            customdata=sub["net_position"],
            hovertemplate="<b>" + sub["country"].astype(str) + "</b><br>Net trade volume: %{customdata:.3g}<br>"
                          "Relations: %{y}<br>rfm: " + str(rfm) + "<extra></extra>",
        ))

    fig.add_vline(x=0, line=dict(color="grey", width=1, dash="solid"))

    xaxis_config = dict(
        title=dict(text="Net trade volume (MWh)  ←  net importer   |   net exporter  →",
                    font=dict(family=FONT_FAMILY, size=LABEL_SIZE, color=FONT_COLOR)),
        tickfont=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR),
        showline=True, linecolor="black", gridcolor="lightgrey", zeroline=False,
    )

    if log_x:
        # manually generate readable tick labels on the transformed axis,
        # since Plotly's native log type can't be used with negative values
        max_val = df["net_position"].abs().max()
        if max_val > 0:
            order = int(np.ceil(np.log10(max_val)))
            tick_raw = [10**k for k in range(0, order + 1)]
            tick_vals = [0] + [np.sign(v) * np.log10(1 + abs(v)) for v in tick_raw + [-t for t in tick_raw]]
            tick_text = ["0"] + [f"{v:,.0f}" for v in tick_raw] + [f"-{v:,.0f}" for v in tick_raw]
            xaxis_config["tickvals"] = tick_vals
            xaxis_config["ticktext"] = tick_text

    fig.update_layout(
        title=dict(text="Trade relations vs. net trade volume across relationship factor magnitude (rfm) sensitivities"
                        + (" (symlog x-axis)" if log_x else ""),
                    font=dict(family=FONT_FAMILY, size=LABEL_SIZE + 2, color=FONT_COLOR)),
        xaxis=xaxis_config,
        yaxis=dict(title=dict(text="Number of active trade relations",
                                font=dict(family=FONT_FAMILY, size=LABEL_SIZE, color=FONT_COLOR)),
                    tickfont=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR)),
        font=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR),
        legend=dict(title=dict(text="Sensitivity run"), font=dict(family=FONT_FAMILY, size=TICK_SIZE)),
        height=800, width=1500,
        plot_bgcolor="white",
    )
    fig.update_yaxes(showline=False, gridcolor="lightgrey")

    os.makedirs(os.path.join(output_path, "figures"), exist_ok=True)
    fig.write_image(os.path.join(output_path, "figures", f"trade_distribution_scatter_{mtd*100:.0f}_{mid*100:.0f}.png"), scale=2)
    fig.write_html(os.path.join(output_path, "figures", f"trade_distribution_scatter_{mtd*100:.0f}_{mid*100:.0f}.html"), include_plotlyjs="cdn")
    return fig

### Helper: filter the full list of model runs by mtd / mid / rfm ###
def filter_runs(all_runs, mtd=None, mid=None, rfm=None):
    """Filter a list of run dicts (as returned by load_all_model_runs) by
    max_total_dependence_rel (mtd), max_indiv_dependence_rel (mid), and/or
    relationship_factor_magnitude (rfm). Any filter left as None is not applied,
    i.e. that dimension is swept over all available values.
    """
    filtered = []
    for run in all_runs:
        meta = run["metadata"]
        if mtd is not None and meta.get("max_total_dependence_rel") != mtd:
            continue
        if mid is not None and meta.get("max_indiv_dependence_rel") != mid:
            continue
        if rfm is not None and meta.get("relationship_factor_magnitude") != rfm:
            continue
        filtered.append(run)
    return filtered

# def plot_hhi_vs_rfm(hhi_df, hhi_col="hhi", rfm_col="relationship_factor_magnitude"):
#     df_sorted = hhi_df.sort_values(rfm_col)
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(df_sorted[rfm_col], df_sorted[hhi_col], marker="o", color="darkred", linewidth=2)
#     ax.set_xlabel("Relationship factor magnitude (rfm)")
#     ax.set_ylabel("HHI")
#     ax.set_title("HHI sensitivity to relationship factor magnitude")
#     ax.grid(True, alpha=0.3)
#     os.makedirs(os.path.join(output_path, "figures"), exist_ok=True)
#     fig.savefig(os.path.join(output_path, "figures", "hhi_vs_rfm.png"), dpi=300, bbox_inches="tight")
#     return fig

def compute_macro_metrics(rfm_sweep_runs, commodity="h2", scenario="Base"):
    records = []

    for rfm, run in rfm_sweep_runs.items():
        solution = run["solution"]
        data_1D = run["data_1D"]
        data_2D = run["data_2D"]

        supply = solution["v_supply_segment"].sel(commodity=commodity, scenario=scenario)
        price = data_1D["price"].sel(commodity=commodity, scenario=scenario)
        transport = solution["v_transport"].sel(commodity=commodity, scenario=scenario)
        transport_cost_rate = data_2D["transport_cost"].sel(commodity=commodity, scenario=scenario)
        vom_multiplier = data_2D["vom_multiplier"].sel(commodity=commodity, scenario=scenario)

        total_supply = float(supply.sum())
        supply_cost = float((supply * price).sum())
        avg_marginal_cost = supply_cost / total_supply if total_supply > 0 else np.nan

        transport_by_route = transport.sum(dim="supply_step")  # dims: region1, region2
        self_loop_mask = xr.DataArray(
            data_2D.region1.values[:, None] == data_2D.region2.values[None, :],
            dims=["region1", "region2"],
            coords={"region1": data_2D.region1, "region2": data_2D.region2},
        )

        total_traded = float(transport_by_route.where(~self_loop_mask, 0).sum())
        transport_cost = float(
            (transport_by_route * transport_cost_rate * vom_multiplier)
            .where(~self_loop_mask, 0).sum()
        )

        # --- NEW: number of active trade relations + flow size distribution ---
        # convert to a flat series and drop self-loops, then look only at nonzero flows
        flows_flat = transport_by_route.where(~self_loop_mask).to_dataframe(name="amount")["amount"]
        nonzero_flows = flows_flat[flows_flat > 1e-6]  # small tolerance instead of exact > 0

        n_active_relations = int(len(nonzero_flows))
        mean_flow_size = float(nonzero_flows.mean()) if n_active_relations > 0 else np.nan
        median_flow_size = float(nonzero_flows.median()) if n_active_relations > 0 else np.nan

        # bottom_up = compute_bottom_up_costs(solution, data_1D, data_2D, commodity=commodity, scenario=scenario)

        # domestic_production_cost = bottom_up["domestic_production_cost"]
        # traded_cost = bottom_up["transport_cost"] if False else bottom_up["traded_cost"]  # (see note below)
        total_domestic_production = total_supply - total_traded
        total_system_costs = supply_cost + transport_cost  # exact, reverted from bottom-up sum

        # bottom-up decomposition kept for reference/diagnostics, not used for the total anymore
        # bottom_up = compute_bottom_up_costs(solution, data_1D, data_2D, commodity=commodity, scenario=scenario)
        # domestic_production_cost = bottom_up["domestic_production_cost"]
        # traded_cost = bottom_up["traded_cost"]

        records.append({
            "rfm": rfm,
            "total_system_costs": total_system_costs,
            "supply_cost": supply_cost,
            "transport_cost": transport_cost,
            # "domestic_production_cost": domestic_production_cost,
            # "traded_cost": traded_cost,
            "average_marginal_cost": avg_marginal_cost,
            "total_supply": total_supply,
            "total_traded_volume": total_traded,
            "total_domestic_production": total_domestic_production,
            "n_active_relations": n_active_relations,
            "mean_flow_size": mean_flow_size,
            "median_flow_size": median_flow_size,
        })

    return pd.DataFrame(records)

def plot_macro_sensitivity(macro_df, hhi_df, output_path,
                            rfm_col="rfm", hhi_rfm_col="relationship_factor_magnitude",
                            baseline_rfm=1.0):
    df = macro_df.sort_values(rfm_col).merge(
        hhi_df.rename(columns={hhi_rfm_col: rfm_col}), on=rfm_col, how="left"
    )

    # relative % change vs. the baseline rfm value
    baseline = df[df[rfm_col] == baseline_rfm].iloc[0]
    rel_cols = ["total_system_costs", "average_marginal_cost", "total_traded_volume", "hhi",
                "n_active_relations", "mean_flow_size"]
    for col in rel_cols:
        df[f"{col}_pct_change"] = (df[col] - baseline[col]) / baseline[col] * 100

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # --- panel 1: total system costs, decomposed into domestic vs. traded ---
    ax = axes[0]
    ax.plot(df[rfm_col], df["total_system_costs"], marker="o", color="steelblue", linewidth=2)
    ax.set_xlabel("Relationship factor magnitude (rfm)")
    ax.set_ylabel("Total system costs")
    ax.set_title("Total system costs")
    ax.grid(True, alpha=0.3)

    # --- panel 2: average marginal cost (absolute) ---
    ax = axes[1]
    ax.plot(df[rfm_col], df["average_marginal_cost"], marker="o", color="darkorange", linewidth=2)
    ax.set_xlabel("Relationship factor magnitude (rfm)")
    ax.set_ylabel("Average marginal cost [€/MWh]")
    ax.set_title("Average marginal cost")
    ax.grid(True, alpha=0.3)

    # --- panel 3: HHI (standalone) ---
    ax = axes[2]
    ax.plot(df[rfm_col], df["hhi"], marker="o", color="darkred", linewidth=2)
    ax.set_xlabel("Relationship factor magnitude (rfm)")
    ax.set_ylabel("HHI")
    ax.set_title("Market concentration (HHI)")
    ax.grid(True, alpha=0.3)

    # --- panel 4: domestic vs traded volume, absolute ---
    ax = axes[3]
    ax.stackplot(df[rfm_col], df["total_domestic_production"], df["total_traded_volume"],
                 labels=["Domestic production", "Traded volume"],
                 colors=["mediumseagreen", "cornflowerblue"], alpha=0.8)
    ax.set_xlabel("Relationship factor magnitude (rfm)")
    ax.set_ylabel("Volume [MWh]")
    ax.set_title("Domestic production vs. traded volume")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # --- panel 5: relative % change from baseline (rfm=1) across all key metrics ---
    ax = axes[4]
    colors = {"total_system_costs": "steelblue", "average_marginal_cost": "darkorange",
              "total_traded_volume": "cornflowerblue", "hhi": "darkred",
              "n_active_relations": "teal", "mean_flow_size": "purple"}
    labels = {"total_system_costs": "System costs", "average_marginal_cost": "Avg. marginal cost",
              "total_traded_volume": "Traded volume", "hhi": "HHI",
              "n_active_relations": "Active relations", "mean_flow_size": "Mean flow size"}
    for col in rel_cols:
        ax.plot(df[rfm_col], df[f"{col}_pct_change"], marker="o", linewidth=2,
                 color=colors[col], label=labels[col])
    ax.axhline(0, color="grey", linewidth=1, linestyle=":")
    ax.set_xlabel("Relationship factor magnitude (rfm)")
    ax.set_ylabel(f"Change vs. rfm={baseline_rfm} (%)")
    ax.set_title("Relative change from baseline")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- panel 6: number of active trade relations + mean/median flow size ---
    ax = axes[5]
    ax2 = ax.twinx()

    l1, = ax.plot(df[rfm_col], df["n_active_relations"], marker="o", color="teal",
                   linewidth=2, label="Active trade relations (count)")

    # annotate each point with its count value, since the line alone can be
    # hard to read against the flow-size axis
    for x, y in zip(df[rfm_col], df["n_active_relations"]):
        ax.annotate(f"{int(y)}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="teal")

    l2, = ax2.plot(df[rfm_col], df["mean_flow_size"], marker="s", color="purple",
                    linewidth=2, linestyle="--", label="Mean flow size (MWh)")
    l3, = ax2.plot(df[rfm_col], df["median_flow_size"], marker="^", color="mediumorchid",
                    linewidth=2, linestyle=":", label="Median flow size (MWh)")

    ax.set_xlabel("Relationship factor magnitude (rfm)")
    ax.set_ylabel("Number of active trade relations", color="teal")
    ax2.set_ylabel("Flow size (MWh)", color="purple")
    ax.set_title("Trade relation count & flow size")
    ax.legend(handles=[l1, l2, l3], loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # headroom on both axes so annotations and legend don't overlap the lines
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin * 0.95, ymax * 1.05)
    ymin2, ymax2 = ax2.get_ylim()
    ax2.set_ylim(ymin2, ymax2 * 1.1)

    fig.suptitle("Macroscopic sensitivity to relationship factor magnitude", fontsize=16)

    os.makedirs(os.path.join(output_path, "figures"), exist_ok=True)
    fig.savefig(os.path.join(output_path, "figures", "macro_sensitivity_rfm.png"),
                dpi=default_dpi, bbox_inches="tight")

    return fig

def plot_macro_sensitivity_condensed(macro_df, output_path,
                                      rfm_col="rfm", baseline_rfm=1.0):
    df = macro_df.sort_values(rfm_col).copy()

    # relative % change vs. the baseline rfm value
    # (HHI dropped entirely; n_active_relations and mean_flow_size dropped here
    # since they're already shown directly in panel 2)
    baseline = df[df[rfm_col] == baseline_rfm].iloc[0]
    rel_cols = ["total_system_costs", "average_marginal_cost", "total_traded_volume"]
    for col in rel_cols:
        df[f"{col}_pct_change"] = (df[col] - baseline[col]) / baseline[col] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- panel 1: relative % change from baseline (rfm=1) across key metrics ---
    ax = axes[0]
    colors = {"total_system_costs": "steelblue", "average_marginal_cost": "darkorange",
              "total_traded_volume": "cornflowerblue"}
    labels = {"total_system_costs": "System costs", "average_marginal_cost": "Avg. marginal cost",
              "total_traded_volume": "Traded volume"}
    for col in rel_cols:
        ax.plot(df[rfm_col], df[f"{col}_pct_change"], marker="o", linewidth=2,
                 color=colors[col], label=labels[col])
    ax.axhline(0, color="grey", linewidth=1, linestyle=":")
    ax.set_xlabel("Relationship factor magnitude (rfm)")
    ax.set_ylabel(f"Change vs. rfm={baseline_rfm} (%)")
    ax.set_title("Relative change from baseline")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- panel 2: number of active trade relations + mean/median flow size ---
    ax = axes[1]
    ax2 = ax.twinx()

    l1, = ax.plot(df[rfm_col], df["n_active_relations"], marker="o", color="teal",
                   linewidth=2, label="Active trade relations (count)")

    for x, y in zip(df[rfm_col], df["n_active_relations"]):
        ax.annotate(f"{int(y)}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="teal")

    l2, = ax2.plot(df[rfm_col], df["mean_flow_size"], marker="s", color="purple",
                    linewidth=2, linestyle="--", label="Mean flow size (MWh)")
    l3, = ax2.plot(df[rfm_col], df["median_flow_size"], marker="^", color="mediumorchid",
                    linewidth=2, linestyle=":", label="Median flow size (MWh)")

    ax.set_xlabel("Relationship factor magnitude (rfm)")
    ax.set_ylabel("Number of active trade relations", color="teal")
    ax2.set_ylabel("Flow size (MWh)", color="purple")
    ax.set_title("Trade relation count & flow size")
    ax.legend(handles=[l1, l2, l3], loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin * 0.95, ymax * 1.05)
    ymin2, ymax2 = ax2.get_ylim()
    ax2.set_ylim(ymin2, ymax2 * 1.1)

    fig.suptitle("Macroscopic sensitivity to relationship factor magnitude", fontsize=15)
    fig.tight_layout()

    os.makedirs(os.path.join(output_path, "figures"), exist_ok=True)
    fig.savefig(os.path.join(output_path, "figures", "macro_sensitivity_rfm_condensed.png"),
                dpi=default_dpi, bbox_inches="tight")

    return fig

data_path = os.path.join(this_dir, "data")
output_path = os.path.join(this_dir, "output")
print("Work in working directory: " + str(this_dir))

### Define static parameters ###
default_epsg_1 = "EPSG:4326"      # default epsg for geoplots
default_epsg_2 = "EPSG:6933"      # default epsg for projections for determining lengths
# reference_region = "EU-DEU"
commodity = "h2"
scenario = "Base"
# static rfm
# rfm = 1.5 # delete as soon as all runs have the frm in the metadata

### Define central parameter values ###
case_study = get_settings(parameter="case_study")
print("Visualise case study: " + str(case_study))
base_step_param = get_settings(parameter="base_step")
print("Visualise with base step: " + str(base_step_param))

mtd_init = 0.8
mid_init = 0.2

### Define a small function to directly call a certain run from the model saves ###
def run_name(n=159, mtd=mtd_init, mid=mid_init, rfm=1.5):
    return f"model_run_{n}n_{case_study}_{mtd*100:.0f}_{mid*100:.0f}_rfm{rfm*100:.0f}"

### Define the individual base run to analyse if you dont make any sweeps ###
model_run = run_name(mtd=mtd_init, mid=mid_init, rfm=1.5)
print("Load base model as " + str(model_run))

# import base model run (used for the one-time visualisations below)
data_1D, data_2D, solution, meta_data = load_model_run(output_path, model_run)
model = load_complete_model(output_path, model_run)
print("Loaded model input data for base model run " + str(model))

regions = solution.region.values
commodities = solution.commodity.values

### define the sweeping lists for e.g. relationship factor, dependency constraints, and country selection ###
rfm_sweep = [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 1.75, 2.0] #[1, 1.25, 1.5, 1.75, 2]
countries_of_interest = ["EU-DEU", "AS-TUR", "SA-BRA"] # "AS-CHN", "AS-KOR", "AF-EGY", "AF-NGA", "AF-ZAF", "AS-TWN"

#%%
# =============================================================================
# SECTION 1 — One-time visualisations (not tied to any specific run)
# =============================================================================
print("Plotting supply curves")
plot_supply_curves(data_1D)
print("Plotted supply curves")
plt.close('all')

#%%
# =============================================================================
# SECTION 2 — Per-run visualisations, looped over filtered model runs
# =============================================================================
all_runs = load_all_model_runs(output_path)

# set any of these to None to sweep over all available values for that dimension
mtd_filter = 0.8 # None #0.8
mid_filter = 0.2 # None # 0.2
rfm_filter = None  # e.g. None to loop over every rfm sensitivity run

selected_runs = filter_runs(all_runs, mtd=mtd_filter, mid=mid_filter, rfm=rfm_filter)
print(f"Running per-run visualisations for {len(selected_runs)} run(s): "
      f"{[r['run_name'] for r in selected_runs]}")

per_run_meta = {}  # lightweight only — keeps just metadata, not the full datasets

for run in selected_runs:
    run_name_str = run["run_name"]

    # skip runs missing the rfm field, since they predate that parameter
    if "relationship_factor_magnitude" not in run["metadata"]:
        print(f"Skipping {run_name_str} — no rfm metadata (legacy run, needs recalculation)")
        continue

    print(f"\n--- Per-run visualisations for {run_name_str} ---")

    run_data_1D, run_data_2D, run_solution, run_meta = load_model_run(output_path, run_name_str)
    run_model = load_complete_model(output_path, run_name_str)

    mtd = run_meta["max_total_dependence_rel"]
    mid = run_meta["max_indiv_dependence_rel"]
    rfm = run_meta["relationship_factor_magnitude"]

    hhi_results = {}
    print("Calculating HHI")
    hhi = calculate_hhi(run_solution)

    print("Plotting supply and demand donut charts")
    plot_supply_demand_donuts(run_solution, mtd, mid, rfm)
    plt.close('all')

    print("Plotting Sankey flow diagram for trade relations")
    # plot_sankey_flows(run_solution, hhi_results, mtd, mid, rfm)
    plot_sankey_flows_selected(run_solution, hhi_results, mtd, mid, rfm, countries_of_interest)
    plt.close('all')

    print("Getting marginals")
    marginals = run_model.constraints["c_balance"].dual.copy()
    marginal_costs_df = marginals_to_df(marginals, commodity=commodity, scenario=scenario)

    print("Getting transport flow values")
    transport_flows_df = get_transport_flows(run_solution)
    print("Load transport paths")
    paths_gdf = load_transport_paths(output_path, transport_flows_df)
    print("Plotting transport flow map")
    plot_transport_flows_map(paths_gdf, output_path, marginal_costs_df, mtd, mid, rfm, projection_type="natural earth")
    plt.close('all')

    print("Plotting supply composition")
    for country in countries_of_interest:
        print(f"  -> {country}")
        plot_supply_composition(run_model, run_solution, mtd, mid, rfm, country)
        plt.close('all')

    # keep only lightweight metadata around after this run
    per_run_meta[run_name_str] = run_meta

    # empty memory after each run so it doesn't accumulate across the loop
    del run_data_1D, run_data_2D, run_solution, run_model
    gc.collect()

#%%
# =============================================================================
# SECTION 3 — Combined visualisations across model runs
# =============================================================================

## A) analyse the HHI/dependence sensitivity across the mtd x mid grid, at a fixed rfm ###
mtd_mid_runs = [
    r for r in all_runs
    if r["metadata"].get("relationship_factor_magnitude", 1.0) == 1.0
]
hhi_df, cost_df = analyse_results(output_path, mtd_mid_runs)

# sanity check — contour/surface needs at least 2 unique values per axis
unique_mtd = hhi_df["mtd"].nunique()
unique_mid = hhi_df["mid"].nunique()
print(f"mtd x mid grid: {unique_mtd} unique mtd values, {unique_mid} unique mid values, "
      f"{len(hhi_df)} total runs")

n = len(regions)

## build the rfm-sweep dataset used for the selected-country sensitivity plot ###
rfm_sweep_runs = {}
rfm_hhi_records = []

for rfm in rfm_sweep:
    name = run_name(rfm=rfm)
    sweep_data_1D, sweep_data_2D, sweep_solution, sweep_meta = load_model_run(output_path, name)

    if sweep_meta is not None:
        assert sweep_meta["relationship_factor_magnitude"] == rfm, (
            f"{name}: metadata rfm={sweep_meta['relationship_factor_magnitude']} != expected {rfm} -- name/meta mismatch"
        )

    rfm_sweep_runs[rfm] = {
        "data_1D": sweep_data_1D, "data_2D": sweep_data_2D,
        "solution": sweep_solution, "meta": sweep_meta,
    }

    rfm_hhi_records.append({"relationship_factor_magnitude": rfm, "hhi": calculate_hhi(sweep_solution)})

rfm_sweep_results_df = load_market_results_in_df(rfm_sweep_runs, rfm_sweep)

## plot HHI across the rfm sweep (1D line, NOT the mtd/mid trade-off curve) ###
rfm_hhi_df = pd.DataFrame(rfm_hhi_records)
# plot_hhi_vs_rfm(rfm_hhi_df)
macro_df = compute_macro_metrics(rfm_sweep_runs)
plot_macro_sensitivity_condensed(macro_df, output_path)
plot_macro_sensitivity(macro_df, rfm_hhi_df, output_path)

plot_rfm_sens_selec_country(rfm_sweep_results_df, rfm_sweep_runs)

print("Plotting country trade relations scatter")
country_stats_all_df, mtd, mid = compute_country_trade_stats_all_rfm(rfm_sweep_runs)
plot_country_trade_scatter(country_stats_all_df, output_path, mtd, mid, log_x = False)

## Generate multi-rfm plots ###
# Collect runs by rfm value
rfm_runs = {}
for r in all_runs:
    rfm_val = r["metadata"].get("relationship_factor_magnitude", 1)
    if rfm_val not in rfm_runs:
        rfm_runs[rfm_val] = []
    rfm_runs[rfm_val].append(r)

# Select which rfm values to include in the multi-plot
rfm_values_to_plot = sorted([rfm for rfm in rfm_runs.keys() if len(rfm_runs[rfm]) > 0])

if len(rfm_values_to_plot) >= 2:  # Need at least 2 to show multiple surfaces
    print(f"\nGenerating multi-rfm plots for rfm values: {rfm_values_to_plot}")

    # Generate DataFrames for each selected rfm
    hhi_dfs = []
    cost_dfs = []
    for rfm in rfm_values_to_plot:
        hhi_df, cost_df = analyse_results(output_path, rfm_runs[rfm])
        hhi_dfs.append(hhi_df)
        cost_dfs.append(cost_df)

    # Generate the multi-rfm plots
    plot_hhi_sens_multi(hhi_dfs, rfm_values_to_plot)
    plot_cost_sens_multi(cost_dfs, rfm_values_to_plot)
    # Generate the multi-rfm trade-off curve
    plot_trade_off_curve_multi(hhi_dfs, cost_dfs, rfm_values_to_plot)
else:
    print("Not enough rfm values with sufficient data to generate multi-rfm plots")

plt.close('all')

#%%
#empty memory
for v in ['run_data_1D','run_data_2D','run_solution','run_model']:
    if v in locals():
        o = locals()[v]
        if hasattr(o, 'close'): o.close()
        del locals()[v]
gc.collect()

STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')
#%%