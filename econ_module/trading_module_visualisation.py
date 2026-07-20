### This is the corresponding visualisation module for the trading module
#%%
import os
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize
import plotly.graph_objects as go
import numpy as np
from model_io import load_model_run, save_model_run, save_complete_model, load_complete_model
import pandas as pd
import seaborn as sns
import math
import linopy
import json

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

MWH_TO_TWH = 1e6  # model results are in MWh; divide by 1e6 for TWh

def plot_supply_curves2(data_1D, commodity='h2', scenario='Base', reference_region=None, MWH_TO_TWH=0.001):
    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.color'] = 'black'

    # Convert data to DataFrame
    data_1D_df = (
        data_1D
        .sel(commodity=commodity, scenario=scenario)
        .to_dataframe()
        .reset_index()
    )

    regions = data_1D_df["region"].unique()
    non_ref_regions = [r for r in regions if r != reference_region]

    # Create a figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create a colormap
    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=len(non_ref_regions))
    color_map = {r: cmap(norm(i)) for i, r in enumerate(non_ref_regions)}

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

        # Plot the supply curve
        ax.plot(cumulative_supply_twh, region_data_sorted["price"],
                color=color, linewidth=line_width, label=region)

        # Add inline label
        last_x = region_data_sorted["supply"].iloc[-1] / MWH_TO_TWH
        last_y = region_data_sorted["price"].iloc[-1]
        ax.text(last_x, last_y, region, fontsize=10, color=color,
                ha='left', va='bottom', fontfamily='Times New Roman')

    # Set the title and labels
    ax.set_title(f"Regional supply curves — {commodity}, {scenario}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Supply potential (TWh/a)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Price (€/MWh)", fontsize=12, fontweight='bold')

    # Set the x-axis to log scale
    x_lower = np.percentile(first_step_twh_by_region, 1)
    x_upper = max(total_twh_by_region.values()) * 1.1
    ax.set_xscale('log')
    ax.set_xlim(x_lower, x_upper)

    # Add grid lines
    ax.grid(True, which="both", linestyle='--', linewidth=0.5, color='lightgrey')

    # Add legend
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Supply curves',
    #           fontsize=10, frameon=True, framealpha=1)

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(right=0.75)
    #export to png
    fig.savefig(os.path.join(output_path, "figures", f"supply_curves_{commodity}_{scenario}_2.png"), dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
    return

def plot_supply_curves(data_1D):
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
                hovertemplate=(
                    f"<b>{region}</b><br>Cumulative supply: %{{x:.2f}} TWh/a<br>"
                    "Price: %{y:.2f} €/MWh<extra></extra>"
                ),
            )
        )

        # ---- inline label for every region, matching the original -----------
        last_x = region_data_sorted["supply"].iloc[-1]/MWH_TO_TWH
        last_y = region_data_sorted["price"].iloc[-1]
        fig.add_annotation(
            x=last_x, y=last_y, text=region, showarrow=False,
            font=dict(family="Times New Roman", size=10, color=color),
            xanchor="left", yanchor="bottom",
        )

    fig.update_layout(
        title=dict(
            text=f"Regional supply curves — {commodity}, {scenario}",
            font=dict(family="Times New Roman", color="black"),
        ),
        xaxis=dict(title=dict(text="Supply potential (TWh/a)", font=dict(family="Times New Roman", color="black")),
                    tickfont=dict(family="Times New Roman", color="black")),
        yaxis=dict(title=dict(text="Price (€/MWh)", font=dict(family="Times New Roman", color="black")),
                    tickfont=dict(family="Times New Roman", color="black")),
        legend=dict(title=dict(text="Supply curves", font=dict(family="Times New Roman", color="black")),
                    font=dict(family="Times New Roman", color="black", size=10),
                    x=1.02, y=1, xanchor="left", yanchor="top"),
        font=dict(family="Times New Roman", color="black"),
        plot_bgcolor="white", paper_bgcolor="white",
        width=1400, height=800, margin=dict(r=200),
    )

    x_lower = np.percentile(first_step_twh_by_region, 1)
    x_upper = max(total_twh_by_region.values()) * 1.1
    fig.update_xaxes(
        type="log",
        range=[np.log10(x_lower), np.log10(x_upper)],
        showgrid=True, gridcolor="lightgrey", gridwidth=1, zeroline=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor="lightgrey", gridwidth=1, zeroline=False)

    fig.write_image(os.path.join(output_path, "figures", f"supply_curves_{commodity}_{scenario}.png"), scale=4)
    fig.show()
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

def plot_supply_composition(model, ds):
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
            data_2D["value"].to_dataframe("transport_cost").reset_index(),
            on=["region1", "region2", "commodity", "scenario", "supply_step"],
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
    ax.set_title(f"Supply composition: {reference_region} - Max import share: {max_total_dependence_rel*100:.0f}% total, {max_indiv_dependence_rel*100:.0f}% per region)", fontfamily="Times New Roman", fontsize=15)
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
    plt.savefig(os.path.join(output_path, "figures", f"supply_composition_{reference_region}_{max_total_dependence_rel*100:.0f}_{max_indiv_dependence_rel*100:.0f}.png"), bbox_inches="tight", dpi=300)
    plt.show()
    return

def plot_sankey_flows(ds, hhi_results):
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
            f"Max import share: {max_total_dependence_rel*100:.0f}% total, "
            f"{max_indiv_dependence_rel*100:.0f}% per region"
        ),
        font_size=15,
        font_family="Times New Roman",
        font_color="black",
        width=1000,
        height=2000,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    os.makedirs("output", exist_ok=True)
    # fig.write_html(os.path.join("output", f"sankey_diagram_constrained_{max_total_dependence_rel*100:.0f}_{max_indiv_dependence_rel*100:.0f}.html"))
    fig.write_image(os.path.join(output_path, "figures", f"sankey_diagram_constrained_{n}n_{max_total_dependence_rel*100:.0f}_{max_indiv_dependence_rel*100:.0f}.png"), scale=4)
    fig.show()

    fig.update_layout(
        width=2000,
        height=1000)
    os.makedirs("output", exist_ok=True)
    fig.write_image(os.path.join(output_path, "figures", f"sankey_diagram_constrained_{n}n_{max_total_dependence_rel*100:.0f}_{max_indiv_dependence_rel*100:.0f}_wide.png"), scale=1)

    fig.show()
    return

def plot_supply_demand_donuts(ds):
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
            f"Max import share: {max_total_dependence_rel*100:.0f}% total, "
            f"{max_indiv_dependence_rel*100:.0f}% per region"
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
    fig.savefig(os.path.join(output_path, "figures", f"donut_charts_{n}n_{commodity}_{max_total_dependence_rel*100:.0f}_{max_indiv_dependence_rel*100:.0f}.png"),
        dpi=300, bbox_inches="tight")

    plt.show()
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
        max_total = run["metadata"]["max_total_dependence_rel"]
        max_indiv = run["metadata"]["max_indiv_dependence_rel"]

        # Load the run solution
        data_1D, data_2D, solution, meta_data = load_model_run(output_path, run_name)

        # Calculate the HHI
        hhi = calculate_hhi(solution)

        hhi_data_list.append({
            "max_total": max_total,
            "max_indiv": max_indiv,
            "HHI": hhi
        })

        # --- Extract total system cost from the full model ---
        model = load_complete_model(output_path, run_name)
        total_system_costs = model.objective.value

        cost_data_list.append({
            "max_total": max_total,
            "max_indiv": max_indiv,
            "total_system_costs": total_system_costs
        })

    # Create DataFrames from the lists
    hhi_df = pd.DataFrame(hhi_data_list)
    cost_df = pd.DataFrame(cost_data_list)

    return hhi_df, cost_df

from matplotlib.lines import Line2D

def plot_hhi_sens(hhi_df, low_thresh=None, high_thresh=None):
    """
    Plot a 3D surface showing HHI as a function of max_total_dependence
    and max_indiv_dependence, with demarcation lines for the standard
    HHI concentration corridors (unconcentrated / moderate / high).

    hhi_df must have columns "max_total", "max_indiv", "HHI".
    low_thresh, high_thresh: override the default corridor thresholds.
        If None, auto-detected as 1500/2500 (0-10000 scale) or
        0.15/0.25 (0-1 scale) based on the max HHI value present.
        The high threshold is only drawn if the surface actually
        exceeds it somewhere.
    """
    # Pivot into a grid: rows = max_total, cols = max_indiv
    pivot = hhi_df.pivot(
        index="max_total",
        columns="max_indiv",
        values="HHI",
    ).sort_index().sort_index(axis=1)

    x = pivot.columns.values  # max_indiv
    y = pivot.index.values    # max_total
    X, Y = np.meshgrid(x, y)
    Z = pivot.values

    zmax = np.nanmax(Z)
    zmin = np.nanmin(Z)

    # --- auto-detect scale and set default thresholds ---
    if low_thresh is None or high_thresh is None:
        if zmax <= 1.5:  # looks like a 0-1 scale
            low_thresh, high_thresh = 0.15, 0.25
        else:  # standard 0-10000 scale
            low_thresh, high_thresh = 1500, 2500

    show_high = zmax > high_thresh  # only demarcate the high corridor if data crosses it

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k",
                            linewidth=0.3, alpha=0.7)
    # ax.invert_xaxis()
    # ax.invert_yaxis()

    ax.zaxis.labelpad = 15
    fig.subplots_adjust(right=0.85)
    ax.set_xlabel("Max individual import dependence")
    ax.set_ylabel("Max total import dependence")
    ax.text2D(0.95, 0.82, "HHI", transform=ax.transAxes, ha="center", fontsize=12)
    ax.set_title("HHI Sensitivity to Dependence Constraints", fontsize=18)

    z_floor = zmin - 0.05 * (zmax - zmin)
    ax.set_zlim(z_floor, zmax)

    thresholds = [(low_thresh, "black", "Treshold HHI > 1500 for moderately concentrated market")]
    if show_high:
        thresholds.append((high_thresh, "black", "Treshold HHI > 2500 for highly concentrated market"))

    legend_lines = []
    xr = (x.min(), x.max())
    yr = (y.min(), y.max())

    for thresh, color, label in thresholds:
        # floor contour: traces where the surface crosses this HHI level
        ax.contour(X, Y, Z, levels=[thresh], zdir="z", offset=thresh,
                   colors=color, linewidths=2, linestyles="--")

        # wall lines: outline the threshold height on the xz/yz background panes
        for y_edge in yr:
            ax.plot(xr, [y_edge, y_edge], [thresh, thresh],
                    color=color, lw=2, ls=":", alpha=1, zorder=2)
        for x_edge in xr:
            ax.plot([x_edge, x_edge], yr, [thresh, thresh],
                    color=color, lw=2, ls=":", alpha=1, zorder=2)

        legend_lines.append(
            Line2D([0], [0], color=color, lw=2, ls="--",
                   label=f"{label} ({thresh})")
        )

    ax.legend(handles=legend_lines, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(output_path, "figures", f"donut_charts_{n}n_{commodity}.png"))
    return fig, ax

def plot_cost_sens(cost_df):
    """
    Plot a 3D surface showing total_system_costs as a function of
    max_total_dependence and max_indiv_dependence.

    cost_df must have columns "max_total", "max_indiv", "total_system_costs".
    """
    currency_unit="€"
    raw_unit_scale=1e9

    pivot = cost_df.pivot(
        index="max_total",
        columns="max_indiv",
        values="total_system_costs",
    ).sort_index().sort_index(axis=1)

    x = pivot.columns.values  # max_indiv
    y = pivot.index.values    # max_total
    X, Y = np.meshgrid(x, y)
    Z = pivot.values / raw_unit_scale 

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k",
                            linewidth=0.3, alpha=0.8)
    # ax.invert_xaxis()
    # ax.invert_yaxis()

    ax.zaxis.labelpad = 15
    fig.subplots_adjust(right=0.85)
    ax.set_xlabel("Max individual import dependence")
    ax.set_ylabel("Max total import dependence")
    ax.text2D(0.95, 0.82, f"Total system\ncosts [bn {currency_unit}]",
              transform=ax.transAxes, ha="center", fontsize=12)
    ax.set_title(f"Total System Cost Sensitivity to Dependence Constraints [bn {currency_unit}]",
                 fontsize=18)

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(output_path, "figures", f"cost_surface_{n}n_{commodity}.png"))
    return fig, ax

import plotly.graph_objects as go

def _curly_brace_path(x, y0, y1, width, q=0.6):
    """
    Build an SVG path string for a vertical curly brace between y0 and y1,
    anchored at horizontal position x, bulging outward by `width`
    (negative width points left/outward from the axis).
    Two cubic beziers meeting at the tip (x - width, midpoint) - the
    standard way to draw a curly brace shape.
    """
    ym = (y0 + y1) / 2
    return (
        f"M {x},{y0} "
        f"C {x+width*q},{y0} {x+width},{y0+(ym-y0)*(1-q)} {x+width},{ym} "
        f"C {x+width},{y1-(y1-ym)*(1-q)} {x+width*q},{y1} {x},{y1}"
    )

import plotly.graph_objects as go

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


def plot_pareto_frontier(hhi_df, cost_df, hhi_threshold=1500):
    """
    Merge hhi_df and cost_df, identify the Pareto frontier (minimizing
    both HHI and total_system_costs), and plot it interactively with
    Plotly. Highlights the HHI concentration threshold as a low-risk
    "zone", marks the best option within the low-concentration zone in
    red with a label of its dependence settings, and annotates the cost
    premium of staying in that zone versus the global cost minimum via
    a curly brace on the right side of the plot.

    Returns fig.
    """
    df = hhi_df.merge(cost_df, on=["max_total", "max_indiv"]).reset_index(drop=True)

    # --- identify Pareto-efficient points ---
    costs = df[["total_system_costs", "HHI"]].values
    n = costs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        dominating = (
            np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        )
        if np.any(dominating):
            is_efficient[i] = False
    df["pareto_efficient"] = is_efficient

    efficient = df[df["pareto_efficient"]].sort_values("HHI").reset_index(drop=True)
    efficient["cost_bn"] = efficient["total_system_costs"] / 1e9
    df["cost_bn"] = df["total_system_costs"] / 1e9

    # --- select best option: lowest cost among frontier points within the
    #     low-concentration zone (HHI <= threshold); fallback to the
    #     frontier point closest to the threshold if none qualify ---
    within_zone = efficient[efficient["HHI"] <= hhi_threshold]
    if not within_zone.empty:
        best = within_zone.loc[within_zone["cost_bn"].idxmin()]
    else:
        best = efficient.loc[(efficient["HHI"] - hhi_threshold).abs().idxmin()]

    global_min_cost = df["cost_bn"].min()
    cost_premium = best["cost_bn"] - global_min_cost
    cost_premium_pct = cost_premium / global_min_cost * 100

    fig = go.Figure()

    # --- transparent blue "safe zone" for HHI <= threshold ---
    fig.add_vrect(
        x0=df["HHI"].min() - 50, x1=hhi_threshold,
        fillcolor="blue", opacity=0.08, line_width=0,
        annotation_text="Low concentration zone", annotation_position="top left"
    )

    # --- vertical threshold line ---
    fig.add_vline(
        x=hhi_threshold, line_width=2, line_dash="dash", line_color="blue",
        annotation_text=f"HHI threshold ({hhi_threshold})", annotation_position="top"
    )

    # --- horizontal line at the minimum optimal cost within the zone ---
    fig.add_hline(
        y=best["cost_bn"], line_width=2, line_dash="dash", line_color="blue",
        annotation_text=f"Min. cost within zone ({best['cost_bn']:.2f} bn €)",
        annotation_position="bottom left"
    )

    # --- smoothed trend/approximation curve: quadratic fit (trend, not
    #     an interpolation through every point) ---
    if len(efficient) >= 3:
        coeffs = np.polyfit(efficient["HHI"], efficient["cost_bn"], deg=2)
        trend = np.poly1d(coeffs)
        x_smooth = np.linspace(efficient["HHI"].min(), efficient["HHI"].max(), 200)
        y_smooth = trend(x_smooth)
        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_smooth, mode="lines",
            line=dict(color="lightgrey", dash="dot", width=2),
            name="Trend approximation", hoverinfo="skip"
        ))

    # --- Pareto frontier ---
    fig.add_trace(go.Scatter(
        x=efficient["HHI"], y=efficient["cost_bn"], mode="lines+markers",
        line=dict(color="black", width=2),
        marker=dict(size=8, color="black"),
        name="Pareto frontier",
        text=efficient[["max_total", "max_indiv"]].apply(lambda row: f"Max Total: {row['max_total']}, Max Indiv: {row['max_indiv']}", axis=1)
    ))

    # --- per-point labels (disabled) ---
    # for row in efficient.itertuples():
    #     fig.add_annotation(
    #         x=row.HHI, y=row.cost_bn,
    #         text=f"({row.max_total:.1f}, {row.max_indiv:.1f})",
    #         showarrow=False,
    #         yshift=36,
    #         font=dict(family="Times New Roman", size=11, color="black")
    #     )

    # --- dominated points (disabled) ---
    # fig.add_trace(go.Scatter(
    #     x=dominated["HHI"], y=dominated["total_system_costs"] / 1e9, mode="markers",
    #     marker=dict(size=6, color="lightgray"),
    #     name="Dominated"
    # ))

    # --- best option, highlighted as a larger red dot, with a clear label ---
    fig.add_trace(go.Scatter(
        x=[best["HHI"]], y=[best["cost_bn"]], mode="markers",
        marker=dict(size=16, color="red", symbol="circle"),
        name="Selected optimum"
        text=best[["max_total", "max_indiv"]].apply(lambda row: f"Max Total: {row['max_total']}, Max Indiv: {row['max_indiv']}", axis=1)
    ))
    fig.add_annotation(
        x=best["HHI"], y=best["cost_bn"],
        text=f"{best['max_total']*100:.0f}% total dependence,<br>{best['max_indiv']*100:.0f}% individual dependence",
        showarrow=True, arrowhead=2, ax=60, ay=-40,
        font=dict(family="Times New Roman", size=15, color="black"),
        bgcolor="white", bordercolor="red", borderwidth=1
    )

    # --- curly brace on the right side, under the legend:
    #     global min cost -> min cost within zone ---
    brace_x = 1.06    # paper coords, just right of the axis
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

    # --- global formatting, matched to your matplotlib rcParams ---
    fig.update_layout(
        title="Total System Costs vs. Import Concentration",
        xaxis_title="HHI (market concentration)",
        yaxis_title="Total system costs [bn €]",
        template="plotly_white",
        font=dict(family="Times New Roman", color="black", size=20),
        xaxis=dict(title_font=dict(size=15), tickfont=dict(size=15)),
        yaxis=dict(title_font=dict(size=15), tickfont=dict(size=15)),
        legend=dict(font=dict(size=15), x=1.0, y=1.0, xanchor="left", yanchor="top"),
        margin=dict(r=160),  # extra room on the right for legend + brace + label
        width=1000, height=650
    )

    fig.write_image(os.path.join(output_path, "figures", f"pareto_front_{n}n_{commodity}.png"), scale=4)

    return fig

from model_settings import get_settings
### Define central parameter values ###
case_study = get_settings(parameter="case_study")
transport_costs_param = get_settings(parameter="transport_costs")
base_step_param = get_settings(parameter="base_step")

# Get the directory where this script is located
script_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data")
output_path = os.path.join(script_dir, "output")

hhi_results = {}

# import model run
data_1D, data_2D, solution, meta_data = load_model_run(os.path.join(output_path), "model_run_159n_h2bb_100_100")
# import model
model = load_complete_model(os.path.join(output_path), "model_run_159n_h2bb_100_100")

regions = solution.region.values
reference_region = "EU-DEU"
commodities = solution.commodity.values
commodity = "h2"
scenario = "Base"
max_total_dependence_rel = meta_data["max_total_dependence_rel"]
max_indiv_dependence_rel = meta_data["max_indiv_dependence_rel"]

print("Plotting supply curves")
# plot_supply_curves(data_1D)Ach ko
# plot_supply_curves2(data_1D)
# print("Calculating HHI")
# hhi = calculate_hhi(solution)
# print("Plotting supply composition")
# plot_supply_composition(model, solution)
# print("Plotting Sankey flow diagram for trade relations")
# plot_sankey_flows(solution, hhi)
# print("Plotting supply and demand donut charts")
# plot_supply_demand_donuts(solution)

### analyse the HHI dependence sensitivity across all model runs ###
runs = load_all_model_runs(output_path)
hhi_df, cost_df = analyse_results(output_path, runs)
regions = solution.region.values.tolist()
n = len(regions)
plot_hhi_sens(hhi_df)
plot_cost_sens(cost_df)
plot_pareto_frontier(hhi_df, cost_df)

#%%
#empty memory
del data_1D, data_2D, solution, meta_data, model, runs, hhi_df

#%%
STOP = time.perf_counter()
print('Total execution time of script',round((STOP-START), 1), 's')