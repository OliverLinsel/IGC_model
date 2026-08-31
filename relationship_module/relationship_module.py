import os
import sys
import time
 
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
#%%
 
# ------------------------------------------------------------------------------
# Stage 1 (expensive, call once): load data and build the full country-pair
# universe with weighted alliance overlap + war/conflict flags. Nothing in this
# stage depends on the sensitivity parameter.
# ------------------------------------------------------------------------------
def build_relationship_base(case_study: str = "h2bb") -> dict:
    """
    Load the alliance/country data and precompute everything that does NOT
    depend on the target ceiling multiplier:
      - the full unordered country-pair universe (including pairs that share
        no alliance at all — needed since the cost multiplier must apply to
        every route, not just allied ones)
      - `alliance_index` per pair: weighted alliance overlap, normalized
        against the best-connected pair
      - `at_war` per pair, from the countries sheet's `;`-separated `enemies`
        column
 
    Returns
    -------
    dict with:
        "base_df"         : DataFrame[country, friends, counter, shared_weight,
                             alliance_index, at_war] — feed this into
                             calculate_relationship_factor()
        "country_df"      : country reference table merged with world geometry
        "geometry_helper" : country -> (lon, lat, geometry) lookup
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(script_dir))
    data_path = os.path.join(script_dir, "data")
 
    print("Start reading input Data\n")
    print(data_path)
 
    world = gpd.read_file(
        os.path.join(
            data_path, "..", "..", "data_module", "Data", "scenario_run_data",
            "h2bb", "triangulation_transport_data", "data_input",
            "naturalearthdata", "ne_110m_admin_0_countries.shp",
        )
    )
    world = world[["SOVEREIGNT", "ISO_A3", "POP_EST", "GDP_MD", "GDP_YEAR", "ECONOMY", "geometry"]]
 
    relations_df = pd.read_excel(os.path.join(data_path, case_study, "country_relations.xlsx"), sheet_name="alliances")
    countries_df = pd.read_excel(os.path.join(data_path, case_study, "country_relations.xlsx"), sheet_name="countries")
 
    country_list = countries_df["country-code"].unique().tolist()
 
    # --- country reference table ---
    countries_df = countries_df.rename(columns={"country-code": "country", "value1": "lon", "value2": "lat"})
    countries_df = countries_df.drop(columns=["attribute", "attribute.1"], errors="ignore")
    country_df = countries_df.merge(world, on="ISO_A3", how="left")
 
    # --- explode alliance membership into a long (alliance, country) table ---
    relations_df["member_states"] = relations_df["member_states"].str.split("; ")
    relations_df["member_unions"] = relations_df["member_unions"].str.split("; ")
 
    alliance_long = (
        relations_df[["alliance", "weight", "member_states"]]
        .explode("member_states")
        .rename(columns={"member_states": "country"})
        .dropna(subset=["country"])
    )
 
    unknown = sorted(set(alliance_long["country"]) - set(country_list))
    if unknown:
        print(f"{len(unknown)} member code(s) not found in the country list: {unknown}")
 
    alliances_per_country = alliance_long.groupby("country")["alliance"].apply(list).rename("alliances")
    country_df = country_df.merge(alliances_per_country, on="country", how="left")
 
    # --- country<->country pairs that share at least one alliance ---
    pairs = alliance_long.merge(alliance_long, on="alliance", suffixes=("", "_friend"))
    pairs = pairs[pairs["country"] != pairs["country_friend"]]
    pairs = pairs.rename(columns={"country_friend": "friends"})[["country", "friends", "alliance", "weight"]]
 
    sorted_pair = np.sort(pairs[["country", "friends"]].to_numpy(), axis=1)
    pairs["country"], pairs["friends"] = sorted_pair[:, 0], sorted_pair[:, 1]
    pairs = pairs.drop_duplicates(subset=["country", "friends", "alliance"])  # drop the A-B/B-A mirror
 
    shared = (
        pairs.groupby(["country", "friends"])
        .agg(counter=("alliance", "nunique"), shared_weight=("weight", "sum"))
        .reset_index()
    )
 
    # --- full pair universe (includes pairs sharing zero alliances) ---
    all_countries = pd.Series(sorted(country_list), name="country")
    full_pairs = all_countries.to_frame().merge(all_countries.rename("friends"), how="cross")
    full_pairs = full_pairs[full_pairs["country"] < full_pairs["friends"]]  # unordered, one row per pair
 
    base_df = full_pairs.merge(shared, on=["country", "friends"], how="left")
    base_df[["counter", "shared_weight"]] = base_df[["counter", "shared_weight"]].fillna(0)
 
    # weighted alliance overlap, normalized against the best-connected pair
    base_df["alliance_index"] = base_df["shared_weight"] / base_df["shared_weight"].max()
 
    # --- war / conflict flag, from the `;`-separated `enemies` column ---
    if "enemies" in countries_df.columns and countries_df["enemies"].notna().any():
        enemy_long = (
            countries_df[["country", "enemies"]]
            .dropna(subset=["enemies"])
            .assign(enemies=lambda d: d["enemies"].str.split("; "))
            .explode("enemies")
            .rename(columns={"enemies": "friends"})
        )
        enemy_pairs = enemy_long[["country", "friends"]].copy()
        sorted_enemy = np.sort(enemy_pairs.to_numpy(), axis=1)
        enemy_pairs["country"], enemy_pairs["friends"] = sorted_enemy[:, 0], sorted_enemy[:, 1]
        enemy_pairs = enemy_pairs.drop_duplicates()
        enemy_pairs["at_war"] = 1
 
        base_df = base_df.merge(enemy_pairs, on=["country", "friends"], how="left")
        base_df["at_war"] = base_df["at_war"].fillna(0).astype(int)
    else:
        base_df["at_war"] = 0
        print("No 'enemies' data found on the countries sheet -- war override will be a no-op.")

    geometry_helper = country_df[["country", "lon", "lat", "geometry"]]
 
    return {"base_df": base_df, "country_df": country_df, "geometry_helper": geometry_helper}
 
def calculate_relationship_factor(base, relationship_factor_magnitude: float, scenario="Base") -> pd.DataFrame:
    """
    Compute the relationship-factor / vom_multiplier table at a given rfm ceiling.

        vom_multiplier_ij = relationship_factor_magnitude ** (1 - alliance_index_ij)

    so vom_multiplier is exactly 1 for the best-connected pair (alliance_index = 1,
    exponent = 0) and exactly relationship_factor_magnitude (rfm) for a pair
    sharing no alliance at all (alliance_index = 0, exponent = 1) -- interpolating
    smoothly on a power curve in between. This is the single, final scaling
    applied to transport costs downstream (trading_module.py consumes
    vom_multiplier directly, with no further rfm scaling applied there) -- so
    rfm should be swept here, at this layer, and nowhere else.

    Pairs flagged `at_war` are overridden to inf (infeasible route) regardless
    of alliance_index -- treated as a hard constraint rather than folded into
    the alliance curve, so trading_module.py can filter those out
    (`rel_df.vom_multiplier < np.inf`) rather than mistaking a war override
    for a legitimately weak alliance score.

    Parameters
    ----------
    base : dict (as returned by build_relationship_base()) or the base_df itself
    relationship_factor_magnitude : float
        The rfm sensitivity parameter. Must be > 1. This is the single
        parameter to sweep in the sensitivity analysis -- it is not
        reapplied anywhere downstream.

    Returns
    -------
    DataFrame[region1, region2, counter, shared_weight, alliance_index,
              vom_multiplier, scenario], sorted by vom_multiplier.
    """
    if relationship_factor_magnitude < 1:
        raise ValueError(f"relationship_factor_magnitude must be > 1, got {relationship_factor_magnitude}")

    base_df = base["base_df"] if isinstance(base, dict) else base
    rel_df = base_df.copy()

    rel_df["vom_multiplier"] = relationship_factor_magnitude ** (1 - rel_df["alliance_index"])
    rel_df.loc[rel_df["at_war"] == 1, "vom_multiplier"] = 10

    rel_df["continent"] = rel_df["country"].str.split("-").str[0]
    rel_df["pairing"] = rel_df["country"] + " - " + rel_df["friends"]

    rel_df["scenario"] = scenario
    rel_df = rel_df.sort_values("vom_multiplier").reset_index(drop=True)
    rel_df = rel_df.rename(columns={"country": "region1", "friends": "region2"})
    rel_df = rel_df.drop(columns={"at_war", "continent", "pairing"})
    return rel_df

def relationship_visualisation(base, rfm_sweep_list, case_study, output_path=None):
    """Visualise how vom_multiplier responds to alliance_index across different
    relationship_factor_magnitude (rfm) sweep values — shows the underlying
    transformation curve, with each region-pair as a point along it."""
    FONT_FAMILY = "Times New Roman"
    FONT_COLOR = "black"

    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "output")

    # --- compute rel_df for every rfm value in the sweep ---
    rfm_dfs = []
    for rfm in rfm_sweep_list:
        df = calculate_relationship_factor(base, relationship_factor_magnitude=rfm)
        df = df.copy()
        df["rfm"] = rfm
        rfm_dfs.append(df)
    combined_df = pd.concat(rfm_dfs, ignore_index=True)

    # --- simple fixed palette, one color per rfm value ---
    rfm_values = sorted(combined_df["rfm"].unique())
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    color_map = {rfm: palette[i % len(palette)] for i, rfm in enumerate(rfm_values)}

    fig = go.Figure()

    for rfm in rfm_values:
        sub = combined_df[combined_df["rfm"] == rfm].sort_values("alliance_index")
        fig.add_trace(go.Scatter(
            x=sub["alliance_index"], y=sub["vom_multiplier"],
            mode="markers",
            marker=dict(size=6, color=color_map[rfm], opacity=0.6,
                         line=dict(width=0.5, color=color_map[rfm])),
            name=f"rfm = {rfm}",
            hovertemplate=(
                sub["region1"].astype(str) + " – " + sub["region2"].astype(str)
                + "<br>Alliance index: %{x:.3f}<br>VOM multiplier: %{y:.3f}"
                + f"<br>rfm: {rfm}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=f"Relationship factor sensitivity — VOM multiplier vs. alliance index ({case_study})",
            font=dict(family=FONT_FAMILY, size=18, color=FONT_COLOR),
        ),
        xaxis=dict(
            title=dict(text="Alliance index", font=dict(family=FONT_FAMILY, size=14, color=FONT_COLOR)),
            tickfont=dict(family=FONT_FAMILY, size=12, color=FONT_COLOR),
            showgrid=True, gridcolor="lightgrey",
            range=[0, 1],
        ),
        yaxis=dict(
            title=dict(text="VOM multiplier", font=dict(family=FONT_FAMILY, size=14, color=FONT_COLOR)),
            tickfont=dict(family=FONT_FAMILY, size=12, color=FONT_COLOR),
            showgrid=True, gridcolor="lightgrey",
            range=[0, 2],
        ),
        legend=dict(title=dict(text="Sensitivity run"), font=dict(family=FONT_FAMILY, size=12, color=FONT_COLOR)),
        font=dict(family=FONT_FAMILY, color=FONT_COLOR),
        plot_bgcolor="white", paper_bgcolor="white",
        width=1200, height=800,
    )

    os.makedirs(os.path.join(output_path, "figures"), exist_ok=True)
    fig.write_image(os.path.join(output_path, "figures", f"relationship_vom_multiplier_sens_{case_study}.png"), scale=4)
    fig.write_html(os.path.join(output_path, "figures", f"relationship_vom_multiplier_sens_{case_study}.html"), include_plotlyjs="cdn")

    return fig, combined_df

def relationship_distribution_visualisation(base, rfm_sweep_list, case_study, output_path=None, n_points=300):
    """Overlaid count distributions of vom_multiplier across region pairs, one
    curve per rfm sweep value, all sharing the same x and y axis --
    differentiated only by color."""
    FONT_FAMILY = "Times New Roman"
    FONT_COLOR = "black"

    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "output")

    rfm_dfs = []
    for rfm in rfm_sweep_list:
        df = calculate_relationship_factor(base, relationship_factor_magnitude=rfm)
        df = df.copy()
        df["rfm"] = rfm
        rfm_dfs.append(df)
    combined_df = pd.concat(rfm_dfs, ignore_index=True)

    plot_df = combined_df[combined_df["vom_multiplier"] < 10].copy()

    rfm_values = sorted(plot_df["rfm"].unique())
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    color_map = {rfm: palette[i % len(palette)] for i, rfm in enumerate(rfm_values)}

    # shared x-range across all curves
    x_min = plot_df["vom_multiplier"].min()
    x_max = plot_df["vom_multiplier"].max()
    x_grid = np.linspace(x_min, x_max, n_points)

    fig = go.Figure()

    for rfm in rfm_values:
        sub = plot_df[plot_df["rfm"] == rfm]
        n_points_in_subset = len(sub)

        # Check if data has sufficient variance
        if n_points_in_subset < 2 or np.var(sub["vom_multiplier"]) < 1e-10:
            # Plot a vertical line at the constant value
            fig.add_trace(go.Scatter(
                x=[sub["vom_multiplier"].iloc[0]] * 2,
                y=[0, n_points_in_subset],  # Use actual count for constant case
                mode="lines",
                line=dict(color=color_map[rfm], width=2, dash="dot"),
                name=f"rfm = {rfm} (constant)",
                hovertemplate=f"rfm = {rfm}<br>Constant VOM multiplier: {sub['vom_multiplier'].iloc[0]:.3f}<br>Count: {n_points_in_subset}<extra></extra>",
            ))
            continue

        try:
            kde = gaussian_kde(sub["vom_multiplier"])
            density = kde(x_grid)
            # Convert density to counts by multiplying by number of points and bin width
            bin_width = (x_max - x_min) / n_points
            counts = density * n_points_in_subset * bin_width

            fig.add_trace(go.Scatter(
                x=x_grid, y=counts,
                mode="lines",
                fill="tozeroy",
                line=dict(color=color_map[rfm], width=2),
                fillcolor=color_map[rfm].replace(")", ", 0.25)").replace("rgb", "rgba")
                            if color_map[rfm].startswith("rgb") else color_map[rfm],
                opacity=0.6,
                name=f"rfm = {rfm}",
                hovertemplate=f"rfm = {rfm}<br>VOM multiplier: %{{x:.3f}}<br>Count: %{{y:.0f}}<extra></extra>",
            ))
        except np.linalg.LinAlgError:
            # Fallback if KDE fails
            fig.add_trace(go.Scatter(
                x=[sub["vom_multiplier"].mean()] * 2,
                y=[0, n_points_in_subset],
                mode="lines",
                line=dict(color=color_map[rfm], width=2, dash="dot"),
                name=f"rfm = {rfm} (error)",
                hovertemplate=f"rfm = {rfm}<br>Mean VOM multiplier: {sub['vom_multiplier'].mean():.3f}<br>Count: {n_points_in_subset}<extra></extra>",
            ))

    fig.update_layout(
        title=dict(
            text=f"Count Distribution of VOM multipliers across region pairs, by rfm ({case_study})",
            font=dict(family=FONT_FAMILY, size=18, color=FONT_COLOR),
        ),
        xaxis=dict(
            title=dict(text="VOM multiplier", font=dict(family=FONT_FAMILY, size=14, color=FONT_COLOR)),
            tickfont=dict(family=FONT_FAMILY, size=12, color=FONT_COLOR),
            showgrid=True, gridcolor="lightgrey",
        ),
        yaxis=dict(
            title=dict(text="Number of relationships", font=dict(family=FONT_FAMILY, size=14, color=FONT_COLOR)),
            tickfont=dict(family=FONT_FAMILY, size=12, color=FONT_COLOR),
            showgrid=True, gridcolor="lightgrey",
        ),
        legend=dict(title=dict(text="Sensitivity run"), font=dict(family=FONT_FAMILY, size=12, color=FONT_COLOR)),
        font=dict(family=FONT_FAMILY, color=FONT_COLOR),
        plot_bgcolor="white", paper_bgcolor="white",
        width=1200, height=800,
    )

    os.makedirs(os.path.join(output_path, "figures"), exist_ok=True)
    fig.write_image(os.path.join(output_path, "figures", f"relationship_vom_multiplier_counts_{case_study}.png"), scale=4)
    fig.write_html(os.path.join(output_path, "figures", f"relationship_vom_multiplier_counts_{case_study}.html"), include_plotlyjs="cdn")

    return fig, combined_df

# ------------------------------------------------------------------------------
# Standalone execution: reproduces the original script's behaviour at a single
# default ceiling. Sensitivity sweeps belong in trading_module.py, calling the
# two functions above directly.
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    START = time.perf_counter()

    case_study = "h2bb"
    relationship_factor_magnitude = 1.5
    relationship_factor_magnitude_sens_list = [1, 1.2, 1.5, 1.8, 2]

    base = build_relationship_base(case_study=case_study)
    rel_df = calculate_relationship_factor(base, relationship_factor_magnitude=relationship_factor_magnitude)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data")
    os.makedirs(os.path.join(data_path, case_study), exist_ok=True)
    rel_df.to_excel(
        os.path.join(data_path, case_study, "relationship_analysis_output.xlsx"),
        sheet_name="relations",
        index=False,
    )

    print("Plotting relationship sensitivity")
    relationship_visualisation(base, relationship_factor_magnitude_sens_list, case_study)

    print("Plotting relationship distribution")
    relationship_distribution_visualisation(base, relationship_factor_magnitude_sens_list, case_study)

    STOP = time.perf_counter()
    print("Total execution time of script", round((STOP - START), 1), "s")
#%%