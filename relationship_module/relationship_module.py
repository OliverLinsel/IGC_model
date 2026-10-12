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
        enemy_pairs_mirrored = enemy_pairs.rename(columns={"country": "friends", "friends": "country"})
        enemy_pairs_full = pd.concat([enemy_pairs, enemy_pairs_mirrored], ignore_index=True).drop_duplicates()

        base_df = base_df.merge(enemy_pairs_full, on=["country", "friends"], how="left")
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

    # --- rainbow palette, one color per rfm value ---
    rfm_values = sorted(combined_df["rfm"].unique())
    rainbow_palette = [
        '#ff0000', '#ff7f00', '#ffff00', '#7fff00', '#00ff00',
        '#00ff7f', '#00ffff', '#007fff', '#0000ff', '#7f00ff'
    ]
    color_map = {rfm: rainbow_palette[i % len(rainbow_palette)] for i, rfm in enumerate(rfm_values)}

    fig = go.Figure()

    for rfm in rfm_values:
        sub = combined_df[combined_df["rfm"] == rfm].sort_values("alliance_index")
        fig.add_trace(go.Scatter(
            x=sub["alliance_index"], y=sub["" \
            "vom_multiplier"],
            mode="markers",
            marker=dict(size=6, color=color_map[rfm], opacity=1.0,
                         line=dict(width=1, color=color_map[rfm])),
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
            range=[0, 2],  # Already showing only positive y-axis
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

def alliance_overlap_distribution_visualisation(
    base,
    case_study,
    output_path=None,
    include_zero=True,
    log_y=True,
    max_bins=40):
    """
    Two adjacent histograms of the raw alliance-overlap measures per country pair:
      left  : `counter`       -- number of alliances a pair shares
      right : `shared_weight` -- summed weight of the shared alliances
 
    Parameters
    ----------
    base : dict (as returned by build_relationship_base()) or the base_df itself
    case_study : str, used in title / file names
    output_path : str, defaults to <script_dir>/output
    include_zero : bool
        Pairs sharing no alliance are usually the largest group and can dwarf
        everything else. Set False to plot only pairs with overlap > 0.
    log_y : bool
        Log-scale the y-axis (recommended: the distributions are heavily skewed).
    max_bins : int
        Upper limit on histogram bins for `shared_weight` (continuous weights).
 
    Also marks the maximum and the 95th percentile of the non-zero shared_weight
    on the right plot: the max is the denominator of the current alliance_index,
    so the gap between the two lines shows how much the max-normalisation
    compresses the bulk of pairs.
 
    Returns
    -------
    fig : plotly Figure
    """
    FONT_FAMILY = "Times New Roman"
    FONT_COLOR = "black"
 
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "output")
 
    base_df = base["base_df"] if isinstance(base, dict) else base
    df = base_df[["counter", "shared_weight"]].copy()
 
    n_total = len(df)
    n_zero = int((df["shared_weight"] == 0).sum())
    nonzero_w = df.loc[df["shared_weight"] > 0, "shared_weight"]
    w_max = nonzero_w.max()
    w_p95 = nonzero_w.quantile(0.95)
 
    if not include_zero:
        df = df[df["shared_weight"] > 0]
 
    # --- left: counter (integer-valued -> one bar per value) ---
    counter_counts = df["counter"].astype(int).value_counts().sort_index()
 
    # --- right: shared_weight (binned; one bar per value if few unique values) ---
    w = df["shared_weight"]
    if w.nunique() <= max_bins:
        w_counts = w.value_counts().sort_index()
        w_x, w_y = w_counts.index.to_numpy(dtype=float), w_counts.values
        w_width = (np.diff(w_x).min() * 0.9) if len(w_x) > 1 else 0.9
    else:
        edges = np.histogram_bin_edges(w, bins="auto")
        if len(edges) - 1 > max_bins:
            edges = np.linspace(w.min(), w.max(), max_bins + 1)
        w_y, edges = np.histogram(w, bins=edges)
        w_x = (edges[:-1] + edges[1:]) / 2
        w_width = np.diff(edges) * 0.95
 
    # one figure, two side-by-side panels built from two x/y axis pairs (x/y and x2/y2)
    fig = go.Figure()
 
    fig.add_trace(go.Bar(
        x=counter_counts.index, y=counter_counts.values,
        xaxis="x", yaxis="y",
        marker=dict(color="#1b9e77", line=dict(width=1, color="black")),
        name="counter",
        hovertemplate="Shared alliances: %{x}<br>Pairs: %{y}<extra></extra>",
    ))
 
    fig.add_trace(go.Bar(
        x=w_x, y=w_y, width=w_width,
        xaxis="x2", yaxis="y2",
        marker=dict(color="#d95f02", line=dict(width=1, color="black")),
        name="shared_weight",
        hovertemplate="Shared weight: %{x:.2f}<br>Pairs: %{y}<extra></extra>",
    ))
 
    # reference lines (and labels) on the shared_weight panel
    for val, label, dash, y_lab in [(w_p95, f"95th pct (non-zero) = {w_p95:.2f}", "dash", 0.97),
                                    (w_max, f"max = {w_max:.2f}", "dot", 0.89)]:
        fig.add_shape(type="line", x0=val, x1=val, y0=0, y1=1,
                      xref="x2", yref="y2 domain",
                      line=dict(color="black", width=1.5, dash=dash))
        fig.add_annotation(x=val, y=y_lab, xref="x2", yref="y2 domain",
                           text=label, showarrow=False, xanchor="right", xshift=-4,
                           font=dict(family=FONT_FAMILY, size=11, color=FONT_COLOR))
 
    # panel titles (placed manually in paper coordinates)
    for x_pos, text in [(0.225, "Shared alliances per pair (counter)"),
                        (0.775, "Summed weight of shared alliances (shared_weight)")]:
        fig.add_annotation(x=x_pos, y=1.0, xref="paper", yref="paper", text=text,
                           showarrow=False, yanchor="bottom",
                           font=dict(family=FONT_FAMILY, size=14, color=FONT_COLOR))
 
    axis_font = dict(family=FONT_FAMILY, size=14, color=FONT_COLOR)
    tick_font = dict(family=FONT_FAMILY, size=12, color=FONT_COLOR)
    y_type = "log" if log_y else "linear"
 
    fig.update_layout(
        xaxis=dict(domain=[0.0, 0.45], anchor="y", dtick=1,
                   title=dict(text="Number of shared alliances", font=axis_font),
                   tickfont=tick_font, showgrid=True, gridcolor="lightgrey"),
        yaxis=dict(domain=[0.0, 1.0], anchor="x", type=y_type,
                   title=dict(text="Number of country pairs", font=axis_font),
                   tickfont=tick_font, showgrid=True, gridcolor="lightgrey"),
        xaxis2=dict(domain=[0.55, 1.0], anchor="y2",
                    title=dict(text="Summed alliance weight", font=axis_font),
                    tickfont=tick_font, showgrid=True, gridcolor="lightgrey"),
        yaxis2=dict(domain=[0.0, 1.0], anchor="x2", type=y_type,
                    title=dict(text="Number of country pairs", font=axis_font),
                    tickfont=tick_font, showgrid=True, gridcolor="lightgrey"),
    )
 
    zero_note = (f"{n_zero} of {n_total} pairs ({n_zero / n_total:.0%}) share no alliance"
                 + ("" if include_zero else " (excluded from plots)"))
    fig.update_layout(
        title=dict(
            text=f"Distribution of alliance overlap across region pairs ({case_study})"
                 f"<br><sup>{zero_note}</sup>",
            font=dict(family=FONT_FAMILY, size=18, color=FONT_COLOR),
        ),
        showlegend=False,
        bargap=0.05,
        font=dict(family=FONT_FAMILY, color=FONT_COLOR),
        plot_bgcolor="white", paper_bgcolor="white",
        width=1400, height=700,
        margin=dict(t=130),
    )
 
    os.makedirs(os.path.join(output_path, "figures"), exist_ok=True)
    fig.write_image(os.path.join(output_path, "figures", f"alliance_overlap_distribution_{case_study}.png"), scale=4)
    fig.write_html(os.path.join(output_path, "figures", f"alliance_overlap_distribution_{case_study}.html"), include_plotlyjs="cdn")
 
    return fig

def relationship_distribution_visualisation(base, rfm_sweep_list, case_study, output_path=None, n_points=300):
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
    relationship_factor_magnitude_sens_list = [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 1.75, 2.0]

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

    print("Plotting alliance overlap distribution")
    alliance_overlap_distribution_visualisation(base, case_study)

    print("Plotting relationship sensitivity")
    relationship_visualisation(base, relationship_factor_magnitude_sens_list, case_study)

    print("Plotting relationship distribution")
    relationship_distribution_visualisation(base, relationship_factor_magnitude_sens_list, case_study)

    STOP = time.perf_counter()
    print("Total execution time of script", round((STOP - START), 1), "s")

## AI statement: During the preparation of this work the author(s) used Claude in order to develop software tools used in this pubication. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.
#%%