import os
import sys
import time
 
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
#%%

# ==============================================================================
# BLOCK 1 (NEW) -- insert above the "Stage 1" comment / build_relationship_base()
# Alternative alliance-index normalisations. To drop a variant: delete its entry
# in ALLIANCE_INDEX_VARIANTS (the function can stay or go). To add a variant: write
# a function f(base_df, ctx) -> Series aligned with base_df and register it.
# Everything downstream (calculate_relationship_factor, both visualisations) picks
# up registered variants automatically via their `alliance_index_<name>` column.
# ==============================================================================

def _alliance_index_total_max(base_df, ctx):
    """
    Weighted alliance overlap, normalised against the best-connected pair
    (global maximum of shared_weight). Former baseline `alliance_index`.
    """
    return base_df["shared_weight"] / base_df["shared_weight"].max()

def _alliance_index_regional_max(base_df, ctx):
    """
    Normalise against "what is usual in the region".
 
    The reference of a region is the highest shared_weight of any pair with at
    least one member in that region. The reference of a pair is the mean of the
    references of its two countries. Since shared_weight_ij is <= both regional
    references, it is also <= their mean, so alliance_index <= 1 by construction.
    """
    region = ctx["countries_df"].drop_duplicates("country").set_index("country")["un_m49_regions"]
    region_i = base_df["country"].map(region)
    region_j = base_df["friends"].map(region)
 
    missing = sorted(set(base_df.loc[region_i.isna(), "country"]) | set(base_df.loc[region_j.isna(), "friends"]))
    if missing:
        print(f"{len(missing)} country code(s) without un_m49_regions (regional index set to 0): {missing}")
 
    incident = pd.concat([
        pd.DataFrame({"region": region_i, "shared_weight": base_df["shared_weight"]}),
        pd.DataFrame({"region": region_j, "shared_weight": base_df["shared_weight"]}),
    ])
    region_max = incident.groupby("region")["shared_weight"].max()
 
    reference = (region_i.map(region_max) + region_j.map(region_max)) / 2
    index = np.where(reference > 0, base_df["shared_weight"] / reference, 0.0)
    return pd.Series(index, index=base_df.index)
  
def _alliance_index_category_max(base_df, ctx):
    """
    Normalise per alliance category.
 
    For every category, the weighted sum of shared alliances of a pair is divided
    by the maximum over all pairs in that category (each category index in [0, 1]).
    The category indices are combined as an unweighted mean. The best-connected pair
    rarely tops every category, so the mean stays below 1; the combined index is
    therefore divided by its own maximum, so that the best-connected pair has
    index 1 (same anchor as total_max).
    Alliances without a category are ignored by this variant.
    """
    by_category = (
        ctx["pairs"]
        .groupby(["country", "friends", "category"])["weight"].sum()
        .unstack("category", fill_value=0)
    )
    by_category = by_category / by_category.max()
 
    category_index = (
        base_df[["country", "friends"]]
        .merge(by_category.reset_index(), on=["country", "friends"], how="left")
        .drop(columns=["country", "friends"])
        .fillna(0)
        .mean(axis=1)
    )
    category_index = category_index / category_index.max()
    return pd.Series(category_index.to_numpy(), index=base_df.index)
 
ALLIANCE_INDEX_VARIANTS = {
    "total_max": _alliance_index_total_max,
    "regional_max": _alliance_index_regional_max,
    "category_max": _alliance_index_category_max,
}

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
        relations_df[["alliance", "category", "weight", "member_states"]]
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
    pairs = pairs.rename(columns={"country_friend": "friends"})[["country", "friends", "alliance", "category", "weight"]]
 
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

    # --- alliance-index variants (see ALLIANCE_INDEX_VARIANTS) ---
    ctx = {"countries_df": countries_df, "pairs": pairs}
    for name, func in ALLIANCE_INDEX_VARIANTS.items():
        base_df[f"alliance_index_{name}"] = func(base_df, ctx)


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

# ==============================================================================
# BLOCK 3 (NEW + REPLACES calculate_relationship_factor)
# ==============================================================================
 
def get_index_variants(base) -> list:
    """Names of all alliance-index variants available in base, i.e. every
    `alliance_index_<name>` column (order = registry order)."""
    base_df = base["base_df"] if isinstance(base, dict) else base
    return [c[len("alliance_index_"):] for c in base_df.columns if c.startswith("alliance_index_")]
 
 
def _resolve_variants(base, variants_to_consider) -> list:
    """None -> all available variants; str -> [str]; list -> checked against what is available."""
    available = get_index_variants(base)
    if variants_to_consider is None:
        return available
    variants = [variants_to_consider] if isinstance(variants_to_consider, str) else list(variants_to_consider)
    unknown = [v for v in variants if v not in available]
    if unknown:
        raise ValueError(f"Unknown alliance index variant(s) {unknown}. Available: {available}")
    return variants

def calculate_relationship_factor(base, relationship_factor_magnitude: float, variants_to_consider="total_max", scenario="Base") -> pd.DataFrame:
    """
    Compute the relationship-factor / vom_multiplier table at a given rfm ceiling,
    for one or several alliance-index variants at once.
 
        vom_multiplier_<variant>_ij = relationship_factor_magnitude ** (1 - alliance_index_<variant>_ij)
 
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
    variants_to_consider : str, list of str or None
        Alliance-index variant(s) to calculate, e.g. "total_max" or
        ["total_max", "regional_max", "category_max"]. None -> all available.
 
    Returns
    -------
    DataFrame[region1, region2, counter, shared_weight,
              alliance_index_<variant> ..., vom_multiplier_<variant> ...,
              scenario], sorted by the vom_multiplier of the first variant.
    """
    if relationship_factor_magnitude < 1:
        raise ValueError(f"relationship_factor_magnitude must be > 1, got {relationship_factor_magnitude}")
 
    base_df = base["base_df"] if isinstance(base, dict) else base
    variants = _resolve_variants(base_df, variants_to_consider)
 
    # keep only the alliance_index columns of the requested variants
    unused = [f"alliance_index_{v}" for v in get_index_variants(base_df) if v not in variants]
    rel_df = base_df.drop(columns=unused).copy()
 
    for variant in variants:
        rel_df[f"vom_multiplier_{variant}"] = relationship_factor_magnitude ** (1 - rel_df[f"alliance_index_{variant}"])
        rel_df.loc[rel_df["at_war"] == 1, f"vom_multiplier_{variant}"] = 10
 
    rel_df["continent"] = rel_df["country"].str.split("-").str[0]
    rel_df["pairing"] = rel_df["country"] + " - " + rel_df["friends"]
 
    rel_df["scenario"] = scenario
    rel_df = rel_df.sort_values(f"vom_multiplier_{variants[0]}").reset_index(drop=True)
    rel_df = rel_df.rename(columns={"country": "region1", "friends": "region2"})
    rel_df = rel_df.drop(columns={"at_war", "continent", "pairing"})
    return rel_df

# ==============================================================================
# BLOCK 4 (NEW) -- insert above relationship_visualisation()
# Generic panel layout: one panel per variant, side by side, in ONE figure
# (axes x/y, x2/y2, ...). Used by both visualisation functions below.
# ==============================================================================
 
def _apply_variant_panels(fig, variants, x_title, y_title, x_range=None, y_range=None, gap=0.05):
    """
    Lay out one panel per index variant side by side. Traces must be added with
    xaxis=f"x{suffix}", yaxis=f"y{suffix}", where suffix is "" for the first panel
    and str(i + 1) for the others. Panels 2..n are matched to panel 1 so the
    variants are directly comparable.
    """
    FONT_FAMILY = "Times New Roman"
    FONT_COLOR = "black"
 
    n = len(variants)
    panel_width = (1 - gap * (n - 1)) / n
 
    for i, variant in enumerate(variants):
        suffix = "" if i == 0 else str(i + 1)
        x0 = i * (panel_width + gap)
 
        fig.update_layout({
            f"xaxis{suffix}": dict(
                domain=[x0, x0 + panel_width], anchor=f"y{suffix}",
                title=dict(text=x_title, font=dict(family=FONT_FAMILY, size=14, color=FONT_COLOR)),
                tickfont=dict(family=FONT_FAMILY, size=12, color=FONT_COLOR),
                showgrid=True, gridcolor="lightgrey",
                range=x_range, matches=None if i == 0 else "x",
            ),
            f"yaxis{suffix}": dict(
                domain=[0, 1], anchor=f"x{suffix}",
                title=dict(text=y_title if i == 0 else "", font=dict(family=FONT_FAMILY, size=14, color=FONT_COLOR)),
                tickfont=dict(family=FONT_FAMILY, size=12, color=FONT_COLOR),
                showgrid=True, gridcolor="lightgrey",
                range=y_range, matches=None if i == 0 else "y",
            ),
        })
        fig.add_annotation(
            x=x0 + panel_width / 2, y=1.0, xref="paper", yref="paper",
            text=f"Index variant: {variant}", showarrow=False, yanchor="bottom",
            font=dict(family=FONT_FAMILY, size=14, color=FONT_COLOR),
        )


# ==============================================================================
# BLOCK 5 (REPLACES relationship_visualisation)
# ==============================================================================
 
def relationship_visualisation(base, rfm_sweep_list, case_study, output_path=None, variants_to_consider=None):
    FONT_FAMILY = "Times New Roman"
    FONT_COLOR = "black"
 
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "output")
 
    # None -> every variant available in base
    variants = _resolve_variants(base, variants_to_consider)
 
    # --- compute rel_df for every rfm value in the sweep (all variants at once) ---
    rfm_dfs = []
    for rfm in rfm_sweep_list:
        df = calculate_relationship_factor(base, relationship_factor_magnitude=rfm, variants_to_consider=variants)
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
 
    # fig = go.Figure()
 
    # for i, variant in enumerate(variants):
    #     suffix = "" if i == 0 else str(i + 1)
    #     index_col = f"alliance_index_{variant}"
    #     vom_col = f"vom_multiplier_{variant}"
    #     for rfm in rfm_values:
    #         sub = combined_df[combined_df["rfm"] == rfm].sort_values(index_col)
    #         fig.add_trace(go.Scatter(
    #             x=sub[index_col], y=sub[vom_col],
    #             xaxis=f"x{suffix}", yaxis=f"y{suffix}",
    #             mode="markers",
    #             marker=dict(size=6, color=color_map[rfm], opacity=1.0,
    #                          line=dict(width=1, color=color_map[rfm])),
    #             name=f"rfm = {rfm}",
    #             legendgroup=f"rfm = {rfm}",
    #             showlegend=(i == 0),  # one legend entry per rfm, not per panel
    #             hovertemplate=(
    #                 sub["region1"].astype(str) + " – " + sub["region2"].astype(str)
    #                 + "<br>Alliance index: %{x:.3f}<br>VOM multiplier: %{y:.3f}"
    #                 + f"<br>rfm: {rfm}<br>variant: {variant}<extra></extra>"
    #             ),
    #         ))
 
    # _apply_variant_panels(fig, variants, x_title="Alliance index", y_title="VOM multiplier",
    #                       x_range=[0, 1], y_range=[0, 2])
 
    # fig.update_layout(
    #     title=dict(
    #         text=f"Relationship factor sensitivity — VOM multiplier vs. alliance index ({case_study})",
    #         font=dict(family=FONT_FAMILY, size=18, color=FONT_COLOR),
    #     ),
    #     legend=dict(title=dict(text="Sensitivity run"), font=dict(family=FONT_FAMILY, size=12, color=FONT_COLOR)),
    #     font=dict(family=FONT_FAMILY, color=FONT_COLOR),
    #     plot_bgcolor="white", paper_bgcolor="white",
    #     width=max(1200, 450 * len(variants) + 300), height=800,
    #     margin=dict(t=120),
    # )
 
    # os.makedirs(os.path.join(output_path, "relationship_analysis", "figures"), exist_ok=True)
    # fig.write_image(os.path.join(output_path, "relationship_analysis", "figures", f"relationship_vom_multiplier_sens_{case_study}.png"), scale=4)
    # fig.write_html(os.path.join(output_path, "relationship_analysis", "figures", f"relationship_vom_multiplier_sens_{case_study}.html"), include_plotlyjs="cdn")
 
    return combined_df

def alliance_overlap_distribution_visualisation(
    base,
    case_study,
    output_path=None,
    include_zero=True,
    log_y=True,
    max_bins=40,
    index_bins=20,
    variants_to_consider=None):
    """
    Histograms of the alliance-overlap measures per country pair, in ONE figure:
      row 1, left  : `counter`       -- number of alliances a pair shares
      row 1, right : `shared_weight` -- summed weight of the shared alliances
      row 2        : `alliance_index` -- one panel per index variant (side by side,
                     y-axes matched so the variants are directly comparable)
 
    `counter` and `shared_weight` are raw measures and do not depend on the
    normalisation; only the alliance_index panels differ between variants.
 
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
    index_bins : int
        Number of equal-width bins between 0 and 1 for the alliance_index panels.
    variants : list of str or None
        Alliance-index variants to show in row 2. None -> every variant available
        in base (baseline "global_max" plus all registered variants).
 
    Also marks the maximum and the 95th percentile of the non-zero shared_weight
    on the shared_weight plot: the max is the denominator of the baseline
    alliance_index (total_max"), so the gap between the two lines shows how
    much the max-normalisation compresses the bulk of pairs.
 
    Returns
    -------
    fig : plotly Figure
    """
    FONT_FAMILY = "Times New Roman"
    FONT_COLOR = "black"
 
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "output")
 
    variants = _resolve_variants(base, variants_to_consider)
 
    base_df = base["base_df"] if isinstance(base, dict) else base
    df = base_df.copy()
 
    n_total = len(df)
    n_zero = int((df["shared_weight"] == 0).sum())
    nonzero_w = df.loc[df["shared_weight"] > 0, "shared_weight"]
    w_max = nonzero_w.max()
    w_p95 = nonzero_w.quantile(0.95)
 
    if not include_zero:
        df = df[df["shared_weight"] > 0]
 
    # --- counter (integer-valued -> one bar per value) ---
    counter_counts = df["counter"].astype(int).value_counts().sort_index()
 
    # --- shared_weight (binned; one bar per value if few unique values) ---
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
 
    # --- panel definitions: a list of rows, each row a list of panels ---
    row_raw = [
        dict(x=counter_counts.index, y=counter_counts.values, width=None, color="#1b9e77", name="counter",
             title="Shared alliances per pair (counter)", x_title="Number of shared alliances",
             dtick=1, x_range=None, refs=[],
             hover="Shared alliances: %{x}<br>Pairs: %{y}"),
        dict(x=w_x, y=w_y, width=w_width, color="#d95f02", name="shared_weight",
             title="Summed weight of shared alliances (shared_weight)", x_title="Summed alliance weight",
             dtick=None, x_range=None,
             refs=[(w_p95, f"95th pct (non-zero) = {w_p95:.2f}", "dash", 0.97),
                   (w_max, f"max = {w_max:.2f}", "dot", 0.89)],
             hover="Shared weight: %{x:.2f}<br>Pairs: %{y}"),
    ]
 
    index_palette = ["#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    row_index = []
    for i, variant in enumerate(variants):
        index_col = f"alliance_index_{variant}"
        if index_col not in df.columns:
            raise ValueError(f"Unknown alliance index variant '{variant}'. Available: {get_index_variants(df)}")
        idx = df[index_col]
        counts, idx_edges = np.histogram(idx, bins=index_bins, range=(0, 1))
        row_index.append(dict(
            x=(idx_edges[:-1] + idx_edges[1:]) / 2, y=counts, width=(1 / index_bins) * 0.95,
            color=index_palette[i % len(index_palette)], name=f"alliance_index ({variant})",
            title=f"Alliance index: {variant} (max = {idx.max():.2f})", x_title="Alliance index",
            dtick=None, x_range=[0, 1], refs=[],
            hover="Alliance index: %{x:.2f}<br>Pairs: %{y}",
        ))
 
    rows = [(row_raw, False), (row_index, True)]  # (panels, match y-axes within row)
 
    # --- one figure, manual grid of panels (axes x/y, x2/y2, ...) ---
    fig = go.Figure()
 
    axis_font = dict(family=FONT_FAMILY, size=14, color=FONT_COLOR)
    tick_font = dict(family=FONT_FAMILY, size=12, color=FONT_COLOR)
    y_type = "log" if log_y else "linear"
 
    n_rows = len(rows)
    h_gap, v_gap = 0.06, 0.16
    row_height = (1 - v_gap * (n_rows - 1)) / n_rows
 
    k = 0  # running panel counter -> axis suffix
    for r, (panels, share_y) in enumerate(rows):
        y1 = 1 - r * (row_height + v_gap)
        y0 = y1 - row_height
        panel_width = (1 - h_gap * (len(panels) - 1)) / len(panels)
        first_suffix = "" if k == 0 else str(k + 1)
 
        for c, p in enumerate(panels):
            suffix = "" if k == 0 else str(k + 1)
            x0 = c * (panel_width + h_gap)
 
            fig.add_trace(go.Bar(
                x=p["x"], y=p["y"], width=p["width"],
                xaxis=f"x{suffix}", yaxis=f"y{suffix}",
                marker=dict(color=p["color"], line=dict(width=1, color="black")),
                name=p["name"],
                hovertemplate=p["hover"] + "<extra></extra>",
            ))
 
            fig.update_layout({
                f"xaxis{suffix}": dict(
                    domain=[x0, x0 + panel_width], anchor=f"y{suffix}",
                    dtick=p["dtick"], range=p["x_range"],
                    title=dict(text=p["x_title"], font=axis_font),
                    tickfont=tick_font, showgrid=True, gridcolor="lightgrey"),
                f"yaxis{suffix}": dict(
                    domain=[y0, y1], anchor=f"x{suffix}", type=y_type,
                    matches=f"y{first_suffix}" if (share_y and c > 0) else None,
                    title=dict(text="Number of country pairs", font=axis_font),
                    tickfont=tick_font, showgrid=True, gridcolor="lightgrey"),
            })
 
            # reference lines (and labels) on this panel
            for val, label, dash, y_lab in p["refs"]:
                fig.add_shape(type="line", x0=val, x1=val, y0=0, y1=1,
                              xref=f"x{suffix}", yref=f"y{suffix} domain",
                              line=dict(color="black", width=1.5, dash=dash))
                fig.add_annotation(x=val, y=y_lab, xref=f"x{suffix}", yref=f"y{suffix} domain",
                                   text=label, showarrow=False, xanchor="right", xshift=-4,
                                   font=dict(family=FONT_FAMILY, size=11, color=FONT_COLOR))
 
            # panel title (placed manually in paper coordinates)
            fig.add_annotation(x=x0 + panel_width / 2, y=y1, xref="paper", yref="paper",
                               text=p["title"], showarrow=False, yanchor="bottom",
                               font=dict(family=FONT_FAMILY, size=14, color=FONT_COLOR))
            k += 1
 
    zero_note = (f"{n_zero} of {n_total} pairs ({n_zero / n_total:.0%}) share no alliance"
                 + ("" if include_zero else " (excluded from plots)"))
    fig.update_layout(
        title=dict(
            text=f"Distribution of alliance overlap and alliance index across region pairs ({case_study})"
                 f"<br><sup>{zero_note}</sup>",
            font=dict(family=FONT_FAMILY, size=18, color=FONT_COLOR),
        ),
        showlegend=False,
        bargap=0.05,
        font=dict(family=FONT_FAMILY, color=FONT_COLOR),
        plot_bgcolor="white", paper_bgcolor="white",
        width=max(1400, 450 * len(variants) + 200), height=1000,
        margin=dict(t=130),
    )
 
    os.makedirs(os.path.join(output_path, "relationship_analysis", "figures"), exist_ok=True)
    fig.write_image(os.path.join(output_path, "relationship_analysis", "figures", f"alliance_overlap_distribution_{case_study}.png"), scale=4)
    fig.write_html(os.path.join(output_path, "relationship_analysis", "figures", f"alliance_overlap_distribution_{case_study}.html"), include_plotlyjs="cdn")
 
    return fig

# ==============================================================================
# BLOCK 6 (REPLACES relationship_distribution_visualisation)
# ==============================================================================
 
def relationship_distribution_visualisation(base, rfm_sweep_list, case_study, output_path=None, n_points=300, variants_to_consider=None):
    FONT_FAMILY = "Times New Roman"
    FONT_COLOR = "black"
 
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "output")
 
    # None -> every variant available in base
    variants = _resolve_variants(base, variants_to_consider)
 
    rfm_dfs = []
    for rfm in rfm_sweep_list:
        df = calculate_relationship_factor(base, relationship_factor_magnitude=rfm, variants_to_consider=variants)
        df = df.copy()
        df["rfm"] = rfm
        rfm_dfs.append(df)
    combined_df = pd.concat(rfm_dfs, ignore_index=True)
 
    rfm_values = sorted(combined_df["rfm"].unique())
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    color_map = {rfm: palette[i % len(palette)] for i, rfm in enumerate(rfm_values)}
 
    # shared x-range across all curves and all variants (at_war pairs, vom = 10, excluded)
    vom_all = combined_df[[f"vom_multiplier_{v}" for v in variants]].to_numpy()
    vom_all = vom_all[vom_all < 10]
    x_min = vom_all.min()
    x_max = vom_all.max()
    x_grid = np.linspace(x_min, x_max, n_points)
 
    fig = go.Figure()
 
    for i, variant in enumerate(variants):
        suffix = "" if i == 0 else str(i + 1)
        vom_col = f"vom_multiplier_{variant}"
 
        for rfm in rfm_values:
            vom = combined_df.loc[combined_df["rfm"] == rfm, vom_col]
            vom = vom[vom < 10]
            n_points_in_subset = len(vom)
 
            # panel assignment + one legend entry per rfm (not per panel)
            panel_kwargs = dict(xaxis=f"x{suffix}", yaxis=f"y{suffix}",
                                legendgroup=f"rfm = {rfm}", showlegend=(i == 0))
 
            # Check if data has sufficient variance
            if n_points_in_subset < 2 or np.var(vom) < 1e-10:
                # Plot a vertical line at the constant value
                fig.add_trace(go.Scatter(
                    x=[vom.iloc[0]] * 2,
                    y=[0, n_points_in_subset],  # Use actual count for constant case
                    mode="lines",
                    line=dict(color=color_map[rfm], width=2, dash="dot"),
                    name=f"rfm = {rfm} (constant)",
                    hovertemplate=f"rfm = {rfm}<br>Constant VOM multiplier: {vom.iloc[0]:.3f}<br>Count: {n_points_in_subset}<br>variant: {variant}<extra></extra>",
                    **panel_kwargs,
                ))
                continue
 
            try:
                kde = gaussian_kde(vom)
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
                    hovertemplate=f"rfm = {rfm}<br>VOM multiplier: %{{x:.3f}}<br>Count: %{{y:.0f}}<br>variant: {variant}<extra></extra>",
                    **panel_kwargs,
                ))
            except np.linalg.LinAlgError:
                # Fallback if KDE fails
                fig.add_trace(go.Scatter(
                    x=[vom.mean()] * 2,
                    y=[0, n_points_in_subset],
                    mode="lines",
                    line=dict(color=color_map[rfm], width=2, dash="dot"),
                    name=f"rfm = {rfm} (error)",
                    hovertemplate=f"rfm = {rfm}<br>Mean VOM multiplier: {vom.mean():.3f}<br>Count: {n_points_in_subset}<br>variant: {variant}<extra></extra>",
                    **panel_kwargs,
                ))
 
    _apply_variant_panels(fig, variants, x_title="VOM multiplier", y_title="Number of relationships")
 
    fig.update_layout(
        title=dict(
            text=f"Count Distribution of VOM multipliers across region pairs, by rfm ({case_study})",
            font=dict(family=FONT_FAMILY, size=18, color=FONT_COLOR),
        ),
        legend=dict(title=dict(text="Sensitivity run"), font=dict(family=FONT_FAMILY, size=12, color=FONT_COLOR)),
        font=dict(family=FONT_FAMILY, color=FONT_COLOR),
        plot_bgcolor="white", paper_bgcolor="white",
        width=max(1200, 450 * len(variants) + 300), height=800,
        margin=dict(t=120),
    )
 
    os.makedirs(os.path.join(output_path, "relationship_analysis", "figures"), exist_ok=True)
    fig.write_image(os.path.join(output_path, "relationship_analysis", "figures", f"relationship_vom_multiplier_counts_{case_study}.png"), scale=4)
    fig.write_html(os.path.join(output_path, "relationship_analysis", "figures", f"relationship_vom_multiplier_counts_{case_study}.html"), include_plotlyjs="cdn")
 
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
    variants_list = ["total_max", "regional_max", "category_max"]
    rel_df = calculate_relationship_factor(base, relationship_factor_magnitude, variants_list)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data")
    output_path = os.path.join(script_dir, "output")
    os.makedirs(os.path.join(data_path, case_study), exist_ok=True)
    rel_df.to_excel(
        os.path.join(output_path, "relationship_analysis", "results", "relationship_analysis_output.xlsx"),
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

## AI statement: During the preparation of this work the author(s) used Claude in order to develop software tools used in this publication. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.
#%%