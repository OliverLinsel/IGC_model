import os
import sys
import time
 
import geopandas as gpd
import numpy as np
import pandas as pd
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
 
def calculate_relationship_factor(base, target_max_multiplier:1.5, scenario="Base") -> pd.DataFrame:
    """
    Compute the relationship-factor / vom_multiplier table at a given ceiling.
 
        vom_multiplier_ij = exp(gamma * (1 - alliance_index_ij))
        gamma = ln(target_max_multiplier)
 
    so vom_multiplier is exactly 1 for the best-connected pair and exactly
    target_max_multiplier for a pair sharing no alliance at all. Pairs flagged
    `at_war` are overridden to inf (infeasible route) regardless of alliance_index
    -- treated as a hard constraint rather than folded into the alliance curve,
    so trading_module.py can filter those out (`rel_df.vom_multiplier < np.inf`)
    rather than mistaking a war override for a legitimately weak alliance score.
 
    Parameters
    ----------
    base : dict (as returned by build_relationship_base()) or the base_df itself
    target_max_multiplier : float
        Must be > 1. This is the parameter to sweep in the sensitivity analysis.
 
    Returns
    -------
    DataFrame[country, friends, counter, shared_weight, alliance_index, at_war,
              gamma, vom_multiplier, continent, pairing], sorted by vom_multiplier.
    """
    if target_max_multiplier < 1:
        raise ValueError(f"target_max_multiplier must be > 1, got {target_max_multiplier}")
 
    base_df = base["base_df"] if isinstance(base, dict) else base
    rel_df = base_df.copy()
 
    gamma = np.log(target_max_multiplier)
    rel_df["gamma"] = gamma
    rel_df["vom_multiplier"] = np.exp(gamma * (1 - rel_df["alliance_index"]))
    rel_df.loc[rel_df["at_war"] == 1, "vom_multiplier"] = 10
 
    rel_df["continent"] = rel_df["country"].str.split("-").str[0]
    rel_df["pairing"] = rel_df["country"] + " - " + rel_df["friends"]

    rel_df["scenario"] = scenario
    rel_df = rel_df.sort_values("vom_multiplier").reset_index(drop=True)
    rel_df = rel_df.rename(columns={"country":"region1", "friends":"region2"})
    rel_df = rel_df.drop(columns={"at_war", "continent", "pairing"}) 
    return rel_df

# ------------------------------------------------------------------------------
# Standalone execution: reproduces the original script's behaviour at a single
# default ceiling. Sensitivity sweeps belong in trading_module.py, calling the
# two functions above directly.
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    START = time.perf_counter()
 
    case_study = "h2bb"
    DEFAULT_TARGET_MAX_MULTIPLIER = 1.5
 
    base = build_relationship_base(case_study=case_study)
    rel_df = calculate_relationship_factor(base, target_max_multiplier=DEFAULT_TARGET_MAX_MULTIPLIER)
 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data")
    os.makedirs(os.path.join(data_path, case_study), exist_ok=True)
    rel_df.to_excel(
        os.path.join(data_path, case_study, "relationship_analysis_output.xlsx"),
        sheet_name="relations",
        index=False,
    )
 
    STOP = time.perf_counter()
    print("Total execution time of script", round((STOP - START), 1), "s")
#%%