"""
trade_analysis_module.py
------------------------
Small helper module to pull bilateral trade flows from the UN Comtrade+ API
(https://comtradeplus.un.org) and plot them.

Output dataframe columns
    origin, destination, commodity, year, trade_volume, trade_value_usd
where trade_volume is in TONNES (Comtrade reports net weight in kg).

Quick start
    export COMTRADE_API_KEY="your-key"        # free key from comtradeplus.un.org

    python trade_analysis_module.py --test     # 1) single-year test run (2023)
    python trade_analysis_module.py            # 2) full run: 2019, 2021, 2023, 2025

    from trade_analysis_module import *
    df = get_trade_data(years=YEARS_TEST)                  # exports, one year
    df = get_trade_data(years=YEARS_LIST, flow="both")     # exports + imports
    plot_flow_map(df, year=2023)                           # -> trade_flow_map.html

flow="X"    exports (reporter = origin)               <- default
flow="M"    imports (reporter = destination)
flow="both" both, adds a `flow` column. NOTE: the same shipment is then reported
            twice (by exporter and by importer) - don't sum across flows.
"""
from __future__ import annotations

import os
os.environ["COMTRADE_API_KEY"] = "a15e3337d50adfb9ffce1596e869de22"
import time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

### Robust path handling: works when run normally and in debug mode / interactive sessions
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
module_dir = os.path.join(project_root, "relationship_module")
data_path = os.path.join(module_dir, "data")
raw_path = os.path.join(data_path, "raw")
output_path = os.path.join(module_dir, "output", "trade_analysis")
figures_path = os.path.join(output_path, "figures")
results_path = os.path.join(output_path, "results")
for _p in (raw_path, figures_path, results_path):
    os.makedirs(_p, exist_ok=True)

# ---- paste into trade_analysis_module.py (replaces the previous plot_flow_map) ----
PATHS_FILE = os.path.join(project_root, "econ_module", "output", "model", "paths.csv")
 
# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
API_KEY = None   # paste your key here, or set the COMTRADE_API_KEY environment variable
 
BASE_URL = "https://comtradeapi.un.org"
DATA_ENDPOINT = f"{BASE_URL}/data/v1/get/C/A/HS"            # needs subscription key
PREVIEW_ENDPOINT = f"{BASE_URL}/public/v1/preview/C/A/HS"   # no key, max 500 rows
 

HS_SELECTION = 27 #10, 27   # default HS chapter(s) / code(s); change here or pass hs=... / --hs
HS_LEVEL = 4        # 2 = chapter total only, 4 = headings, 6 = subheadings
 
# cmdCode="10" alone only returns the chapter total, so the sub-codes are requested explicitly.
HS_CEREALS_4DIGIT = {
    "1001": "Wheat and meslin",
    "1002": "Rye",
    "1003": "Barley",
    "1004": "Oats",
    "1005": "Maize (corn)",
    "1006": "Rice",
    "1007": "Grain sorghum",
    "1008": "Buckwheat, millet, canary seed, other cereals",
}
 
YEARS_TEST = [2023]
YEARS_LIST = [2019, 2021, 2023]
 
 
# Optional readable names for file naming (unknown chapters simply become "HS<nn>")
CHAPTER_LABELS = {"10": "cereals", "27": "mineral_fuels", "12": "oil_seeds", "31": "fertilisers",
                  "72": "iron_steel", "84": "machinery", "85": "electrical_equipment"}
 
# Offline fallback if the UN reference table can't be downloaded (level-4 headings only)
FALLBACK_HEADINGS = {
    "10": HS_CEREALS_4DIGIT,
    "27": {
        "2701": "Coal; briquettes made from coal",
        "2702": "Lignite",
        "2703": "Peat",
        "2704": "Coke and semi-coke",
        "2705": "Coal gas, water gas and similar gases",
        "2706": "Tar distilled from coal, lignite or peat",
        "2707": "Oils from high-temperature coal tar",
        "2708": "Pitch and pitch coke from coal tar",
        "2709": "Petroleum oils, crude",
        "2710": "Petroleum oils, not crude (refined products)",
        "2711": "Petroleum gases (incl. natural gas, LNG, LPG)",
        "2712": "Petroleum jelly, paraffin wax",
        "2713": "Petroleum coke, bitumen residues",
        "2714": "Bitumen and asphalt, natural",
        "2715": "Bituminous mixtures",
        "2716": "Electrical energy",
    },
}
 
HS_REFERENCE_URL = f"{BASE_URL}/files/v1/app/reference/HS.json"

 
def _norm_code(x) -> str:
    s = str(x).strip()
    return s.zfill(2) if len(s) == 1 else s       # 1 -> "01"
 
 
def hs_reference(refresh: bool = False) -> dict[str, str]:
    """HS code -> description from the UN Comtrade reference table (cached in data/)."""
    import json
 
    cache = os.path.join(data_path, "hs_reference.json")
    if os.path.exists(cache) and not refresh:
        with open(cache) as fh:
            return json.load(fh)
    try:
        r = requests.get(HS_REFERENCE_URL, timeout=60)
        r.raise_for_status()
        payload = r.json()
        items = payload.get("results", []) if isinstance(payload, dict) else payload
        ref = {}
        for it in items:
            code = str(it.get("id", "")).strip()
            if code.isdigit() and len(code) in (2, 4, 6):
                text = str(it.get("text") or it.get("description") or "")
                ref[code] = text.split(" - ", 1)[1] if text.startswith(code + " - ") else text
        if ref:
            with open(cache, "w") as fh:
                json.dump(ref, fh)
            return ref
        print("WARNING: HS reference table was empty/unreadable.")
    except Exception as e:                                   # noqa: BLE001
        print(f"WARNING: could not download HS reference table ({e}).")
    return {}
 
 
def resolve_commodities(hs=HS_SELECTION, level: int = HS_LEVEL) -> dict[str, str]:
    """
    Turn a selection into {code: description} to request from the API.
      hs = 27 | "27" | [10, 27]         -> all codes of that chapter at `level` (2/4/6)
      hs = ["2709", "2710"]              -> exactly these codes
      hs = {"2709": "Crude oil", ...}    -> used as is (your own names)
    """
    if isinstance(hs, dict):
        return {_norm_code(k): v for k, v in hs.items()}
    items = [hs] if isinstance(hs, (int, str)) else list(hs)
 
    ref = None
    out: dict[str, str] = {}
    for code in map(_norm_code, items):
        if ref is None:
            ref = hs_reference()
        if len(code) != 2:                                   # explicit 4/6-digit code
            out[code] = ref.get(code) or next(
                (v.get(code, "") for v in FALLBACK_HEADINGS.values() if code in v), "")
        elif level == 2:
            out[code] = ref.get(code) or CHAPTER_LABELS.get(code, "")
        else:
            heads = {k: v for k, v in ref.items() if len(k) == level and k.startswith(code)}
            if not heads and level == 4 and code in FALLBACK_HEADINGS:
                print(f"Using built-in heading list for chapter {code}.")
                heads = FALLBACK_HEADINGS[code]
            if not heads:
                raise ValueError(
                    f"No HS{level} codes found for chapter {code}. Pass explicit codes instead, "
                    f"e.g. hs=['{code}01', '{code}02'], or check the chapter number.")
            out.update(heads)
    return out
 
 
def make_group_name(hs=HS_SELECTION, level: int = HS_LEVEL) -> str:
    """Commodity label used in file names, e.g. HS10_cereals, HS27_mineral_fuels, HS10_12."""
    items = list(hs) if isinstance(hs, dict) else ([hs] if isinstance(hs, (int, str)) else list(hs))
    codes = [_norm_code(i) for i in items]
    if all(len(c) == 2 for c in codes):
        name = "HS" + "_".join(codes)
        if len(codes) == 1 and codes[0] in CHAPTER_LABELS:
            name += "_" + CHAPTER_LABELS[codes[0]]
        if level != 4:
            name += f"_L{level}"                               # keeps caches of different levels apart
        return name
    return "HS_" + "_".join(codes[:4]) + ("_etc" if len(codes) > 4 else "")
 
 
def _tag(group: str, year: int, flow: str) -> str:
    return f"{group}_{year}_{flow}"
 
 
# --------------------------------------------------------------------------- #
# API access
# --------------------------------------------------------------------------- #
def _request(url: str, params: dict, api_key: str | None, retries: int = 4) -> list[dict]:
    """GET with simple retry/back-off (Comtrade returns 429 when you go too fast)."""
    headers = {"Ocp-Apim-Subscription-Key": api_key} if api_key else {}
    for attempt in range(retries):
        r = requests.get(url, params=params, headers=headers, timeout=120)
        if r.status_code == 200:
            payload = r.json()
            if payload.get("error"):
                raise RuntimeError(f"Comtrade API error: {payload['error']}")
            return payload.get("data", [])
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt * 5)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Comtrade API: too many retries")
 
 
def fetch_raw(
    commodity_codes: Sequence[str],
    years: Iterable[int],
    flow: str = "X",
    group: str = "HS_custom",
    reporters: Sequence[int | str] | None = None,
    partners: Sequence[int | str] | None = None,
    api_key: str | None = None,
    force_refresh: bool = False,
    pause: float = 1.0,
) -> pd.DataFrame:
    """
    Raw annual HS data. One raw file per year in data/raw/; existing files are reused.
    Within a year, one API request per commodity code (stays below the record limit).
    """
    api_key = api_key or API_KEY or os.environ.get("COMTRADE_API_KEY")
    url = DATA_ENDPOINT if api_key else PREVIEW_ENDPOINT
    if not api_key:
        print("No API key found -> using public preview endpoint (max 500 rows/request).")
 
    frames = []
    for year in years:
        raw_file = os.path.join(raw_path, f"comtrade_raw_{_tag(group, year, flow)}.csv")
        if os.path.exists(raw_file) and not force_refresh:
            print(f"Raw data for {year} found, skipping download: {os.path.basename(raw_file)}")
            frames.append(pd.read_csv(raw_file, dtype={"cmdCode": str}, low_memory=False))
            continue
 
        year_frames = []
        for code in commodity_codes:
            params = {
                "period": year,
                "cmdCode": code,
                "flowCode": flow,
                "partner2Code": 0,      # no third-country breakdown
                "motCode": 0,           # all modes of transport (avoids double counting)
                "customsCode": "C00",   # total customs procedures
                "includeDesc": "true",
            }
            if reporters:
                params["reporterCode"] = ",".join(map(str, reporters))
            if partners:
                params["partnerCode"] = ",".join(map(str, partners))
            print(f"Fetching {flow} {year} HS{code} ...")
            year_frames.append(pd.DataFrame(_request(url, params, api_key)))
            time.sleep(pause)
 
        year_frames = [f for f in year_frames if not f.empty]
        if not year_frames:
            print(f"WARNING: no data returned for {year}")
            continue
        df_year = pd.concat(year_frames, ignore_index=True)
        df_year.to_csv(raw_file, index=False)
        print(f"Saved raw data: {raw_file}")
        frames.append(df_year)
 
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
 
 
# --------------------------------------------------------------------------- #
# Cleaning -> tidy dataframe
# --------------------------------------------------------------------------- #
def tidy(raw: pd.DataFrame, flow: str = "X", label: str = "iso") -> pd.DataFrame:
    """
    Raw Comtrade records -> origin, destination, commodity, year, trade_volume (t), trade_value_usd.
    flow="X": reporter = exporter -> origin = reporter, destination = partner
    flow="M": reporter = importer -> origin = partner,  destination = reporter
    label:    "iso" (ISO3 codes) or "desc" (country names)
    """
    cols = ["origin", "destination", "commodity", "year", "trade_volume", "trade_value_usd"]
    if raw.empty:
        return pd.DataFrame(columns=cols)
    col = {"iso": ("reporterISO", "partnerISO"), "desc": ("reporterDesc", "partnerDesc")}[label]
    df = raw.copy()
    df = df[df["partnerCode"] != 0]                       # drop "World" aggregate
    df["cmdCode"] = df["cmdCode"].astype(str)
 
    rep, par = df[col[0]], df[col[1]]
    df["origin"], df["destination"] = (rep, par) if flow == "X" else (par, rep)
    df["commodity"] = df["cmdCode"]
    df["year"] = df["refYear"].astype(int)
    df["trade_volume"] = pd.to_numeric(df["netWgt"], errors="coerce") / 1000.0   # kg -> t
    df["trade_value_usd"] = pd.to_numeric(df["primaryValue"], errors="coerce")
 
    return (df.groupby(["origin", "destination", "commodity", "year"], as_index=False)
              [["trade_volume", "trade_value_usd"]].sum(min_count=1))
 
 
def get_trade_data(
    hs=HS_SELECTION,
    years: Iterable[int] = YEARS_TEST,
    flow: str = "X",
    label: str = "iso",
    level: int = HS_LEVEL,
    group: str | None = None,
    save: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch (or load cached raw) + tidy + save. flow = "X", "M" or "both".
    hs: chapter(s) / code(s) / {code: name} - see resolve_commodities().
    Saves one processed file per year (and flow) to output/trade_analysis/results/.
    Extra kwargs go to fetch_raw (api_key, force_refresh, reporters, partners, pause).
    """
    codes = list(resolve_commodities(hs, level))
    group = group or make_group_name(hs, level)
    years = list(years)
    flows = ["X", "M"] if flow == "both" else [flow]
    out = []
    for f in flows:
        raw = fetch_raw(codes, years, flow=f, group=group, **kwargs)
        t = tidy(raw, flow=f, label=label)
        if save:
            for y, part in t.groupby("year"):
                fp = os.path.join(results_path, f"trade_{_tag(group, y, f)}.csv")
                part.to_csv(fp, index=False)
                print(f"Saved results: {fp}")
        if flow == "both":
            t.insert(0, "flow", f)
        out.append(t)
    return pd.concat(out, ignore_index=True)
 
 
def load_from_file(path: str, flow: str = "X", label: str = "iso") -> pd.DataFrame:
    """Alternative to the API: load a CSV downloaded manually from the Comtrade+ web UI."""
    raw = pd.read_csv(path, dtype={"cmdCode": str}, low_memory=False)
    return tidy(raw, flow=flow, label=label)
 
 
# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _save(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
 
 
def plot_top_exporters(df: pd.DataFrame, year: int | None = None, top_n: int = 20,
                       commodity_names: dict | None = None, save_path: str | None = None):
    """Horizontal stacked bars: volume of the top-N origins, split by commodity."""
    import matplotlib.pyplot as plt
 
    names = commodity_names or {}
    d = df if year is None else df[df["year"] == year]
    top = d.groupby("origin")["trade_volume"].sum().nlargest(top_n).index
    pivot = (d[d["origin"].isin(top)].groupby(["origin", "commodity"])["trade_volume"].sum()
               .unstack(fill_value=0).loc[top[::-1]] / 1e6)
    pivot.columns = [f"{c} {names.get(c, '')}".strip() for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Volume (million tonnes)")
    ax.set_ylabel("Origin")
    ax.set_title(f"Top {top_n} origins by commodity ({year or 'all years'})")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
 
 
def plot_time_series(df: pd.DataFrame, top_n: int = 20, save_path: str | None = None):
    """Yearly volume of the top-N origins."""
    import matplotlib.pyplot as plt
 
    top = df.groupby("origin")["trade_volume"].sum().nlargest(top_n).index
    ts = (df[df["origin"].isin(top)].groupby(["year", "origin"])["trade_volume"].sum()
            .unstack() / 1e6)
    fig, ax = plt.subplots(figsize=(20, 10))
    ts.plot(ax=ax, marker="o")
    ax.set_xticks(ts.index)
    ax.set_ylabel("Million tonnes")
    ax.set_title(f"Trade volume over time - top {top_n} origins")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
 
 
# def plot_flow_heatmap(df: pd.DataFrame, top_n: int = 15, year: int | None = None,
#                       save_path: str | None = None):
#     """Origin x destination heatmap (million tonnes) for the top-N on each side."""
#     import matplotlib.pyplot as plt
 
#     d = df if year is None else df[df["year"] == year]
#     top_o = d.groupby("origin")["trade_volume"].sum().nlargest(top_n).index
#     top_d = d.groupby("destination")["trade_volume"].sum().nlargest(top_n).index
#     m = (d[d["origin"].isin(top_o) & d["destination"].isin(top_d)]
#            .pivot_table(index="origin", columns="destination", values="trade_volume",
#                         aggfunc="sum", fill_value=0) / 1e6)
#     m = m.loc[top_o.intersection(m.index)]
#     fig, ax = plt.subplots(figsize=(11, 7))
#     im = ax.imshow(m.values, cmap="viridis", aspect="auto")
#     ax.set_xticks(range(len(m.columns)), m.columns, rotation=90)
#     ax.set_yticks(range(len(m.index)), m.index)
#     ax.set_xlabel("Destination")
#     ax.set_ylabel("Origin")
#     ax.set_title(f"Flows, million tonnes ({year or 'all years'})")
#     fig.colorbar(im, label="Million tonnes")
#     fig.tight_layout()
#     _save(fig, save_path)
#     return fig
 
 
# --------------------------------------------------------------------------- #
# World map
# --------------------------------------------------------------------------- #
NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
          "geojson/ne_50m_admin_0_countries.geojson")
 
 
def _ring_centroid(ring):
    """Area-weighted centroid of a polygon ring (lon/lat degrees)."""
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        cr = x0 * y1 - x1 * y0
        a += cr
        cx += (x0 + x1) * cr
        cy += (y0 + y1) * cr
    a *= 0.5
    if a == 0:
        return None
    return cx / (6 * a), cy / (6 * a), abs(a)
 
 
def country_centroids() -> dict[str, tuple[float, float]]:
    """ISO3 -> (lon, lat) from Natural Earth polygons. Downloaded once, cached in data/."""
    cache = os.path.join(data_path, "country_centroids.csv")
    if os.path.exists(cache):
        c = pd.read_csv(cache)
        return {r.iso3: (r.lon, r.lat) for r in c.itertuples()}
 
    print("Downloading Natural Earth country shapes (one-off) ...")
    gj = requests.get(NE_URL, timeout=120).json()
    rows = []
    for ft in gj["features"]:
        pr = ft["properties"]
        iso = pr.get("ISO_A3_EH")
        if not iso or iso == "-99":
            iso = pr.get("ADM0_A3")
        geom = ft["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        best = max((r for r in (_ring_centroid(p[0]) for p in polys) if r), key=lambda r: r[2])
        rows.append({"iso3": iso, "lon": best[0], "lat": best[1]})
    c = pd.DataFrame(rows).drop_duplicates("iso3")
    c.to_csv(cache, index=False)
    return {r.iso3: (r.lon, r.lat) for r in c.itertuples()}
 
 
# def plot_flow_map(
#     df: pd.DataFrame,
#     year: int | None = None,
#     top_n: int = 100,
#     commodity: str | None = None,
#     html_path: str | None = None,
#     title_prefix: str = "Trade flows",
# ):
#     """
#     Interactive world map (plotly, `pip install plotly`): countries shaded by total
#     origin volume, lines for the top-N bilateral flows with width ~ volume.
#     """
#     import plotly.graph_objects as go
 
#     d = df
#     if "flow" in d.columns and d["flow"].nunique() > 1:   # flow="both" -> use exporter-reported
#         d = d[d["flow"] == "X"]
#     if year is not None:
#         d = d[d["year"] == year]
#     if commodity is not None:
#         d = d[d["commodity"] == commodity]
 
#     totals = d.groupby("origin")["trade_volume"].sum() / 1e6
#     flows = (d.groupby(["origin", "destination"], as_index=False)["trade_volume"].sum()
#                .nlargest(top_n, "trade_volume"))
 
#     cent = country_centroids()
#     missing = set(flows["origin"]).union(flows["destination"]) - set(cent)
#     if missing:
#         print(f"No coordinates for {sorted(missing)} - those flows are skipped on the map.")
 
#     fig = go.Figure(go.Choropleth(
#         locations=totals.index, z=totals.values, colorscale="Blues",
#         marker_line_width=0.3, colorbar=dict(title="Origin volume<br>(Mt)"),
#         hovertemplate="%{location}: %{z:.2f} Mt<extra></extra>"))
 
#     vmax = flows["trade_volume"].max()
#     for r in flows.itertuples():
#         if r.origin not in cent or r.destination not in cent:
#             continue
#         (lo0, la0), (lo1, la1) = cent[r.origin], cent[r.destination]
#         fig.add_trace(go.Scattergeo(
#             lon=[lo0, lo1], lat=[la0, la1], mode="lines+markers",
#             line=dict(width=0.8 + 6 * r.trade_volume / vmax, color="rgba(214,39,40,0.55)"),
#             marker=dict(size=[0, 5], color="rgb(214,39,40)"),
#             hovertemplate=f"{r.origin} -> {r.destination}: {r.trade_volume / 1e6:.2f} Mt<extra></extra>",
#             showlegend=False))
 
#     fig.update_layout(title=f"{title_prefix} ({year or 'all years'}"
#                             f"{', HS' + commodity if commodity else ''})",
#                       height=650, margin=dict(l=0, r=0, t=50, b=0),
#                       geo=dict(projection_type="natural earth", showframe=False,
#                                showcountries=True, countrycolor="lightgrey"))
#     if html_path:
#         fig.write_html(html_path)
#         print(f"Saved map: {html_path}")
#     return fig

def _iso(region) -> str:
    """'AF-AGO' -> 'AGO'"""
    return str(region).split("-")[-1]


def load_paths(paths_file: str = PATHS_FILE) -> pd.DataFrame:
    """paths.csv -> one row per directed pair: origin, destination (ISO3), geometry (shapely), path_cost, path_efficiency."""
    from shapely import wkt

    p = pd.read_csv(paths_file, usecols=["source", "sink", "path_cost", "path_geometry", "path_efficiency"])
    p["origin"], p["destination"] = p["source"].map(_iso), p["sink"].map(_iso)
    p["geometry"] = p["path_geometry"].map(wkt.loads)
    p = p.drop(columns="path_geometry")
    dup = p.duplicated(["origin", "destination"], keep=False)
    if dup.any():
        print(f"NOTE: {int(dup.sum())} rows in paths.csv share an ISO3 pair - keeping the cheapest path.")
        p = p.sort_values("path_cost").drop_duplicates(["origin", "destination"])
    return p


def _reverse_geom(g):
    try:
        return g.reverse()                                   # shapely >= 2.0
    except AttributeError:
        from shapely.geometry import LineString
        return LineString(list(g.coords)[::-1])


def _filter_trade(df: pd.DataFrame, year: int | None, commodity: str | None) -> pd.DataFrame:
    d = df
    if "flow" in d.columns and d["flow"].nunique() > 1:      # flow="both" -> use exporter-reported
        d = d[d["flow"] == "X"]
    if year is not None:
        d = d[d["year"] == year]
    if commodity is not None:
        d = d[d["commodity"] == commodity]
    return d


def merge_paths_onto_trade(df: pd.DataFrame, year: int | None = None, commodity=None,
                           paths: pd.DataFrame | None = None, paths_file: str = PATHS_FILE,
                           verbose: bool = True) -> pd.DataFrame:
    """
    One row per directed trade pair (all selected commodities summed) with the transport route attached:
    origin, destination, trade_volume, [trade_value_usd], source, sink, path_cost, path_efficiency,
    geometry, has_path, reversed.
    Matching: origin->destination path first; if only the opposite direction exists in paths.csv,
    that path is used in reverse (reversed=True). Pairs without any path get has_path=False.
    """
    d = _filter_trade(df, year, commodity)
    vals = [c for c in ("trade_volume", "trade_value_usd") if c in d.columns]
    flows = d.groupby(["origin", "destination"], as_index=False)[vals].sum()
    flows = flows[flows["trade_volume"] > 0]
    paths = load_paths(paths_file) if paths is None else paths

    direct = flows.merge(paths, on=["origin", "destination"], how="inner")
    direct["reversed"] = False

    rev_paths = paths.rename(columns={"origin": "destination", "destination": "origin",
                                      "source": "sink", "sink": "source"})
    done = set(zip(direct["origin"], direct["destination"]))
    rest = flows[[k not in done for k in zip(flows["origin"], flows["destination"])]]
    rev = rest.merge(rev_paths, on=["origin", "destination"], how="inner")
    rev["geometry"] = rev["geometry"].map(_reverse_geom)
    rev["reversed"] = True

    matched = pd.concat([direct, rev], ignore_index=True)
    done = set(zip(matched["origin"], matched["destination"]))
    none = flows[[k not in done for k in zip(flows["origin"], flows["destination"])]].copy()
    none["reversed"] = False

    out = pd.concat([matched.assign(has_path=True), none.assign(has_path=False)], ignore_index=True)
    out = out.sort_values("trade_volume", ascending=False).reset_index(drop=True)

    if verbose and len(out):
        share = out.loc[out["has_path"], "trade_volume"].sum() / out["trade_volume"].sum()
        print(f"Paths merged: {int(out['has_path'].sum()):,} of {len(out):,} trade pairs have a route "
              f"({int(out['reversed'].sum()):,} via reversed path), covering {share:.1%} of trade volume.")
        if len(none):
            top = none.nlargest(8, "trade_volume")
            print("  Largest pairs without route:",
                  ", ".join(f"{o}->{t}" for o, t in zip(top["origin"], top["destination"])))
    return out

def plot_flow_map(
    df: pd.DataFrame,
    year: int | None = None,
    top_n: int | None = None,
    commodity=None,
    commodity_names: dict | None = None,
    html_path: str | None = None,
    png_path: str | None = None,
    title_prefix: str = "Trade flows",
    projection_type: str = "natural earth",
    shared_scale: bool = False,
    global_volume_min: float | None = None,
    global_volume_max: float | None = None,
    paths_file: str = PATHS_FILE,
    unmatched: str = "skip",
    return_merged: bool = False,
):
    """
    Interactive world map (plotly). The dropdown switches between "All commodities" and each single
    commodity code (e.g. 2709, 2710, 2711): countries shaded by origin volume (YlGn), the top-N
    bilateral flows of the selection drawn along the routes from paths.csv, coloured (Rainbow, log
    scale) and sized by volume.

    commodity:        optional pre-filter (code or list, prefix match); the dropdown then only offers these
    commodity_names:  {code: description} for the dropdown labels
    shared_scale:     False = every selection has its own colour/width scale (best readability);
                      True  = one common scale across all selections (comparable between commodities)
    global_volume_min/max (tonnes): fixed bounds that override the scale (e.g. to compare several maps)
    unmatched:        pairs without a route: "skip" (default) or "straight" (dashed centroid line)
    return_merged:    also return the merged trade+route dataframe (column `selection`) -> (fig, merged)
    """
    import numpy as np
    import plotly.graph_objects as go
    import plotly.colors as pc

    FONT_FAMILY, FONT_COLOR, LABEL_SIZE, TICK_SIZE = "Times New Roman", "black", 14, 12

    d = _filter_trade(df, year, commodity)
    d = d.assign(commodity=d["commodity"].astype(str))
    names = commodity_names or {}
    codes = sorted(d["commodity"].unique())
    paths = load_paths(paths_file)
    cent = country_centroids() if unmatched == "straight" else {}

    selections = [(f"{c} {names.get(c, '')}".strip(), c) for c in codes]
    if len(codes) > 1:
        selections.insert(0, ("All commodities", None))

    # ---- prepare data for every dropdown selection ----
    prepared, skipped = [], 0
    for label, code in selections:
        dd = d if code is None else d[d["commodity"] == code]
        merged = merge_paths_onto_trade(dd, paths=paths, verbose=(code is None))
        totals = dd.groupby("origin")["trade_volume"].sum() / 1e6          # Mt
        if merged.empty or totals.sum() <= 0:
            print(f"Skipping '{label}': no trade volume.")
            continue
        n_no_route = int((~merged["has_path"]).sum())
        fl = merged[merged["has_path"]] if unmatched == "skip" else merged   # drop route-less pairs BEFORE top_n
        fl = (fl.head(top_n) if top_n else fl).sort_values("trade_volume").reset_index(drop=True)
        skipped += n_no_route        # biggest lines drawn last
        if unmatched == "skip":
            skipped += int((~fl["has_path"]).sum())
            fl = fl[fl["has_path"]].reset_index(drop=True)
        prepared.append(dict(label=label, code=code, totals=totals, flows=fl, merged=merged))
    if prepared:
        p0 = prepared[0]
        print(f"Map '{p0['label']}': {len(p0['merged']):,} trade pairs, "
              f"{int(p0['merged']['has_path'].sum()):,} with a route in paths.csv, "
              f"{len(p0['flows']):,} drawn (top_n={top_n}, unmatched='{unmatched}').")
    if not prepared:
        raise ValueError("Nothing to plot - no commodity has trade volume for this selection.")

    # ---- colour/width scale ----
    def log_bounds(frames):
        v = np.log1p(np.concatenate([f["trade_volume"].values.astype(float) for f in frames]))
        return (np.nanmin(v), np.nanmax(v)) if len(v) else (0.0, 1.0)

    if shared_scale:
        shared = log_bounds([p["flows"] for p in prepared if len(p["flows"])])

    fig = go.Figure()
    group_idx, route_count = {}, 0
    for sel_i, p in enumerate(prepared):
        vis, idxs = (sel_i == 0), []
        totals, fl = p["totals"], p["flows"]

        # country shading: origin volume of this selection
        vmin, vmax = totals.min(), totals.max()
        fig.add_trace(go.Choropleth(
            locations=totals.index, z=totals.values, locationmode="ISO-3", colorscale="YlGn",
            marker_line_color="white", marker_line_width=0.5, zmin=vmin, zmax=vmax,
            colorbar=dict(
                title=dict(text="", font=dict(family=FONT_FAMILY, size=LABEL_SIZE, color=FONT_COLOR)),
                len=0.4, thickness=10, x=0.9, y=0.5, xanchor="left",
                tickvals=[vmin, vmax], ticktext=[f"{vmin:.1f}", f"{vmax:.1f}"],
                tickfont=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR)),
            hovertemplate="%{location}: %{z:.2f} Mt<extra></extra>",
            name="Origin volume", showlegend=False, visible=vis))
        idxs.append(len(fig.data) - 1)

        node_coords = {}
        if len(fl):
            amounts = fl["trade_volume"].values.astype(float)
            log_amounts = np.log1p(amounts)
            amin, amax = shared if shared_scale else (np.nanmin(log_amounts), np.nanmax(log_amounts))
            if global_volume_min is not None:
                amin = np.log1p(global_volume_min)
            if global_volume_max is not None:
                amax = np.log1p(global_volume_max)
            norm = np.clip((log_amounts - amin) / (amax - amin + 1e-12), 0, 1)
            widths = 1 + 5 * norm
            colors = pc.sample_colorscale("Rainbow", norm.tolist())

            for r, w, color in zip(fl.itertuples(), widths, colors):
                if r.has_path:
                    lines = list(r.geometry.geoms) if r.geometry.geom_type == "MultiLineString" else [r.geometry]
                    lon, lat = [], []
                    for k, ln in enumerate(lines):
                        xs, ys = ln.xy
                        if k:
                            lon.append(None), lat.append(None)
                        lon += list(xs)
                        lat += list(ys)
                    dash = "solid"
                    extra = f"<br>path cost: {r.path_cost:.3g}<br>efficiency: {r.path_efficiency:.3f}"
                else:                                                       # unmatched == "straight"
                    if r.origin not in cent or r.destination not in cent:
                        continue
                    lon = [cent[r.origin][0], cent[r.destination][0]]
                    lat = [cent[r.origin][1], cent[r.destination][1]]
                    dash, extra = "dash", "<br>(no route in paths.csv)"
                node_coords.setdefault(r.origin, (lon[0], lat[0]))
                node_coords.setdefault(r.destination, (lon[-1], lat[-1]))
                fig.add_trace(go.Scattergeo(
                    lon=lon, lat=lat, mode="lines+markers",
                    line=dict(width=w, color=color, dash=dash),
                    marker=dict(size=[0] * (len(lon) - 1) + [5], color=color),   # dot at destination = direction
                    showlegend=False, hoverinfo="text", opacity=0.8, visible=vis,
                    text=f"{r.origin} → {r.destination}<br>{p['label']}"
                         f"<br>volume: {r.trade_volume / 1e6:.3g} Mt{extra}"))
                idxs.append(len(fig.data) - 1)
                route_count += 1

            tick_vals = np.linspace(amin, amax, 5)
            tick_text = [f"{v / 1e6:.2g}" for v in np.expm1(tick_vals)]
            fig.add_trace(go.Scattergeo(
                lon=[None], lat=[None], mode="markers",
                marker=dict(size=0.1, color=[amin, amax], colorscale="Rainbow", cmin=amin, cmax=amax,
                            colorbar=dict(
                                title=dict(text="", font=dict(family=FONT_FAMILY, size=LABEL_SIZE, color=FONT_COLOR)),
                                len=0.4, thickness=10, x=0.98, y=0.5, xanchor="left",
                                tickvals=tick_vals, ticktext=tick_text,
                                tickfont=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR)),
                            showscale=True),
                showlegend=False, hoverinfo="skip", visible=vis))
            idxs.append(len(fig.data) - 1)

        names_n = sorted(node_coords)
        fig.add_trace(go.Scattergeo(
            lon=[node_coords[n][0] for n in names_n], lat=[node_coords[n][1] for n in names_n],
            mode="markers", marker=dict(size=6, color="darkblue"), name="Nodes",
            text=names_n, hoverinfo="text", showlegend=False, visible=vis))
        idxs.append(len(fig.data) - 1)
        group_idx[p["label"]] = idxs

    annotations = [
        dict(text=label, xref="paper", yref="paper", x=x_pos - 0.03, y=0.5, xanchor="center",
             yanchor="middle", textangle=-90, showarrow=False,
             font=dict(family=FONT_FAMILY, size=LABEL_SIZE - 1, color=FONT_COLOR))
        for x_pos, label in zip([0.9, 0.98], ["Origin volume (Mt)", "Trade volume (Mt)"])]

    # ---- dropdown: switch between commodities ----
    def title_for(label):
        return f"{title_prefix} ({year or 'all years'}) – {label}"

    n_traces = len(fig.data)
    buttons = []
    for p in prepared:
        vis = [False] * n_traces
        for i in group_idx[p["label"]]:
            vis[i] = True
        buttons.append(dict(label=p["label"], method="update",
                            args=[{"visible": vis}, {"title.text": title_for(p["label"])}]))

    fig.update_layout(
        updatemenus=[dict(buttons=buttons, direction="down", x=0.01, y=0.98, xanchor="left", yanchor="top",
                          showactive=True, pad=dict(t=0, r=0),
                          font=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR))],
        annotations=annotations)
    fig.update_geos(
        projection_type=projection_type, showland=True, landcolor="lightgrey",
        showocean=True, oceancolor="lightblue", showcountries=True, countrycolor="white",
        showcoastlines=True, coastlinecolor="white", domain=dict(x=[0, 0.85], y=[0, 1]))
    fig.update_layout(
        title=dict(text=title_for(prepared[0]["label"]),
                   font=dict(family=FONT_FAMILY, size=LABEL_SIZE + 2, color=FONT_COLOR),
                   y=0.98, yanchor="top", automargin=True),
        height=700, width=1300, font=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR),
        showlegend=False, margin=dict(t=40, b=0, l=20, r=90))

    if html_path:
        fig.write_html(html_path, include_plotlyjs="cdn")
        print(f"Saved map: {html_path}  ({len(prepared)} selections, {route_count:,} route traces)")
        if png_path is None:
            png_path = os.path.splitext(html_path)[0] + ".png"
    if png_path:
        try:
            fig.write_image(png_path, scale=2)      # static export = first selection ("All commodities")
            print(f"Saved static map: {png_path}")
        except Exception as e:                                       # noqa: BLE001
            print(f"PNG export failed ({e}). Install kaleido: pip install --upgrade \"kaleido>=1\"")
    if return_merged:
        return fig, pd.concat([p["merged"].assign(selection=p["label"]) for p in prepared], ignore_index=True)
    return fig
 
# --------------------------------------------------------------------------- #
# Batch figure export
# --------------------------------------------------------------------------- #
def make_all_figures(df: pd.DataFrame, group: str, commodity_names: dict | None = None,
                     with_map: bool = True, show: bool = False, map_top_n: int | None = None):
    """Per year and flow: top exporters, heatmap, map. Plus one time series per flow."""
    import matplotlib.pyplot as plt

    flows = sorted(df["flow"].unique()) if "flow" in df.columns else ["X"]
    for f in flows:
        d = df[df["flow"] == f] if "flow" in df.columns else df
        years = sorted(d["year"].unique())
        for y in years:
            tag = _tag(group, y, f)
            fp = lambda suffix: os.path.join(figures_path, f"{tag}_{suffix}")
            dy = d[d["year"] == y]
            plot_top_exporters(dy, year=y, commodity_names=commodity_names,
                               save_path=fp("top_exporters.png"))
            # plot_flow_heatmap(dy, year=y, save_path=fp("flow_heatmap.png"))
            if with_map:
                plot_flow_map(dy, year=y, top_n=map_top_n, commodity_names=commodity_names,
                              html_path=fp("flow_map.html"), title_prefix=f"{group} flows")
            if not show:
                plt.close("all")
        if len(years) > 1:
            plot_time_series(d, save_path=os.path.join(
                figures_path, f"{group}_{f}_{years[0]}-{years[-1]}_timeseries.png"))
            if not show:
                plt.close("all")
    if show:
        plt.show()
 
 
# --------------------------------------------------------------------------- #
# One-call workflow
# --------------------------------------------------------------------------- #
def run(hs=HS_SELECTION, years: Iterable[int] = YEARS_TEST, flow: str = "X", level: int = HS_LEVEL,
        group: str | None = None, figures: bool = True, with_map: bool = True, show: bool = False,
        **kwargs) -> pd.DataFrame:
    """Fetch/load -> save raw + results -> figures, for any HS selection."""
    names = resolve_commodities(hs, level)
    group = group or make_group_name(hs, level)
    print(f"Group: {group} | {len(names)} commodity codes: {', '.join(names)}")
    df = get_trade_data(hs, years, flow=flow, level=level, group=group, **kwargs)
    print(df.head(), f"\n{len(df):,} rows")
    if figures and not df.empty:
        make_all_figures(df, group, commodity_names=names, with_map=with_map, show=show)
    return df
 
 
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
 
    ap = argparse.ArgumentParser(description="Comtrade+ trade analysis")
    ap.add_argument("--hs", nargs="+", default=[str(HS_SELECTION)],
                    help="HS chapter(s) or codes, e.g. --hs 27   --hs 10 12   --hs 2709 2710")
    ap.add_argument("--level", type=int, default=HS_LEVEL, choices=[2, 4, 6],
                    help="detail level for chapters (2 = chapter total, 4 = headings, 6 = subheadings)")
    ap.add_argument("--test", action="store_true", help="single-year test run (2023)")
    ap.add_argument("--flow", default="X", choices=["X", "M", "both"])
    ap.add_argument("--refresh", action="store_true", help="re-download even if raw files exist")
    ap.add_argument("--no-map", action="store_true", help="skip the world maps")
    ap.add_argument("--show", action="store_true", help="show matplotlib figures")
    args = ap.parse_args()
 
    run(hs=args.hs, years=YEARS_TEST if args.test else YEARS_LIST, flow=args.flow,
        level=args.level, force_refresh=args.refresh, with_map=not args.no_map, show=args.show)

## AI statement: During the preparation of this work the author(s) used Claude in order to develop software tools used in this publication. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.
