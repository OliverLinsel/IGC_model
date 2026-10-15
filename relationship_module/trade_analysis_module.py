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
 
# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
API_KEY = None   # paste your key here, or set the COMTRADE_API_KEY environment variable
 
BASE_URL = "https://comtradeapi.un.org"
DATA_ENDPOINT = f"{BASE_URL}/data/v1/get/C/A/HS"            # needs subscription key
PREVIEW_ENDPOINT = f"{BASE_URL}/public/v1/preview/C/A/HS"   # no key, max 500 rows
 

HS_SELECTION = 10 #10, 27   # default HS chapter(s) / code(s); change here or pass hs=... / --hs
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
 
 
def plot_flow_map(
    df: pd.DataFrame,
    year: int | None = None,
    top_n: int = 100,
    commodity: str | None = None,
    html_path: str | None = None,
    title_prefix: str = "Trade flows",
):
    """
    Interactive world map (plotly, `pip install plotly`): countries shaded by total
    origin volume, lines for the top-N bilateral flows with width ~ volume.
    """
    import plotly.graph_objects as go
 
    d = df
    if "flow" in d.columns and d["flow"].nunique() > 1:   # flow="both" -> use exporter-reported
        d = d[d["flow"] == "X"]
    if year is not None:
        d = d[d["year"] == year]
    if commodity is not None:
        d = d[d["commodity"] == commodity]
 
    totals = d.groupby("origin")["trade_volume"].sum() / 1e6
    flows = (d.groupby(["origin", "destination"], as_index=False)["trade_volume"].sum()
               .nlargest(top_n, "trade_volume"))
 
    cent = country_centroids()
    missing = set(flows["origin"]).union(flows["destination"]) - set(cent)
    if missing:
        print(f"No coordinates for {sorted(missing)} - those flows are skipped on the map.")
 
    fig = go.Figure(go.Choropleth(
        locations=totals.index, z=totals.values, colorscale="Blues",
        marker_line_width=0.3, colorbar=dict(title="Origin volume<br>(Mt)"),
        hovertemplate="%{location}: %{z:.2f} Mt<extra></extra>"))
 
    vmax = flows["trade_volume"].max()
    for r in flows.itertuples():
        if r.origin not in cent or r.destination not in cent:
            continue
        (lo0, la0), (lo1, la1) = cent[r.origin], cent[r.destination]
        fig.add_trace(go.Scattergeo(
            lon=[lo0, lo1], lat=[la0, la1], mode="lines+markers",
            line=dict(width=0.8 + 6 * r.trade_volume / vmax, color="rgba(214,39,40,0.55)"),
            marker=dict(size=[0, 5], color="rgb(214,39,40)"),
            hovertemplate=f"{r.origin} -> {r.destination}: {r.trade_volume / 1e6:.2f} Mt<extra></extra>",
            showlegend=False))
 
    fig.update_layout(title=f"{title_prefix} ({year or 'all years'}"
                            f"{', HS' + commodity if commodity else ''})",
                      height=650, margin=dict(l=0, r=0, t=50, b=0),
                      geo=dict(projection_type="natural earth", showframe=False,
                               showcountries=True, countrycolor="lightgrey"))
    if html_path:
        fig.write_html(html_path)
        print(f"Saved map: {html_path}")
    return fig
 
 
# --------------------------------------------------------------------------- #
# Batch figure export
# --------------------------------------------------------------------------- #
def make_all_figures(df: pd.DataFrame, group: str, commodity_names: dict | None = None,
                     with_map: bool = True, show: bool = False):
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
                plot_flow_map(dy, year=y, html_path=fp("flow_map.html"),
                              title_prefix=f"{group} flows")
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
