"""
fbic_visualization.py
---------------------
Plain visualisation of the bilateral fbic table (countrya, countryb, year, fbic):

  1. world map with DIRECT connection lines between country pairs (great-circle arcs),
     colour and line width coded by fbic (log scale), year slider
  2. histogram of the fbic values (log10 axis), year slider

Both are written as interactive HTML (all years, slider starts at the latest year) and as
static PNG (latest year only).

Input    relationship_module/data/fbic/fbic_tab.csv
Output   relationship_module/output/fbic_visualization/figures/
             fbic_map.html            fbic_map_<latest year>.png
             fbic_histogram.html      fbic_histogram_<latest year>.png

Country names are matched to ISO3 + map coordinates via Natural Earth (downloaded once, cached
in relationship_module/data/country_lookup.csv). Names that cannot be matched are listed in the
console - add them to NAME_ALIASES below.

Usage    python fbic_visualization.py
         python fbic_visualization.py --top-n 1000 --no-png
Needs    pip install pandas numpy requests plotly kaleido   (kaleido>=1 only for the PNG export)
"""
from __future__ import annotations

import argparse
import os
import re
import unicodedata

import numpy as np
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
fbic_file = os.path.join(data_path, "fbic", "fbic_tab.csv")
lookup_file = os.path.join(data_path, "country_lookup.csv")
figures_path = os.path.join(module_dir, "output", "fbic_visualization", "figures")
os.makedirs(figures_path, exist_ok=True)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MAP_TOP_N = 500          # lines per year: the N pairs with the highest fbic; None = all pairs (huge HTML!)
MIN_FBIC = None          # optionally ignore pairs below this fbic
UNDIRECTED = True        # A-B and B-A are merged into one pair (mean of both values)
N_CLASSES = 10           # line colour/width classes (lines are batched per class -> fast, small HTML)
LINE_COLORSCALE = "Rainbow"
N_BINS = 40              # histogram bins (on log10(fbic), identical for all years)
FONT_FAMILY, FONT_COLOR, LABEL_SIZE, TICK_SIZE = "Times New Roman", "black", 14, 12

# table name -> name as used by Natural Earth (extend when the console lists unmatched names)
NAME_ALIASES = {
    "Cote d'Ivoire": "Ivory Coast", "Czech Republic": "Czechia", "Swaziland": "eSwatini",
    "Burma": "Myanmar", "East Timor": "Timor-Leste", "Macedonia": "North Macedonia",
    "Cape Verde": "Cabo Verde", "Congo, Democratic Republic of the": "Democratic Republic of the Congo",
    "Congo, Republic of the": "Republic of the Congo", "Congo (Kinshasa)": "Democratic Republic of the Congo",
    "Congo (Brazzaville)": "Republic of the Congo", "Congo": "Republic of the Congo",
    "DR Congo": "Democratic Republic of the Congo", "Gambia, The": "The Gambia", "Gambia": "The Gambia",
    "Bahamas, The": "The Bahamas", "Bahamas": "The Bahamas", "Korea, South": "South Korea",
    "Korea, Republic of": "South Korea", "Republic of Korea": "South Korea", "Korea": "South Korea",
    "Korea, North": "North Korea", "Korea, Democratic People's Republic of": "North Korea",
    "Russian Federation": "Russia", "Syrian Arab Republic": "Syria", "Iran, Islamic Republic of": "Iran",
    "Viet Nam": "Vietnam", "Lao PDR": "Laos", "Lao People's Democratic Republic": "Laos",
    "Brunei Darussalam": "Brunei", "Micronesia, Federated States of": "Federated States of Micronesia",
    "Micronesia": "Federated States of Micronesia", "Slovak Republic": "Slovakia",
    "Kyrgyz Republic": "Kyrgyzstan", "Turkiye": "Turkey", "Turkey (Turkiye)": "Turkey",
    "Tanzania": "United Republic of Tanzania", "United States": "United States of America",
    "USA": "United States of America", "UK": "United Kingdom", "Moldova, Republic of": "Moldova",
    "Bolivia, Plurinational State of": "Bolivia", "Venezuela, Bolivarian Republic of": "Venezuela",
    "Serbia and Montenegro": "Serbia", "Yemen, Rep.": "Yemen", "Egypt, Arab Rep.": "Egypt",
    "Hong Kong SAR": "Hong Kong", "Macao": "Macau", "Palestine, State of": "Palestine",
    "West Bank and Gaza": "Palestine", "Sao Tome and Principe": "Sao Tome and Principe",
    "Saint Vincent and Grenadines": "Saint Vincent and the Grenadines",
}


# --------------------------------------------------------------------------- #
# Country lookup: name -> ISO3 + coordinates (Natural Earth)
# --------------------------------------------------------------------------- #
NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
          "geojson/ne_50m_admin_0_countries.geojson")
# names tried per country in this priority order (SOVEREIGNT deliberately left out: it maps territories to their state)
NE_NAME_FIELDS = ["ADMIN", "NAME", "NAME_LONG", "NAME_EN", "BRK_NAME", "FORMAL_EN", "NAME_CIAWF", "NAME_SORT", "ABBREV"]


def _norm(name) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\bst\.?(?=\s)", "saint", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\bthe\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _ring_centroid(ring):
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        cr = x0 * y1 - x1 * y0
        a += cr
        cx += (x0 + x1) * cr
        cy += (y0 + y1) * cr
    a *= 0.5
    return None if a == 0 else (cx / (6 * a), cy / (6 * a), abs(a))


def build_country_lookup() -> pd.DataFrame:
    """One row per (normalised) country name: key, iso3, lon, lat. Cached in data/country_lookup.csv."""
    if os.path.exists(lookup_file):
        return pd.read_csv(lookup_file, keep_default_na=False)
    print("Downloading Natural Earth country shapes (one-off) ...")
    feats = requests.get(NE_URL, timeout=120).json()["features"]
    info = []
    for ft in feats:
        pr, geom = ft["properties"], ft["geometry"]
        iso = pr.get("ISO_A3_EH")
        if not iso or iso == "-99":
            iso = pr.get("ADM0_A3")
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        best = max((r for r in (_ring_centroid(p[0]) for p in polys) if r), key=lambda r: r[2])
        lon, lat = best[0], best[1]
        if pr.get("LABEL_X") is not None and pr.get("LABEL_Y") is not None:   # NE label point: nicer than centroid
            lon, lat = float(pr["LABEL_X"]), float(pr["LABEL_Y"])
        info.append((pr, iso, lon, lat))
    rows, seen = [], set()
    for field in NE_NAME_FIELDS:                       # higher-priority fields win across all countries
        for pr, iso, lon, lat in info:
            if pr.get(field):
                key = _norm(pr[field])
                if key and key not in seen:
                    seen.add(key)
                    rows.append({"key": key, "iso3": iso, "lon": lon, "lat": lat})
    out = pd.DataFrame(rows)
    out.to_csv(lookup_file, index=False)
    return out


def resolve_countries(names) -> tuple[dict, list]:
    """names -> ({name: (iso3, lon, lat)}, [unmatched names])"""
    lk = build_country_lookup().set_index("key")
    aliases = {_norm(k): _norm(v) for k, v in NAME_ALIASES.items()}
    found, missing = {}, []
    for n in names:
        key = _norm(n)
        key = key if key in lk.index else aliases.get(key, key)
        if key in lk.index:
            r = lk.loc[key]
            found[n] = (r["iso3"], float(r["lon"]), float(r["lat"]))
        else:
            missing.append(n)
    return found, missing


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_fbic(path: str = fbic_file, years=None, min_fbic=MIN_FBIC, undirected: bool = UNDIRECTED):
    """
    Returns (pairs, coords): pairs = iso1, iso2, name1, name2, year, fbic (one row per pair and year);
    coords = {iso3: (lon, lat)}.
    """
    df = pd.read_csv(path)                                    # "NA" is read as missing
    df["fbic"] = pd.to_numeric(df["fbic"], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["fbic"])
    df = df[df["fbic"] > 0]                                   # log scale needs positive values
    print(f"fbic table: {n0:,} rows, {len(df):,} with a positive value "
          f"({n0 - len(df):,} missing/zero dropped), years {df['year'].min()}-{df['year'].max()}.")
    if min_fbic:
        df = df[df["fbic"] >= min_fbic]
    if years:
        df = df[df["year"].isin(years)]

    geo, missing = resolve_countries(sorted(set(df["countrya"]) | set(df["countryb"])))
    if missing:
        print(f"WARNING: {len(missing)} country names could not be matched and are skipped "
              f"(add them to NAME_ALIASES): {missing}")
    df = df[df["countrya"].isin(geo) & df["countryb"].isin(geo)].copy()
    df["iso1"] = df["countrya"].map(lambda n: geo[n][0])
    df["iso2"] = df["countryb"].map(lambda n: geo[n][0])
    df = df[df["iso1"] != df["iso2"]]
    coords = {iso: (lon, lat) for iso, lon, lat in geo.values()}

    if undirected:
        a, b = df["iso1"].to_numpy(), df["iso2"].to_numpy()
        df["iso1"], df["iso2"] = np.where(a < b, a, b), np.where(a < b, b, a)
        n_dir = len(df)
        df = df.groupby(["iso1", "iso2", "year"], as_index=False).agg(fbic=("fbic", "mean"))
        if len(df) < n_dir:
            print(f"Merged {n_dir - len(df):,} A-B / B-A duplicates into one pair (mean fbic).")
    return df, coords


def _midpoint(lon1, lat1, lon2, lat2):
    """Great-circle midpoint (so hover points sit on the drawn arcs)."""
    p1, p2 = np.radians([lon1, lat1]), np.radians([lon2, lat2])
    def vec(p):
        return np.array([np.cos(p[1]) * np.cos(p[0]), np.cos(p[1]) * np.sin(p[0]), np.sin(p[1])])
    m = vec(p1) + vec(p2)
    n = np.linalg.norm(m)
    if n < 1e-9:
        return (lon1 + lon2) / 2, (lat1 + lat2) / 2
    m /= n
    return float(np.degrees(np.arctan2(m[1], m[0]))), float(np.degrees(np.arcsin(m[2])))


def _log_ticks(lo: float, hi: float):
    ticks = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1))
    if len(ticks) < 2:
        ticks = list(np.linspace(lo, hi, 4))
        return ticks, [f"{10 ** t:.1e}" for t in ticks]
    return ticks, [f"1e{t}" for t in ticks]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def build_map_figure(pairs: pd.DataFrame, coords: dict, years, top_n=MAP_TOP_N, slider: bool = True,
                     bounds: tuple[float, float] | None = None):
    """Direct connection lines, colour + width = log10(fbic); one batch of traces per year and class."""
    import plotly.colors as pc
    import plotly.graph_objects as go

    years = sorted(years)
    shown = {}
    for y in years:
        d = pairs[pairs["year"] == y]
        shown[y] = (d.nlargest(top_n, "fbic") if top_n else d).assign(
            lf=lambda x: np.log10(x["fbic"]))
    lmin, lmax = bounds or (min(s["lf"].min() for s in shown.values() if len(s)),
                            max(s["lf"].max() for s in shown.values() if len(s)))
    edges = np.linspace(lmin, lmax, N_CLASSES + 1)
    colors = pc.sample_colorscale(LINE_COLORSCALE, [(k + 0.5) / N_CLASSES for k in range(N_CLASSES)])

    fig = go.Figure()
    year_traces = {}
    for y in years:
        s = shown[y]
        idxs = []
        if len(s):
            cls = np.clip(np.digitize(s["lf"].to_numpy(), edges[1:-1]), 0, N_CLASSES - 1)
            vis = (y == years[-1])
            for k in range(N_CLASSES):                         # weakest class first -> strongest drawn on top
                sub = s[cls == k]
                if sub.empty:
                    continue
                lon, lat = [], []
                for r in sub.itertuples():
                    (lo1, la1), (lo2, la2) = coords[r.iso1], coords[r.iso2]
                    lon += [lo1, lo2, None]
                    lat += [la1, la2, None]
                frac = k / max(N_CLASSES - 1, 1)
                fig.add_trace(go.Scattergeo(
                    lon=lon, lat=lat, mode="lines", hoverinfo="skip", showlegend=False, visible=vis,
                    line=dict(width=0.6 + 4.4 * frac, color=colors[k]), opacity=0.35 + 0.5 * frac))
                idxs.append(len(fig.data) - 1)
            mids = [_midpoint(*coords[r.iso1], *coords[r.iso2]) for r in s.itertuples()]
            fig.add_trace(go.Scattergeo(                       # invisible hover points on the arcs
                lon=[m[0] for m in mids], lat=[m[1] for m in mids], mode="markers", showlegend=False,
                marker=dict(size=8, color="rgba(0,0,0,0)"), visible=vis,
                text=[f"{r.iso1} – {r.iso2} ({y})<br>fbic: {r.fbic:.3e}" for r in s.itertuples()],
                hovertemplate="%{text}<extra></extra>"))
            idxs.append(len(fig.data) - 1)
        year_traces[y] = idxs

    tick_vals, tick_text = _log_ticks(lmin, lmax)
    fig.add_trace(go.Scattergeo(                               # colorbar carrier (always visible)
        lon=[None], lat=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(size=0.1, color=[lmin, lmax], colorscale=LINE_COLORSCALE, cmin=lmin, cmax=lmax,
                    showscale=True,
                    colorbar=dict(title=dict(text=""), len=0.5, thickness=10, x=0.93, y=0.5, xanchor="left",
                                  tickvals=tick_vals, ticktext=tick_text,
                                  tickfont=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR)))))
    always = len(fig.data) - 1

    def title_for(y):
        n_all = int((pairs["year"] == y).sum())
        return (f"fbic – direct bilateral connections, {y} "
                f"(top {len(shown[y]):,} of {n_all:,} pairs)" if top_n else
                f"fbic – direct bilateral connections, {y} ({n_all:,} pairs)")

    n_traces = len(fig.data)
    steps = []
    for y in years:
        vis = [False] * n_traces
        vis[always] = True
        for i in year_traces[y]:
            vis[i] = True
        steps.append(dict(method="update", label=str(y),
                          args=[{"visible": vis}, {"title.text": title_for(y)}]))

    fig.update_geos(projection_type="natural earth", showland=True, landcolor="lightgrey",
                    showocean=True, oceancolor="lightblue", showcountries=True, countrycolor="white",
                    showcoastlines=True, coastlinecolor="white",
                    domain=dict(x=[0, 0.9], y=[0.1 if slider else 0, 1]))
    fig.update_layout(
        title=dict(text=title_for(years[-1]), y=0.98, yanchor="top", automargin=True,
                   font=dict(family=FONT_FAMILY, size=LABEL_SIZE + 2, color=FONT_COLOR)),
        annotations=[dict(text="fbic (log scale)", xref="paper", yref="paper", x=0.90, y=0.5,
                          xanchor="center", yanchor="middle", textangle=-90, showarrow=False,
                          font=dict(family=FONT_FAMILY, size=LABEL_SIZE - 1, color=FONT_COLOR))],
        height=700, width=1300, showlegend=False,
        font=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR),
        margin=dict(t=40, b=60 if slider else 0, l=20, r=90))
    if slider and len(years) > 1:
        fig.update_layout(sliders=[dict(active=len(years) - 1, steps=steps, x=0.05, len=0.85, y=0.02,
                                        pad=dict(t=10, b=0), currentvalue=dict(prefix="Year: "))])
    return fig


def build_histogram_figure(pairs: pd.DataFrame, years, slider: bool = True):
    """Histogram of log10(fbic), identical bins and axis ranges for all years."""
    import plotly.graph_objects as go

    years = sorted(years)
    lf = np.log10(pairs["fbic"].to_numpy())
    edges = np.linspace(np.floor(lf.min()), np.ceil(lf.max()), N_BINS + 1)
    centers, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    counts = {y: np.histogram(np.log10(pairs.loc[pairs["year"] == y, "fbic"]), bins=edges)[0] for y in years}
    ymax = max(c.max() for c in counts.values()) * 1.08

    fig = go.Figure()
    for y in years:
        fig.add_trace(go.Bar(
            x=centers, y=counts[y], width=width * 0.95, marker=dict(color="steelblue", line=dict(width=0)),
            visible=(y == years[-1]), showlegend=False,
            hovertemplate=f"{y}<br>fbic ≈ 10^%{{x:.2f}}<br>pairs: %{{y:,}}<extra></extra>"))

    def title_for(y):
        v = pairs.loc[pairs["year"] == y, "fbic"]
        return f"Distribution of fbic, {y} (n = {len(v):,}, median = {v.median():.2e})"

    tick_vals, tick_text = _log_ticks(edges[0], edges[-1])
    steps = [dict(method="update", label=str(y),
                  args=[{"visible": [i == k for i in range(len(years))]}, {"title.text": title_for(y)}])
             for k, y in enumerate(years)]
    fig.update_layout(
        title=dict(text=title_for(years[-1]), font=dict(family=FONT_FAMILY, size=LABEL_SIZE + 2, color=FONT_COLOR)),
        xaxis=dict(title="fbic (log scale)", tickvals=tick_vals, ticktext=tick_text, range=[edges[0], edges[-1]]),
        yaxis=dict(title="Number of country pairs", range=[0, ymax]),
        bargap=0, height=600, width=1000, plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, size=TICK_SIZE, color=FONT_COLOR),
        margin=dict(t=60, b=90 if slider else 50, l=70, r=30))
    fig.update_xaxes(showline=True, linecolor="black", gridcolor="lightgrey")
    fig.update_yaxes(showline=True, linecolor="black", gridcolor="lightgrey")
    if slider and len(years) > 1:
        fig.update_layout(sliders=[dict(active=len(years) - 1, steps=steps, x=0.05, len=0.9, y=-0.12,
                                        pad=dict(t=10, b=0), currentvalue=dict(prefix="Year: "))])
    return fig


def _write(fig, name: str, latest: int, png_fig=None, png: bool = True, scale: int = 2):
    html = os.path.join(figures_path, f"{name}.html")
    fig.write_html(html, include_plotlyjs="cdn")
    print(f"Saved: {html}")
    if png:
        path = os.path.join(figures_path, f"{name}_{latest}.png")
        try:
            (png_fig or fig).write_image(path, scale=scale)    # needs: pip install --upgrade "kaleido>=1"
            print(f"Saved: {path}")
        except Exception as e:                                 # noqa: BLE001
            print(f"PNG export failed ({e}). Install kaleido: pip install --upgrade \"kaleido>=1\"")


def run(top_n=MAP_TOP_N, years=None, png: bool = True, path: str = fbic_file):
    pairs, coords = load_fbic(path, years=years)
    all_years = sorted(pairs["year"].unique())
    latest = all_years[-1]
    print(f"{len(pairs):,} country-pair observations, years {all_years[0]}-{latest}, latest = {latest}.")

    map_fig = build_map_figure(pairs, coords, all_years, top_n=top_n)
    # static PNG: latest year only, no slider, same colour scale as the interactive map
    lo, hi = None, None
    d = pairs[pairs["year"].isin(all_years)]
    sel = [(y_df.nlargest(top_n, "fbic") if top_n else y_df) for _, y_df in d.groupby("year")]
    lf = np.log10(pd.concat(sel)["fbic"])
    lo, hi = lf.min(), lf.max()
    map_png = build_map_figure(pairs, coords, [latest], top_n=top_n, slider=False, bounds=(lo, hi))
    _write(map_fig, "fbic_map", latest, png_fig=map_png, png=png)

    hist_fig = build_histogram_figure(pairs, all_years)
    hist_png = build_histogram_figure(pairs, [latest], slider=False)
    _write(hist_fig, "fbic_histogram", latest, png_fig=hist_png, png=png, scale=2)
    return pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Visualise the fbic table")
    ap.add_argument("--top-n", type=int, default=MAP_TOP_N, help="lines per year (0 = all pairs)")
    ap.add_argument("--years", nargs="+", type=int, default=None, help="restrict to these years")
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()
    run(top_n=a.top_n or None, years=a.years, png=not a.no_png)