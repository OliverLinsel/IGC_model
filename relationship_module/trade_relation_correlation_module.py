"""
trade_alliance_correlation.py
-----------------------------
Correlates bilateral trade (output of trade_analysis_module.py) with the alliance
indices from relationship_analysis_output.xlsx (sheet "relations").

Steps
  1. read trade results (one file per HS group and year), drop self-trade
  2. aggregate to ONE ROW PER UNORDERED COUNTRY PAIR (all commodities, both directions,
     averaged over the selected years)
  3. per country: maximum pair trade of that country; every pair is divided by it
     -> each pair gets a value from each side; they are combined into one 0-1 value (PAIR_NORM)
  4. join with the alliance table (region "EU-DEU" -> ISO3 "DEU"), per scenario
  5. Pearson / Spearman correlation + scatter plots, for every HS code and alliance variant

Inputs   relationship_module/output/trade_analysis/results/trade_HS<code>_<label>_<year>_<flow>.csv
         relationship_module/output/relationship_analysis/relationship_analysis_output.xlsx
Outputs  relationship_module/output/trade_alliance_correlation/results/  (merged data, summary)
         relationship_module/output/trade_alliance_correlation/figures/  (scatter plots)

Highlighting  HIGHLIGHT_PAIRS = ["EU-DEU_NA-USA", "NA-USA_NA-CAN"]   # "<region1>_<region2>", order irrelevant
         -> shown as large coloured, labelled points in the scatter plots, marked in column
            `highlight` of the merged files, and saved to trade_alliance_highlighted_pairs_<years>.csv

Usage    python trade_alliance_correlation.py
         python trade_alliance_correlation.py --hs 10 27 --years 2023 --variants alliance_index_total_max
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

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
trade_results_path = os.path.join(module_dir, "output", "trade_analysis", "results")
relations_file = os.path.join(module_dir, "output", "relationship_analysis", "results", "relationship_analysis_output.xlsx")
output_path = os.path.join(module_dir, "output", "trade_alliance_correlation")
results_path = os.path.join(output_path, "results")
figures_path = os.path.join(output_path, "figures")
for _p in (results_path, figures_path):
    os.makedirs(_p, exist_ok=True)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# HS commodity codes to analyze (10 = cereals, 27 = mineral fuels)
HS_CODES = ["10", "27"]

# Alliance index variants to include (from the Excel file)
ALLIANCE_VARIANTS = ["alliance_index_total_max", "alliance_index_regional_max", "alliance_index_category_max"]

TRADE_YEARS = [2023]            # trade years to use; several years -> yearly average per pair
TRADE_FLOW = "X"                # flow of the trade result files (X = exports)
TRADE_METRIC = "trade_volume"   # or "trade_value_usd" (useful where weight is missing, e.g. electricity)
PAIR_NORM = "max"               # how the two country-normalised values of a pair are combined:
                                #   "max"  = trade / smaller of the two country maxima
                                #   "min"  = trade / larger of the two country maxima
                                #   "mean" = average of the two normalised values
MISSING_TRADE = "zero"          # alliance pair without any trade record: "zero" = trade 0, "drop" = remove
SCENARIOS = None                # e.g. ["Base"]; None = all scenarios in the Excel sheet
RELATIONS_SHEET = "relations"
ISO_SEPARATOR = "-"             # region "EU-DEU" -> ISO3 = text after the last separator

# Country pairs to highlight in the scatter plots: "<region1>_<region2>" (order irrelevant;
# matched on the ISO3 codes, so "EU-DEU_NA-USA" == "NA-USA_EU-DEU")
HIGHLIGHT_PAIRS = ["EU-DEU_NA-USA", "NA-USA_NA-CAN", "EU-FRA_EU-DEU", "AS-CHN_AS-IDN"]


# --------------------------------------------------------------------------- #
# Trade side
# --------------------------------------------------------------------------- #
def _find_trade_file(hs_code: str, year: int, flow: str) -> str | None:
    """trade_HS10_cereals_2023_X.csv, trade_HS27_mineral_fuels_2023_X.csv, ... (label is optional)."""
    pat = re.compile(rf"^trade_HS{hs_code}(?:_[A-Za-z_]+)?_{year}_{flow}\.csv$")
    hits = [f for f in os.listdir(trade_results_path) if pat.match(f)]
    if len(hits) > 1:
        print(f"WARNING: several files match HS{hs_code} {year}: {hits} - using {hits[0]}")
    return os.path.join(trade_results_path, hits[0]) if hits else None


def load_trade(hs_code: str, years, flow: str = TRADE_FLOW) -> pd.DataFrame:
    frames = []
    for y in years:
        fp = _find_trade_file(hs_code, y, flow)
        if fp is None:
            print(f"WARNING: no trade result file for HS{hs_code}, {y}, flow {flow} in {trade_results_path}")
            continue
        frames.append(pd.read_csv(fp, dtype={"commodity": str}))
    if not frames:
        raise FileNotFoundError(f"No trade files found for HS{hs_code} and years {list(years)}")
    return pd.concat(frames, ignore_index=True)


def aggregate_pairs(trade: pd.DataFrame, metric: str, n_years: int) -> pd.DataFrame:
    """One row per UNORDERED pair: all commodities, both directions, yearly average."""
    t = trade[trade["origin"] != trade["destination"]].copy()
    t["iso1"] = np.where(t["origin"] < t["destination"], t["origin"], t["destination"])
    t["iso2"] = np.where(t["origin"] < t["destination"], t["destination"], t["origin"])
    pairs = t.groupby(["iso1", "iso2"], as_index=False)[metric].sum()
    pairs[metric] = pairs[metric] / n_years
    return pairs.rename(columns={metric: "trade"})


def normalise_pairs(pairs: pd.DataFrame, pair_norm: str = PAIR_NORM) -> pd.DataFrame:
    """Normalise each pair by the maximum pair trade of each of its two countries (-> 0..1)."""
    long = pd.concat([pairs[["iso1", "trade"]].rename(columns={"iso1": "iso"}),
                      pairs[["iso2", "trade"]].rename(columns={"iso2": "iso"})])
    country_max = long.groupby("iso")["trade"].max().replace(0, np.nan)
    p = pairs.copy()
    p["country_max_1"] = p["iso1"].map(country_max)
    p["country_max_2"] = p["iso2"].map(country_max)
    n1, n2 = p["trade"] / p["country_max_1"], p["trade"] / p["country_max_2"]
    p["trade_norm"] = {"max": np.fmax(n1, n2), "min": np.fmin(n1, n2), "mean": (n1 + n2) / 2}[pair_norm]
    return p


# --------------------------------------------------------------------------- #
# Alliance side
# --------------------------------------------------------------------------- #
def load_relations(variants, scenarios=SCENARIOS, path: str = relations_file) -> pd.DataFrame:
    rel = pd.read_excel(path, sheet_name=RELATIONS_SHEET)
    missing = [v for v in variants if v not in rel.columns]
    if missing:
        raise KeyError(f"Alliance variants not in sheet '{RELATIONS_SHEET}': {missing}. "
                       f"Available: {list(rel.columns)}")
    if scenarios:
        rel = rel[rel["scenario"].isin(scenarios)]

    a = rel["region1"].astype(str).str.split(ISO_SEPARATOR).str[-1]
    b = rel["region2"].astype(str).str.split(ISO_SEPARATOR).str[-1]
    rel = rel.assign(iso1=np.where(a < b, a, b), iso2=np.where(a < b, b, a))   # unordered pair
    rel = rel[rel["iso1"] != rel["iso2"]]

    keep_num = [c for c in ["counter", "shared_weight", *variants] if c in rel.columns]
    agg = {"region1": "first", "region2": "first", **{c: "mean" for c in keep_num}}
    dup = rel.duplicated(["iso1", "iso2", "scenario"]).sum()
    if dup:
        print(f"NOTE: {dup} duplicate country pairs in the relations sheet - values averaged.")
    return rel.groupby(["scenario", "iso1", "iso2"], as_index=False).agg(agg)


# --------------------------------------------------------------------------- #
# Highlighting
# --------------------------------------------------------------------------- #
def parse_highlight_pairs(highlight_pairs) -> dict[str, tuple[str, str]]:
    """["EU-DEU_NA-USA", ...] -> {"EU-DEU / NA-USA": ("DEU", "USA")}  (ISO3 pair, sorted)."""
    if not highlight_pairs:
        return {}
    if isinstance(highlight_pairs, str):
        highlight_pairs = [highlight_pairs]
    items = []
    for h in highlight_pairs:                       # also accepts "A_B; C_D" in one string
        items += [x.strip() for x in re.split(r"[;,]", h) if x.strip()]
    out = {}
    for item in items:
        parts = item.split("_")
        if len(parts) != 2:
            raise ValueError(f"Highlight pair '{item}' must look like 'REGION1_REGION2' "
                             f"(e.g. 'EU-DEU_NA-USA'), exactly one '_' between the two regions.")
        isos = sorted(p.split(ISO_SEPARATOR)[-1] for p in parts)
        out[f"{parts[0]} / {parts[1]}"] = (isos[0], isos[1])
    return out


def tag_highlights(merged: pd.DataFrame, highlights: dict[str, tuple[str, str]], hs: str) -> pd.DataFrame:
    """Add column `highlight` (label or empty string)."""
    merged["highlight"] = ""
    for label, (i1, i2) in highlights.items():
        mask = (merged["iso1"] == i1) & (merged["iso2"] == i2)
        if not mask.any():
            print(f"  WARNING: highlighted pair '{label}' not found in the relations sheet "
                  f"(after scenario filter) - not shown.")
        merged.loc[mask, "highlight"] = label
    return merged


# --------------------------------------------------------------------------- #
# Merge, correlate, plot
# --------------------------------------------------------------------------- #
def merge_trade_alliance(rel: pd.DataFrame, pairs: pd.DataFrame, missing_trade: str = MISSING_TRADE):
    how = "left" if missing_trade == "zero" else "inner"
    m = rel.merge(pairs, on=["iso1", "iso2"], how=how)
    n_no_trade = int(m["trade"].isna().sum())
    if missing_trade == "zero":
        m[["trade", "trade_norm"]] = m[["trade", "trade_norm"]].fillna(0.0)
    print(f"  {len(m):,} pairs merged ({n_no_trade:,} alliance pairs without trade record, "
          f"{'set to 0' if missing_trade == 'zero' else 'dropped'}).")
    return m


def correlate(merged: pd.DataFrame, variants) -> pd.DataFrame:
    rows = []
    for (hs, sc), g in merged.groupby(["hs_code", "scenario"]):
        for v in variants:
            d = g[["trade_norm", v]].dropna()
            rows.append({"hs_code": hs, "scenario": sc, "alliance_variant": v, "n_pairs": len(d),
                         "pearson_r": d["trade_norm"].corr(d[v]),
                         "spearman_rho": d["trade_norm"].corr(d[v], method="spearman")})
    return pd.DataFrame(rows)


def plot_scatter(merged: pd.DataFrame, summary: pd.DataFrame, variants, hs_codes, tag: str,
                 show: bool = False):
    """One figure per scenario: rows = HS codes, columns = alliance variants."""
    import matplotlib.pyplot as plt

    for sc in merged["scenario"].unique():
        fig, axes = plt.subplots(len(hs_codes), len(variants), squeeze=False,
                                 figsize=(4.6 * len(variants), 4.0 * len(hs_codes)), sharey=True)
        for i, hs in enumerate(hs_codes):
            g = merged[(merged["hs_code"] == hs) & (merged["scenario"] == sc)]
            for j, v in enumerate(variants):
                ax = axes[i][j]
                d = g[[v, "trade_norm"]].dropna()
                ax.scatter(d[v], d["trade_norm"], s=10, alpha=0.4, edgecolors="none")
                if len(d) > 2 and d[v].nunique() > 1:
                    k, c0 = np.polyfit(d[v], d["trade_norm"], 1)
                    xs = np.array([d[v].min(), d[v].max()])
                    ax.plot(xs, k * xs + c0, color="crimson", lw=1.2)
                s = summary[(summary["hs_code"] == hs) & (summary["scenario"] == sc)
                            & (summary["alliance_variant"] == v)]
                if len(s):
                    ax.text(0.03, 0.97, f"r = {s['pearson_r'].iloc[0]:.2f}\n"
                                        f"rho = {s['spearman_rho'].iloc[0]:.2f}\nn = {int(s['n_pairs'].iloc[0])}",
                            transform=ax.transAxes, va="top", fontsize=9,
                            bbox=dict(boxstyle="round", fc="white", ec="lightgrey"))
                hl = g[g["highlight"] != ""][[v, "trade_norm", "highlight"]].dropna() \
                    if "highlight" in g.columns else g.iloc[0:0]
                for k, (label, hg) in enumerate(hl.groupby("highlight")):
                    ax.scatter(hg[v], hg["trade_norm"], s=90, color=f"C{k + 1}", edgecolors="black",
                               linewidths=1.0, zorder=5, label=label)
                if len(hl):
                    ax.legend(fontsize=7, loc="best", framealpha=0.85, title="highlighted", title_fontsize=7)
                ax.set_ylim(-0.02, 1.02)
                if i == len(hs_codes) - 1:
                    ax.set_xlabel(v)
                if j == 0:
                    ax.set_ylabel(f"HS{hs}: normalised trade (0-1)")
        fig.suptitle(f"Normalised bilateral trade vs. alliance index - scenario {sc}", y=1.0)
        fig.tight_layout()
        fp = os.path.join(figures_path, f"trade_alliance_scatter_{sc}_{tag}.png")
        fig.savefig(fp, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {fp}")
        if not show:
            plt.close(fig)
    if show:
        plt.show()


def run_correlation(hs_codes=HS_CODES, alliance_variants=ALLIANCE_VARIANTS, years=TRADE_YEARS,
                    flow: str = TRADE_FLOW, metric: str = TRADE_METRIC, pair_norm: str = PAIR_NORM,
                    missing_trade: str = MISSING_TRADE, scenarios=SCENARIOS,
                    highlight_pairs=HIGHLIGHT_PAIRS, show: bool = False):
    highlights = parse_highlight_pairs(highlight_pairs)
    hs_codes = [str(c).zfill(2) if len(str(c)) == 1 else str(c) for c in hs_codes]
    years = list(years)
    tag = "_".join(map(str, years)) if len(years) <= 3 else f"{min(years)}-{max(years)}"

    rel = load_relations(alliance_variants, scenarios)
    print(f"Relations: {len(rel):,} country pairs, scenarios: {sorted(rel['scenario'].unique())}")

    merged_all = []
    for hs in hs_codes:
        print(f"HS{hs}:")
        try:
            trade = load_trade(hs, years, flow)
        except FileNotFoundError as e:
            print(f"  skipped - {e}")
            continue
        pairs = normalise_pairs(aggregate_pairs(trade, metric, len(years)), pair_norm)
        m = merge_trade_alliance(rel, pairs, missing_trade)
        m.insert(0, "hs_code", hs)
        m = tag_highlights(m, highlights, hs)
        m.to_csv(os.path.join(results_path, f"trade_alliance_merged_HS{hs}_{tag}.csv"), index=False)
        merged_all.append(m)
    if not merged_all:
        raise RuntimeError("No trade data could be loaded - nothing to correlate.")

    merged = pd.concat(merged_all, ignore_index=True)
    if highlights:
        hl_rows = merged[merged["highlight"] != ""]
        hl_rows.to_csv(os.path.join(results_path, f"trade_alliance_highlighted_pairs_{tag}.csv"), index=False)
        print("\nHighlighted pairs:\n", hl_rows[["hs_code", "scenario", "highlight", "trade", "trade_norm",
                                                  *alliance_variants]].round(3).to_string(index=False))
    summary = correlate(merged, alliance_variants)
    summary.to_csv(os.path.join(results_path, f"trade_alliance_correlation_summary_{tag}.csv"), index=False)
    print("\n", summary.round(3).to_string(index=False))
    plot_scatter(merged, summary, alliance_variants, [h for h in hs_codes if h in set(merged["hs_code"])],
                 tag, show=show)
    return merged, summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Correlate trade with alliance indices")
    ap.add_argument("--hs", nargs="+", default=HS_CODES)
    ap.add_argument("--variants", nargs="+", default=ALLIANCE_VARIANTS)
    ap.add_argument("--years", nargs="+", type=int, default=TRADE_YEARS)
    ap.add_argument("--metric", default=TRADE_METRIC, choices=["trade_volume", "trade_value_usd"])
    ap.add_argument("--pair-norm", default=PAIR_NORM, choices=["max", "min", "mean"])
    ap.add_argument("--missing-trade", default=MISSING_TRADE, choices=["zero", "drop"])
    ap.add_argument("--scenario", nargs="+", default=SCENARIOS)
    ap.add_argument("--highlight", nargs="*", default=HIGHLIGHT_PAIRS,
                    help='pairs to highlight, e.g. --highlight EU-DEU_NA-USA NA-USA_NA-CAN (none: --highlight)')
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    run_correlation(a.hs, a.variants, a.years, metric=a.metric, pair_norm=a.pair_norm,
                    missing_trade=a.missing_trade, scenarios=a.scenario,
                    highlight_pairs=a.highlight, show=a.show)