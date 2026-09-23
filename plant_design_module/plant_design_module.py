### This is the plant_design_module that is supposed to be used to corretly preprocess technology data and unify different technologies and datasets into one homogenous datasaet for energy systems modelling
#%%

import pandas as pd
import time
import os
import sys
import re
from dataclasses import dataclass, field
from typing import Optional

START = time.perf_counter()
print("Execute in Directory:")
print(os.getcwd() + "\n")

try:
    module_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    module_dir = os.getcwd()

module_dir = os.getcwd()
input_path = os.path.join(module_dir, "input_data")
output_path = os.path.join(module_dir, "output")

# --- global parameters (step 8 discussion: to be moved to a config/other module later) ---
TARGET_UNIT = "MW"
TARGET_CAPACITY_MW = 1.0
TARGET_CURRENCY = "EUR"  # alias: €
TARGET_YEAR = 2026

# derived target compound units used throughout the pipeline
TARGET_ENERGY_UNIT = f"{TARGET_UNIT}h"                    # e.g. "MWh"
TARGET_CAPEX_UNIT = f"{TARGET_CURRENCY}/{TARGET_UNIT}"     # e.g. "EUR/MW"
TARGET_VOM_UNIT = f"{TARGET_CURRENCY}/{TARGET_ENERGY_UNIT}"  # e.g. "EUR/MWh"

# step 3: fractional deviation allowed between a reported amount_per_unit_output
# and the theoretically calculated stoichiometric value before it gets flagged
STOICHIOMETRY_TOLERANCE = 0.01  # 1%


# ---------------------------------------------------------------------------
# Unit handling (step 2): currency + physical unit unification
# ---------------------------------------------------------------------------
currency_units = [
    "EUR", "USD",
]

_CURRENCY_ALIASES = {"€": "EUR", "$": "USD"}

energy_units = [
    "kWh", "MWh", "GWh", "TWh",
    "MJ", "GJ",
]

capacity_units = [
    "kW", "MW", "GW", "TW",
]

mass_units = [
    "kg", "t",
]

time_units = ["h", "yr"]

UNIT_CATEGORIES = {
    "currency": currency_units,
    "energy": energy_units,
    "capacity": capacity_units,
    "mass": mass_units,
    "time": time_units,
}

METRIC_PREFIXES = {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}

BASE_SYMBOL_TO_CANONICAL = {
    "currency": {"EUR": 1.0, "USD": None},          # canonical base: EUR
    "energy": {"Wh": 1.0, "J": 1.0 / 3600.0},        # canonical base: Wh (1 kWh = 3.6 MJ)
    "capacity": {"W": 1.0},                          # canonical base: W
    "mass": {"g": 1.0, "t": 1.0e6},                  # canonical base: g (1 t = 1e6 g)
    "time": {"h": 1.0, "yr": 8760.0},                # canonical base: h (non-leap year)
}


def decompose_token(token: str) -> tuple[str, str, float]:
    """One unit token (e.g. 'MWh') -> (category, base_symbol, metric_prefix_factor).
    Exposed at module level (not just inside unit_unifying) since it's also
    needed standalone, e.g. to work out what unit capex_scale_base is in."""
    token = _CURRENCY_ALIASES.get(token.strip(), token.strip())
    # try longest base symbols first, so e.g. 'MWh' matches base 'Wh'
    # (energy) before the unrelated, shorter 'h' (time) would match.
    all_base_symbols = sorted(
        ((category, base_symbol)
         for category, bases in BASE_SYMBOL_TO_CANONICAL.items()
         for base_symbol in bases),
        key=lambda cb: len(cb[1]), reverse=True,
    )
    for category, base_symbol in all_base_symbols:
        if token.endswith(base_symbol):
            prefix = token[: len(token) - len(base_symbol)]
            if prefix in METRIC_PREFIXES:
                return category, base_symbol, METRIC_PREFIXES[prefix]
    raise ValueError(
        f"Unrecognized unit token: '{token}'. Known base symbols: "
        f"{sorted(set(bs for bases in BASE_SYMBOL_TO_CANONICAL.values() for bs in bases))} "
        f"(each optionally preceded by one of {sorted(METRIC_PREFIXES)})."
    )


def split_compound_unit(unit_str: str) -> tuple[list[str], list[str]]:
    """'EUR/kW/yr' -> (['EUR'], ['kW', 'yr']); 'kg*m/s' -> (['kg','m'], ['s']).
    Exposed at module level so callers (e.g. Technology.unify_units, and
    later the capex input/output rebasing in step 4) can inspect a compound
    unit's numerator/denominator without going through unit_unifying."""
    unit_str = unit_str.strip().replace(" ", "")
    parts = unit_str.split("/")
    numerator = parts[0].split("*") if parts[0] else []
    denominator = []
    for p in parts[1:]:
        denominator.extend(p.split("*"))
    return numerator, denominator


def unit_unifying(target_unit: str = "EUR/MWh", input_unit: Optional[str] = None,
                   input_value: Optional[float] = None,
                   fx_rate_usd_per_eur: Optional[float] = None) -> tuple[float, str]:
    """
    Convert input_value (given in input_unit) into target_unit.

    Handles simple compound units built from the categories above (currency,
    energy, capacity/power, mass, time), with '/' for division and '*' for
    multiplication within one side (e.g. 'EUR/kW/yr'). Metric prefixes
    (k, M, G, T) are resolved generically against the "even steps" pattern,
    so any prefixed combination of a known base symbol works even if it
    isn't explicitly listed in currency_units/energy_units/etc. above.

    Raises ValueError if input_unit and target_unit don't share the same
    physical dimension, or if a EUR<->USD conversion is needed but
    fx_rate_usd_per_eur wasn't supplied.
    """
    if input_unit is None or input_value is None:
        raise ValueError("input_unit and input_value are required.")

    def token_to_canonical(category: str, base_symbol: str, prefix_factor: float) -> float:
        """Value, in the category's canonical base unit, of 1 [prefix+base_symbol]."""
        base_to_canonical = BASE_SYMBOL_TO_CANONICAL[category][base_symbol]
        if base_to_canonical is None:  # currently only USD: not a fixed ratio to EUR
            if fx_rate_usd_per_eur is None:
                raise ValueError(
                    f"Converting '{base_symbol}' requires fx_rate_usd_per_eur "
                    f"(USD per 1 EUR) to be supplied."
                )
            base_to_canonical = 1.0 / fx_rate_usd_per_eur  # 1 USD -> EUR
        return prefix_factor * base_to_canonical

    def scale_to_canonical(numerator: list[str], denominator: list[str]) -> float:
        """Value, in canonical units, of 1 [numerator / denominator]."""
        scale = 1.0
        for tok in numerator:
            category, base_symbol, prefix_factor = decompose_token(tok)
            scale *= token_to_canonical(category, base_symbol, prefix_factor)
        for tok in denominator:
            category, base_symbol, prefix_factor = decompose_token(tok)
            scale /= token_to_canonical(category, base_symbol, prefix_factor)
        return scale

    # --- split target unit into its components ---
    target_num, target_den = split_compound_unit(target_unit)          # above / below the division line
    target_num_categories = sorted(decompose_token(t)[0] for t in target_num)   # compare to unit lists
    target_den_categories = sorted(decompose_token(t)[0] for t in target_den)   # compare to unit lists

    # --- split input unit into its components ---
    input_num, input_den = split_compound_unit(input_unit)             # above / below the division line
    input_num_categories = sorted(decompose_token(t)[0] for t in input_num)     # compare to unit lists
    input_den_categories = sorted(decompose_token(t)[0] for t in input_den)     # compare to unit lists

    # --- test that target and input units even have the same dimension ---
    if (target_num_categories, target_den_categories) != (input_num_categories, input_den_categories):
        raise ValueError(
            f"Dimension mismatch: input unit '{input_unit}' is "
            f"{input_num_categories}/{input_den_categories}, target unit '{target_unit}' is "
            f"{target_num_categories}/{target_den_categories}."
        )

    # --- define the way to bring input to target unit ---
    scale_input = scale_to_canonical(input_num, input_den)
    scale_target = scale_to_canonical(target_num, target_den)
    conversion_factor = scale_input / scale_target

    # --- apply this way to the input value ---
    output_value = input_value * conversion_factor
    output_unit = target_unit
    return output_value, output_unit


def get_fx_rate(fx_rates_df: pd.DataFrame, year, currency_pair: str = "USD_per_EUR") -> Optional[float]:
    """
    Look up the exchange rate for a given year and currency pair from the
    fx_rates sheet. Returns None (and prints a note) if no matching row exists,
    e.g. because the sheet doesn't cover that year yet.
    """
    match = fx_rates_df.loc[
        (fx_rates_df["year"] == year) & (fx_rates_df["currency_pair"] == currency_pair)
    ]
    if match.empty:
        print(f"[get_fx_rate] No '{currency_pair}' rate found for year {year}.")
        return None
    return match["rate"].iloc[0]


# ---------------------------------------------------------------------------
# Stoichiometry handling (step 3)
# ---------------------------------------------------------------------------
atomic_weights = {
    "H": 1, "He": 4, "C": 12, "N": 14, "O": 16,
    "Li": 7, "Na": 23, "K": 39, "Mg": 24, "Al": 27, "Si": 28,
    "P": 31, "S": 32, "Cl": 35, "Ca": 40, "Fe": 56,
    }

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_chemical_formula(formula: str) -> dict[str, int]:
    """'H2O' -> {'H': 2, 'O': 1}; 'CH3OH' -> {'C': 1, 'H': 4, 'O': 1}.
    Only handles simple (non-nested, no parentheses) formulas - sufficient
    for the reaction rows used here."""
    matches = _FORMULA_TOKEN_RE.findall(formula)
    if not matches or "".join(el + count for el, count in matches) != formula.strip():
        raise ValueError(f"Could not parse chemical formula: '{formula}'.")
    counts: dict[str, int] = {}
    for element, count_str in matches:
        counts[element] = counts.get(element, 0) + (int(count_str) if count_str else 1)
    return counts


def molar_mass(formula: str) -> float:
    """Molar mass of a chemical formula, in g/mol, using the atomic_weights table above."""
    counts = parse_chemical_formula(formula)
    try:
        return sum(atomic_weights[element] * n for element, n in counts.items())
    except KeyError as e:
        raise ValueError(f"Unknown element {e} in formula '{formula}'. Add it to atomic_weights.") from e


# ---------------------------------------------------------------------------
# Technology object (steps 2-7 will be implemented as methods on this class)
# ---------------------------------------------------------------------------
@dataclass
class Technology:
    tech_id: str  # required - rows without this are skipped in build_technologies()
    tech_name: Optional[str] = None
    category: Optional[str] = None
    capex: Optional[float] = None
    capex_unit: Optional[str] = None
    capex_ref_year: Optional[int] = None
    capex_scale_base: Optional[float] = None
    degression_exponent: Optional[float] = None
    lifetime_years: Optional[float] = None
    wacc: Optional[float] = None
    fom_factor: Optional[float] = None
    vom: Optional[float] = None
    vom_unit: Optional[str] = None
    full_load_hours: Optional[float] = None

    # populated by step 3 (efficiency / stoichiometry)
    efficiency_energetic: Optional[float] = None
    energy_losses: Optional[float] = None          # MWh per unit output
    efficiency_material: Optional[float] = None

    # each Technology carries its own slice of commodity_efficiencies
    efficiencies: pd.DataFrame = field(default_factory=pd.DataFrame)

    # populated later by validation steps (e.g. stoichiometry check, step 3 discussion)
    warnings: list = field(default_factory=list)

    # --- steps 2-7 will live here as methods, e.g.: ---
    # def rebase_capex(self): ...              # step 4
    # def escalate_capex(self, cepci_df): ...   # step 5
    # def scale_capex(self): ...                # step 6
    # def compute_fom(self): ...                # step 7
    # def efficiency_energetic(self): ...
    # def efficiency_material(self): ...
    # def check_stoichiometry(self): ...

    def unify_units(self, fx_rates_df: pd.DataFrame) -> None:
        """
        Step 2: convert capex, capex_scale_base and vom to this module's
        target units (TARGET_CAPEX_UNIT, TARGET_UNIT, TARGET_VOM_UNIT).

        The fx rate used for any EUR<->USD conversion is looked up for this
        technology's capex_ref_year (not TARGET_YEAR), so old cost figures
        are retraced at the exchange rate from when they were published.
        vom is assumed to share the same reference year as capex, since the
        input schema doesn't carry a separate vom_ref_year.

        Updates capex/capex_unit/capex_scale_base/vom/vom_unit in place.
        Anything that can't be converted (missing value, missing fx rate,
        unrecognized unit, mismatched dimension) is left untouched and
        recorded in self.warnings instead of raising.
        """
        # capture the *original* capex_unit before it gets overwritten below -
        # capex_scale_base's implicit unit is this unit's denominator
        original_capex_unit = self.capex_unit

        fx_rate = None
        if self.capex_ref_year is not None:
            fx_rate = get_fx_rate(fx_rates_df, self.capex_ref_year)

        # --- CAPEX ---
        if self.capex is not None and self.capex_unit is not None:
            try:
                self.capex, self.capex_unit = unit_unifying(
                    target_unit=TARGET_CAPEX_UNIT,
                    input_unit=self.capex_unit,
                    input_value=self.capex,
                    fx_rate_usd_per_eur=fx_rate,
                )
            except ValueError as e:
                self.warnings.append(
                    f"unify_units: could not convert capex ({original_capex_unit} -> {TARGET_CAPEX_UNIT}): {e}"
                )
        else:
            self.warnings.append("unify_units: capex or capex_unit missing, capex not converted.")

        # --- capex_scale_base: same capacity unit as capex_unit's original denominator ---
        if self.capex_scale_base is not None and original_capex_unit is not None:
            _, capacity_tokens = split_compound_unit(original_capex_unit)
            if len(capacity_tokens) == 1:
                try:
                    self.capex_scale_base, _ = unit_unifying(
                        target_unit=TARGET_UNIT,
                        input_unit=capacity_tokens[0],
                        input_value=self.capex_scale_base,
                    )
                except ValueError as e:
                    self.warnings.append(
                        f"unify_units: could not convert capex_scale_base ({capacity_tokens[0]} -> {TARGET_UNIT}): {e}"
                    )
            else:
                self.warnings.append(
                    f"unify_units: capex_unit denominator '{original_capex_unit}' has "
                    f"{len(capacity_tokens)} tokens, expected exactly 1 - capex_scale_base not converted."
                )
        elif self.capex_scale_base is not None:
            self.warnings.append("unify_units: capex_unit missing, capex_scale_base not converted.")

        # --- VOM ---
        if self.vom is not None and self.vom_unit is not None:
            try:
                self.vom, self.vom_unit = unit_unifying(
                    target_unit=TARGET_VOM_UNIT,
                    input_unit=self.vom_unit,
                    input_value=self.vom,
                    fx_rate_usd_per_eur=fx_rate,
                )
            except ValueError as e:
                self.warnings.append(f"unify_units: could not convert vom ({self.vom_unit} -> {TARGET_VOM_UNIT}): {e}")
        else:
            self.warnings.append("unify_units: vom or vom_unit missing, vom not converted.")

    def check_stoichiometry(self, tolerance: float = STOICHIOMETRY_TOLERANCE) -> None:
        """
        Step 3 (data-quality check, not a conversion): for every reaction row
        with chemical_formula + stoichiometric_coefficient set, compute the
        theoretical amount_per_unit_output from the is_stoichiometric_basis
        row via molar mass ratios, and compare it to the *reported* value.
        Flags (appends to self.warnings) any participant whose reported value
        deviates from the theoretical value by more than `tolerance`
        (fractional, default STOICHIOMETRY_TOLERANCE). Does not modify any
        values - real-world yield loss is handled separately via
        compute_efficiency_material()'s material_eff row, not here.
        The material_eff marker row is excluded (it has no chemical_formula).
        """
        rows = self.efficiencies.loc[self.efficiencies["commodity"] != "material_eff"]
        rows = rows.loc[rows["chemical_formula"].notna() & rows["reaction_id"].notna()]

        for reaction_id, group in rows.groupby("reaction_id"):
            basis_rows = group.loc[group["is_stoichiometric_basis"] == 1]
            if basis_rows.empty:
                self.warnings.append(
                    f"check_stoichiometry: reaction '{reaction_id}' has no is_stoichiometric_basis row, skipped."
                )
                continue
            if len(basis_rows) > 1:
                self.warnings.append(
                    f"check_stoichiometry: reaction '{reaction_id}' has multiple is_stoichiometric_basis rows, using the first."
                )
            basis = basis_rows.iloc[0]

            try:
                basis_molar_mass = molar_mass(basis["chemical_formula"])
            except ValueError as e:
                self.warnings.append(
                    f"check_stoichiometry: reaction '{reaction_id}': {e}"
                )
                continue

            for _, participant in group.iterrows():
                if participant["is_stoichiometric_basis"] == 1:
                    continue  # basis row compared against itself is trivial

                try:
                    participant_molar_mass = molar_mass(participant["chemical_formula"])
                except ValueError as e:
                    self.warnings.append(f"check_stoichiometry: reaction '{reaction_id}': {e}")
                    continue

                ratio = (participant["stoichiometric_coefficient"] * participant_molar_mass) / \
                        (basis["stoichiometric_coefficient"] * basis_molar_mass)
                theoretical_value = ratio * basis["amount_per_unit_output"]
                actual_value = participant["amount_per_unit_output"]

                if theoretical_value == 0:
                    continue
                deviation = abs(actual_value - theoretical_value) / abs(theoretical_value)
                if deviation > tolerance:
                    self.warnings.append(
                        f"check_stoichiometry: '{participant['commodity']}' in reaction '{reaction_id}' "
                        f"deviates {deviation:.1%} from the theoretical stoichiometric value "
                        f"(reported={actual_value}, theoretical={theoretical_value:.4g})."
                    )

    def compute_efficiency_energetic(self) -> None:
        """
        Step 3: efficiency_energetic = output_energy / sum(input_energy_equivalents)
        energy_losses = sum(input_energy_equivalents) - output_energy  (MWh per unit output)

        Input energy equivalents include:
          - every input row with flow_type == "energy" (already in MWh terms)
          - every input row with flow_type == "material" AND a non-null
            lhv_mwh_per_t (a material that carries usable energy, e.g. a fuel
            logged by mass), converted via amount_per_unit_output * lhv_mwh_per_t
        Material inputs without an lhv (e.g. water) are excluded - they don't
        carry usable energy and can't be expressed in energy terms.
        The material_eff marker row is excluded.
        """
        rows = self.efficiencies.loc[self.efficiencies["commodity"] != "material_eff"]

        output_rows = rows.loc[(rows["is_reference_output"] == 1) & (rows["flow_type"] == "energy")]
        if output_rows.empty:
            self.warnings.append("compute_efficiency_energetic: no reference-output energy row found, skipped.")
            return
        if len(output_rows) > 1:
            self.warnings.append(
                "compute_efficiency_energetic: multiple reference-output energy rows found, using the first."
            )
        output_energy = output_rows.iloc[0]["amount_per_unit_output"]

        input_rows = rows.loc[rows["direction"] == "input"]
        input_energy = 0.0
        for _, row in input_rows.iterrows():
            if row["flow_type"] == "energy":
                input_energy += row["amount_per_unit_output"]
            elif row["flow_type"] == "material" and pd.notna(row["lhv_mwh_per_t"]):
                input_energy += row["amount_per_unit_output"] * row["lhv_mwh_per_t"]
            # else: material input without an lhv (e.g. water) - excluded

        if input_energy == 0:
            self.warnings.append("compute_efficiency_energetic: total input energy is zero, cannot compute efficiency.")
            return

        self.efficiency_energetic = output_energy / input_energy
        self.energy_losses = input_energy - output_energy

    def compute_efficiency_material(self) -> None:
        """
        Step 3: efficiency_material comes from a dedicated 'material_eff'
        marker row (commodity == 'material_eff') if present for this
        tech_id - a literature-sourced yield factor covering real-world
        losses that stoichiometry alone can't capture (side reactions,
        purge losses, incomplete conversion). If no such row exists,
        efficiency_material defaults to 1.0 (the reported flows are assumed
        fully converted, i.e. only the stoichiometric balance applies).
        """
        material_eff_rows = self.efficiencies.loc[self.efficiencies["commodity"] == "material_eff"]
        if material_eff_rows.empty:
            self.efficiency_material = 1.0
            return
        if len(material_eff_rows) > 1:
            self.warnings.append("compute_efficiency_material: multiple material_eff rows found, using the first.")
        self.efficiency_material = material_eff_rows.iloc[0]["amount_per_unit_output"]


def build_technologies(technologies_df: pd.DataFrame,
                        commodity_efficiencies_df: pd.DataFrame) -> list[Technology]:
    """
    Build one Technology object per row of the technologies sheet, attaching
    each technology's own slice of commodity_efficiencies.

    Rows without a tech_id are skipped and flagged (printed), since tech_id
    is the join key used throughout the pipeline and later for aggregation.
    All other fields are optional to allow partial records (e.g. a row that
    only supplies capex, or only full_load_hours) to still be built.
    """
    def clean(value):
        # turn pandas/NaN missing values into plain None
        return None if pd.isna(value) else value

    technologies = []
    skipped_row_count = 0

    for idx, row in technologies_df.iterrows():
        tech_id = clean(row.get("tech_id"))
        if tech_id is None:
            skipped_row_count += 1
            print(f"[build_technologies] Skipping row {idx}: missing tech_id.")
            continue

        eff_slice = commodity_efficiencies_df.loc[
            commodity_efficiencies_df["tech_id"] == tech_id
        ].copy()

        tech = Technology(
            tech_id=tech_id,
            tech_name=clean(row.get("tech_name")),
            category=clean(row.get("category")),
            capex=clean(row.get("capex")),
            capex_unit=clean(row.get("capex_unit")),
            capex_ref_year=clean(row.get("capex_ref_year")),
            capex_scale_base=clean(row.get("capex_scale_base")),
            degression_exponent=clean(row.get("degression_exponent")),
            lifetime_years=clean(row.get("lifetime_years")),
            wacc=clean(row.get("WACC")),
            fom_factor=clean(row.get("fom_factor")),
            vom=clean(row.get("vom")),
            vom_unit=clean(row.get("vom_unit")),
            full_load_hours=clean(row.get("full_load_hours")),
            efficiencies=eff_slice,
        )
        technologies.append(tech)

    if skipped_row_count:
        print(f"[build_technologies] Skipped {skipped_row_count} row(s) total due to missing tech_id.")

    return technologies


def get_technology(technologies: list[Technology], tech_id: str) -> Optional[Technology]:
    """
    Look up a single Technology by tech_id. Also copies its fields to the
    clipboard (as a pandas Series) for quick inspection while developing.
    Returns the first match, or None if tech_id isn't found. Note: if
    tech_ids are ever duplicated (step 8), this only returns the first hit.
    """
    for tech in technologies:
        if tech.tech_id == tech_id:
            pd.Series(vars(tech)).to_clipboard()
            print("Copied " + str(tech_id) + " to clipboard")
            return tech
    return None


def technologies_to_df(technologies: list[Technology]) -> pd.DataFrame:
    """
    Flatten a list of Technology objects back into a DataFrame.
    Used later for step 8 (aggregation across duplicate tech_ids) and
    step 10 (final export). Not yet implemented.
    """
    # TODO: implement once steps 4-7 populate the values that need to end up
    # in the flattened row (rebased/escalated/scaled capex, fom, efficiencies, ...)
    raise NotImplementedError


def main():
    """""
    1. read all objects from excel input into objects
    2. unify the units in which the techno-economic data is given (mainly CAPEX and OPEX)
    2.1 unify currencies
    2.2 unify physical units
    3. read, check and calculate the energy and material efficiencey (stochiometrics)
    once those steps are cleanly done and the base is laid out, we can go to manipulating the data as we please
    4. correct invest cost reference depending on whether the invest cost is given as EUR/unit_in or EUR/unit_out
    5. index invest costs using cepci to target year
    6. capacity scaling to target capacity (default 1MW)
    7. calculate fom cost from fom-factor and determined invest costs (CAPEX)
    8. determine the max, min, mean and median cost values (invest, fom and vom) for technologies that appear multiple times in the dataset, so that the base dataset can be expanded as desired
    9 output the complete homogenized dataset in technology_data_output.xlsx with "techno_economic_data" and "commodity_efficiencies" as output sheets
    10. output the dataset in another excel sheet and the following format rows=technologies columns=invest_cost    full_load_hours lifetime    efficiency_substantial  efficiency_energetic    efficiency_dynamic  fom_factor  electricity_consumption fuel_consumption
                                                                                                    €/MW    h/a a   -   -   %/km    %/invest_cost   MWh_el/MWh_x    MWh_fuel/MWh_x
    Dont worry, if you are missing some of the information, we can work this out later
    """""

    # --- step 1: read all input sheets ---
    technologies_df = pd.read_excel(os.path.join(input_path, "technology_input.xlsx"), sheet_name="technologies")
    commodity_efficiencies_df = pd.read_excel(os.path.join(input_path, "technology_input.xlsx"), sheet_name="commodity_efficiencies")
    fx_rates_df = pd.read_excel(os.path.join(input_path, "technology_input.xlsx"), sheet_name="fx_rates")
    cepci_df = pd.read_excel(os.path.join(input_path, "technology_input.xlsx"), sheet_name="cepci_index")

    sheets = {
        "technologies": technologies_df,
        "commodity_efficiencies": commodity_efficiencies_df,
        "fx_rates": fx_rates_df,
        "cepci_index": cepci_df,
    }

    technologies = build_technologies(technologies_df, commodity_efficiencies_df)

    # --- step 2: unify units ---
    for tech in technologies:
        tech.unify_units(fx_rates_df)
    # --- step 3: efficiency / stoichiometry check ---
    for tech in technologies:
        tech.check_stoichiometry()
        tech.compute_efficiency_energetic()
        tech.compute_efficiency_material()
    # --- step 4: correct capex reference basis (TODO) ---
    # --- step 5: CEPCI escalation to TARGET_YEAR (TODO) ---
    # --- step 6: capacity scaling to TARGET_CAPACITY_MW (TODO) ---
    # --- step 7: FOM calculation (TODO) ---
    # --- step 8: aggregation across duplicate tech_ids (TODO) ---
    # --- step 9/10: export (TODO) ---

    STOP = time.perf_counter()
    print('Total execution time of script', round((STOP - START), 1), 's')

    return sheets, technologies


if __name__ == "__main__":
    sheets, technologies = main()

## AI statement: During the preparation of this work the author(s) used Claude in order to develop software tools used in this pubication. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.
#%%