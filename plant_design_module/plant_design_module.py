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

### This part is to guarantee execution in normal and in debug mode to cleanly call scripts from neighbouring directories
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
module_dir = os.path.join(project_root, "plant_design_module")
input_path = os.path.join(module_dir, "input")
output_path = os.path.join(module_dir, "output")

# --- global parameters (step 8 discussion: to be moved to a config/other module later) ---
TARGET_UNIT = "MW"
TARGET_CAPACITY_MW = 1.0
TARGET_CURRENCY = "EUR"  # alias: €
TARGET_YEAR = 2040
DEFAULT_DEGRESSION_EXPONENT = 0.7 # used only when a record gives no degression_exponent; an assumption, flagged in warnings.

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

# ---------------------------------------------------------------------------
# Assigning units for export
# ---------------------------------------------------------------------------

LIFETIME_UNIT = "yr"
FULL_LOAD_HOURS_UNIT = "h/yr"
FOM_FACTOR_UNIT = "1/yr"
ENERGY_RATIO_UNIT = f"{TARGET_ENERGY_UNIT}/{TARGET_ENERGY_UNIT}_out"

# unit shown next to each column in the step 9 sheet "techno_economic_data"
FIELD_UNITS = {
    "capex": TARGET_CAPEX_UNIT, "capex_scale_base": TARGET_UNIT,
    "fom_factor": FOM_FACTOR_UNIT, "fom": f"{TARGET_CAPEX_UNIT}/yr", "vom": TARGET_VOM_UNIT,
    "lifetime_years": LIFETIME_UNIT, "full_load_hours": FULL_LOAD_HOURS_UNIT,
    "efficiency_energetic": "-", "efficiency_material": "-",
    "energy_losses": ENERGY_RATIO_UNIT,
}
EFFICIENCY_COLUMNS = ["efficiency_energetic", "energy_losses", "efficiency_material"]

UNIT_COLUMN_NAMES = {"lifetime_years": "lifetime_unit"}   # default name is <column>_unit

# unit shown next to each column in the step 10 sheet "model_input"
MODEL_INPUT_UNITS = {
    "invest_cost": TARGET_CAPEX_UNIT, "full_load_hours": FULL_LOAD_HOURS_UNIT,
    "lifetime": LIFETIME_UNIT, "efficiency_material": "-", "efficiency_energetic": "-",
    "fom_factor": FOM_FACTOR_UNIT, "electricity_consumption": f"{TARGET_ENERGY_UNIT}_el/{TARGET_ENERGY_UNIT}_out",
}

# ---------------------------------------------------------------------------
# Helping functions for data handling
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Function definitions
# ---------------------------------------------------------------------------

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
    # populated by step 7 (FOM)
    fom: Optional[float] = None          # EUR/MW/yr, based on final capex
    fom_unit: Optional[str] = None
    # populated by step 7b (commodity cost)
    # commodity_cost: Optional[float] = None       # EUR/MWh_out, inputs only
    # commodity_cost_note: Optional[str] = None
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
        original_capex_unit = self.capex_unit # capex_scale_base's implicit unit is its denominator

        fx_rate = get_fx_rate(fx_rates_df, self.capex_ref_year) if self.capex_ref_year is not None else None

        # --- CAPEX ---
        if self.capex is None:
            self._missing("capex not provided.")
        elif self.capex_unit is None:
            self._fail("capex given without capex_unit.", "capex")
        else:
            try:
                self.capex, self.capex_unit = unit_unifying(
                    target_unit=TARGET_CAPEX_UNIT, input_unit=self.capex_unit,
                    input_value=self.capex, fx_rate_usd_per_eur=fx_rate)
            except ValueError as e:
                self._fail(f"capex ({original_capex_unit} -> {TARGET_CAPEX_UNIT}): {e}", "capex", "capex_unit")

        # --- capex_scale_base ---
        if self.capex_scale_base is None:
            self._missing("capex_scale_base not provided.")
        elif original_capex_unit is None:
            self._fail("capex_scale_base given without capex_unit.", "capex_scale_base")
        else:
            _, capacity_tokens = split_compound_unit(original_capex_unit)
            if len(capacity_tokens) != 1:
                self._fail(f"capex_unit '{original_capex_unit}' has {len(capacity_tokens)} "
                        f"denominator tokens, expected 1.", "capex_scale_base")
            else:
                try:
                    self.capex_scale_base, _ = unit_unifying(
                        target_unit=TARGET_UNIT, input_unit=capacity_tokens[0],
                        input_value=self.capex_scale_base)
                except ValueError as e:
                    self._fail(f"capex_scale_base ({capacity_tokens[0]} -> {TARGET_UNIT}): {e}",
                            "capex_scale_base")

        # --- VOM ---
        if self.vom is None:
            self._missing("vom not provided.")
        elif self.vom_unit is None:
            self._fail("vom given without vom_unit.", "vom")
        else:
            original_vom_unit = self.vom_unit
            try:
                self.vom, self.vom_unit = unit_unifying(
                    target_unit=TARGET_VOM_UNIT, input_unit=self.vom_unit,
                    input_value=self.vom, fx_rate_usd_per_eur=fx_rate)
            except ValueError as e:
                self._fail(f"vom ({original_vom_unit} -> {TARGET_VOM_UNIT}): {e}", "vom", "vom_unit")

    def check_stoichiometry(self, tolerance: float = STOICHIOMETRY_TOLERANCE) -> None:
        """
        Step 3 (data-quality check only - does not modify any values): for
        every reaction row with chemical_formula + stoichiometric_coefficient
        set, compute the theoretical amount_per_unit_output from the
        is_stoichiometric_basis row via molar mass ratios, and compare it to
        the *reported* value. Flags (self.warnings) any participant whose
        reported value deviates from the theoretical value by more than
        `tolerance` (fractional, default STOICHIOMETRY_TOLERANCE).

        If the basis row or a participant row is missing its
        amount_per_unit_output, this is explicitly flagged (rather than
        silently skipped) so it's visible that a value needs to be supplied
        or calculated manually - no value is derived or filled in here.
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

            if pd.isna(basis["amount_per_unit_output"]):
                self.warnings.append(
                    f"check_stoichiometry: reaction '{reaction_id}' basis '{basis['commodity']}' has no "
                    f"amount_per_unit_output - reaction cannot be validated, please supply a value."
                )
                continue

            try:
                basis_molar_mass = molar_mass(basis["chemical_formula"])
            except ValueError as e:
                self.warnings.append(f"check_stoichiometry: reaction '{reaction_id}': {e}")
                continue

            for _, participant in group.iterrows():
                if participant["is_stoichiometric_basis"] == 1:
                    continue  # basis row compared against itself is trivial

                if pd.isna(participant["amount_per_unit_output"]):
                    self.warnings.append(
                        f"check_stoichiometry: '{participant['commodity']}' in reaction '{reaction_id}' has no "
                        f"amount_per_unit_output - cannot validate, please supply or calculate a value manually."
                    )
                    continue

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

    def escalate_capex(self, cepci_df: pd.DataFrame) -> None:
        """Step 5. Fail closed: capex that cannot be moved to TARGET_YEAR becomes None."""
        if self.capex is None:
            return  # already flagged upstream
        if self.capex_ref_year is None:
            self._fail("capex_ref_year missing, capex cannot be placed in time.", "capex")
            return
        if self.capex_ref_year == TARGET_YEAR:
            return
        escalated = cepci_escalate(cepci_df, self.capex, self.capex_ref_year, TARGET_YEAR)
        if escalated is None:
            self._fail(f"no CEPCI value for {self.capex_ref_year} or {TARGET_YEAR}.", "capex")
            return
        self.capex = escalated
        self.capex_ref_year = TARGET_YEAR

    def scale_capex(self) -> None:
        """Step 6. capex (EUR/MW) -> total cost -> size_scale() -> EUR/MW at TARGET_CAPACITY_MW.
        Fail closed: capex that cannot be scaled becomes None."""
        if self.capex is None:
            return  # already flagged upstream
        if self.capex_unit != TARGET_CAPEX_UNIT:
            self._fail(f"capex_unit is '{self.capex_unit}', expected '{TARGET_CAPEX_UNIT}'.", "capex")
            return
        if self.capex_scale_base is None:
            self._fail("capex_scale_base missing, capex cannot be scaled to the target capacity.", "capex")
            return
        if abs(self.capex_scale_base - TARGET_CAPACITY_MW) < 1e-9:
            return  # already at target capacity, no exponent needed

        d = self.degression_exponent
        if d is None:
            d = DEFAULT_DEGRESSION_EXPONENT  # local only, so statistics never contain assumed exponents
            self.warnings.append(f"[assumed] degression_exponent missing, using default {d}.")
        try:
            total_target = size_scale(self.capex * self.capex_scale_base,
                                      self.capex_scale_base, TARGET_CAPACITY_MW, d)
        except ValueError as e:
            self._fail(str(e), "capex")
            return
        self.capex = total_target / TARGET_CAPACITY_MW
        self.capex_scale_base = TARGET_CAPACITY_MW

    def compute_fom(self) -> None:
        """Step 7: fom = fom_factor * capex (per year), computed per record. Fail closed."""
        self.fom, self.fom_unit = None, None
        if self.capex is None:
            return  # reason already recorded upstream
        if self.fom_factor is None:
            self._missing("fom_factor not provided, fom not computed.")
            return
        if (self.capex_unit != TARGET_CAPEX_UNIT or self.capex_ref_year != TARGET_YEAR
                or self.capex_scale_base is None
                or abs(self.capex_scale_base - TARGET_CAPACITY_MW) > 1e-9):
            self._fail("capex not in final state (steps 2/5/6 incomplete), fom not computed.")
            return
        self.fom = self.fom_factor * self.capex
        self.fom_unit = f"{TARGET_CAPEX_UNIT}/yr"

    record_id: int = 0                        # row order in the technologies sheet (internal)
    raw: dict = field(default_factory=dict)   # input values as read, for debugging

    def __post_init__(self):
        self.raw = {k: v for k, v in vars(self).items()
                    if k not in ("efficiencies", "warnings", "raw")}

    def _fail(self, msg: str, *fields: str) -> None:
        """Operation failed: record why and set the affected fields to None (fail closed)."""
        self.warnings.append(f"[failed] {msg}")
        for f in fields:
            setattr(self, f, None)

    def _missing(self, msg: str) -> None:
        """Input never provided (normal for partial records, not an error)."""
        self.warnings.append(f"[missing] {msg}")

def check_efficiency_duplicates(commodity_efficiencies_df: pd.DataFrame) -> None:
    """One efficiency set per tech_id is expected; stop if a row is duplicated."""
    key = ["tech_id", "commodity", "direction", "flow_type", "reaction_id"]
    dup = commodity_efficiencies_df[commodity_efficiencies_df.duplicated(subset=key, keep=False)]
    if not dup.empty:
        raise ValueError("commodity_efficiencies has duplicated rows:\n" + dup[key].to_string())

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
            record_id=idx,
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

# ---------------------------------------------------------------------------
# Step 8: aggregation across records of the same tech_id
# ---------------------------------------------------------------------------
RECORD_FIELDS = ["capex", "fom_factor", "fom", "vom", "lifetime_years",
                 "wacc", "full_load_hours", "degression_exponent"]
CARRIED_FIELDS = ["tech_name", "category", "efficiency_energetic", "energy_losses",
                  "efficiency_material"] #, "commodity_cost", "commodity_cost_note"]


def _valid_value(tech: Technology, field_name: str) -> Optional[float]:
    """Value of a field if it may enter the statistics, else None.
    Belt and braces on top of fail-closed steps 2-7: capex/fom/vom must also be in
    their final unit/year/scale, so a leftover value can never be averaged in."""
    value = getattr(tech, field_name)
    if value is None or pd.isna(value):
        return None
    if field_name in ("capex", "fom") and not (
            tech.capex_unit == TARGET_CAPEX_UNIT and tech.capex_ref_year == TARGET_YEAR
            and tech.capex_scale_base is not None
            and abs(tech.capex_scale_base - TARGET_CAPACITY_MW) < 1e-9):
        return None
    if field_name == "vom" and tech.vom_unit != TARGET_VOM_UNIT:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def aggregate_technologies(technologies: list[Technology]) -> tuple[pd.DataFrame, dict]:
    """
    Step 8 (capex-led): the variants are defined by CAPEX only, and every variant is ONE real source
    record, so all its values (fom_factor, fom, vom, lifetime, ...) stay coherent:
      <tech_id>_min     record with the lowest capex
      <tech_id>_max     record with the highest capex
      <tech_id>_median  record whose capex is closest to the median capex (ties -> lower capex, then record_id)
    min/max exist only if capex actually differs between records. Records without a valid capex cannot be
    ranked and are NOT used (listed in the audit). Nothing is interpolated or filled from other records.
    Fallback: a tech_id with no valid capex in any record still gets a _median row built from its first
    record (capex empty), so it does not silently disappear from the output.
    Efficiencies/commodity cost are identical for all records of a tech_id and are carried through once.
    Returns (aggregated_df, audit_info).
    """
    groups: dict[str, list[Technology]] = {}
    for tech in technologies:
        groups.setdefault(tech.tech_id, []).append(tech)

    rows, audit = [], {}
    for tech_id, records in groups.items():
        ranked = []
        for r in records:
            c = _valid_value(r, "capex")
            if c is not None:
                ranked.append((c, r.record_id, r))
        ranked.sort(key=lambda x: (x[0], x[1]))

        notes, selected = [], {}
        if ranked:
            capexes = [c for c, _, _ in ranked]
            median_capex = float(pd.Series(capexes).median())
            selected["median"] = min(ranked, key=lambda x: (abs(x[0] - median_capex), x[0], x[1]))[2]
            if capexes[0] != capexes[-1]:
                selected["min"] = ranked[0][2]
                selected["max"] = next(r for c, _, r in ranked if c == capexes[-1])
        else:
            selected["median"] = records[0]
            notes.append(f"no valid capex in any record, row built from record {records[0].record_id}")

        categories = {r.category for r in records if r.category is not None}
        if len(categories) > 1:
            notes.append(f"records disagree on category: {sorted(categories)}")

        carried = {f: next((getattr(r, f) for r in records if getattr(r, f) is not None), None)
                   for f in CARRIED_FIELDS}
        for stat, rec in selected.items():
            rows.append({"tech_id": f"{tech_id}_{stat}", "base_tech_id": tech_id, "statistic": stat,
                         "source_record_id": rec.record_id, "n_records": len(records),
                         "n_with_capex": len(ranked), **carried,
                         **{f: _valid_value(rec, f) for f in RECORD_FIELDS}})

        audit[tech_id] = {
            "n_records": len(records), "n_with_capex": len(ranked),
            "selected": {s: rec.record_id for s, rec in selected.items()},
            # fom is derived and an empty degression_exponent just means the default was used -> not listed
            "empty": {s: [f for f in RECORD_FIELDS if f not in ("fom", "degression_exponent")
                          and _valid_value(rec, f) is None] for s, rec in selected.items()},
            "no_capex": [r.record_id for r in records
                         if _valid_value(r, "capex") is None and r.raw.get("capex") is None],
            "capex_failed": [r.record_id for r in records
                             if _valid_value(r, "capex") is None and r.raw.get("capex") is not None],
            "notes": notes,
        }
    return pd.DataFrame(rows), audit


def print_aggregation_audit(audit: dict) -> None:
    """Compact console summary of step 8. Delete this function and its call in main() to remove it."""
    for tech_id, a in audit.items():
        sel = " ".join(f"{s}=rec{i}" for s, i in a["selected"].items())
        empty = "; ".join(f"{s}: {','.join(fs)}" for s, fs in a["empty"].items() if fs)
        print(f"[step 8] {tech_id}: {a['n_records']} rec, {a['n_with_capex']} with capex | {sel}"
              + (f" | EMPTY {empty}" if empty else "")
              + (f" | NO CAPEX rec {a['no_capex']} (unused)" if a["no_capex"] else "")
              + (f" | CAPEX FAILED rec {a['capex_failed']} (unused)" if a["capex_failed"] else "")
              + "".join(f" | NOTE {n}" for n in a["notes"]))

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

# ---------------------------------------------------------------------------
# Cost development function using CEPCI Index
# ---------------------------------------------------------------------------

def cepci_escalate(cepci_df: pd.DataFrame, cost_base: float,
                    cepci_base_year, cepci_target_year) -> Optional[float]:
    """
    Chemical Engineering Plant Cost Index escalation: rescale cost_base
    (given in cepci_base_year terms) to cepci_target_year terms.
    Takes the already-loaded cepci_df (sheet 'cepci_index') rather than
    reading the input file itself, so it can be called once per technology
    without re-reading the Excel file every time.
    Returns None (and prints a note) if either year isn't covered by the
    CEPCI table.
    """
    base_match = cepci_df.loc[cepci_df["year"] == cepci_base_year]
    target_match = cepci_df.loc[cepci_df["year"] == cepci_target_year]
    if base_match.empty:
        print(f"[cepci_escalate] No CEPCI value found for base year {cepci_base_year}.")
        return None
    if target_match.empty:
        print(f"[cepci_escalate] No CEPCI value found for target year {cepci_target_year}.")
        return None

    cepci_base = base_match["cepci_value"].item()
    cepci_target = target_match["cepci_value"].item()
    if cepci_base <= 0:
        raise ValueError("cepci_base must be > 0")
    return cost_base * (cepci_target / cepci_base)

# ---------------------------------------------------------------------------
# Economy of scale function
# ---------------------------------------------------------------------------

def size_scale(cost_base: float, scale_base: float, scale_design: float,
               degression_exponent: float) -> float:
    # applying economy of scale effects to a TOTAL cost
    # Cost_design = Cost_base * (Scale_design / Scale_base) ** d

    if scale_base <= 0:
        raise ValueError("scale_base must be > 0")
    if scale_design <= 0:
        raise ValueError("scale_design must be > 0")
    return cost_base * (scale_design / scale_base) ** degression_exponent

# ---------------------------------------------------------------------------
# Data preparation and export functions
# ---------------------------------------------------------------------------

def add_unit_columns(df: pd.DataFrame, field_units: dict) -> pd.DataFrame:
    """Insert a unit column directly after every column listed in field_units."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        out[col] = df[col]
        if col in field_units:
            out[UNIT_COLUMN_NAMES.get(col, f"{col}_unit")] = field_units[col]
    return out


def build_techno_economic_sheet(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """Step 9, sheet 1: aggregated data, self-describing (reference year and capacity basis)."""
    df = aggregated_df.drop(columns=EFFICIENCY_COLUMNS)
    i = df.columns.get_loc("capex") + 1
    df.insert(i, "capex_ref_year", TARGET_YEAR)
    df.insert(i + 1, "capex_scale_base", TARGET_CAPACITY_MW)
    return add_unit_columns(df, FIELD_UNITS)


def build_model_input_table(aggregated_df: pd.DataFrame,
                            commodity_efficiencies_df: pd.DataFrame) -> pd.DataFrame:
    """Step 10: one row per tech_id variant (_median/_min/_max); efficiencies repeat per variant."""
    el_rows = commodity_efficiencies_df[(commodity_efficiencies_df["commodity"] == "el")
                                        & (commodity_efficiencies_df["direction"] == "input")
                                        & (commodity_efficiencies_df["flow_type"] == "energy")]
    el_sum = el_rows.groupby("tech_id")["amount_per_unit_output"].sum(min_count=1)

    table = pd.DataFrame({
        "tech_id": aggregated_df["tech_id"],
        "invest_cost": aggregated_df["capex"],
        "full_load_hours": aggregated_df["full_load_hours"],
        "lifetime": aggregated_df["lifetime_years"],
        "efficiency_material": aggregated_df["efficiency_material"],
        "efficiency_energetic": aggregated_df["efficiency_energetic"],
        "fom_factor": aggregated_df["fom_factor"],
        "electricity_consumption": aggregated_df["base_tech_id"].map(el_sum),  # empty if no el input
    })
    return add_unit_columns(table, MODEL_INPUT_UNITS)


def export_to_excel(aggregated_df: pd.DataFrame, commodity_efficiencies_df: pd.DataFrame,
                    out_dir: str) -> str:
    """Steps 9 and 10: write technology_data_output.xlsx with three sheets."""
    if aggregated_df.empty:
        print("[export] Nothing to export: aggregated dataset is empty.")
        return ""
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, "technology_data_output.xlsx")

    base_ids = set(aggregated_df["base_tech_id"])
    orphans = set(commodity_efficiencies_df["tech_id"].dropna()) - base_ids
    if orphans:
        print(f"[export] commodity_efficiencies has tech_ids without a technologies record "
              f"(not exported): {sorted(orphans)}")
    # exported once per technology; joins to the other sheets via base_tech_id
    eff_out = (commodity_efficiencies_df[commodity_efficiencies_df["tech_id"].isin(base_ids)]
               .rename(columns={"tech_id": "base_tech_id"}))
    # per-technology efficiency results go on this sheet only (same value on every row of a technology)
    eff_summary = aggregated_df.drop_duplicates("base_tech_id").set_index("base_tech_id")[EFFICIENCY_COLUMNS]
    eff_out = add_unit_columns(eff_out.join(eff_summary, on="base_tech_id"), {"energy_losses": ENERGY_RATIO_UNIT})

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        build_techno_economic_sheet(aggregated_df).to_excel(writer, sheet_name="techno_economic_data", index=False)
        eff_out.to_excel(writer, sheet_name="commodity_efficiencies", index=False)
        build_model_input_table(aggregated_df, commodity_efficiencies_df).to_excel(
            writer, sheet_name="model_input", index=False)
    print(f"[export] Written: {file_path}")
    return file_path

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

    check_efficiency_duplicates(commodity_efficiencies_df)
    technologies = build_technologies(technologies_df, commodity_efficiencies_df)

    # --- step 2: unify units ---
    for tech in technologies:
        tech.unify_units(fx_rates_df)
    # --- step 3: efficiency / stoichiometry check ---
    for tech in technologies:
        tech.check_stoichiometry()
        tech.compute_efficiency_energetic()
        tech.compute_efficiency_material()
    # --- step 4: cancelled - CAPEX input/output basis correction is handled manually ---
    # --- step 5: CEPCI escalation to TARGET_YEAR ---
    for tech in technologies:
        tech.escalate_capex(cepci_df)
    # --- step 6: capacity scaling to TARGET_CAPACITY_MW ---
    for tech in technologies:
        tech.scale_capex()
    # --- step 7: FOM calculation ---
    for tech in technologies:
        tech.compute_fom()
    # --- step 8: aggregation across records of the same tech_id ---
    aggregated_df, aggregation_audit = aggregate_technologies(technologies)
    sheets["aggregated"] = aggregated_df                # keeps main()'s return signature unchanged
    print_aggregation_audit(aggregation_audit)          # optional: comment out or delete
    # --- step 9/10: export ---
    export_to_excel(aggregated_df, commodity_efficiencies_df, output_path)

    STOP = time.perf_counter()
    print('Total execution time of script', round((STOP - START), 1), 's')

    return sheets, technologies


if __name__ == "__main__":
    sheets, technologies = main()

## AI statement: During the preparation of this work the author(s) used Claude in order to develop software tools used in this pubication. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.
#%%