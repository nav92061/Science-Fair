"""Feature lists and engineered features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .generate import bp_derived

STRAIN = [
    "gls", "gcs", "grs", "basal_strain", "mid_strain", "apical_strain",
    "strain_rate_s", "strain_rate_e", "strain_rate_a", "regional_variation",
    "twist", "twist_rate", "untwist_rate", "torsion",
    "e_a_ratio", "e_prime_septal", "e_prime_lateral", "e_e_prime",
    "wall_thickness", "lv_mass_index",
]
STANDARD_ECHO = ["EDV", "ESV", "EF", "lv_cavity"]
GEOMETRY = ["relative_wall_thickness", "asym_ratio", "cavity_wall_ratio"]
ENGINEERED = [
    "strain_gradient", "diastolic_function_index", "systolic_efficiency",
    "geometric_index", "strain_ratio",
]
BLOOD_PRESSURE = [
    "systolic_bp", "diastolic_bp", "resting_hr", "pulse_pressure",
    "map", "bp_response_exercise", "hypertensive_history",
]
TROPONIN = [
    "troponin_value", "hours_post_exercise", "troponin_fold_change",
    "troponin_above_url", "normalization_trend_slow", "troponin_pattern",
]

#: The 45 model inputs, in a fixed order.
FEATURE_COLUMNS = STRAIN + STANDARD_ECHO + GEOMETRY + ENGINEERED + BLOOD_PRESSURE + TROPONIN

#: Features computed from other features (never filled with a median).
COMPUTED = ENGINEERED + ["cavity_wall_ratio"]

#: Columns the synthesizer learns from: everything that is not computed,
#: plus the troponin baseline, which is needed to re-derive the pattern.
SYNTH_COLUMNS = [c for c in FEATURE_COLUMNS if c not in COMPUTED] + ["troponin_baseline_estimated"]

#: Columns that hold whole numbers or 0/1 flags.
DISCRETE = ["hours_post_exercise", "hypertensive_history", "troponin_above_url",
            "normalization_trend_slow", "troponin_pattern"]

#: Short descriptions with units, used in the report and the docs.
DESCRIPTIONS = {
    "wall_thickness": "Max LV wall thickness (mm)",
    "lv_cavity": "LV end-diastolic cavity diameter (mm)",
    "gls": "Global longitudinal strain (%)",
    "gcs": "Global circumferential strain (%)",
    "grs": "Global radial strain (%)",
    "regional_variation": "Regional strain variation (%)",
    "e_e_prime": "E/e' ratio",
    "e_prime_septal": "Septal e' (cm/s)",
    "EF": "Ejection fraction (%)",
    "EDV": "End-diastolic volume (mL)",
    "ESV": "End-systolic volume (mL)",
    "systolic_bp": "Systolic BP (mmHg)",
    "diastolic_bp": "Diastolic BP (mmHg)",
    "resting_hr": "Resting heart rate (bpm)",
    "troponin_value": "hs-troponin (ng/L)",
    "hours_post_exercise": "Hours since exercise when troponin was drawn",
}


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add the computed features. Works on a DataFrame of any length."""
    df = df.copy()
    df["strain_gradient"] = df["apical_strain"] - df["basal_strain"]
    df["diastolic_function_index"] = df["strain_rate_e"] * df["e_a_ratio"]
    df["systolic_efficiency"] = df["gls"] * df["EF"] / 100
    df["geometric_index"] = df["wall_thickness"] / np.sqrt(df["EDV"])
    df["strain_ratio"] = df["gls"] / df["gcs"]
    df["cavity_wall_ratio"] = df["lv_cavity"] / df["wall_thickness"]
    return df


def complete_case(values: dict, medians: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Turn a partial set of measurements into one full model input row.

    - Regional strains default to fixed fractions of GLS (same ratios the
      generator centers on).
    - Pulse pressure and MAP are computed from BP when both are given.
    - Any other missing input gets the training-set median. The median is
      taken over all training cases (HCM and athlete mixed), so it does not
      push the case toward either diagnosis.
    - Engineered features are always recomputed.

    Returns the row and the list of features that were filled with a median.
    """
    unknown = set(values) - set(FEATURE_COLUMNS) - {"troponin_baseline_estimated"}
    if unknown:
        raise ValueError(f"Unknown feature(s): {sorted(unknown)}")

    f = dict(values)
    if "gls" in f:
        f.setdefault("basal_strain", f["gls"] * 0.725)
        f.setdefault("mid_strain", f["gls"] * 0.90)
        f.setdefault("apical_strain", f["gls"] * 1.05)
    if "systolic_bp" in f and "diastolic_bp" in f:
        for k, v in bp_derived(f["systolic_bp"], f["diastolic_bp"]).items():
            f.setdefault(k, v)

    filled = []
    for col in FEATURE_COLUMNS:
        if col not in f and col not in COMPUTED:
            f[col] = float(medians[col])
            filled.append(col)

    row = add_engineered(pd.DataFrame([f]))
    return row[FEATURE_COLUMNS], filled
