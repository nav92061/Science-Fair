"""Seed-case generators.

Each seed case pairs one real EchoNet-Dynamic row (EF, EDV, ESV) with synthetic
strain, cavity, blood-pressure and troponin values drawn from distributions
based on published cohort summaries. Only EF, EDV and ESV are real.

Parameters with a published source say so in a comment (checked against each
paper's abstract and tables; see docs/METHODS.md). Parameters without one are
modeling assumptions and are marked as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIAGNOSES = ("HCM", "Athlete", "Normal")
LABELS = {"HCM": 1, "Athlete": 0, "Normal": 2}

# Upper reference limit for high-sensitivity troponin (ng/L), male cutoff.
TROPONIN_URL = 22.0


def generate_bp_features(diagnosis: str, rng: np.random.Generator) -> dict:
    """Blood pressure and resting heart rate.

    The useful signal is resting HR: trained athletes tend to have a low
    resting rate. BP history helps rule out hypertension as the cause of a
    thick wall.
    """
    noise = rng.uniform(0.95, 1.05)
    if diagnosis == "HCM":
        f = {
            "systolic_bp": rng.normal(125, 15) * noise,
            "diastolic_bp": rng.normal(78, 10) * noise,
            "resting_hr": rng.normal(70, 12) * noise,
            "bp_response_exercise": rng.uniform(1.2, 1.5),
            "hypertensive_history": int(rng.binomial(1, 0.20)),
        }
    elif diagnosis == "Athlete":
        f = {
            "systolic_bp": rng.normal(115, 10) * noise,
            "diastolic_bp": rng.normal(68, 8) * noise,
            "resting_hr": rng.normal(50, 8) * noise,
            "bp_response_exercise": rng.uniform(1.5, 1.9),
            "hypertensive_history": int(rng.binomial(1, 0.02)),
        }
    else:
        f = {
            "systolic_bp": rng.normal(120, 12) * noise,
            "diastolic_bp": rng.normal(75, 8) * noise,
            "resting_hr": rng.normal(70, 10) * noise,
            "bp_response_exercise": rng.uniform(1.4, 1.6),
            "hypertensive_history": int(rng.binomial(1, 0.25)),
        }
    f.update(bp_derived(f["systolic_bp"], f["diastolic_bp"]))
    return f


def bp_derived(systolic, diastolic) -> dict:
    """Pulse pressure and mean arterial pressure."""
    pulse_pressure = systolic - diastolic
    return {"pulse_pressure": pulse_pressure, "map": diastolic + pulse_pressure / 3}


def classify_troponin_pattern(value: float, hours: float, baseline: float) -> int:
    """Rule-based reading of a troponin value given time since exercise.

    Returns 0 (looks exercise-induced), 1 (indeterminate) or 2 (concerning).
    Exercise-induced rises peak within about 6 h and fall back within 24-72 h;
    a value that stays high after 24 h, or is very high, is treated as
    concerning.
    """
    fold = value / baseline if baseline > 0 else 1.0
    if hours <= 6 and fold > 2 and value < 100:
        return 0
    if hours > 12 and fold < 1.5:
        return 0
    if hours > 24 and fold > 1.5:
        return 2
    if value > 100:
        return 2
    return 1


def troponin_derived(value: float, hours: float, baseline: float) -> dict:
    """Troponin features that are computed from value, timing and baseline."""
    return {
        "troponin_fold_change": value / baseline if baseline > 0 else 1.0,
        "troponin_above_url": int(value > TROPONIN_URL),
        "troponin_pattern": classify_troponin_pattern(value, hours, baseline),
    }


def generate_troponin_features(diagnosis: str, rng: np.random.Generator) -> dict:
    """Troponin value with exercise timing.

    Athletes: low baseline, about half show a transient post-exercise rise
    that clears quickly. HCM: higher baseline and slower clearance.
    """
    hours = int(rng.choice([0, 1, 3, 6, 12, 24, 48]))
    noise = rng.uniform(0.85, 1.15)

    if diagnosis == "Athlete":
        baseline = rng.uniform(3, 6)  # Li 2021: pre-exercise hs-cTnT mean 4.4 ng/L
        # Shave 2007: cTnT detectable after exercise in 47%. Peak within hours and return
        # to baseline in 24-72 h: Shave 2010, Aengevaeren 2021. Factors are assumptions.
        if rng.random() < 0.47:
            if hours <= 6:
                factor = rng.uniform(3, 8)
            elif hours <= 24:
                factor = rng.uniform(1.5, 3)
            else:
                factor = rng.uniform(0.9, 1.2)
        else:
            factor = rng.uniform(1.0, 1.5)
        slow = 0
    elif diagnosis == "HCM":
        baseline = rng.uniform(7, 24)  # Jenab 2014: HCM median 12.4 ng/L (IQR 6.9-24.3)
        if hours <= 6:
            factor = rng.uniform(1.5, 3)
        elif hours <= 24:
            factor = rng.uniform(1.2, 2)
        else:
            factor = rng.uniform(1.0, 1.5)
        slow = 1
    else:
        baseline = rng.uniform(2, 8)
        factor = rng.uniform(1.0, 2.0) if hours <= 6 else rng.uniform(0.9, 1.1)
        slow = 0

    value = baseline * factor * noise
    f = {
        "troponin_value": value,
        "hours_post_exercise": hours,
        "troponin_baseline_estimated": baseline,
        "normalization_trend_slow": slow,
    }
    f.update(troponin_derived(value, hours, baseline))
    return f


def generate_strain_features(row: pd.Series, diagnosis: str, rng: np.random.Generator) -> dict:
    """Strain, wall, cavity and diastolic features for one seed case.

    About 25% of HCM and athlete cases are drawn as grey-zone cases
    (mild HCM, or athletes with a 13-15 mm wall).
    """
    grey = rng.random() > 0.75
    noise = rng.uniform(0.9, 1.1)

    if diagnosis == "HCM":
        if grey:  # mild HCM: Sheikh 2015, athletes with HCM
            wall = rng.normal(15.8, 3.4) * noise  # Sheikh 2015
            cavity = rng.normal(47.8, 6.0) * noise  # Sheikh 2015
            gls = rng.normal(-15.8, 3.0) * noise  # assumption (Sheikh 2015 has no strain)
            e_e_prime = rng.normal(7.9, 2.4) * noise  # Sheikh 2015
            e_prime_septal = rng.normal(9.0, 3.0) * noise  # Sheikh 2015: 0.09 +/- 0.03 m/s
        else:  # typical HCM (assumptions, except as noted)
            wall = rng.normal(19.0, 3.0) * noise
            cavity = rng.normal(46.0, 5.0) * noise
            gls = rng.normal(-10.8, 3.8) * noise  # Butz 2011 reports -8.1 +/- 3.8 (n = 15)
            e_e_prime = rng.normal(12.0, 5.0) * noise
            e_prime_septal = rng.normal(7.0, 2.0) * noise
        gcs = rng.normal(-18.4, 4.8) * noise  # assumption
        grs = rng.normal(30.8, 11.6) * noise  # assumption
        wall = np.clip(wall, 13, 30)
        cavity = np.clip(cavity, 38, 60)
        gls = np.clip(gls, -20, -5)
        gcs = np.clip(gcs, -25, -12)
        grs = np.clip(grs, 15, 45)
        e_a_ratio = rng.uniform(0.6, 1.2)
        regional_variation = rng.uniform(3.0, 7.0)

    elif diagnosis == "Athlete":
        wall = (rng.uniform(13, 15) if grey else rng.normal(11.0, 2.0)) * noise
        cavity = rng.normal(58.0, 5.0) * noise  # assumption; Caselli 2014: 60 +/- 3 mm
        gls = rng.normal(-19.0, 2.0) * noise  # assumption; Butz 2011: -15.2 +/- 3.6
        gcs = rng.normal(-23.0, 3.0) * noise
        grs = rng.normal(40.0, 10.0) * noise
        wall = np.clip(wall, 8, 16)
        cavity = np.clip(cavity, 50, 70)
        gls = np.clip(gls, -24, -14)
        gcs = np.clip(gcs, -28, -18)
        grs = np.clip(grs, 25, 60)
        e_prime_septal = rng.normal(12.5, 1.9) * noise  # Caselli 2014 (e' site not stated)
        e_e_prime = rng.normal(6.0, 1.5) * noise  # assumption; Caselli 2014: 6.6 +/- 1.2
        e_a_ratio = rng.uniform(1.3, 2.3)
        regional_variation = rng.uniform(0.8, 2.5)

    else:
        wall = rng.normal(10.0, 1.5) * noise
        cavity = rng.normal(52.0, 4.0) * noise
        gls = rng.normal(-19.0, 2.0) * noise
        gcs = rng.normal(-22.0, 3.0) * noise
        grs = rng.normal(40.0, 8.0) * noise
        e_prime_septal = rng.normal(11.0, 2.0) * noise
        e_e_prime = rng.normal(7.0, 2.0) * noise
        e_a_ratio = rng.uniform(0.95, 1.9)
        regional_variation = rng.uniform(1.0, 2.5)

    athlete = diagnosis == "Athlete"
    return {
        "wall_thickness": wall,
        "lv_cavity": cavity,
        "gls": gls,
        "gcs": gcs,
        "grs": grs,
        "basal_strain": gls * rng.uniform(0.65, 0.80),
        "mid_strain": gls * rng.uniform(0.85, 0.95),
        "apical_strain": gls * rng.uniform(0.95, 1.15),
        "strain_rate_s": rng.uniform(-1.3, -0.7) * noise,
        "strain_rate_e": rng.uniform(0.9, 1.6) if athlete else rng.uniform(0.6, 1.0),
        "strain_rate_a": rng.uniform(0.4, 0.9) * noise,
        "regional_variation": regional_variation,
        "twist": rng.uniform(9, 19) * noise,
        "twist_rate": rng.uniform(95, 135) * noise,
        "untwist_rate": rng.uniform(-130, -80) * noise,
        "torsion": rng.uniform(1.8, 4.5) * noise,
        "e_a_ratio": e_a_ratio,
        "e_prime_septal": e_prime_septal,
        "e_prime_lateral": e_prime_septal * rng.uniform(1.3, 1.6),
        "e_e_prime": e_e_prime,
        "lv_mass_index": rng.uniform(70, 120) if athlete else rng.uniform(90, 150),
        "relative_wall_thickness": rng.uniform(0.30, 0.44) if athlete else rng.uniform(0.42, 0.68),
        "asym_ratio": rng.uniform(0.9, 1.2) if athlete else rng.uniform(1.2, 2.2),
    }


def _pick_rows(df: pd.DataFrame, mask: pd.Series, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Take up to n rows matching mask, topping up with random rows if short."""
    rows = df[mask].head(n)
    if len(rows) < n:
        extra = df.sample(n=n - len(rows), random_state=int(rng.integers(1 << 31)))
        rows = pd.concat([rows, extra])
    return rows


def make_seed_cases(
    echonet: pd.DataFrame,
    n_per_class: int = 500,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build the seed dataset: n_per_class HCM, athlete and normal cases."""
    rng = rng or np.random.default_rng(42)
    pool = echonet.sample(
        n=min(len(echonet), 3 * n_per_class), random_state=int(rng.integers(1 << 31))
    )
    picks = {
        # Smaller hearts with preserved EF for HCM, larger for athletes.
        "HCM": _pick_rows(pool, (pool["EF"] > 55) & (pool["EDV"] < 100), n_per_class, rng),
        "Athlete": _pick_rows(pool, (pool["EF"] > 55) & (pool["EDV"] > 90), n_per_class, rng),
        "Normal": pool.head(n_per_class),
    }

    cases = []
    for diagnosis, rows in picks.items():
        for _, row in rows.iterrows():
            f = generate_strain_features(row, diagnosis, rng)
            f.update(generate_bp_features(diagnosis, rng))
            f.update(generate_troponin_features(diagnosis, rng))
            f.update({
                "EF": row["EF"], "EDV": row["EDV"], "ESV": row["ESV"],
                "diagnosis": diagnosis, "label": LABELS[diagnosis],
            })
            cases.append(f)
    return pd.DataFrame(cases)
