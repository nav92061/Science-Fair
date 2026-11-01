"""Label synthetic cases with a weighted HCM score.

The synthesizer does not keep labels, so each sampled case is scored and then
labeled by a coin flip weighted by that score. This keeps grey-zone cases in
the data, and it also means no model can score close to perfect.

Important: the models are partly learning to reproduce this formula. Good
test-set numbers show the models recovered the pattern in the synthetic data,
not that they work on real patients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Weight of each component in the HCM score. Sums to 1.0.
WEIGHTS = {
    "wall": 0.20,
    "gls": 0.18,
    "cavity": 0.15,
    "regional_variation": 0.12,
    "e_e_prime": 0.12,
    "edv": 0.08,
    "resting_hr": 0.08,
    "troponin_pattern": 0.07,
}

#: Center of the cavity sigmoid (mm). Cavities below this push toward HCM.
CAVITY_CUTOFF_MM = 54.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def component_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Each component is 0-1, where higher means more HCM-like."""
    return pd.DataFrame({
        "wall": _sigmoid(2.0 * (df["wall_thickness"] - 13.5)),
        "gls": _sigmoid(0.5 * (df["gls"] + 17)),  # less negative GLS = worse
        "cavity": _sigmoid(-0.3 * (df["lv_cavity"] - CAVITY_CUTOFF_MM)),
        "regional_variation": _sigmoid(1.5 * (df["regional_variation"] - 3.0)),
        "e_e_prime": _sigmoid(0.8 * (df["e_e_prime"] - 10)),
        "edv": _sigmoid(-0.03 * (df["EDV"] - 95)),
        "resting_hr": _sigmoid(0.1 * (df["resting_hr"] - 72.5)),
        # pattern 2 (concerning) -> 0.5, 1 (indeterminate) -> 0.25, 0 -> 0
        "troponin_pattern": df["troponin_pattern"].round().clip(0, 2) * 0.25,
    }, index=df.index)


def hcm_score(df: pd.DataFrame, rng: np.random.Generator, noise_sd: float = 0.1) -> pd.Series:
    """Weighted score plus noise, clipped to 0-1."""
    parts = component_scores(df)
    score = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)
    score = score + rng.normal(0, noise_sd, len(df))
    return score.clip(0, 1)


def assign_labels(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy with ``hcm_score`` and ``label`` (1 = HCM, 0 = athlete)."""
    out = df.copy()
    out["hcm_score"] = hcm_score(out, rng)
    out["label"] = rng.binomial(1, out["hcm_score"].to_numpy())
    return out
