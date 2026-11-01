"""Generate new cases from the seed cases.

Two methods:

- ``ctgan``: train SDV's CTGAN on the seed features (no labels) and sample.
  This is the method used for reported results. Needs ``pip install sdv``.
- ``bootstrap``: resample seed rows and add Gaussian jitter. Seconds instead
  of minutes, so it is used by the tests and for quick experiments. Do not
  report results from it as CTGAN results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import DISCRETE, SYNTH_COLUMNS
from .generate import bp_derived, troponin_derived

# Physiological limits applied after sampling.
CLIP = {
    "gls": (-26, -5),
    "EF": (15, 85),
    "EDV": (30, 250),
    "ESV": (10, 200),
    "wall_thickness": (7, 30),
    "lv_cavity": (38, 70),
    "regional_variation": (0.5, 8),
    "systolic_bp": (90, 180),
    "diastolic_bp": (50, 110),
    "resting_hr": (35, 120),
    "troponin_value": (1, 200),
    "troponin_baseline_estimated": (0.5, 50),
    "hours_post_exercise": (0, 48),
}

# Real EchoNet values and discrete columns get no measurement noise.
NO_NOISE = {"EF", "EDV", "ESV", *DISCRETE}


def _detect_metadata(df: pd.DataFrame):
    """Table metadata across SDV versions (1.x and 2.x name this differently)."""
    try:
        from sdv.metadata import Metadata
    except ImportError:
        Metadata = None
    if Metadata is not None and hasattr(Metadata, "detect_from_dataframes"):  # SDV >= 1.15
        return Metadata.detect_from_dataframes({"cases": df}, infer_keys=None)
    if Metadata is not None and hasattr(Metadata, "detect_from_dataframe"):
        return Metadata.detect_from_dataframe(df, table_name="cases")
    from sdv.metadata import SingleTableMetadata  # SDV 1.0-1.14

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    return metadata


def sample_ctgan(seed: pd.DataFrame, n: int, epochs: int = 300, seed_value: int = 42) -> pd.DataFrame:
    try:
        from sdv.single_table import CTGANSynthesizer
    except ImportError as e:
        raise ImportError(
            "CTGAN needs the SDV package: pip install 'hcm-athlete[ctgan]' "
            "(or pass --synth bootstrap for a quick run)."
        ) from e
    metadata = _detect_metadata(seed)

    try:
        import torch

        torch.manual_seed(seed_value)
    except ImportError:
        pass

    synth = CTGANSynthesizer(metadata, epochs=epochs, batch_size=500, verbose=False)
    try:
        synth.fit(seed)
    except Exception as e:  # SDV 2.x wants {table_name: DataFrame}
        if "dictionary" not in str(e):
            raise
        synth.fit({"cases": seed})
    sampled = synth.sample(num_rows=n)
    if isinstance(sampled, dict):
        sampled = next(iter(sampled.values()))
    return sampled


def sample_bootstrap(seed: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    idx = rng.integers(0, len(seed), size=n)
    out = seed.iloc[idx].reset_index(drop=True).astype(float)
    for col in out.columns:
        if col not in DISCRETE:
            out[col] += rng.normal(0, 0.15 * seed[col].std(), n)
    return out


def postprocess(df: pd.DataFrame, rng: np.random.Generator, noise_frac: float = 0.05) -> pd.DataFrame:
    """Add measurement noise, clip to physiological ranges, and recompute
    features that are defined from other features so they stay consistent."""
    df = df.copy().astype(float)

    for col in df.columns:
        if col not in NO_NOISE:
            df[col] += rng.normal(0, abs(df[col].mean()) * noise_frac, len(df))

    for col, (lo, hi) in CLIP.items():
        df[col] = df[col].clip(lo, hi)

    for col in ["hypertensive_history", "normalization_trend_slow"]:
        df[col] = df[col].round().clip(0, 1)
    df["hours_post_exercise"] = df["hours_post_exercise"].round()

    bp = bp_derived(df["systolic_bp"], df["diastolic_bp"])
    df["pulse_pressure"], df["map"] = bp["pulse_pressure"], bp["map"]

    trop = [troponin_derived(v, h, b) for v, h, b in zip(
        df["troponin_value"], df["hours_post_exercise"], df["troponin_baseline_estimated"])]
    for key in ("troponin_fold_change", "troponin_above_url", "troponin_pattern"):
        df[key] = [t[key] for t in trop]
    return df


def synthesize(
    seed_cases: pd.DataFrame,
    n: int = 1000,
    method: str = "ctgan",
    rng: np.random.Generator | None = None,
    ctgan_epochs: int = 300,
) -> pd.DataFrame:
    """Sample n new unlabeled cases from the seed cases."""
    rng = rng or np.random.default_rng(42)
    seed = seed_cases[SYNTH_COLUMNS]
    if method == "ctgan":
        raw = sample_ctgan(seed, n, epochs=ctgan_epochs, seed_value=int(rng.integers(1 << 31)))
    elif method == "bootstrap":
        raw = sample_bootstrap(seed, n, rng)
    else:
        raise ValueError(f"Unknown synth method: {method!r} (use 'ctgan' or 'bootstrap')")
    return postprocess(raw[SYNTH_COLUMNS], rng)
