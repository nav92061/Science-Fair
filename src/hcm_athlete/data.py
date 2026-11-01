"""Load EchoNet-Dynamic measurements.

EchoNet-Dynamic is distributed by Stanford under a research use agreement and
cannot be redistributed, so it is not included in this repository. Request it
at https://echonet.github.io/dynamic/ and point the pipeline at FileList.csv.
"""

from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["EF", "EDV", "ESV"]

COLAB_GUESSES = [
    "/content/drive/MyDrive/EchoNet-Dynamic/FileList.csv",
    "/content/drive/My Drive/EchoNet-Dynamic/FileList.csv",
]


def find_filelist(path: str | os.PathLike | None = None) -> Path | None:
    """Return the first FileList.csv found, or None.

    Search order: the given path, $ECHONET_FILELIST, ./FileList.csv,
    ./data/FileList.csv, then the usual Google Drive locations in Colab.
    """
    candidates = [path, os.environ.get("ECHONET_FILELIST"), "FileList.csv",
                  "data/FileList.csv", *COLAB_GUESSES]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    if Path("/content/drive").is_dir():
        hits = glob.glob("/content/drive/**/FileList.csv", recursive=True)
        if hits:
            return Path(hits[0])
    return None


def load_echonet(path: str | os.PathLike) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}; is this EchoNet FileList.csv?")
    df = df.dropna(subset=REQUIRED)
    return df[(df["EF"] > 0) & (df["EDV"] > 0) & (df["ESV"] > 0)].reset_index(drop=True)


def demo_echonet(n: int = 1500, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Fake EF/EDV/ESV values with roughly EchoNet-like spread.

    For trying the code without the dataset. Results from demo data are not
    results and must not be reported.
    """
    rng = rng or np.random.default_rng(0)
    edv = rng.lognormal(np.log(85), 0.4, n).clip(25, 300)
    ef = rng.normal(57, 11, n).clip(15, 85)
    esv = edv * (1 - ef / 100)
    return pd.DataFrame({"EF": ef, "EDV": edv, "ESV": esv})


def get_echonet(path: str | os.PathLike | None = None, allow_demo: bool = False,
                rng: np.random.Generator | None = None) -> tuple[pd.DataFrame, str]:
    """Load EchoNet, or demo data if allowed. Returns (data, source)."""
    found = find_filelist(path)
    if found:
        return load_echonet(found), str(found)
    if path:
        raise FileNotFoundError(f"FileList.csv not found at {path}")
    if not allow_demo:
        raise FileNotFoundError(
            "EchoNet FileList.csv not found. Pass --echonet PATH, set ECHONET_FILELIST, "
            "or use --demo to run on fake data."
        )
    warnings.warn("Using FAKE demo EF/EDV/ESV values. Do not report these results.", stacklevel=2)
    return demo_echonet(rng=rng), "demo"
