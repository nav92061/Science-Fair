"""End-to-end run: seed cases -> synthesizer -> labels -> models -> checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import __version__
from .data import get_echonet
from .features import FEATURE_COLUMNS, add_engineered
from .generate import make_seed_cases
from .labeling import assign_labels
from .literature import check_pairs, check_profiles
from .model import HCMModel, compare_models, importance
from .synth import synthesize


@dataclass
class RunConfig:
    echonet: str | None = None
    demo: bool = False
    out_dir: str = "results"
    seed: int = 42
    n_seed_per_class: int = 500
    n_synthetic: int = 1000
    synth: str = "ctgan"
    ctgan_epochs: int = 300
    test_size: float = 0.2
    plots: bool = True


def build_dataset(cfg: RunConfig, rng: np.random.Generator, log=print):
    echonet, source = get_echonet(cfg.echonet, allow_demo=cfg.demo, rng=rng)
    log(f"EchoNet rows: {len(echonet)}  (source: {source})")
    if source == "demo":
        log("WARNING: demo data is fake. Do not report these results.")

    seed_cases = make_seed_cases(echonet, cfg.n_seed_per_class, rng)
    log(f"Seed cases: {len(seed_cases)} ({cfg.n_seed_per_class} each HCM / athlete / normal)")

    log(f"Sampling {cfg.n_synthetic} cases with {cfg.synth}...")
    synthetic = synthesize(seed_cases, cfg.n_synthetic, cfg.synth, rng, cfg.ctgan_epochs)
    labeled = add_engineered(assign_labels(synthetic, rng))
    return labeled, source


def run(cfg: RunConfig, log=print) -> dict:
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data, source = build_dataset(cfg, rng, log)
    n_hcm = int(data["label"].sum())
    grey = int(data["hcm_score"].between(0.4, 0.6).sum())
    log(f"Labels: {n_hcm} HCM, {len(data) - n_hcm} athlete, {grey} with score 0.4-0.6")

    X, y = data[FEATURE_COLUMNS], data["label"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)
    log(f"Split: {len(X_train)} train / {len(X_test)} test (stratified, before training)")

    rows, fitted, curves = compare_models(X_train, y_train, X_test, y_test, seed=cfg.seed)
    table = pd.DataFrame(rows)
    best_name = table.dropna(subset=["auc"]).sort_values("auc", ascending=False)["model"].iloc[0]
    log("\n" + table.round(3).to_string(index=False))
    log(f"\nBest model by test AUC: {best_name}")

    info = {
        "version": __version__, "data_source": source, "synth": cfg.synth,
        "n_train": len(X_train), "n_test": len(X_test), "seed": cfg.seed,
    }
    reference = X_train[["wall_thickness", "lv_cavity", "gls"]].assign(label=y_train)
    model = HCMModel(fitted[best_name], best_name, X_train.median(), info, reference=reference)
    model.save(out / "model.joblib")

    imp = importance(fitted[best_name], X_test, y_test, seed=cfg.seed)
    profiles = pd.DataFrame(check_profiles(model))
    n_ok = int(profiles["correct"].sum())
    log("\nLiterature-based profiles (sanity check, not a test set):")
    log(profiles[["profile", "expected", "predicted", "p_hcm"]].to_string(index=False))
    log(f"{n_ok} of {len(profiles)} profiles classified as expected.")
    pairs = check_pairs(profiles.to_dict(orient="records"))
    n_pairs_ok = sum(p["ranked_right"] for p in pairs)
    for p in pairs:
        log(f"  {p['pair']} pair: HCM {p['p_hcm_hcm']:.2f} vs athlete {p['p_hcm_athlete']:.2f} "
            f"({'ranked right' if p['ranked_right'] else 'ranked wrong'})")
    log(f"{n_pairs_ok} of {len(pairs)} same-study pairs ranked the right way round.")

    table.to_csv(out / "model_comparison.csv", index=False)
    imp.to_csv(out / "feature_importance.csv", index=False)
    profiles.to_csv(out / "literature_profiles.csv", index=False)
    data.to_csv(out / "synthetic_cases.csv", index=False)

    summary = {
        **info,
        "config": asdict(cfg),
        "best_model": best_name,
        "label_counts": {"hcm": n_hcm, "athlete": len(data) - n_hcm, "grey_zone": grey},
        "models": table.replace({np.nan: None}).to_dict(orient="records"),
        "top_features": imp.head(10)[["feature", "importance"]].to_dict(orient="records"),
        "profiles_correct": n_ok,
        "profiles_total": len(profiles),
        "pairs_ranked_right": n_pairs_ok,
        "pairs_total": len(pairs),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    if cfg.plots:
        from .plots import plot_importance, plot_roc

        baseline = table[table["auc"].isna()].iloc[0]
        plot_roc(curves, table, baseline, out / "roc_curves.png")
        plot_importance(imp, out / "feature_importance.png")

    log(f"\nSaved results to {out.resolve()}")
    return summary
