"""Plain-text report for one case."""

from __future__ import annotations

from .features import DESCRIPTIONS
from .generate import classify_troponin_pattern
from .labeling import CAVITY_CUTOFF_MM

DISCLAIMER = ("Research prototype trained on synthetic data. Not a medical device and "
              "not for clinical decisions.")

TROPONIN_TEXT = {
    0: "fits a normal post-exercise rise",
    1: "indeterminate",
    2: "persistent or high; would need follow-up",
}


def band(p: float) -> str:
    if p < 0.2:
        return "Low - athlete's heart more likely"
    if p < 0.5:
        return "Intermediate - grey zone"
    return "Elevated - HCM cannot be excluded"


def make_report(values: dict, p_hcm: float, filled: list[str], model_name: str = "") -> str:
    lines = [
        "HCM vs athlete's heart - model output",
        "=" * 44,
        f"P(HCM): {p_hcm:.0%}   [{band(p_hcm)}]",
    ]
    if model_name:
        lines.append(f"Model:  {model_name}")
    lines += ["", "Measurements given:"]
    for key, value in values.items():
        label = DESCRIPTIONS.get(key, key)
        lines.append(f"  {label:<46} {value:g}")

    notes = []
    cavity = values.get("lv_cavity")
    if cavity is not None:
        side = "above" if cavity >= CAVITY_CUTOFF_MM else "below"
        leans = "athlete's heart" if side == "above" else "HCM"
        notes.append(f"LV cavity {cavity:g} mm is {side} ~{CAVITY_CUTOFF_MM:g} mm, "
                     f"which leans toward {leans}.")
    if {"troponin_value", "hours_post_exercise", "troponin_baseline_estimated"} <= values.keys():
        pattern = classify_troponin_pattern(values["troponin_value"], values["hours_post_exercise"],
                                            values["troponin_baseline_estimated"])
        notes.append(f"Troponin timing pattern: {TROPONIN_TEXT[pattern]}.")
    if notes:
        lines += ["", "Notes:"] + [f"  - {n}" for n in notes]

    if filled:
        lines += ["", f"{len(filled)} inputs were not given and were set to the training median "
                      "(favors neither diagnosis)."]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)
