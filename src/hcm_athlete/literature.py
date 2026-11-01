"""Six profiles built from published cohort summaries.

Each profile holds only values reported in its paper (checked against the
abstract and published tables in November 2026; see docs/METHODS.md). Every
other model input is set to the training-set median, which favors neither
diagnosis. So a profile asks: given only what the paper reported, which way
does the model lean?

These are a sanity check, not a test set. Most are cohort averages, not
individual patients. Where a paper reports a range instead of a mean, the
midpoint is used and the note says so.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    citation: str
    expected: str  # "HCM" or "Athlete"
    values: dict  # only values reported in the paper
    note: str
    pair: str = ""  # profiles from the same study share a pair name


PROFILES = [
    Profile(
        name="Sheikh 2015 - athletes with HCM",
        citation="Sheikh N, et al. Circ Cardiovasc Imaging 2015;8:e003454",
        expected="HCM",
        values=dict(wall_thickness=15.8, lv_cavity=47.8, e_e_prime=7.9, e_prime_septal=9.0),
        note="Means for 106 athletes with HCM. Septal e' 0.09 m/s in the paper.",
        pair="Sheikh 2015",
    ),
    Profile(
        name="Sheikh 2015 - athletes with physiological LVH",
        citation="Sheikh N, et al. Circ Cardiovasc Imaging 2015;8:e003454",
        expected="Athlete",
        values=dict(wall_thickness=14.5, lv_cavity=54.9, e_e_prime=5.5),
        note="Means for 55 athletes with a 13-16 mm wall (wall = range midpoint).",
        pair="Sheikh 2015",
    ),
    Profile(
        name="Caselli 2014 - grey-zone athletes",
        citation="Caselli S, et al. Am J Cardiol 2014;114:1383-1389",
        expected="Athlete",
        values=dict(wall_thickness=14.0, lv_cavity=60.0, e_prime_septal=12.5, e_e_prime=6.6, EF=63),
        note="Same 13-15 mm wall as the next case (wall = midpoint), large cavity.",
        pair="Caselli 2014",
    ),
    Profile(
        name="Caselli 2014 - grey-zone HCM",
        citation="Caselli S, et al. Am J Cardiol 2014;114:1383-1389",
        expected="HCM",
        values=dict(wall_thickness=14.0, lv_cavity=45.0, e_prime_septal=9.3, e_e_prime=9.2, EF=64),
        note="Same 13-15 mm wall as the previous case (wall = midpoint), small cavity.",
        pair="Caselli 2014",
    ),
    Profile(
        name="Butz 2011 - HCM",
        citation="Butz T, et al. Int J Cardiovasc Imaging 2011;27:91-100",
        expected="HCM",
        values=dict(gls=-8.1),
        note="Mean GLS for 15 patients with non-obstructive HCM; nothing else used.",
    ),
    Profile(
        name="Pelliccia 1991 - athletes with a 13-16 mm wall",
        citation="Pelliccia A, et al. N Engl J Med 1991;324:295-301",
        expected="Athlete",
        values=dict(wall_thickness=14.5, lv_cavity=59.0),
        note="16 of 947 elite athletes; wall 13-16 mm, cavity 55-63 mm (range midpoints).",
    ),
]


def check_profiles(model, profiles=PROFILES) -> list[dict]:
    """Run each profile through a fitted :class:`~hcm_athlete.model.HCMModel`."""
    rows = []
    for p in profiles:
        prob, filled = model.predict_one(p.values)
        predicted = "HCM" if prob >= 0.5 else "Athlete"
        rows.append({
            "profile": p.name,
            "expected": p.expected,
            "predicted": predicted,
            "p_hcm": round(prob, 3),
            "correct": predicted == p.expected,
            "values_from_paper": len(p.values),
            "values_filled": len(filled),
            "pair": p.pair,
        })
    return rows


def check_pairs(rows: list[dict]) -> list[dict]:
    """For each same-study pair, is the HCM profile scored higher than the athlete one?"""
    out = []
    for pair in dict.fromkeys(r["pair"] for r in rows if r["pair"]):
        members = {r["expected"]: r for r in rows if r["pair"] == pair}
        if {"HCM", "Athlete"} <= members.keys():
            hcm, ath = members["HCM"]["p_hcm"], members["Athlete"]["p_hcm"]
            out.append({"pair": pair, "p_hcm_hcm": hcm, "p_hcm_athlete": ath, "ranked_right": hcm > ath})
    return out
