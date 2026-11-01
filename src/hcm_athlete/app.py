"""Web app: score one case and see where it sits against the training data.

Run with ``hcm-athlete app`` (or ``streamlit run src/hcm_athlete/app.py``).
Needs ``pip install -e ".[app]"``.
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from hcm_athlete.labeling import CAVITY_CUTOFF_MM
from hcm_athlete.literature import PROFILES, check_pairs, check_profiles
from hcm_athlete.model import BASELINE_WALL_MM, HCMModel
from hcm_athlete.report import DISCLAIMER, band

MODEL_PATH = Path(os.environ.get("HCM_MODEL", "results/model.joblib"))

# (key, label, unit, min, max, step)
FIELDS = {
    "Echo": [
        ("wall_thickness", "Max wall thickness", "mm", 5.0, 35.0, 0.5),
        ("lv_cavity", "LV cavity (end-diastolic)", "mm", 30.0, 75.0, 0.5),
        ("gls", "Global longitudinal strain", "%", -30.0, 0.0, 0.5),
        ("e_e_prime", "E/e'", "", 2.0, 30.0, 0.1),
        ("EF", "Ejection fraction", "%", 10.0, 90.0, 1.0),
        ("EDV", "End-diastolic volume", "mL", 20.0, 300.0, 1.0),
        ("ESV", "End-systolic volume", "mL", 5.0, 250.0, 1.0),
    ],
    "More strain": [
        ("gcs", "Global circumferential strain", "%", -35.0, 0.0, 0.5),
        ("grs", "Global radial strain", "%", 5.0, 70.0, 0.5),
        ("e_prime_septal", "Septal e'", "cm/s", 2.0, 20.0, 0.1),
    ],
    "Clinical": [
        ("resting_hr", "Resting heart rate", "bpm", 30.0, 130.0, 1.0),
        ("systolic_bp", "Systolic BP", "mmHg", 80.0, 200.0, 1.0),
        ("diastolic_bp", "Diastolic BP", "mmHg", 40.0, 120.0, 1.0),
    ],
    "Troponin": [
        ("troponin_value", "hs-troponin", "ng/L", 0.5, 300.0, 0.5),
        ("hours_post_exercise", "Hours since exercise", "h", 0.0, 72.0, 1.0),
    ],
}
ALL_KEYS = [f[0] for group in FIELDS.values() for f in group]
PRESETS = {p.name: p for p in PROFILES}
DEFAULT_PRESET = "Caselli 2014 - grey-zone athletes"

COLORS = {"Athlete": "#2e6da4", "HCM": "#c0392b"}
BAND_COLORS = {"Low": "#1e8449", "Intermediate": "#b9770e", "Elevated": "#c0392b"}

st.set_page_config(page_title="HCM vs Athlete's Heart", page_icon="🫀", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
h1 {padding-top: 0;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- model
@st.cache_resource(show_spinner=False)
def load_model(path: str, mtime: float) -> HCMModel:
    return HCMModel.load(path)


def train_demo_model() -> None:
    from hcm_athlete.pipeline import RunConfig, run

    run(RunConfig(demo=True, synth="bootstrap", out_dir=str(MODEL_PATH.parent)), log=lambda *a: None)


if not MODEL_PATH.exists():
    st.title("HCM vs Athlete's Heart")
    st.warning(f"No trained model found at `{MODEL_PATH}`.")
    st.markdown(
        "Train one first:\n\n"
        "```bash\nhcm-athlete run --echonet path/to/FileList.csv\n```\n\n"
        "Or train a quick demo model on fake data to try the app (about 30 seconds)."
    )
    if st.button("Train demo model", type="primary"):
        with st.spinner("Training on demo data..."):
            train_demo_model()
        st.rerun()
    st.stop()

model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
info = model.info


# --------------------------------------------------------------------------- sidebar
def apply_preset() -> None:
    preset = PRESETS.get(st.session_state["preset"])
    for key in ALL_KEYS:
        st.session_state[key] = float(preset.values[key]) if preset and key in preset.values else None


if "preset" not in st.session_state:
    st.session_state["preset"] = DEFAULT_PRESET
    apply_preset()

with st.sidebar:
    st.header("Case")
    st.selectbox("Start from a literature profile", ["Custom", *PRESETS], key="preset",
                 on_change=apply_preset)
    st.caption("Leave a field blank if it was not measured. Blank inputs are set to the "
               "training median, which favors neither diagnosis.")
    for group, fields in FIELDS.items():
        with st.expander(group, expanded=group == "Echo"):
            for key, label, unit, lo, hi, step in fields:
                st.number_input(f"{label} ({unit})" if unit else label, min_value=lo, max_value=hi,
                                step=step, key=key, value=None, format="%g")

values = {k: st.session_state[k] for k in ALL_KEYS if st.session_state.get(k) is not None}


# --------------------------------------------------------------------------- header
st.title("HCM vs Athlete's Heart")
st.caption(DISCLAIMER)
if info.get("data_source") == "demo":
    st.warning("This model was trained on **fake demo data**. Train on EchoNet-Dynamic for real "
               "results: `hcm-athlete run --echonet path/to/FileList.csv`")

tab_case, tab_profiles, tab_model, tab_about = st.tabs(
    ["Score a case", "Literature profiles", "Model performance", "About"])


# --------------------------------------------------------------------------- score
def probability_bar(p: float, color: str) -> str:
    return f"""
<div style="position:relative;height:14px;border-radius:7px;margin:6px 0 4px;
     background:linear-gradient(90deg,#1e8449 0%,#b9770e 35%,#c0392b 100%);opacity:.9">
  <div style="position:absolute;left:calc({p * 100:.1f}% - 9px);top:-5px;width:18px;height:24px;
       border-radius:5px;background:white;border:3px solid {color};box-sizing:border-box"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:12px;opacity:.7">
  <span>0% · athlete's heart</span><span>20%</span><span>50%</span><span>100% · HCM</span>
</div>"""


def scatter(reference: pd.DataFrame | None, case: dict) -> alt.Chart | None:
    if reference is None or "wall_thickness" not in case or "lv_cavity" not in case:
        return None
    ref = reference.sample(min(len(reference), 800), random_state=0).assign(
        group=lambda d: d["label"].map({1: "HCM", 0: "Athlete"}))
    color = alt.Color("group:N", title="Training label",
                      scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())))
    x = alt.X("wall_thickness:Q", title="Max wall thickness (mm)", scale=alt.Scale(domain=[6, 31]))
    y = alt.Y("lv_cavity:Q", title="LV cavity (mm)", scale=alt.Scale(domain=[36, 72]))
    points = alt.Chart(ref).mark_circle(size=28, opacity=0.35).encode(
        x=x, y=y, color=color, tooltip=["group", "wall_thickness", "lv_cavity"])
    rule_x = alt.Chart(pd.DataFrame({"v": [BASELINE_WALL_MM]})).mark_rule(
        strokeDash=[5, 4], color="#555").encode(x="v:Q")
    rule_y = alt.Chart(pd.DataFrame({"v": [CAVITY_CUTOFF_MM]})).mark_rule(
        strokeDash=[2, 3], color="#555").encode(y="v:Q")
    you = pd.DataFrame([{"wall_thickness": case["wall_thickness"], "lv_cavity": case["lv_cavity"],
                         "name": "This case"}])
    marker = alt.Chart(you).mark_point(shape="diamond", size=320, filled=True, color="black",
                                       stroke="white", strokeWidth=2).encode(x=x, y=y, tooltip=["name"])
    label = alt.Chart(you).mark_text(dx=14, align="left", fontWeight="bold").encode(
        x=x, y=y, text="name")
    return (points + rule_x + rule_y + marker + label).properties(height=380)


with tab_case:
    if not values:
        st.info("Enter at least one measurement in the sidebar.")
    else:
        p, filled = model.predict_one(values)
        band_text = band(p)
        color = BAND_COLORS[band_text.split(" ")[0]]

        left, right = st.columns([2, 3], gap="large")
        with left:
            st.markdown(f"<div style='font-size:14px;opacity:.75'>Probability of HCM</div>"
                        f"<div style='font-size:56px;font-weight:700;line-height:1.1;color:{color}'>"
                        f"{p:.0%}</div>"
                        f"<div style='font-size:18px;font-weight:600;color:{color}'>{band_text}</div>",
                        unsafe_allow_html=True)
            st.markdown(probability_bar(p, color), unsafe_allow_html=True)
            st.write("")

            wall = values.get("wall_thickness")
            if wall is not None:
                rule = "HCM" if wall > BASELINE_WALL_MM else "athlete's heart"
                st.markdown(f"**Wall-thickness rule** (> {BASELINE_WALL_MM:g} mm) alone would say: "
                            f"**{rule}**")
            cavity = values.get("lv_cavity")
            if cavity is not None:
                leans = "athlete's heart" if cavity >= CAVITY_CUTOFF_MM else "HCM"
                side = "above" if cavity >= CAVITY_CUTOFF_MM else "below"
                st.markdown(f"**LV cavity** {cavity:g} mm is {side} ~{CAVITY_CUTOFF_MM:g} mm, "
                            f"which leans toward {leans}.")
            hr = values.get("resting_hr")
            if hr is not None and hr < 55:
                st.markdown(f"**Resting HR** {hr:g} bpm fits athletic training.")
            st.caption(f"{len(values)} inputs given · {len(filled)} set to the training median · "
                       f"model: {model.name}")

        with right:
            chart = scatter(model.reference, values)
            if chart is not None:
                st.markdown("**Where this case sits among the training cases**")
                st.altair_chart(chart, use_container_width=True)
                st.caption(f"Dashed line: {BASELINE_WALL_MM:g} mm wall rule. "
                           f"Dotted line: ~{CAVITY_CUTOFF_MM:g} mm cavity. "
                           "Labels come from the synthetic labeling formula.")
            else:
                st.info("Enter wall thickness and LV cavity to see the case on the chart.")


# --------------------------------------------------------------------------- profiles
with tab_profiles:
    st.markdown("Six profiles built from published cohort results. Each uses only values "
                "reported in its paper; every other input is set to the training median. They "
                "are a sanity check, not a test set: most are cohort averages, not patients.")
    rows = pd.DataFrame(check_profiles(model))
    rows["result"] = rows["correct"].map({True: "✓", False: "✗"})
    st.dataframe(
        rows[["result", "profile", "expected", "predicted", "p_hcm", "values_from_paper",
              "values_filled"]],
        hide_index=True, use_container_width=True,
        column_config={
            "result": st.column_config.TextColumn("", width="small"),
            "profile": "Profile", "expected": "Expected", "predicted": "Predicted",
            "p_hcm": st.column_config.ProgressColumn("P(HCM)", min_value=0, max_value=1, format="%.3f"),
            "values_from_paper": "Values from paper", "values_filled": "Set to median",
        })
    pairs = check_pairs(rows.to_dict(orient="records"))
    st.caption(f"{int(rows['correct'].sum())} of {len(rows)} classified as expected at the 50% "
               f"cutoff. {sum(p['ranked_right'] for p in pairs)} of {len(pairs)} same-study pairs "
               "(Sheikh 2015, Caselli 2014) have the HCM profile scored higher than the athlete one.")


# --------------------------------------------------------------------------- model
with tab_model:
    out = MODEL_PATH.parent
    st.markdown(f"Best model: **{model.name}** · data: `{info.get('data_source', '?')}` · "
                f"sampler: `{info.get('synth', '?')}` · {info.get('n_train', '?')} train / "
                f"{info.get('n_test', '?')} test cases")
    if (out / "model_comparison.csv").exists():
        table = pd.read_csv(out / "model_comparison.csv").round(3).astype(object)
        table = table.where(table.notna(), "–")
        table.columns = ["Model", "Test AUC", "Accuracy", "Sensitivity", "Specificity",
                         "CV AUC (mean)", "CV AUC (sd)"]
        st.dataframe(table, hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    for col, name in [(c1, "roc_curves.png"), (c2, "feature_importance.png")]:
        if (out / name).exists():
            col.image(str(out / name), use_container_width=True)
    st.caption("Scores are on held-out synthetic cases. They show the models recovered the pattern "
               "in the synthetic data, not performance on real patients.")


# --------------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**What this is.** A research and teaching project that asks whether cavity size, strain,
resting heart rate, blood pressure and troponin timing help separate hypertrophic
cardiomyopathy from athlete's heart in the 13-15 mm wall-thickness grey zone.

**How the model was built.** Real EF/EDV/ESV values from EchoNet-Dynamic are combined with
synthetic strain, cavity, BP and troponin values drawn from published cohort summaries. A CTGAN
samples new cases, which are labeled by a weighted scoring formula, and four models are compared
against the wall-thickness rule.

**Limitations.** The training data is synthetic and the labels come from a formula, so the model
has not been tested on real patients. It is not a medical device and must not be used for
clinical decisions.
""")
