import json

import numpy as np
import pandas as pd
import pytest

from hcm_athlete.cli import main
from hcm_athlete.data import demo_echonet
from hcm_athlete.features import COMPUTED, FEATURE_COLUMNS, SYNTH_COLUMNS, complete_case
from hcm_athlete.generate import classify_troponin_pattern, make_seed_cases
from hcm_athlete.labeling import WEIGHTS, assign_labels, component_scores
from hcm_athlete.model import HCMModel
from hcm_athlete.pipeline import RunConfig, run
from hcm_athlete.synth import synthesize


@pytest.fixture(scope="module")
def seed_cases():
    rng = np.random.default_rng(0)
    return make_seed_cases(demo_echonet(rng=rng), n_per_class=100, rng=rng)


def test_feature_lists():
    assert len(FEATURE_COLUMNS) == 45
    assert len(set(FEATURE_COLUMNS)) == 45
    assert len(SYNTH_COLUMNS) == 40
    assert not set(COMPUTED) & set(SYNTH_COLUMNS)


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_seed_cases_shape_and_ranges(seed_cases):
    assert len(seed_cases) == 300
    assert set(seed_cases["diagnosis"]) == {"HCM", "Athlete", "Normal"}
    hcm = seed_cases[seed_cases["diagnosis"] == "HCM"]
    ath = seed_cases[seed_cases["diagnosis"] == "Athlete"]
    assert hcm["wall_thickness"].between(13, 30).all()
    assert ath["lv_cavity"].between(50, 70).all()
    assert hcm["lv_cavity"].mean() < ath["lv_cavity"].mean()
    assert ath["resting_hr"].mean() < hcm["resting_hr"].mean()


def test_seed_cases_reproducible():
    df = demo_echonet(rng=np.random.default_rng(1))
    a = make_seed_cases(df, 20, np.random.default_rng(5))
    b = make_seed_cases(df, 20, np.random.default_rng(5))
    pd.testing.assert_frame_equal(a, b)


def test_bootstrap_synth_keeps_derived_features_consistent(seed_cases):
    out = synthesize(seed_cases, n=200, method="bootstrap", rng=np.random.default_rng(0))
    assert list(out.columns) == SYNTH_COLUMNS
    assert np.allclose(out["pulse_pressure"], out["systolic_bp"] - out["diastolic_bp"])
    assert set(out["troponin_pattern"].unique()) <= {0, 1, 2}
    assert set(out["troponin_above_url"].unique()) <= {0, 1}
    assert out["wall_thickness"].between(7, 30).all()


def test_cavity_component_direction():
    df = pd.DataFrame({
        "wall_thickness": [14, 14], "gls": [-16, -16], "lv_cavity": [45, 62],
        "regional_variation": [3, 3], "e_e_prime": [9, 9], "EDV": [95, 95],
        "resting_hr": [60, 60], "troponin_pattern": [1, 1],
    })
    parts = component_scores(df)
    assert parts.loc[0, "cavity"] > 0.9 > 0.1 > parts.loc[1, "cavity"]


def test_labels_are_binary(seed_cases):
    rng = np.random.default_rng(0)
    out = assign_labels(synthesize(seed_cases, 100, "bootstrap", rng), rng)
    assert set(out["label"].unique()) <= {0, 1}
    assert out["hcm_score"].between(0, 1).all()


@pytest.mark.parametrize("value,hours,baseline,expected", [
    (20, 3, 4, 0),    # early rise, fits exercise
    (5, 24, 4, 0),    # back near baseline
    (30, 48, 10, 2),  # still high after 24 h
    (150, 3, 80, 2),  # very high
    (10, 12, 6, 1),   # in between
])
def test_troponin_pattern(value, hours, baseline, expected):
    assert classify_troponin_pattern(value, hours, baseline) == expected


def test_check_pairs():
    from hcm_athlete.literature import check_pairs
    rows = [{"pair": "A", "expected": "HCM", "p_hcm": 0.6},
            {"pair": "A", "expected": "Athlete", "p_hcm": 0.4},
            {"pair": "", "expected": "HCM", "p_hcm": 0.9}]
    assert check_pairs(rows) == [{"pair": "A", "p_hcm_hcm": 0.6, "p_hcm_athlete": 0.4,
                                  "ranked_right": True}]


def test_complete_case_fills_and_rejects():
    medians = pd.Series(1.0, index=FEATURE_COLUMNS)
    row, filled = complete_case({"wall_thickness": 14, "lv_cavity": 56, "gls": -18}, medians)
    assert list(row.columns) == FEATURE_COLUMNS
    assert row["cavity_wall_ratio"].iloc[0] == pytest.approx(4.0)
    assert row["basal_strain"].iloc[0] == pytest.approx(-18 * 0.725)
    assert "wall_thickness" not in filled and "EF" in filled
    with pytest.raises(ValueError):
        complete_case({"not_a_feature": 1}, medians)


def test_end_to_end(tmp_path):
    cfg = RunConfig(demo=True, out_dir=str(tmp_path), n_seed_per_class=150,
                    n_synthetic=400, synth="bootstrap", plots=True)
    summary = run(cfg, log=lambda *a: None)

    assert summary["data_source"] == "demo"
    assert summary["n_train"] == 320 and summary["n_test"] == 80
    aucs = [m["auc"] for m in summary["models"] if m["auc"] is not None]
    assert len(aucs) == 4 and all(0.5 < a <= 1 for a in aucs)
    for name in ["model.joblib", "model_comparison.csv", "feature_importance.csv",
                 "literature_profiles.csv", "summary.json", "roc_curves.png",
                 "feature_importance.png"]:
        assert (tmp_path / name).exists(), name

    model = HCMModel.load(tmp_path / "model.joblib")
    p_ath, _ = model.predict_one({"wall_thickness": 14, "lv_cavity": 62, "gls": -20, "resting_hr": 45})
    p_hcm, _ = model.predict_one({"wall_thickness": 14, "lv_cavity": 44, "gls": -11, "resting_hr": 72})
    assert p_hcm > p_ath


def test_cli_predict(tmp_path, capsys):
    run(RunConfig(demo=True, out_dir=str(tmp_path), n_seed_per_class=100, n_synthetic=300,
                  synth="bootstrap", plots=False), log=lambda *a: None)
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"_comment": "x", "wall_thickness": 14, "lv_cavity": 60}))
    assert main(["predict", str(case), "--model", str(tmp_path / "model.joblib"), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert 0 <= out["p_hcm"] <= 1
    assert main(["predict", str(case), "--model", str(tmp_path / "model.joblib")]) == 0
    assert "Not a medical device" in capsys.readouterr().out
