from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from hcm_athlete.pipeline import RunConfig, run  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "hcm_athlete" / "app.py")


def test_app_scores_default_profile(tmp_path, monkeypatch):
    run(RunConfig(demo=True, out_dir=str(tmp_path), n_seed_per_class=100, n_synthetic=300,
                  synth="bootstrap"), log=lambda *a: None)
    monkeypatch.setenv("HCM_MODEL", str(tmp_path / "model.joblib"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Probability of HCM" in text
    assert "Wall-thickness rule" in text


def test_app_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HCM_MODEL", str(tmp_path / "missing.joblib"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("No trained model" in w.value for w in at.warning)
