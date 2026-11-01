"""Train, compare and save models."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FEATURE_COLUMNS, complete_case

#: Wall-thickness cutoff for the baseline rule (mm).
BASELINE_WALL_MM = 13.0


def candidate_models(seed: int = 42) -> dict:
    """Each model is wrapped with a scaler, so scaling is fit on training folds only."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                                random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                        random_state=seed),
        # Platt-scaled probabilities (same idea as the old SVC(probability=True)).
        "SVM": CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed), ensemble=False),
    }


def _rates(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
    }


def baseline_rule(X: pd.DataFrame, y) -> dict:
    """Wall thickness > 13 mm means HCM."""
    pred = (X["wall_thickness"] > BASELINE_WALL_MM).astype(int)
    return {"model": f"Baseline (wall > {BASELINE_WALL_MM:g} mm)", "auc": None,
            **_rates(y, pred)}


def compare_models(X_train, y_train, X_test, y_test, seed: int = 42, cv_folds: int = 5):
    """Fit every candidate, score on the test set, and cross-validate on train.

    Returns (rows, fitted, curves): one metrics row per model plus the
    baseline, the fitted pipelines, and ROC curve points for plotting.
    """
    rows, fitted, curves = [], {}, {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for name, est in candidate_models(seed).items():
        pipe = make_pipeline(StandardScaler(), est)
        cv_auc = cross_val_score(clone(pipe), X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        rows.append({
            "model": name,
            "auc": roc_auc_score(y_test, prob),
            **_rates(y_test, (prob >= 0.5).astype(int)),
            "cv_auc_mean": cv_auc.mean(),
            "cv_auc_sd": cv_auc.std(),
        })
        fitted[name] = pipe
        curves[name] = (fpr, tpr)
    rows.append(baseline_rule(X_test, y_test))
    return rows, fitted, curves


def importance(pipe: Pipeline, X_test, y_test, seed: int = 42) -> pd.DataFrame:
    """Permutation importance on the test set: the drop in AUC when one
    feature is shuffled. Works for any model, unlike tree-only importances."""
    r = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                               n_repeats=10, random_state=seed, n_jobs=-1)
    return (pd.DataFrame({"feature": X_test.columns, "importance": r.importances_mean,
                          "sd": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


@dataclass
class HCMModel:
    """A fitted pipeline plus what is needed to score a partial case."""

    pipeline: Pipeline
    name: str
    medians: pd.Series
    info: dict = field(default_factory=dict)
    feature_columns: list = field(default_factory=lambda: list(FEATURE_COLUMNS))
    #: Training cases (wall, cavity, GLS, label) kept for plotting in the app.
    reference: pd.DataFrame | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.feature_columns])[:, 1]

    def predict_one(self, values: dict) -> tuple[float, list[str]]:
        """P(HCM) for one case given any subset of features.

        Returns the probability and the features filled with training medians.
        """
        row, filled = complete_case(values, self.medians)
        return float(self.predict_proba(row)[0]), filled

    def save(self, path: str | os.PathLike) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | os.PathLike) -> HCMModel:
        obj = joblib.load(path)
        if not isinstance(obj, HCMModel):
            raise TypeError(f"{path} does not contain an HCMModel")
        return obj
