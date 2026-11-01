"""The two figures saved by a run."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUP_COLORS = {"cavity": "#c98a00", "troponin": "#c0392b", "bp": "#d35400", "other": "#2e6da4"}


def _group(feature: str) -> str:
    if "cavity" in feature:
        return "cavity"
    if "troponin" in feature or feature == "hours_post_exercise":
        return "troponin"
    if feature in {"systolic_bp", "diastolic_bp", "resting_hr", "pulse_pressure", "map",
                   "bp_response_exercise", "hypertensive_history"}:
        return "bp"
    return "other"


def plot_roc(curves, table, baseline, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = dict(zip(table["model"], table["auc"]))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {aucs[name]:.2f})")
    ax.scatter([1 - baseline["specificity"]], [baseline["sensitivity"]], marker="X", s=90,
               color="black", zorder=5, label=baseline["model"])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
           title="HCM vs athlete's heart: held-out synthetic test set")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_importance(imp, path, top=20):
    top_imp = imp.head(top).iloc[::-1]
    colors = [GROUP_COLORS[_group(f)] for f in top_imp["feature"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top_imp["feature"], top_imp["importance"], xerr=top_imp["sd"], color=colors)
    ax.set(xlabel="Drop in test AUC when the feature is shuffled",
           title=f"Top {top} features (permutation importance)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    ax.legend(handles, ["LV cavity", "Troponin", "BP / HR", "Other"], loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
