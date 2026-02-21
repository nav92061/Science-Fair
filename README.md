# HCM vs Athlete's Heart Classification

Classifies hypertrophic cardiomyopathy against athlete's heart using real echocardiogram data from the EchoNet-Dynamic dataset combined with synthetic strain features. The main challenge this addresses is the "grey zone" — cases where wall thickness alone is not enough to differentiate the two conditions.

## What it does

Loads real EF/EDV/ESV measurements from EchoNet-Dynamic, generates physiologically correlated strain features (GLS, regional variation, diastolic indices, twist mechanics etc.), then trains CTGAN on those feature distributions to produce realistic synthetic cases. Labels are assigned probabilistically based on a weighted clinical scoring function rather than deterministically, which creates genuine diagnostic ambiguity in the dataset.

Four traditional ML models plus a small feedforward neural network are then trained and compared. Random forest came out on top in testing (AUC 0.848) with regional variation, wall thickness, and GLS as the top discriminators.

## Requirements

```
pip install sdv tensorflow scikit-learn pandas numpy matplotlib seaborn joblib
```

SDV v1.0+ is preferred but the notebook handles v0.18 as a fallback automatically.

## Setup

Requires access to the EchoNet-Dynamic dataset. You can request it at echonet.github.io/dynamic. Once you have it, upload the folder to google drive and the notebook will mount drive and look for it at:

```
/content/drive/MyDrive/EchoNet-Dynamic/FileList.csv
```

If your path is different just update the `csv_path` variable in cell 2.

## Pipeline

1. Load FileList.csv (real EF, EDV, ESV for ~10k echo studies)
2. Generate strain features correlated with real measurements (500 HCM, 500 athlete, 500 normal)
3. Train CTGAN on feature distributions without labels
4. Sample 1000 synthetic cases from CTGAN, apply probabilistic labeling
5. Engineer 5 additional composite features
6. Train and evaluate logistic regression, random forest, gradient boosting, SVM, and neural net
7. Output ROC curves, confusion matrices, feature importances, and clinical metrics

## Results (on test set)

| model | accuracy | auc |
|---|---|---|
| random forest | 0.815 | 0.849 |
| gradient boosting | 0.805 | 0.835 |
| logistic regression | 0.795 | 0.834 |
| neural network | 0.795 | 0.827 |
| svm | 0.790 | 0.804 |

Sensitivity 65%, specificity 91% for the best model. The sensitivity is lower than youd want in a real clinical tool but this is a proof of concept on synthetic data — the point is showing the methodology works.

## Output files

Running the full notebook saves: 8 plots (png), trained models (pkl and h5), feature scaler, feature importance csv, and model comparison csv.

## Notes

CTGAN training takes a few minutes even on CPU. The grey zone cases (~11% of the synthetic set) are the interesting ones to look at — those are where the models disagree and where the probabilistic labeling creates realistic uncertainty. Results will vary slightly between runs because of the stochastic labeling step.
