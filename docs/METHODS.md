# Methods

## 1. Pipeline

1. Load real EchoNet-Dynamic measurements (EF, EDV, ESV) from `FileList.csv`.
2. For each sampled echo, generate synthetic strain, cavity, BP and troponin
   features from literature-based distributions: 500 HCM, 500 athlete and 500
   normal = 1,500 seed cases (`generate.py`).
3. Train CTGAN on the 1,500 seed cases (40 features, no labels) and sample 1,000
   new cases (`synth.py`).
4. Add 5% measurement noise, clip to physiological limits, and recompute the
   features defined from other features (pulse pressure, MAP, troponin fold
   change, above-URL flag and pattern) so they stay consistent.
5. Label the 1,000 cases with a weighted HCM score (Section 3).
6. Split 80/20, stratified, before any training: 800 train, 200 test.
7. Train and compare models (Section 4).
8. Score the best model on six literature-based profiles (Section 5).

CTGAN produces some values outside the physiological limits. Step 4 clips them,
so a noticeable number of cases sit exactly at a limit (for example a 38 mm
cavity or a 30 mm wall). You can see this as a line of points along the edges
of the app's scatter plot. Tightening this, for example by resampling instead of
clipping, is an open improvement.

`--synth bootstrap` replaces step 3 with resampling plus Gaussian jitter. It
exists for tests and quick experiments. Do not report its results as CTGAN results.

## 2. The 45 features

| Group | Count | Features |
|---|---|---|
| Strain and diastolic | 20 | gls, gcs, grs, basal/mid/apical_strain, strain_rate_s/e/a, regional_variation, twist, twist_rate, untwist_rate, torsion, e_a_ratio, e_prime_septal, e_prime_lateral, e_e_prime, wall_thickness, lv_mass_index |
| Standard echo | 4 | EDV, ESV, EF (real EchoNet), lv_cavity (synthetic) |
| Geometry | 3 | relative_wall_thickness, asym_ratio, cavity_wall_ratio |
| Engineered | 5 | strain_gradient, diastolic_function_index, systolic_efficiency, geometric_index, strain_ratio |
| Blood pressure | 7 | systolic_bp, diastolic_bp, resting_hr, pulse_pressure, map, bp_response_exercise, hypertensive_history |
| Troponin | 6 | troponin_value, hours_post_exercise, troponin_fold_change, troponin_above_url, normalization_trend_slow, troponin_pattern |

Only EF, EDV and ESV are real measurements.

Engineered features:

```
strain_gradient          = apical_strain - basal_strain
diastolic_function_index = strain_rate_e * e_a_ratio
systolic_efficiency      = gls * EF / 100
geometric_index          = wall_thickness / sqrt(EDV)
strain_ratio             = gls / gcs
cavity_wall_ratio        = lv_cavity / wall_thickness
```

### Key generator settings

"Source" means the value matches a published figure (see References).
"Assumption" means it is a modeling choice with no single published source.

| Setting | HCM | Athlete | Source |
|---|---|---|---|
| Wall thickness (mm) | mild 15.8 ± 3.4; typical 19 ± 3; clipped 13-30 | typical 11 ± 2; grey zone 13-15; clipped 8-16 | Mild HCM: Sheikh 2015. Athlete upper limit 16 mm: Pelliccia 1991. Rest: assumption |
| LV cavity (mm) | mild 47.8 ± 6.0; typical 46 ± 5 | 58 ± 5 | Mild HCM: Sheikh 2015. Rest: assumption (Caselli 2014 reports 60 ± 3 vs 45 ± 5) |
| GLS (%) | mild −15.8 ± 3; typical −10.8 ± 3.8 | −19 ± 2 | Assumption (Butz 2011 reports −8.1 ± 3.8 vs −15.2 ± 3.6) |
| GCS, GRS (%) | −18.4 ± 4.8; 30.8 ± 11.6 | −23 ± 3; 40 ± 10 | Assumption |
| E/e' | mild 7.9 ± 2.4; typical 12 ± 5 | 6.0 ± 1.5 | Mild HCM: Sheikh 2015. Rest: assumption (Caselli 2014: 9.2 vs 6.6) |
| Septal e' (cm/s) | mild 9.0 ± 3.0; typical 7 ± 2 | 12.5 ± 1.9 | Mild HCM: Sheikh 2015. Athlete: Caselli 2014. Typical HCM: assumption |
| Systolic BP (mmHg) | 125 ± 15 | 115 ± 10 | Assumption |
| Resting HR (bpm) | 70 ± 12 | 50 ± 8 | Assumption |
| BP rise with exercise | 1.2-1.5× | 1.5-1.9× | Assumption |
| Resting hs-troponin (ng/L) | 7-24 | 3-6 | HCM: Jenab 2014 (median 12.4, IQR 6.9-24.3). Athlete: Li 2021 (mean 4.4) |
| Post-exercise troponin rise | slower clearance | 47% show a rise that clears in 24-72 h | Shave 2007 (47% detectable), Shave 2010 and Aengevaeren 2021 (timing). Size of rise: assumption |

About 25% of HCM and athlete seed cases are drawn as grey-zone cases.

**Why cavity size matters.** Training enlarges the athlete's chamber along with
the wall (volume load). HCM thickens the wall without enlarging the chamber. At
the same 13-15 mm wall, a large cavity points to athlete's heart and a small
one to HCM. In Caselli 2014 (28 athletes and 25 HCM patients, all with a 13-15
mm wall), an LV end-diastolic diameter below 54 mm separated the two groups
with 100% sensitivity and specificity. That is a small study, so the labeling
formula uses 54 mm as the center of a smooth curve rather than a hard cutoff.

**Troponin timing.** After endurance exercise, troponin becomes detectable in
about half of athletes (47% in the Shave 2007 meta-analysis of 26 studies). The
rise peaks within a few hours and returns to baseline within 24-72 h (Shave
2010; Aengevaeren 2021). The rule in
`classify_troponin_pattern` reads a value together with time since exercise:
0 = fits exercise, 1 = indeterminate, 2 = persistent or very high.

## 3. Labels

The synthesizer does not keep labels. Each sampled case gets an HCM score:

```
HCM score = 0.20 wall + 0.18 GLS + 0.15 cavity + 0.12 regional variation
          + 0.12 E/e' + 0.08 EDV + 0.08 resting HR + 0.07 troponin pattern
          + noise (sd 0.1), clipped to 0-1
```

| Component | Sigmoid | Leans HCM when |
|---|---|---|
| wall | 2.0 × (wall − 13.5) | wall > 13.5 mm |
| GLS | 0.5 × (GLS + 17) | GLS less negative than −17% |
| cavity | −0.3 × (cavity − 54) | cavity < 54 mm |
| regional variation | 1.5 × (rv − 3.0) | rv > 3% |
| E/e' | 0.8 × (E/e' − 10) | E/e' > 10 |
| EDV | −0.03 × (EDV − 95) | EDV < 95 mL |
| resting HR | 0.1 × (HR − 72.5) | HR > 72.5 bpm |
| troponin pattern | 0 / 0.25 / 0.5 for pattern 0 / 1 / 2 | pattern 2 |

The label is then drawn at random with probability equal to the score
(`label ~ Bernoulli(score)`). This keeps grey-zone cases in the data. It also
means no model can get close to perfect.

The weights live in `labeling.WEIGHTS`.

## 4. Model comparison

- Models: Logistic Regression, Random Forest (200 trees, depth 10), Gradient
  Boosting (200 estimators, depth 5), SVM (RBF, Platt-scaled probabilities).
  Each is a scikit-learn pipeline with a `StandardScaler`, so scaling is fit on
  training data only, including inside cross-validation.
- All models are scored on the same 200 test cases: AUC-ROC, accuracy,
  sensitivity, specificity (threshold 0.5).
- 5-fold stratified cross-validation AUC on the 800 training cases (mean ± sd).
- Baseline: wall thickness > 13 mm.
- Best model = highest test AUC. It is not always the same model.
- Feature importance is permutation importance of the best model on the test
  set: the drop in AUC when one feature is shuffled (10 repeats).

**What this shows:** the models learned the patterns in the synthetic data
better than the wall-thickness rule did.

**What it does not show:** performance on real patients. The labels come from
the formula in Section 3, so the models are partly learning that formula.

## 5. Literature profiles

Each profile uses **only values reported in its paper**. Every other input is
set to the training-set median, which favors neither diagnosis.

| # | Source | Expected | Values from the paper |
|---|---|---|---|
| 1 | Sheikh 2015, 106 athletes with HCM | HCM | wall 15.8, cavity 47.8, E/e' 7.9, septal e' 9.0 |
| 2 | Sheikh 2015, 55 athletes with physiological LVH | Athlete | wall 14.5*, cavity 54.9, E/e' 5.5 |
| 3 | Caselli 2014, 28 grey-zone athletes | Athlete | wall 14.0*, cavity 60, e' 12.5, E/e' 6.6, EF 63 |
| 4 | Caselli 2014, 25 grey-zone HCM patients | HCM | wall 14.0*, cavity 45, e' 9.3, E/e' 9.2, EF 64 |
| 5 | Butz 2011, 15 patients with HCM | HCM | GLS −8.1 |
| 6 | Pelliccia 1991, 16 athletes with a ≥13 mm wall | Athlete | wall 14.5*, cavity 59* |

\* Midpoint of a reported range: 13-16 mm (Sheikh subgroup, Pelliccia), 13-15 mm
(Caselli matching criterion), 55-63 mm (Pelliccia cavity). Caselli 2014 does not
say in its abstract which wall site e' was measured at; it is used as septal e'.

Pairs 1-2 and 3-4 come from the same paper, so they are the most useful
comparison: the model should score the HCM member of each pair higher.

How to describe these accurately:

- They are cohort averages, not individual patients.
- Because the papers report few of the model's 45 inputs (none reports strain
  except Butz, and none reports troponin), most inputs are set to the median.
  HCM profiles can therefore land near 50%, and the result can change between
  runs. Report both the count at the 50% cutoff and whether each pair is
  ranked the right way round.
- Six cases is a sanity check, not a performance estimate. Say "the model
  classified N of 6 literature-based profiles as expected," not "N/6 accuracy."

## 6. Changes from the original notebook

- Split into a tested Python package with a command line and a thin Colab notebook.
- All randomness goes through one seeded generator, so runs are reproducible.
- Derived features (pulse pressure, MAP, troponin fold change, above-URL flag,
  troponin pattern) are recomputed after sampling and noise. Before, CTGAN and
  the noise step could make them disagree with the values they come from.
- Whole-number and 0/1 features are no longer given continuous noise.
- Scaling is inside each model pipeline, so cross-validation does not leak.
- Feature importance uses the best model (permutation importance), not always
  the Random Forest.
- SVM probabilities use `CalibratedClassifierCV`, since `SVC(probability=True)`
  is deprecated in recent scikit-learn.
- The neural network was dropped to keep dependencies small. It did not beat
  the tree models in earlier runs.
- The single-case report no longer gives treatment or sports-participation
  advice.

### Corrections from the November 2026 reference check

The original values came from an AI-generated research summary. Checking them
against the papers found these problems, now fixed:

- **Profiles used values the papers do not report.** Strain (except Butz), BP,
  heart rate, troponin and LV volumes had been filled in by the summary. The
  profiles now contain only published values.
- **"Zhao 2024" was misdescribed.** The real paper (Zhao K et al., JACC
  Cardiovasc Imaging 2024; 758 patients) uses cardiac MRI feature tracking, not
  echo, and its −17.3% GCS and 25.8% GRS values are tertile boundaries, not
  risk thresholds. The strain means attributed to it could not be found. It is
  no longer used; its profile was replaced by Sheikh 2015's physiological-LVH group.
- **Sheikh 2015 reports no strain.** The "GLS −15.8%" was the wall-thickness
  value reused. Septal e' is 9 cm/s, not 10.
- **Pelliccia 1991 cavity range is 55-63 mm**, not 55-70 mm.
- **Caselli 2014 HCM E/e' is 9.2**, not 11; EF is 63% (athletes) and 64% (HCM).
- **Shave 2007** supports the 47% figure (as *detectable* troponin), not the
  timing; timing now cites Shave 2010 and Aengevaeren 2021. Resting troponin now
  follows Li 2021 (athletes) and Jenab 2014 (HCM). The claim that 30% of HCM
  patients exceed the upper reference limit had no source and was removed;
  published figures range from 24% to 54%.

## References

Every value was checked against the paper's abstract and published tables in
November 2026 (via literature search). The table lists what each source is used
for and what the check found.

| Source | Used for | Check result |
|---|---|---|
| Pelliccia A, Maron BJ, Spataro A, et al. The upper limit of physiologic cardiac hypertrophy in highly trained elite athletes. *N Engl J Med* 1991;324(5):295-301. doi:10.1056/NEJM199101313240504 | 16 mm upper limit; 16 of 947 athletes with a ≥13 mm wall, all with enlarged cavities (55-63 mm); profile 6 | Confirmed. Cavity range corrected from 55-70 to 55-63 mm |
| Butz T, van Buuren F, Mellwig KP, et al. Two-dimensional strain analysis of the global and regional myocardial function for the differentiation of pathologic and physiologic left ventricular hypertrophy. *Int J Cardiovasc Imaging* 2011;27(1):91-100. doi:10.1007/s10554-010-9665-5 | GLS −8.1 ± 3.8% (15 HCM) vs −15.2 ± 3.6% (20 athletes); profile 5 | Confirmed. Other values in the old profile were not from this paper and were removed |
| Caselli S, Maron MS, Urbano-Moral JA, et al. Differentiating left ventricular hypertrophy in athletes from that in patients with hypertrophic cardiomyopathy. *Am J Cardiol* 2014;114(9):1383-1389. doi:10.1016/j.amjcard.2014.07.070 | 13-15 mm grey zone; LVEDD 60 ± 3 vs 45 ± 5 mm; <54 mm cutoff (100%/100%); e' 12.5 ± 1.9 vs 9.3 ± 2.3 cm/s; E/e' 6.6 vs 9.2; EF 63 vs 64%; profiles 3-4 | Confirmed. HCM E/e' corrected from 11 to 9.2; EF corrected |
| Sheikh N, Papadakis M, Schnell F, et al. Clinical profile of athletes with hypertrophic cardiomyopathy. *Circ Cardiovasc Imaging* 2015;8(7):e003454. doi:10.1161/CIRCIMAGING.114.003454 | 106 athletes with HCM: wall 15.8 ± 3.4 mm, LVEDD 47.8 ± 6.0 mm, E/e' 7.9 ± 2.4, septal e' 0.09 ± 0.03 m/s; 55 athletes with physiological LVH: LVEDD 54.9 ± 4.7 mm, E/e' 5.5 ± 1.1; profiles 1-2 | Confirmed. GLS removed (not reported); septal e' corrected from 10 to 9 cm/s |
| Shave R, George KP, Atkinson G, et al. Exercise-induced cardiac troponin T release: a meta-analysis. *Med Sci Sports Exerc* 2007;39(12):2099-2106. doi:10.1249/mss.0b013e318153ff78 | cTnT detectable after exercise in 47% (95% CI 39-56%) of 1,120 athletes | Confirmed (as "detectable", not "elevated") |
| Shave R, Baggish A, George K, et al. Exercise-induced cardiac troponin elevation: evidence, mechanisms, and implications. *J Am Coll Cardiol* 2010;56(3):169-176. doi:10.1016/j.jacc.2010.03.037 | Early peak, return to baseline within 24-72 h | Added in this check |
| Aengevaeren VL, et al. Exercise-induced cardiac troponin elevations: from underlying mechanisms to clinical relevance. *Circulation* 2021. doi:10.1161/CIRCULATIONAHA.121.056208 | Return to baseline within 48-72 h | Added in this check |
| Li et al. Meta-analysis of the hs-cTnT time course after exercise. *Front Physiol* 2021;12:651851. doi:10.3389/fphys.2021.651851 | Athlete pre-exercise hs-cTnT mean 4.4 ng/L | Added in this check |
| Jenab Y, Pourjafari M, Darabi F, et al. Prevalence and determinants of elevated high-sensitivity cardiac troponin T in hypertrophic cardiomyopathy. *J Cardiol* 2014;63(2):140-144. doi:10.1016/j.jjcc.2013.07.008 | HCM hs-cTnT 12.4 ng/L (IQR 6.9-24.3) | Added in this check |
| Ouyang D, et al. Video-based AI for beat-to-beat assessment of cardiac function. *Nature* 2020;580(7802):252-256. doi:10.1038/s41586-020-2145-8 | EchoNet-Dynamic: 10,030 echo videos with EF, EDV, ESV | Confirmed. Research use agreement: personal, non-commercial research only; not redistributed here |
