# Wildfire Evacuation Threat Prediction -- WiDS Datathon 2026

**Hybrid Score: 0.939** (C-index + Weighted Brier Score)

---

## Overview

This repository contains my solution for the **Women in Data Science (WiDS) Datathon 2026**, a global data science competition that challenges participants to tackle real-world problems using machine learning.

This year's challenge focused on **wildfire evacuation threat prediction** -- a high-stakes problem with direct public safety implications. Given early-stage fire perimeter observations from the first five hours after detection, the goal was to predict the probability that a fire comes within 5 km of an evacuation zone centroid at four forecast horizons: **12h, 24h, 48h, and 72h**.

---

## The Problem

The dataset contains **316 verified wildfire events** with confirmed outcomes, representing fires that had both reliable early perimeter updates (within the first 5 hours) and sufficient follow-up observations to confirm whether and when they threatened evacuation zones.

This is a **right-censored survival analysis** problem:
- If a fire reached an evacuation zone within 72 hours, `event = 1` and `time_to_hit_hours` is the observed time
- If a fire did not reach a zone within 72 hours, `event = 0` and `time_to_hit_hours` is censored at the last observation

The task was not to predict a single time, but to output **four calibrated cumulative probabilities** per fire, with monotonicity enforced: `prob_12h <= prob_24h <= prob_48h <= prob_72h`

---

## Evaluation Metric

```
Hybrid Score = 0.3 x C-index + 0.7 x (1 - Weighted Brier Score)
```

- **C-index (30%)** -- measures how well the model ranks fires by urgency
- **Weighted Brier Score (70%)** -- measures calibration at each horizon
  - `0.3 x Brier@24h + 0.4 x Brier@48h + 0.3 x Brier@72h`
  - 48h is weighted highest because 24-48 hours is the strongest operational decision window for evacuation planning

---

## My Approach

### Survival Analysis Framing
Rather than treating this as a standard regression or classification task, I framed it as a **censoring-aware binary classification problem** at each horizon. For each horizon H:
- **Positive**: `event=1` AND `time_to_hit_hours <= H` (confirmed hit)
- **Negative**: `time_to_hit_hours > H` (survived past horizon)
- **Excluded**: `event=0` AND `time_to_hit_hours <= H` (censored before horizon -- truth unknown)

Excluding censored-before-horizon cases from each horizon's training set is what separates a proper survival model from a naive one.

### Feature Engineering
Built domain-driven features on top of the raw early-spread dynamics:
- **Danger score**: growth rate / distance to nearest evacuation zone
- **Naive ETA**: distance to zone / closing speed (direct survival-time signal)
- **Spread geometry**: alignment of fire direction toward evacuation zones, accelerating/decelerating signals
- **Binary risk flags**: fast grower, close to evac, low data resolution
- **Temporal context**: peak fire season months, afternoon ignition hours
- **Log-transformed** skewed distance and speed features

### Model
- **LightGBM** with one model trained per horizon (4 total)
- **5-fold stratified cross-validation** with out-of-fold (OOF) predictions for evaluation and calibration
- **Optuna** hyperparameter tuning (50 trials) optimized on the 48h horizon
- **Platt scaling** (logistic regression on OOF predictions) to correct probability calibration and reduce Brier score
- **Monotonicity enforcement** via cumulative max across horizons

### Key Engineering Decisions
- Stored OOF predictions via `event_id` dictionary mapping to avoid silent pandas index alignment bugs
- Handled degenerate horizons (72h has 100% hit rate among included samples) by skipping calibration and capping predictions at 0.97 to avoid maximum Brier penalty
- Guards for single-class validation folds that would crash `binary_logloss` early stopping

---

## Results

| Metric | Score |
|--------|-------|
| Hybrid Score | **0.939** |
| Metric | 0.3 x C-index + 0.7 x (1 - Weighted Brier) |

---

## Project Structure

```
.
├── train.csv                  # Training set (221 fires) with labels
├── test.csv                   # Test set (95 fires) for submission
├── metaData.csv               # Feature descriptions and metadata
├── sample_submission.csv      # Submission format reference
├── submission.csv             # Submission for the datathon
├── BTT_Keystone_LGB.py      # Full modeling pipeline
└── README.md
```

---

## How to Run

1. Clone the repository and upload the data files to your environment
2. Install dependencies:
```bash
pip install lightgbm optuna shap lifelines scikit-learn pandas numpy
```
3. Run the pipeline:
```bash
python wildfire_lgb_model.py
```
Or paste directly into a Google Colab notebook cell by cell.

---

## About WiDS

The [Women in Data Science (WiDS) Datathon](https://www.widsconference.org/datathon.html) is an annual global competition that encourages women and allies to use data science for social good. Participants from over 100 countries compete on real-world datasets across domains including healthcare, climate, and disaster response.

---

## Tools & Libraries

`Python` `LightGBM` `Optuna` `SHAP` `lifelines` `scikit-learn` `pandas` `numpy`
