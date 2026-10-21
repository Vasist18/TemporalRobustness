# Temporal Robustness of Fraud-Detection Models

**Dataset shift, calibration and explanation stability on real and simulated transaction data**

This repository contains the code and result artifacts for a study of how well common fraud-detection classifiers hold up when they are evaluated the way they are deployed: trained on the past, tested on the future, with no retraining.

---

## Research questions

| # | Question | Where answered |
|---|----------|----------------|
| RQ1 | How optimistic is a random train/test split compared with a chronological one (the *optimism gap*)? | Notebook §5 |
| RQ2 | How does performance decay across successive future time blocks, and is the model ranking stable over time? | Notebook §5 |
| RQ3 | How well calibrated are the scores under drift, and does class weighting (vs. none) and Platt scaling change that? | Notebook §7 |
| RQ4 | How stable are the model explanations (permutation importance) between the validation period and the late test period? | Notebook §8 |
| RQ5 | Do the conclusions agree across a real and a simulated dataset? | Notebook §9 |

Drift is also quantified directly (adversarial AUC and PSI, with a random-halves control) in §6.

## Datasets

Both are downloaded automatically through the Kaggle Hub API (`kagglehub`).

| Tag | Source | Type | Notes |
|-----|--------|------|-------|
| `creditcard` | [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | Real, anonymised (PCA features V1–V28 + amount) | Sorted by `Time`; `Time` is **excluded** as a feature; `LogAmount` added. |
| `sparkov` | [`kartik2112/fraud-detection`](https://www.kaggle.com/datasets/kartik2112/fraud-detection) | Simulated (Sparkov generator) | Train and test files concatenated, sorted by `unix_time`, then uniformly subsampled to 300,000 rows (preserves prevalence and time order). Engineered features: log amount, log city population, customer–merchant distance (haversine), age, cyclical hour, day of week, gender, one-hot merchant category. |

Please follow each dataset's licence and terms on Kaggle. The raw data is **not** redistributed here.

## Experimental design

- **Temporal protocol:** rows are sorted by time. The first 50% is the training period (the last 15% of it, in time, is held out as the calibration/validation set). The remaining 50% is cut into **5 consecutive test blocks**. Models are never retrained between blocks.
- **Random-split baseline:** stratified random split with the same training size; its test set is cut into 5 equal random blocks. 5 random splits are run.
- **Seeds:** 3 seeds (0, 1, 2) for the temporal protocol.
- **Models (main):** Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, MLP. All except the MLP use class weighting.
- **Models (ablation, RQ3):** unweighted Logistic Regression, HistGB and XGBoost.
- **Calibration:** Platt scaling fitted on the validation period, applied to every test block.
- **Metrics:** average precision (primary), ROC-AUC, recall at 1% FPR, Brier score, ECE (15 equal-mass bins), calibration-in-the-large (CITL = mean predicted / observed rate; 1 is perfect), risk-tier calibration.
- **Statistics:** bootstrap CIs (300 resamples) for the optimism gap and pooled AP; bootstrap CI (100 resamples) for the per-block AP slope; paired bootstrap for best-vs-rest comparison; Kendall's W for ranking agreement across blocks.
- **Explanations:** permutation importance (scored by AP) on validation, early-test and late-test subsets; compared by Spearman correlation and top-5 Jaccard overlap.

## Repository contents

```
.
├── fraud_temporal_robustness_v2_colab.ipynb   # full pipeline, runs top to bottom
├── README.md
└── outputs/                                   # (paper_outputs_v2.zip, unzipped)
    ├── environment.json                       # software versions and run configuration
    ├── cross_dataset_summary.csv              # RQ5 summary table
    ├── fig0_prevalence.png                    # fraud rate by chronological decile
    ├── fig_drift.png                          # adversarial AUC / PSI across blocks
    ├── fig_stability.png                      # explanation stability per model
    ├── fig_accuracy_vs_stability.png          # accuracy vs. explanation stability
    ├── {creditcard,sparkov}_results_raw.csv   # every metric, split, block, model, seed
    ├── {…}_performance.csv                    # RQ1/RQ2 summary (gap, slope, worst block)
    ├── {…}_ap_by_block.csv                    # AP per model per test block
    ├── {…}_pooled_ap_and_paired_tests.csv     # pooled AP with paired bootstrap tests
    ├── {…}_drift.csv                          # adversarial AUC and PSI vs. training
    ├── {…}_calibration.csv                    # Brier, ECE, CITL, raw and Platt-scaled
    ├── {…}_risk_tier_calibration.csv          # observed vs. predicted rate per risk tier
    ├── {…}_explanation_stability.csv          # RQ4 summary
    ├── {…}_fig1_ap_blocks.png                 # AP over time vs. random-split AP
    └── {…}_fig2_risk_tier_calibration.png     # reliability by risk tier
```

## Reproducing the results

### Option A: Google Colab (recommended)

1. Upload `fraud_temporal_robustness_v2_colab.ipynb` to Colab.
2. Run all cells. The notebook installs `kagglehub` and `xgboost`, downloads both datasets, runs everything, writes to `outputs/`, and triggers a download of `paper_outputs_v2.zip`.
3. If the dataset download asks for credentials, uncomment `kagglehub.login()` in the data-loading cell.

### Option B: Local

```bash
pip install kagglehub xgboost scikit-learn pandas numpy scipy matplotlib jupyter
jupyter notebook fraud_temporal_robustness_v2_colab.ipynb
```

Notes:

- Set `QUICK = True` in the config cell for a fast smoke test. **Quick-mode numbers must not be used for reported results.**
- The experiment step (§4) is the slow one: 8 models × 3 temporal seeds plus 5 random splits, per dataset, plus permutation importance.
- Outside Colab the final `google.colab` download step is skipped automatically; results remain in `./outputs`.
- Everything is seeded, but exact numbers can still shift slightly with library versions and thread scheduling.

### Environment of the reported run

Python 3.13, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, XGBoost 3.4.1, Linux x86_64 (Colab). Full details in `outputs/environment.json`.

## Headline results

Mean over the 5 temporal test blocks (average precision; full tables in the CSVs).

| Model | creditcard AP (random → temporal) | sparkov AP (random → temporal) |
|-------|-----------------------------------|--------------------------------|
| LogReg | 0.731 → 0.688 | 0.190 → 0.231 |
| RandomForest | 0.842 → 0.773 | 0.778 → 0.757 |
| HistGB | 0.753 → 0.732 | 0.809 → 0.827 |
| XGBoost | 0.854 → 0.771 | 0.810 → 0.796 |
| MLP | 0.814 → 0.694 | 0.836 → 0.810 |

What the outputs show:

- **Optimism gap (RQ1):** on `creditcard`, random splits overstate AP for every model, and the 95% CI excludes zero for RandomForest, XGBoost and MLP. On `sparkov` the gap is small, and it is significant only for the MLP (the LogReg gap is negative).
- **Decay and ranking (RQ2):** block-to-block AP is noisy on `creditcard` (e.g. LogReg ranges 0.51–0.86; few frauds per block). No model has a per-block AP slope whose CI excludes zero in either dataset. On `sparkov`, HistGB has the best pooled AP and is significantly better than all other models.
- **Drift:** adversarial AUC vs. training is about 0.50 for the random-halves control. It rises to 0.96–1.00 on `creditcard` test blocks and from 0.72 to 0.82 on `sparkov`, so shift is much stronger in the real data.
- **Calibration (RQ3):** raw scores from class-weighted models are badly miscalibrated (e.g. LogReg CITL ≈ 55 on both datasets), while unweighted counterparts are close to calibrated. Platt scaling on the validation period brings ECE down to roughly 0.0004–0.002 for all models.
- **Explanation stability (RQ4):** much lower on `creditcard` (Spearman val-vs-late 0.18–0.61) than on `sparkov` (0.81–0.97).

> **Caveat:** `creditcard` has very few positives per test block, so block-level estimates are high-variance and many differences are not statistically distinguishable. Cross-dataset correlations in §9 use only 5 models (and pooled points are not independent), so treat them as descriptive. Section 10 of the notebook auto-generates draft findings. Verify every sentence against the CSV tables before using it in a paper.

## Limitations

- Only two public datasets; `sparkov` is simulated, and `creditcard` features are anonymised, so feature-level explanations are not human-interpretable there.
- `sparkov` is subsampled to 300,000 rows for tractability.
- Fixed hyperparameters (no tuning), single training window, no retraining or online-update strategies.
- Permutation importance uses all positives plus up to 8,000 sampled negatives per evaluation set.

## Authors

- Kuntapalli Vasist (independent researcher, Hyderabad, India), corresponding author: learningwithvashistkuntapalli@gmail.com
- G V S Sathwik (independent researcher, Ongole, India)

## Citation

```bibtex
@misc{fraud_temporal_robustness,
  title  = {Temporal Robustness of Fraud-Detection Models: Dataset Shift, Calibration and Explanation Stability on Real and Simulated Transaction Data},
  author = {Kuntapalli, Vasist and Sathwik, G V S},
  year   = {2026},
  note   = {Manuscript in preparation. Code and results: <add repository / DOI link>}
}
```

## License

Code: add a licence of your choice (e.g. MIT or Apache-2.0) as `LICENSE`.
Data: governed by the original Kaggle dataset licences; not included.
