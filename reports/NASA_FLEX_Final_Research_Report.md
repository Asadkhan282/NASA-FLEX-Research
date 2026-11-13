# NASA PSI-69 FLEX Research Project
## Experimental Data Analysis, Statistical Investigation, and Predictive Modeling

**Project:** NASA FLEX — Flame Extinguishment Experiment Research  
**Dataset:** NASA Physical Sciences Informatics (PSI), PSI-69 Version 5 — Flame Extinguishment Experiment (FLEX)  
**Dataset DOI:** [10.60555/mbq8-0451](https://doi.org/10.60555/mbq8-0451)  
**Project type:** Research prototype and experimental-dataset analysis  
**Status:** Research analysis and interactive prototype completed; independent external validation and final submission review remain outstanding.

---

## Abstract

This project analyzes the NASA PSI-69 Flame Extinguishment Experiment (FLEX) dataset to investigate experimental outcomes associated with flame extinction, disruption, and completion. The local dataset contains 274 experimental records and 15 original columns. The work includes data validation, exploratory data analysis, statistical testing, multivariable analysis, predictive-model development, date-grouped cross-validation, error analysis, and an interactive Streamlit research dashboard.

The observed outcome distribution is 186 Extinction records, 61 Disruption records, and 27 Completion records. Fuel type and initial oxygen composition show statistically significant associations with the recorded outcome in the analyses performed. Three predictive approaches were compared: a majority-class baseline, logistic regression, and random forest. Under date-grouped cross-validation, the random forest achieved an accuracy of 0.7692 and macro F1 of 0.5841, while logistic regression achieved a higher balanced accuracy of 0.5997. The random forest had limited recall for the minority Completion class (0.1852), demonstrating that aggregate accuracy does not adequately describe performance across all outcomes.

The project is a research prototype, not a validated operational fire-safety predictor. The evaluation uses internal cross-validation on one dataset; no independent experimental campaign was available for external validation. Statistical associations are observational and do not establish causation.

## 1. Research objectives

The project was designed to:

1. Validate the integrity and structure of the downloaded NASA FLEX experimental dataset.
2. Characterize the distribution of experimental outcomes and measured conditions.
3. Examine associations between fuel, oxygen composition, and experimental outcome.
4. Build and compare baseline and machine-learning classifiers.
5. Evaluate generalization using experiment-date-grouped cross-validation to reduce the risk of related experiments appearing in both training and validation folds.
6. Examine class-specific errors and feature importance.
7. Provide an interactive dashboard for data exploration, statistical results, model evaluation, and illustrative predictions.

## 2. Dataset and data preparation

### 2.1 Source and scope

The source is NASA PSI-69 Version 5, the Flame Extinguishment Experiment (FLEX) investigation. The local CSV contains 274 experimental records and 15 columns. The dataset is associated with spacecraft fire-safety and combustion research in a microgravity context.

The source data were retained as the original input. Processing created a separate cleaned dataset; the original CSV was not modified.

### 2.2 Record and outcome counts

| Outcome | Records | Share |
|---|---:|---:|
| Extinction | 186 | 67.88% |
| Disruption | 61 | 22.26% |
| Completion | 27 | 9.85% |
| **Total** | **274** | **100.00%** |

Fuel distribution:

| Fuel | Records |
|---|---:|
| Methanol | 157 |
| Heptane | 117 |
| **Total** | **274** |

The local file contains test numbers 1–274, with no missing or duplicated test numbers. The NASA portal's displayed range extending to 275 should not be interpreted as evidence of a 275th record in this local CSV. This analysis is based on the 274 records actually present in the downloaded file.

### 2.3 Data-quality checks

The validation pipeline checked row and column counts, duplicate rows, duplicate test numbers, outcome labels, missing measurements, approximate measurements, pressure values, and gas-composition totals.

| Data-quality item | Result |
|---|---:|
| Rows | 274 |
| Original columns | 15 |
| Duplicate rows | 0 |
| Duplicate test numbers | 0 |
| Zero-pressure records | 1 |
| Approximate burn-time entries | 2 |
| Missing initial droplet diameter | 12 |
| Missing visible flame-extinction diameter | 77 |
| Missing burning rate | 16 |
| Missing burn time | 6 |

Two burn-time entries are approximate rather than exact: Test 223 is recorded as approximately 6 seconds and Test 224 as approximately 9 seconds. The numeric estimates were retained with approximation flags in the quality report. They should not be treated as exact measurements.

Unicode dash entries were treated as unavailable measurements. The cleaned dataset preserves the 274 experimental records and does not fabricate replacement measurements. Missing predictor values used in model pipelines are imputed within the training pipeline rather than by changing the source data.

### 2.4 Gas-composition totals

The four recorded gas-composition columns were summed for each record where all four values were available. The mean total was 0.9987956, the minimum was 0.98, and the maximum was 1.01.

A strict floating-point comparison counted 35 totals below 1.0. The EDA uses a numerical tolerance of \(10^{-9}\), which counted 28. Seven records display as 1.000000000000 but are infinitesimally below 1.0 in floating-point representation. For interpretation, the tolerance-based count is preferable to treating numerical representation noise as a meaningful physical discrepancy.

Six records fell outside the interval 0.99–1.01. These are quality-control flags for review, not proof that the underlying experimental records are invalid. The EDA and validation summaries should use the same tolerance convention when reporting the below-one count.

## 3. Exploratory data analysis

The EDA stage generated seven visualizations:

1. `01_outcome_distribution.png` — outcome distribution.
2. `02_fuel_outcomes.png` — outcomes by fuel.
3. `03_oxygen_outcomes.png` — oxygen composition by outcome.
4. `04_pressure_burntime.png` — pressure and burn-time relationship.
5. `05_droplet_behavior.png` — droplet-related measurements.
6. `06_gas_correlation.png` — gas-composition relationships.
7. `07_burning_rate.png` — burning-rate analysis.

The EDA includes 274 experiments. Its summary also includes two derived columns, which is why the EDA dataframe has 17 columns while the original and cleaned datasets have 15.

### 3.1 Outcome distribution

Extinction is the most frequent recorded outcome (186/274; 67.88%). Disruption accounts for 61 records (22.26%), and Completion accounts for 27 (9.85%). This imbalance is important for model evaluation: a classifier can obtain apparently strong accuracy by favoring the majority class while performing poorly on minority outcomes.

### 3.2 Fuel and outcome

The observed outcome distribution differs between the two fuels:

| Fuel | Extinction | Disruption | Completion |
|---|---:|---:|---:|
| Heptane (n=117) | 60 | 38 | 19 |
| Methanol (n=157) | 126 | 23 | 8 |

The association between fuel and outcome was statistically significant: chi-square = 26.3107, df = 2, p = 0.00000194, Cramér's V = 0.3099. This is an association in the observed dataset; it does not establish that changing fuel alone causes a particular outcome.

### 3.3 Oxygen composition and outcome

Initial oxygen composition differed across the three outcome groups:

| Outcome | n | Mean oxygen fraction | Median oxygen fraction |
|---|---:|---:|---:|
| Extinction | 186 | 0.18274 | 0.18 |
| Disruption | 61 | 0.24984 | 0.24 |
| Completion | 27 | 0.21296 | 0.21 |

A Kruskal–Wallis test found a statistically significant difference across groups (H = 84.99556, p approximately 3.5 × 10⁻¹⁹; epsilon-squared = 0.30626). Pairwise Mann–Whitney comparisons with Holm adjustment were also significant:

- Extinction vs Disruption: adjusted p = 3.8729 × 10⁻¹⁹.
- Extinction vs Completion: adjusted p = 0.0024136.
- Disruption vs Completion: adjusted p = 0.0012587.

These are observational group differences. They should not be interpreted as causal effects.

## 4. Multivariable statistical analysis

A multinomial logistic regression was fitted to all 274 complete-case records for the selected variables, with Extinction as the reference outcome. Predictors were initial oxygen composition and fuel (Methanol versus Heptane). The model converged.

| Model statistic | Result |
|---|---:|
| Pseudo R² | 0.27965 |
| Log-likelihood | -162.98366 |
| Null log-likelihood | -226.25732 |
| AIC | 337.9673 |
| BIC | 359.6461 |
| Likelihood-ratio test p-value | 2.1312 × 10⁻²⁶ |

Adjusted associations relative to Extinction:

| Comparison | Predictor | Odds ratio | 95% CI | p-value |
|---|---|---:|---:|---:|
| Disruption vs Extinction | Oxygen, per 0.01 increase | 1.5309 | 1.3605–1.7225 | 1.4886 × 10⁻¹² |
| Disruption vs Extinction | Methanol vs Heptane | 0.2050 | 0.0935–0.4494 | 0.000075972 |
| Completion vs Extinction | Oxygen, per 0.01 increase | 1.2881 | 1.1357–1.4608 | 0.000080770 |
| Completion vs Extinction | Methanol vs Heptane | 0.1902 | 0.0761–0.4755 | 0.000385249 |

The model indicates adjusted associations between the selected predictors and outcome categories. The odds ratios are conditional on the specified model and included predictors. They are not causal effects and do not account for every possible experimental or temporal factor.

## 5. Predictive modeling

### 5.1 Modeling dataset and features

One record with non-positive ambient pressure was excluded from predictive modeling, leaving 273 records across 80 experiment dates. The modeling outcome counts were 185 Extinction, 61 Disruption, and 27 Completion.

Predictors:

- Ambient pressure.
- Initial oxygen composition.
- Initial carbon-monoxide composition.
- Initial helium composition.
- Initial droplet diameter.
- Fuel.

The following were excluded to avoid leakage or inappropriate identifiers: test number, experiment identifier, date/GMT, nitrogen composition (dependent on the other gas fractions), and measurements recorded during or after the experiment outcome, including visible flame-extinction diameter, burning rate, and burn time.

Numeric missing values were imputed within the modeling pipeline; numeric features were standardized; fuel was imputed if needed and one-hot encoded. The models were a majority-class dummy baseline, class-weighted logistic regression, and class-weighted random forest.

### 5.2 Validation design

Two five-fold validation strategies were used:

- Stratified random cross-validation.
- Stratified group cross-validation grouped by experiment date.

Date-grouped validation is the more conservative estimate used for the main generalization discussion because experiments from the same date are kept together rather than distributed across training and validation folds. Both are internal cross-validation on the same dataset, not independent external validation.

### 5.3 Model comparison

| Validation strategy | Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|
| Stratified random | Majority baseline | 0.6777 | 0.3333 | 0.2693 |
| Stratified random | Logistic regression | 0.6374 | 0.5609 | 0.5265 |
| Stratified random | Random forest | 0.7546 | 0.5842 | 0.5864 |
| Date-grouped | Majority baseline | 0.6777 | 0.3333 | 0.2693 |
| Date-grouped | Logistic regression | 0.6520 | **0.5997** | 0.5518 |
| Date-grouped | Random forest | **0.7692** | 0.5776 | **0.5841** |

The majority baseline's 67.77% accuracy is misleading because it predicts the majority Extinction class and has zero recall for the other two classes. Under date-grouped validation, random forest achieved the highest accuracy and macro F1, while logistic regression achieved the highest balanced accuracy. No single metric tells the whole story.

### 5.4 Date-grouped random-forest class performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Extinction | 0.8462 | 0.8919 | 0.8684 |
| Disruption | 0.6250 | 0.6557 | 0.6400 |
| Completion | 0.3571 | 0.1852 | 0.2439 |

Confusion matrix (rows = actual class; columns = predicted class; order = Extinction, Disruption, Completion):

| Actual \ Predicted | Extinction | Disruption | Completion |
|---|---:|---:|---:|
| Extinction | 165 | 15 | 5 |
| Disruption | 17 | 40 | 4 |
| Completion | 13 | 9 | 5 |

The model identifies Extinction more reliably than Completion. It correctly identified 5 of 27 Completion records and missed 22. This minority-class limitation must be prominent in any presentation of the model.

## 6. Model error and feature analysis

The date-grouped random-forest predictions covered 273 records: 210 correct and 63 incorrect.

Main misclassification paths:

| Actual outcome | Predicted outcome | Count |
|---|---|---:|
| Disruption | Extinction | 17 |
| Extinction | Disruption | 15 |
| Completion | Extinction | 13 |
| Completion | Disruption | 9 |
| Extinction | Completion | 5 |
| Disruption | Completion | 4 |

Permutation feature importance, averaged across folds:

| Feature | Mean importance | Fold standard deviation |
|---|---:|---:|
| Initial oxygen composition | 0.131060 | 0.105058 |
| Initial droplet diameter | 0.099994 | 0.055768 |
| Fuel | 0.073564 | 0.045891 |
| Initial CO composition | 0.030301 | 0.016184 |
| Ambient pressure | 0.009336 | 0.056729 |
| Initial helium composition | 0.007874 | 0.034645 |

Oxygen composition, initial droplet diameter, and fuel had the largest mean permutation importance in this analysis. The relatively large fold-to-fold variation for some features indicates that importance estimates are not perfectly stable. Permutation importance measures predictive contribution in this fitted evaluation setup; it does not demonstrate physical causality.

The error analysis also confirms that Completion recall is a key weakness. More balanced experimental coverage and independent validation are needed before relying on the classifier for any operational decision.

## 7. Interactive prediction demonstration: fuel comparison

The dashboard includes an illustrative comparison using the same fixed input conditions for both fuels:

- Ambient pressure: 759.60 mmHg.
- Initial oxygen fraction: 0.2000.
- Initial CO fraction: 0.0000.
- Initial helium fraction: 0.0000.
- Initial droplet diameter: 3.0100 mm.

| Model output | Heptane | Methanol |
|---|---:|---:|
| Completion score | 14.37% | 0.39% |
| Disruption score | 43.17% | 11.99% |
| Extinction score | 42.46% | 87.62% |
| Predicted class | Disruption | Extinction |

For this single input configuration, the model's Methanol score for Extinction is 45.16 percentage points higher than its Heptane score. This is a demonstration of the trained model's response to changing the fuel input, not a new experimental observation, a causal estimate, or a calibrated physical probability. The scores must not be interpreted as guaranteed real-world outcome probabilities.

## 8. Software and dashboard

The project includes a Streamlit dashboard with the following pages:

1. Research Overview.
2. Experimental Data Explorer.
3. Exploratory Data Analysis.
4. Statistical Analysis.
5. Model Evaluation.
6. Explainability & Errors.
7. Data Quality & Limitations.
8. Prediction Lab.

A deployable random-forest model was saved at `models/flex_random_forest.joblib`, with accompanying metadata at `models/flex_random_forest_metadata.json`. The saved model was trained on all 273 eligible records after excluding the zero-pressure record. The metadata's date-grouped cross-validation metrics describe the earlier evaluation procedure; they are not an independent test of the final all-data fitted model.

## 9. Limitations

1. **Dataset scope:** The analysis uses one NASA FLEX dataset version and 274 locally available records. It is not an independent multi-campaign validation.
2. **Class imbalance:** Completion has only 27 records in the full dataset and 5 correct predictions in the date-grouped random-forest evaluation.
3. **External validation:** No separate experimental campaign or external holdout dataset was used. Cross-validation is internal.
4. **Prediction uncertainty:** Model scores are not established as calibrated physical probabilities.
5. **Observational analysis:** Statistical associations do not establish causation.
6. **Missing measurements:** Several physical measurements are missing, particularly visible flame-extinction diameter. Imputation is performed within the predictive pipeline where required.
7. **Approximate measurements:** Two burn-time entries are approximate values and should not be treated as exact.
8. **Composition quality checks:** Six gas-composition totals fall outside 0.99–1.01. The below-one count depends on the numerical tolerance; seven totals are represented as infinitesimally below one due to floating-point precision.
9. **Model stability:** Feature importance and minority-class performance can vary across folds. Additional data are needed to establish robust generalization.
10. **Operational use:** This research prototype is not suitable for spacecraft fire-safety decisions, certification, or operational prediction without substantially more validation.

## 10. Conclusions

The analysis identifies meaningful associations between FLEX experimental outcomes, fuel type, and initial oxygen composition. The multivariable analysis supports the presence of adjusted associations for oxygen composition and fuel in the fitted model. Predictive modeling shows that machine-learning classifiers can distinguish some outcome patterns better than a majority-class baseline, particularly for Extinction and Disruption.

The date-grouped random forest produced an accuracy of 0.7692 and macro F1 of 0.5841, while logistic regression achieved a higher balanced accuracy of 0.5997. The Completion recall of the random forest was only 0.1852, which limits the model's usefulness for that outcome. These results support continued research, not operational deployment.

The most important next research steps are to obtain additional FLEX experiments, improve representation of minority outcomes, test alternative models and calibration methods, and validate the approach on an independent experimental campaign. The dashboard provides a transparent interface for reviewing the dataset, analyses, model behavior, and limitations.

## 11. Reproducibility and submission checklist

Before submission, verify the following in the project environment:

- [ ] The original NASA CSV remains unchanged.
- [ ] `validate_flex_dataset.py` runs successfully and regenerates the cleaned CSV and quality reports.
- [ ] `run_flex_eda.py` runs successfully and regenerates all seven figures and `eda_summary.json`.
- [ ] Statistical-analysis scripts run from the project root and regenerate their JSON/CSV outputs.
- [ ] The prediction evaluation and error-analysis scripts run successfully.
- [ ] The saved model and metadata files exist and load successfully.
- [ ] The Streamlit dashboard starts and all eight pages can be opened.
- [ ] Prediction Lab produces outputs and clearly displays the limitations of model scores.
- [ ] The report's figures and tables match the latest generated outputs.
- [ ] The report and dashboard identify the work as a research prototype and do not claim independent external validation.
- [ ] All project dependencies and run commands are documented.
- [ ] The final submission folder contains the report, source code, cleaned data, model artifacts, figures, and a README.

### Suggested final verification commands

Run from the project root with the virtual environment activated:

```powershell
python -m py_compile .\validate_flex_dataset.py
python .\validate_flex_dataset.py

python -m py_compile .\run_flex_eda.py
python .\run_flex_eda.py

python -m py_compile .\train_flex_prediction.py
python .\train_flex_prediction.py

python -m py_compile .\analyze_flex_model_errors.py
python .\analyze_flex_model_errors.py

python -m py_compile .\save_flex_model.py
python .\save_flex_model.py
```

Run the statistical scripts by their actual filenames in the project. After regenerating results, compare the report tables with the corresponding JSON files. Start the dashboard with:

```powershell
streamlit run .\app.py
```

Do not rerun scripts that have not been verified against the current project state if doing so would overwrite a known-good artifact. Preserve a copy of the final reports and model artifacts before submission.

## References

1. NASA Physical Sciences Informatics. **PSI-69 Version 5: Flame Extinguishment Experiment (FLEX)**. Dataset DOI: [https://doi.org/10.60555/mbq8-0451](https://doi.org/10.60555/mbq8-0451). Research area: Combustion Science; sub-research area: Spacecraft Fire Safety. Dataset license as listed in the NASA PSI record: CC0-1.0.
2. scikit-learn documentation. Model evaluation, preprocessing pipelines, cross-validation, and permutation feature importance. [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/).
3. pandas documentation. Data loading, cleaning, and tabular analysis. [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/).
4. Streamlit documentation. Interactive application framework. [https://docs.streamlit.io/](https://docs.streamlit.io/).

---

**Reporting note:** This document summarizes the project results recorded during development. Before formal submission, run the reproducibility checklist and reconcile all report values against the latest machine-generated JSON outputs. Any change in regenerated results must be reflected in this report.
