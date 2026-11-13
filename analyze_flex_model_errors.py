
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "processed" / "flex_cleaned.csv"
PREDICTION_PATH = ROOT / "reports" / "prediction" / "out_of_fold_predictions.csv"

OUTPUT_DIR = ROOT / "reports" / "model_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 20

TARGET = "Test end"
DATE_COL = "Test Date"

NUMERIC_FEATURES = [
    "Ambient pressure; mmHg",
    "O initial ambient composition; mole fraction",
    "CO initial ambient composition; mole fraction",
    "He initial ambient composition; mole fraction",
    "Droplet initial diameter; mm",
]

CATEGORICAL_FEATURES = ["Fuel"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLASS_ORDER = ["Extinction", "Disruption", "Completion"]


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

print("\nNASA FLEX MODEL ERROR AND FEATURE ANALYSIS")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
predictions = pd.read_csv(PREDICTION_PATH)

required = FEATURES + [
    TARGET,
    DATE_COL,
    "FLEX Test #",
    "FLEX Identifier",
]

missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    raise ValueError(
        f"Missing columns in cleaned dataset: {missing_columns}"
    )

# Exclude the known non-positive pressure record.
df["Ambient pressure; mmHg"] = pd.to_numeric(
    df["Ambient pressure; mmHg"],
    errors="coerce",
)

df = df[
    df["Ambient pressure; mmHg"] > 0
].copy()

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL],
    format="mixed",
    errors="coerce",
)

df = df.dropna(
    subset=[DATE_COL, TARGET]
).copy()

print(f"Records analyzed: {len(df)}")
print(f"Unique experiment dates: {df[DATE_COL].nunique()}")

print("\nOutcome distribution:")
print(df[TARGET].value_counts().to_string())


# ============================================================
# DATE-GROUPED RANDOM FOREST PREDICTION ERROR ANALYSIS
# ============================================================

print("\n1. DATE-GROUPED RANDOM FOREST ERROR ANALYSIS")
print("-" * 65)

selected = predictions[
    (predictions["Validation"] == "Date grouped")
    & (predictions["Model"] == "Random forest")
].copy()

if selected.empty:
    raise ValueError(
        "No date-grouped Random Forest predictions found."
    )

# Match model predictions with original experimental records.
error_data = selected.merge(
    df,
    on="FLEX Test #",
    how="left",
    suffixes=("_prediction", ""),
    validate="one_to_one",
)

if error_data[TARGET].isna().any():
    raise ValueError(
        "Some predictions could not be matched to the dataset."
    )

# Verify actual outcomes agree between the two files.
if not (
    error_data["Actual outcome"].astype(str).values
    == error_data[TARGET].astype(str).values
).all():
    raise ValueError(
        "Actual outcomes do not match between prediction and data files."
    )

error_data["Correct prediction"] = (
    error_data["Actual outcome"]
    == error_data["Predicted outcome"]
)

error_data["Prediction status"] = np.where(
    error_data["Correct prediction"],
    "Correct",
    "Incorrect",
)

error_data["Error type"] = np.where(
    error_data["Correct prediction"],
    "Correct",
    error_data["Actual outcome"].astype(str)
    + " predicted as "
    + error_data["Predicted outcome"].astype(str),
)

error_data.to_csv(
    OUTPUT_DIR / "date_grouped_random_forest_all_predictions.csv",
    index=False,
)

misclassified = error_data[
    ~error_data["Correct prediction"]
].copy()

misclassified.to_csv(
    OUTPUT_DIR / "date_grouped_random_forest_misclassified.csv",
    index=False,
)

print(
    f"Predictions analyzed: {len(error_data)}"
)

print(
    "Correct predictions: "
    f"{error_data['Correct prediction'].sum()}"
)

print(
    f"Incorrect predictions: {len(misclassified)}"
)

print("\nMisclassification types:")
print(
    misclassified["Error type"]
    .value_counts()
    .to_string()
)


# ============================================================
# COMPLETION CLASS ERROR ANALYSIS
# ============================================================

print("\n2. COMPLETION CLASS ANALYSIS")
print("-" * 65)

completion = error_data[
    error_data["Actual outcome"] == "Completion"
].copy()

missed_completion = completion[
    completion["Predicted outcome"] != "Completion"
].copy()

correct_completion = completion[
    completion["Predicted outcome"] == "Completion"
].copy()

missed_completion.to_csv(
    OUTPUT_DIR / "missed_completion_experiments.csv",
    index=False,
)

correct_completion.to_csv(
    OUTPUT_DIR / "correct_completion_experiments.csv",
    index=False,
)

print(
    f"Actual Completion experiments: {len(completion)}"
)

print(
    f"Correctly identified: {len(correct_completion)}"
)

print(
    f"Missed: {len(missed_completion)}"
)

if len(completion):
    completion_recall = (
        len(correct_completion) / len(completion)
    )

    print(
        f"Completion recall: {completion_recall:.4f}"
    )
else:
    completion_recall = None

print("\nMissed Completion prediction destinations:")

if len(missed_completion):
    print(
        missed_completion["Predicted outcome"]
        .value_counts()
        .to_string()
    )
else:
    print("No missed Completion records.")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n3. CONFUSION MATRIX")
print("-" * 65)

y_true = error_data["Actual outcome"]
y_pred = error_data["Predicted outcome"]

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASS_ORDER,
)

print(
    pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in CLASS_ORDER],
        columns=[f"Predicted {c}" for c in CLASS_ORDER],
    ).to_string()
)

fig, ax = plt.subplots(figsize=(8, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_ORDER,
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)

ax.set_title(
    "NASA FLEX - Date-Grouped Random Forest"
)

ax.set_xlabel("Predicted outcome")
ax.set_ylabel("Actual outcome")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "date_grouped_random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    labels=CLASS_ORDER,
    output_dict=True,
    zero_division=0,
)

with open(
    OUTPUT_DIR / "date_grouped_random_forest_classification_report.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(report, f, indent=4)

print("\nClassification report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0,
    )
)


# ============================================================
# REBUILD RANDOM FOREST PIPELINE
# ============================================================

# Important:
# This now matches the original model preprocessing:
# median imputation + StandardScaler for numeric features.
# Categorical preprocessing remains most-frequent imputation
# followed by one-hot encoding.

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent"),
    ),
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore"),
    ),
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        NUMERIC_FEATURES,
    ),
    (
        "categorical",
        categorical_pipeline,
        CATEGORICAL_FEATURES,
    ),
])

random_forest_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor,
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
])


# ============================================================
# DATE-GROUPED CROSS-VALIDATED PERMUTATION IMPORTANCE
# ============================================================

print("\n4. PERMUTATION FEATURE IMPORTANCE")
print("-" * 65)

X = df[FEATURES].copy()
y = df[TARGET].copy()

groups = df[DATE_COL].dt.normalize()

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

importance_records = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1,
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = clone(random_forest_pipeline)

    model.fit(
        X_train,
        y_train,
    )

    fold_predictions = model.predict(X_test)

    fold_macro_f1 = f1_score(
        y_test,
        fold_predictions,
        average="macro",
        zero_division=0,
    )

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
    )

    print(f"\nFold {fold}:")
    print(f"  Training records: {len(train_idx)}")
    print(f"  Testing records: {len(test_idx)}")
    print(f"  Macro F1: {fold_macro_f1:.4f}")

    for feature, mean, std in zip(
        FEATURES,
        result.importances_mean,
        result.importances_std,
    ):

        importance_records.append({
            "Fold": fold,
            "Feature": feature,
            "Importance mean": float(mean),
            "Importance std": float(std),
        })


# ============================================================
# SAVE FEATURE IMPORTANCE RESULTS
# ============================================================

importance_by_fold = pd.DataFrame(
    importance_records
)

importance_by_fold.to_csv(
    OUTPUT_DIR / "permutation_importance_by_fold.csv",
    index=False,
)

importance_summary = (
    importance_by_fold
    .groupby("Feature")
    .agg(
        Mean_importance=("Importance mean", "mean"),
        Std_across_folds=("Importance mean", "std"),
        Mean_permutation_std=("Importance std", "mean"),
    )
    .reset_index()
    .sort_values(
        "Mean_importance",
        ascending=False,
    )
)

importance_summary.to_csv(
    OUTPUT_DIR / "permutation_feature_importance.csv",
    index=False,
)

print(
    "\nAverage feature importance across date-grouped folds:"
)

print(
    importance_summary.to_string(index=False)
)


# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================

plot_data = importance_summary.sort_values(
    "Mean_importance",
    ascending=True,
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    plot_data["Feature"],
    plot_data["Mean_importance"],
    xerr=plot_data["Std_across_folds"].fillna(0),
    capsize=4,
)

ax.axvline(
    0,
    linewidth=1,
)

ax.set_title(
    "NASA FLEX - Date-Grouped Random Forest\n"
    "Permutation Feature Importance"
)

ax.set_xlabel(
    "Decrease in validation macro F1 after permutation"
)

ax.set_ylabel(
    "Input feature"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "permutation_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {
    "records_analyzed": int(len(error_data)),

    "correct_predictions": int(
        error_data["Correct prediction"].sum()
    ),

    "incorrect_predictions": int(
        len(misclassified)
    ),

    "completion_records": int(
        len(completion)
    ),

    "correct_completion": int(
        len(correct_completion)
    ),

    "missed_completion": int(
        len(missed_completion)
    ),

    "completion_recall": (
        float(completion_recall)
        if completion_recall is not None
        else None
    ),

    "date_grouped_random_forest_macro_f1": float(
        f1_score(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )
    ),

    "date_grouped_random_forest_confusion_matrix": (
        cm.tolist()
    ),

    "feature_importance": (
        importance_summary.to_dict(orient="records")
    ),
}

with open(
    OUTPUT_DIR / "model_error_analysis_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED")
print("=" * 65)

print(f"\nOutput directory:\n{OUTPUT_DIR}")

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    if path.is_file():
        print(f"  {path.name}")

print(
    "\nNASA FLEX feature and error analysis completed."
)
