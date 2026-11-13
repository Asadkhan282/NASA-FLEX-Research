
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "data" / "processed" / "flex_cleaned.csv"
OUTPUT_DIR = ROOT / "reports" / "prediction"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Test end"
DATE_COLUMN = "Test Date"
PRESSURE = "Ambient pressure; mmHg"

NUMERIC_FEATURES = [
    PRESSURE,
    "O initial ambient composition; mole fraction",
    "CO initial ambient composition; mole fraction",
    "He initial ambient composition; mole fraction",
    "Droplet initial diameter; mm"
]

CATEGORICAL_FEATURES = ["Fuel"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLASS_ORDER = [
    "Extinction",
    "Disruption",
    "Completion"
]

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# DATA PREPARATION
# ============================================================

def load_data():

    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT}")

    df = pd.read_csv(INPUT)

    required = FEATURES + [TARGET, DATE_COLUMN, "FLEX Test #"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()

    # Keep only the three known outcome classes.
    df = df[df[TARGET].isin(CLASS_ORDER)].copy()

    # Convert predictors to numeric.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Exclude non-positive pressure.
    # Test 114 is the previously identified zero-pressure record.
    invalid_pressure = df[PRESSURE] <= 0

    print("\nDATA QUALITY")
    print("=" * 65)
    print("Rows before pressure filtering:", len(df))
    print("Non-positive pressure records:", int(invalid_pressure.sum()))

    df = df.loc[~invalid_pressure].copy()

    # Parse mixed date formats.
    df["_parsed_date"] = pd.to_datetime(
        df[DATE_COLUMN],
        format="mixed",
        errors="coerce"
    )

    if df["_parsed_date"].isna().any():
        bad_dates = df.loc[
            df["_parsed_date"].isna(),
            [DATE_COLUMN, "FLEX Test #"]
        ]

        raise ValueError(
            "Some test dates could not be parsed. Inspect:\n"
            + bad_dates.to_string(index=False)
        )

    # Same date = same validation group.
    groups = df["_parsed_date"].dt.strftime("%Y-%m-%d")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    print("Rows retained:", len(df))
    print("Unique test dates:", groups.nunique())

    print("\nOutcome counts:")
    print(y.value_counts().reindex(CLASS_ORDER, fill_value=0))

    print("\nMissing predictor values:")
    print(X.isna().sum().to_string())

    return df, X, y, groups


# ============================================================
# MODEL PIPELINES
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


def build_models():

    models = {}

    models["Majority baseline"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", DummyClassifier(strategy="most_frequent"))
        ]
    )

    models["Multinomial logistic regression"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    models["Random forest"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )

    return models


# ============================================================
# VALIDATION
# ============================================================

def evaluate_model(model, X, y, groups, validation_name):

    if validation_name == "Stratified random":
        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        split_iterator = splitter.split(X, y)

    elif validation_name == "Date grouped":
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        split_iterator = splitter.split(X, y, groups)

    else:
        raise ValueError("Unknown validation method.")

    predictions = np.empty(len(y), dtype=object)

    fold_f1 = []
    fold_balanced_accuracy = []

    for fold_number, (train_idx, test_idx) in enumerate(
        split_iterator,
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Every training fold must contain all classes.
        missing_classes = set(CLASS_ORDER) - set(y_train.unique())

        if missing_classes:
            raise ValueError(
                f"{validation_name}, fold {fold_number}: "
                f"training fold is missing classes {missing_classes}"
            )

        fitted = clone(model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted.fit(X_train, y_train)

        fold_predictions = fitted.predict(X_test)

        predictions[test_idx] = fold_predictions

        fold_f1.append(
            f1_score(
                y_test,
                fold_predictions,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0
            )
        )

        fold_balanced_accuracy.append(
            balanced_accuracy_score(
                y_test,
                fold_predictions
            )
        )

        print(
            f"  Fold {fold_number}: "
            f"train={len(train_idx)}, "
            f"test={len(test_idx)}, "
            f"macro-F1={fold_f1[-1]:.4f}, "
            f"balanced-accuracy={fold_balanced_accuracy[-1]:.4f}"
        )

    predictions = pd.Series(
        predictions,
        index=y.index,
        name="Predicted outcome"
    )

    results = {
        "validation": validation_name,
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "mean_fold_macro_f1": float(np.mean(fold_f1)),
        "std_fold_macro_f1": float(np.std(fold_f1, ddof=1)),
        "mean_fold_balanced_accuracy": float(
            np.mean(fold_balanced_accuracy)
        ),
        "std_fold_balanced_accuracy": float(
            np.std(fold_balanced_accuracy, ddof=1)
        ),
        "classification_report": classification_report(
            y,
            predictions,
            labels=CLASS_ORDER,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y,
            predictions,
            labels=CLASS_ORDER
        ).tolist()
    }

    return results, predictions


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nNASA FLEX PREDICTIVE MODEL DEVELOPMENT")
    print("=" * 65)

    df, X, y, groups = load_data()

    models = build_models()

    all_results = {}
    prediction_frames = []

    validation_methods = [
        "Stratified random",
        "Date grouped"
    ]

    for validation_name in validation_methods:

        print("\n")
        print("#" * 65)
        print(validation_name.upper(), "VALIDATION")
        print("#" * 65)

        all_results[validation_name] = {}

        for model_name, model in models.items():

            print(f"\nMODEL: {model_name}")
            print("-" * 65)

            results, predictions = evaluate_model(
                model,
                X,
                y,
                groups,
                validation_name
            )

            all_results[validation_name][model_name] = results

            print("\nOverall out-of-fold results:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(
                f"Balanced accuracy: "
                f"{results['balanced_accuracy']:.4f}"
            )
            print(f"Macro F1: {results['macro_f1']:.4f}")

            print("\nClassification report:")
            print(
                classification_report(
                    y,
                    predictions,
                    labels=CLASS_ORDER,
                    zero_division=0
                )
            )

            print("Confusion matrix:")
            print(
                pd.DataFrame(
                    results["confusion_matrix"],
                    index=CLASS_ORDER,
                    columns=CLASS_ORDER
                )
            )

            frame = pd.DataFrame({
                "FLEX Test #": df["FLEX Test #"],
                "FLEX Identifier": df["FLEX Identifier"],
                "Test Date": df[DATE_COLUMN],
                "Actual outcome": y,
                "Predicted outcome": predictions,
                "Validation": validation_name,
                "Model": model_name
            })

            prediction_frames.append(frame)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    json_path = OUTPUT_DIR / "model_comparison_results.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    predictions_path = OUTPUT_DIR / "out_of_fold_predictions.csv"

    pd.concat(
        prediction_frames,
        ignore_index=True
    ).to_csv(
        predictions_path,
        index=False
    )

    # Compact comparison table.
    summary_rows = []

    for validation_name, model_results in all_results.items():

        for model_name, metrics in model_results.items():

            summary_rows.append({
                "Validation": validation_name,
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Balanced accuracy": metrics["balanced_accuracy"],
                "Macro F1": metrics["macro_f1"],
                "Mean fold Macro F1": metrics["mean_fold_macro_f1"],
                "Std fold Macro F1": metrics["std_fold_macro_f1"]
            })

    summary = pd.DataFrame(summary_rows)

    summary_path = OUTPUT_DIR / "model_comparison_summary.csv"

    summary.to_csv(summary_path, index=False)

    print("\nFINAL MODEL COMPARISON")
    print("=" * 65)
    print(summary.to_string(index=False))

    print("\nSAVED FILES")
    print("=" * 65)
    print(json_path)
    print(predictions_path)
    print(summary_path)

    print("\nPREDICTIVE MODEL EVALUATION COMPLETED.")


if __name__ == "__main__":
    main()
