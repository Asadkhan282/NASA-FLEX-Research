from pathlib import Path
import json
import warnings

import joblib

from train_flex_prediction import (
    load_data,
    build_models,
    CLASS_ORDER,
    FEATURES,
    RANDOM_STATE,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "flex_random_forest.joblib"
METADATA_PATH = MODEL_DIR / "flex_random_forest_metadata.json"


def main():
    print("\nNASA FLEX DEPLOYMENT MODEL")
    print("=" * 65)

    df, X, y, groups = load_data()

    models = build_models()
    model = models["Random forest"]

    print("\nTraining Random Forest on all eligible records...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    # Confirm that the fitted pipeline can produce predictions.
    sample = X.iloc[[0]]
    sample_prediction = model.predict(sample)[0]
    sample_probabilities = model.predict_proba(sample)[0]

    metadata = {
        "project": "NASA PSI-69 FLEX",
        "model": "Random Forest",
        "purpose": "Exploratory research demonstration",
        "training_records": int(len(X)),
        "unique_test_dates": int(groups.nunique()),
        "excluded_nonpositive_pressure_records": 1,
        "features": FEATURES,
        "class_order_for_display": CLASS_ORDER,
        "model_classes": model.classes_.tolist(),
        "random_state": RANDOM_STATE,
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "validation_reference": {
            "method": "StratifiedGroupKFold",
            "groups": "Test Date",
            "folds": 5,
            "out_of_fold_macro_f1": 0.5841,
            "out_of_fold_balanced_accuracy": 0.5776,
            "out_of_fold_accuracy": 0.7692
        },
        "limitations": [
            "This deployment model was fitted on the full eligible dataset.",
            "Its interactive predictions are not independent validation results.",
            "The dataset contains only 27 Completion records before pressure filtering.",
            "This is a research prototype, not a certified fire-safety prediction system."
        ]
    }

    bundle = {
        "model": model,
        "metadata": metadata
    }

    joblib.dump(bundle, MODEL_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("\nMODEL CHECK")
    print("-" * 65)
    print("Sample prediction:", sample_prediction)
    print("\nSample class probabilities:")

    for class_name, probability in zip(
        model.classes_,
        sample_probabilities
    ):
        print(f"{class_name}: {probability:.4f}")

    print("\nMODEL SAVED")
    print("-" * 65)
    print(MODEL_PATH)
    print(METADATA_PATH)

    print("\nDeployment model creation completed.")


if __name__ == "__main__":
    main()