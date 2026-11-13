
from pathlib import Path
import json

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent

MODEL_FILE = ROOT / "models" / "flex_random_forest.joblib"
OUTPUT_DIR = ROOT / "reports" / "prediction_lab"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]

conditions = {
    "Ambient pressure; mmHg": 759.60,
    "O initial ambient composition; mole fraction": 0.2000,
    "CO initial ambient composition; mole fraction": 0.0000,
    "He initial ambient composition; mole fraction": 0.0000,
    "Droplet initial diameter; mm": 3.0100,
}

results = []

for fuel in ["Heptane", "Methanol"]:

    record = {
        **conditions,
        "Fuel": fuel,
    }

    input_data = pd.DataFrame([record])

    prediction = model.predict(input_data)[0]
    scores = model.predict_proba(input_data)[0]

    for class_name, score in zip(model.classes_, scores):

        results.append({
            "Fuel": fuel,
            "Predicted outcome": prediction,
            "Outcome": class_name,
            "Model score": float(score),
        })


results_df = pd.DataFrame(results)

# Save detailed results.
csv_path = OUTPUT_DIR / "fuel_comparison_scores.csv"
results_df.to_csv(csv_path, index=False)

# Calculate score differences.
wide = results_df.pivot(
    index="Outcome",
    columns="Fuel",
    values="Model score",
)

wide["Methanol minus Heptane"] = (
    wide["Methanol"] - wide["Heptane"]
)

wide["Difference percentage points"] = (
    wide["Methanol minus Heptane"] * 100
)

wide = wide.reset_index()

difference_path = OUTPUT_DIR / "fuel_comparison_differences.csv"
wide.to_csv(difference_path, index=False)

predictions = {}

for fuel in ["Heptane", "Methanol"]:

    subset = results_df[results_df["Fuel"] == fuel]

    predictions[fuel] = {
        "predicted_outcome": subset[
            "Predicted outcome"
        ].iloc[0],
        "scores": {
            row["Outcome"]: float(row["Model score"])
            for _, row in subset.iterrows()
        },
    }

report = {
    "experiment": "Controlled fuel-input comparison",
    "model": "Saved NASA FLEX Random Forest pipeline",
    "conditions": conditions,
    "predictions": predictions,
    "interpretation": (
        "At the selected input conditions, the model produces "
        "different outcome scores for Heptane and Methanol. "
        "This is a model-behavior comparison, not causal evidence "
        "or an independently validated physical prediction."
    ),
    "limitations": [
        "The model was fitted on all 273 eligible records.",
        "Model scores are not calibrated physical probabilities.",
        "This comparison uses one selected set of input conditions.",
        "The result does not establish a general causal fuel effect.",
        "The model is not suitable for operational fire-safety decisions."
    ],
}

json_path = OUTPUT_DIR / "fuel_comparison_summary.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)


markdown_path = OUTPUT_DIR / "fuel_comparison_report.md"

with open(markdown_path, "w", encoding="utf-8") as file:

    file.write("# NASA FLEX: Controlled Fuel-Input Comparison\n\n")

    file.write("## Experimental conditions\n\n")

    for name, value in conditions.items():
        file.write(f"- {name}: {value}\n")

    file.write("\n## Prediction results\n\n")

    for fuel, result in predictions.items():

        file.write(f"### {fuel}\n\n")
        file.write(
            f"Predicted outcome: **{result['predicted_outcome']}**\n\n"
        )

        for outcome, score in sorted(
            result["scores"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            file.write(f"- {outcome}: {score:.6f} ({score:.2%})\n")

        file.write("\n")

    file.write("## Score differences\n\n")

    file.write(
        "| Outcome | Heptane | Methanol | "
        "Difference (percentage points) |\n"
    )
    file.write("|---|---:|---:|---:|\n")

    for _, row in wide.iterrows():

        file.write(
            f"| {row['Outcome']} "
            f"| {row['Heptane']:.4%} "
            f"| {row['Methanol']:.4%} "
            f"| {row['Difference percentage points']:+.2f} |\n"
        )

    file.write("\n## Interpretation\n\n")
    file.write(report["interpretation"] + "\n\n")

    file.write("## Limitations\n\n")

    for limitation in report["limitations"]:
        file.write(f"- {limitation}\n")


print("\nNASA FLEX FUEL COMPARISON SAVED")
print("=" * 60)

print(f"Detailed scores: {csv_path}")
print(f"Score differences: {difference_path}")
print(f"JSON summary: {json_path}")
print(f"Research report: {markdown_path}")

print("\nPredicted outcomes:")

for fuel, result in predictions.items():
    print(f"{fuel}: {result['predicted_outcome']}")

print("\nAll comparison artifacts generated successfully.")
