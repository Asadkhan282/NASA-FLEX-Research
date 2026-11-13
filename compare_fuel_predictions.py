
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent

MODEL_FILE = ROOT / "models" / "flex_random_forest.joblib"

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]

conditions = {
    "Ambient pressure; mmHg": 759.60,
    "O initial ambient composition; mole fraction": 0.2000,
    "CO initial ambient composition; mole fraction": 0.0000,
    "He initial ambient composition; mole fraction": 0.0000,
    "Droplet initial diameter; mm": 3.0100,
}

print("\nNASA FLEX FUEL COMPARISON")
print("=" * 65)

for fuel in ["Heptane", "Methanol"]:

    record = {
        **conditions,
        "Fuel": fuel,
    }

    input_data = pd.DataFrame([record])

    prediction = model.predict(input_data)[0]
    scores = model.predict_proba(input_data)[0]

    print(f"\nFuel: {fuel}")
    print(f"Predicted outcome: {prediction}")
    print("\nModel class scores:")

    for class_name, score in sorted(
        zip(model.classes_, scores),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"  {class_name:<15} {score:.6f} ({score:.2%})")

print("\n" + "=" * 65)
print("Comparison completed.")
