
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

FILE = ROOT / "data" / "raw" / "PSI-69_Experimental table_FLEX.csv"

df = pd.read_csv(FILE, encoding="cp1252", dtype=str)

test_ids = [69, 75, 76, 79, 80, 81, 114, 223, 224]

selected = df[
    pd.to_numeric(df["FLEX Test #"], errors="coerce").isin(test_ids)
].copy()

gas_columns = [
    "O initial ambient composition; mole fraction",
    "N initial ambient composition; mole fraction",
    "CO initial ambient composition; mole fraction",
    "He initial ambient composition; mole fraction",
]

print("\nNASA FLEX FLAGGED RECORDS")
print("=" * 75)

for _, row in selected.iterrows():
    print(f"\nTEST: {row['FLEX Test #']} | {row['FLEX Identifier']}")
    print(f"Fuel: {row['Fuel']}")
    print(f"Outcome: {row['Test end']}")
    print(f"Pressure: {row['Ambient pressure; mmHg']} mmHg")

    print("\nGas composition:")
    total = 0.0
    valid = True

    for col in gas_columns:
        value = row[col]
        print(f"  {col}: {value}")

        try:
            total += float(value)
        except (ValueError, TypeError):
            valid = False

    if valid:
        print(f"  Composition total: {total:.5f}")
    else:
        print("  Composition total: unavailable")

    print(f"Initial droplet diameter: {row['Droplet initial diameter; mm']}")
    print(f"Extinction diameter: {row['Visible flame extinction diameter; mm']}")
    print(f"Burning rate: {row['Burning rate; mm']}")
    print(f"Burn time: {row['Burn time; s']}")

print("\nINSPECTION COMPLETED")
