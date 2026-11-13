
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "data" / "raw" / "PSI-69_Experimental table_FLEX.csv"
OUTPUT = ROOT / "reports" / "flex_anomaly_audit.csv"

df = pd.read_csv(
    INPUT,
    encoding="cp1252",
    na_values=["-", "–", "—", ""],
)

numeric_columns = [
    "Ambient pressure; mmHg",
    "O initial ambient composition; mole fraction",
    "N initial ambient composition; mole fraction",
    "CO initial ambient composition; mole fraction",
    "He initial ambient composition; mole fraction",
    "Droplet initial diameter; mm",
    "Visible flame extinction diameter; mm",
    "Burning rate; mm",
    "Burn time; s",
]

raw = df.copy()
clean = df.copy()

for col in numeric_columns:
    clean[col] = pd.to_numeric(
        clean[col].astype("string").str.strip(),
        errors="coerce",
    )

composition_columns = [
    "O initial ambient composition; mole fraction",
    "N initial ambient composition; mole fraction",
    "CO initial ambient composition; mole fraction",
    "He initial ambient composition; mole fraction",
]

composition_sum = clean[composition_columns].sum(
    axis=1,
    min_count=4,
)

issues = []

for index, row in clean.iterrows():
    flags = []

    if row["Ambient pressure; mmHg"] == 0:
        flags.append("ZERO_AMBIENT_PRESSURE")

    if pd.notna(composition_sum.iloc[index]):
        if abs(composition_sum.iloc[index] - 1.0) > 0.02:
            flags.append("COMPOSITION_SUM_OUTSIDE_TOLERANCE")

    for col in numeric_columns:
        if pd.notna(raw.at[index, col]) and pd.isna(row[col]):
            flags.append(f"NON_NUMERIC_VALUE:{col}")

    if flags:
        record = {
            "FLEX Test #": row["FLEX Test #"],
            "FLEX Identifier": row["FLEX Identifier"],
            "Fuel": row["Fuel"],
            "Test end": row["Test end"],
            "Composition sum": composition_sum.iloc[index],
            "Issues": " | ".join(flags),
        }

        for col in numeric_columns:
            record[f"Original: {col}"] = raw.at[index, col]

        issues.append(record)

audit = pd.DataFrame(issues)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUTPUT, index=False)

print("=" * 55)
print("NASA FLEX ANOMALY INVESTIGATION")
print("=" * 55)

print("Total flagged experiments:", len(audit))

if not audit.empty:
    print("\nFlagged experiments:")
    print(
        audit[
            ["FLEX Test #", "FLEX Identifier", "Issues"]
        ].to_string(index=False)
    )

print("\nAudit saved to:")
print(OUTPUT)
