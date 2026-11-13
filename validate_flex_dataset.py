
import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# NASA PSI-69 FLEX DATA VALIDATION
# Flame in Freefall | NASA Space Apps 2026
# ============================================================

ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

INPUT_FILE = RAW_DIR / "PSI-69_Experimental table_FLEX.csv"
OUTPUT_FILE = PROCESSED_DIR / "flex_cleaned.csv"
REPORT_FILE = REPORT_DIR / "flex_data_quality_report.json"
SUMMARY_FILE = REPORT_DIR / "flex_data_quality_summary.txt"

EXPECTED_COLUMNS = [
    "FLEX Test #",
    "FLEX Identifier",
    "Test Date",
    "Test GMT",
    "Fuel",
    "Ambient pressure; mmHg",
    "O initial ambient composition; mole fraction",
    "N initial ambient composition; mole fraction",
    "CO initial ambient composition; mole fraction",
    "He initial ambient composition; mole fraction",
    "Droplet initial diameter; mm",
    "Visible flame extinction diameter; mm",
    "Burning rate; mm",
    "Burn time; s",
    "Test end",
]

NUMERIC_COLUMNS = [
    "FLEX Test #",
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

MEASUREMENT_COLUMNS = [
    "Droplet initial diameter; mm",
    "Visible flame extinction diameter; mm",
    "Burning rate; mm",
    "Burn time; s",
]

GAS_COLUMNS = [
    "O initial ambient composition; mole fraction",
    "N initial ambient composition; mole fraction",
    "CO initial ambient composition; mole fraction",
    "He initial ambient composition; mole fraction",
]

VALID_OUTCOMES = [
    "Extinction",
    "Disruption",
    "Completion",
]

# Recognize common missing-value dash representations.
MISSING_TOKENS = {
    "",
    "-",
    "–",
    "—",
    "−",
    "â€“",
    "â€”",
    "nan",
    "none",
    "null",
}

NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def parse_measurement(value):
    """
    Return (numeric_value, is_approximate).

    Examples:
        "3.5"  -> (3.5, False)
        "~6"   -> (6.0, True)
        "–"    -> (None, False)
        ""     -> (None, False)
    """
    if pd.isna(value):
        return None, False

    text = str(value).strip()

    if text.lower() in MISSING_TOKENS:
        return None, False

    approximate = "~" in text
    match = NUMBER_PATTERN.search(text)

    if not match:
        return None, False

    return float(match.group()), approximate


def load_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"NASA dataset not found: {INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE,
        encoding="cp1252",
        dtype=str,
        keep_default_na=False,
    )

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in data.columns
    ]

    unexpected_columns = [
        column for column in data.columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if unexpected_columns:
        print("WARNING: Unexpected columns:", unexpected_columns)

    return data


def clean_dataset(original):
    cleaned = original.copy()
    approximate_flags = {}

    for column in NUMERIC_COLUMNS:
        parsed = cleaned[column].apply(parse_measurement)

        cleaned[column] = parsed.apply(
            lambda item: item[0]
        )

        approximate_flags[column] = parsed.apply(
            lambda item: item[1]
        )

    for column in ["Fuel", "Test end", "FLEX Identifier"]:
        cleaned[column] = (
            cleaned[column].astype("string").str.strip()
        )

    # Keep the same 15-column CSV structure.
    # Approximation flags are recorded in the JSON report instead.
    return cleaned, approximate_flags


def generate_report(original, cleaned, approximate_flags):
    report = {
        "project": "Flame in Freefall",
        "dataset": "NASA PSI-69 FLEX",
        "source_file": INPUT_FILE.name,
        "rows": int(len(cleaned)),
        "columns": int(len(cleaned.columns)),
        "duplicate_rows": int(cleaned.duplicated().sum()),
        "duplicate_test_numbers": int(
            cleaned["FLEX Test #"].duplicated().sum()
        ),
        "missing_values": {
            column: int(cleaned[column].isna().sum())
            for column in cleaned.columns
        },
        "approximate_measurements": {},
        "fuel_distribution": {
            str(key): int(value)
            for key, value in cleaned["Fuel"].value_counts(
                dropna=False
            ).items()
        },
        "outcome_distribution": {
            str(key): int(value)
            for key, value in cleaned["Test end"].value_counts(
                dropna=False
            ).items()
        },
    }

    # Record approximate measurements separately.
    for column in MEASUREMENT_COLUMNS:
        flags = approximate_flags[column]
        positions = flags[flags].index

        records = []

        for index in positions:
            records.append({
                "test_number": int(cleaned.loc[index, "FLEX Test #"]),
                "flex_identifier": str(
                    cleaned.loc[index, "FLEX Identifier"]
                ),
                "estimated_numeric_value": float(
                    cleaned.loc[index, column]
                ),
            })

        report["approximate_measurements"][column] = {
            "count": len(records),
            "records": records,
        }

    observed_outcomes = set(
        cleaned["Test end"].dropna().unique()
    )

    report["unexpected_outcomes"] = sorted(
        observed_outcomes - set(VALID_OUTCOMES)
    )

    report["test_number_range"] = {
        "minimum": int(cleaned["FLEX Test #"].min()),
        "maximum": int(cleaned["FLEX Test #"].max()),
    }

    test_numbers = set(
        cleaned["FLEX Test #"].dropna().astype(int)
    )

    report["missing_test_numbers"] = sorted(
        set(
            range(
                int(cleaned["FLEX Test #"].min()),
                int(cleaned["FLEX Test #"].max()) + 1,
            )
        ) - test_numbers
    )

    report["zero_pressure_records"] = (
        cleaned.loc[
            cleaned["Ambient pressure; mmHg"] == 0,
            ["FLEX Test #", "FLEX Identifier"],
        ]
        .astype(str)
        .to_dict(orient="records")
    )

    gas_totals = cleaned[GAS_COLUMNS].sum(
        axis=1,
        min_count=len(GAS_COLUMNS),
    )

    report["gas_composition_totals"] = {
        "valid_records": int(gas_totals.notna().sum()),
        "mean": float(gas_totals.mean()),
        "minimum": float(gas_totals.min()),
        "maximum": float(gas_totals.max()),
        "below_1": int((gas_totals < 1.0).sum()),
        "above_1": int((gas_totals > 1.0).sum()),
        "outside_0.99_to_1.01": int(
            ((gas_totals < 0.99) | (gas_totals > 1.01)).sum()
        ),
    }

    report["notes"] = [
        "Original NASA dataset was not modified.",
        "Unicode dash entries are treated as unavailable measurements.",
        "Approximate numeric measurements are retained as numeric estimates.",
        "Approximation flags and original test identifiers are recorded separately.",
        "Missing measurements are not replaced with zero.",
        "Approximate values should be interpreted as estimates, not exact measurements.",
        "Zero pressure measurements require scientific verification.",
        "The local CSV contains 274 rows; the NASA portal displays 275 records.",
        "No missing 275th experiment has been fabricated.",
    ]

    return report


def save_outputs(cleaned, report):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    summary = [
        "NASA PSI-69 FLEX DATA QUALITY REPORT",
        "=" * 55,
        f"Dataset rows: {report['rows']}",
        f"Dataset columns: {report['columns']}",
        f"Duplicate rows: {report['duplicate_rows']}",
        f"Duplicate test numbers: {report['duplicate_test_numbers']}",
        "",
        "FUEL DISTRIBUTION",
        str(report["fuel_distribution"]),
        "",
        "EXPERIMENT OUTCOMES",
        str(report["outcome_distribution"]),
        "",
        "MISSING VALUES",
    ]

    for column, count in report["missing_values"].items():
        summary.append(f"{column}: {count}")

    summary.extend([
        "",
        "APPROXIMATE MEASUREMENTS",
    ])

    for column, details in report["approximate_measurements"].items():
        summary.append(
            f"{column}: {details['count']} approximate entries"
        )

        for record in details["records"]:
            summary.append(
                f"  Test {record['test_number']} "
                f"({record['flex_identifier']}): "
                f"approximately {record['estimated_numeric_value']}"
            )

    summary.extend([
        "",
        "GAS COMPOSITION TOTALS",
        str(report["gas_composition_totals"]),
        "",
        "ZERO PRESSURE RECORDS",
        str(report["zero_pressure_records"]),
        "",
        "MISSING TEST NUMBERS",
        str(report["missing_test_numbers"]),
        "",
        "IMPORTANT",
        "The local CSV contains 274 rows.",
        "The NASA portal displays 275 records.",
        "No missing 275th experiment has been fabricated.",
        "Approximate measurements are estimates, not exact values.",
    ])

    SUMMARY_FILE.write_text(
        "\n".join(summary),
        encoding="utf-8",
    )


def main():
    print("\nNASA PSI-69 FLEX VALIDATION")
    print("=" * 55)

    original = load_dataset()
    print("Original dataset loaded.")

    cleaned, approximate_flags = clean_dataset(original)
    report = generate_report(
        original,
        cleaned,
        approximate_flags,
    )

    save_outputs(cleaned, report)

    print("\nVALIDATION RESULTS")
    print("=" * 55)
    print("Rows:", report["rows"])
    print("Columns:", report["columns"])
    print("Duplicate rows:", report["duplicate_rows"])
    print(
        "Duplicate test numbers:",
        report["duplicate_test_numbers"],
    )

    print("\nOutcome distribution:")
    for outcome, count in report["outcome_distribution"].items():
        print(f"{outcome}: {count}")

    print("\nMissing values:")
    for column, count in report["missing_values"].items():
        if count > 0:
            print(f"{column}: {count}")

    print("\nApproximate measurements:")
    for column, details in report["approximate_measurements"].items():
        if details["count"] > 0:
            print(f"{column}: {details['count']}")
            for record in details["records"]:
                print(
                    f"  Test {record['test_number']}: "
                    f"approximately {record['estimated_numeric_value']}"
                )

    print("\nGas composition totals:")
    print(report["gas_composition_totals"])

    print("\nFiles created:")
    print(OUTPUT_FILE)
    print(REPORT_FILE)
    print(SUMMARY_FILE)

    print("\nVALIDATION PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
