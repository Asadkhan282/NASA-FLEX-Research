
from pathlib import Path
import re
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# NASA PSI-69 FLEX: SCIENTIFIC EXPLORATORY DATA ANALYSIS
# Project: Flame in Freefall
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "data" / "raw" / "PSI-69_Experimental table_FLEX.csv"
OUTPUT = ROOT / "reports" / "eda"
OUTPUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = ["Extinction", "Disruption", "Completion"]

GAS_COLUMNS = [
    "O initial ambient composition; mole fraction",
    "N initial ambient composition; mole fraction",
    "CO initial ambient composition; mole fraction",
    "He initial ambient composition; mole fraction",
]

NUMERIC_COLUMNS = [
    "FLEX Test #",
    "Ambient pressure; mmHg",
    *GAS_COLUMNS,
    "Droplet initial diameter; mm",
    "Visible flame extinction diameter; mm",
    "Burning rate; mm",
    "Burn time; s",
]


def parse_measurement(value):
    """Extract numeric estimate while preserving approximation flags."""
    if pd.isna(value):
        return np.nan, False

    value = str(value).strip()

    if value in ["", "-", "–", "—", "nan"]:
        return np.nan, False

    approximate = "~" in value

    match = re.search(r"[-+]?\d*\.?\d+", value)

    if not match:
        return np.nan, False

    return float(match.group()), approximate


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(
        INPUT,
        encoding="cp1252",
        dtype=str,
        keep_default_na=False,
    )

    approximate_flags = {}

    for col in NUMERIC_COLUMNS:
        parsed = df[col].apply(parse_measurement)

        df[col] = parsed.apply(lambda x: x[0])
        approximate_flags[col] = parsed.apply(lambda x: x[1])

    df["Burn time approximate"] = approximate_flags[
        "Burn time; s"
    ]

    df["Composition total"] = df[GAS_COLUMNS].sum(
        axis=1,
        min_count=4,
    )

    return df


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def plot_outcome_distribution(df):
    counts = df["Test end"].value_counts().reindex(
        OUTCOMES,
        fill_value=0,
    )

    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values)

    plt.title("NASA FLEX: Experimental Outcome Distribution")
    plt.xlabel("Experimental outcome")
    plt.ylabel("Number of experiments")

    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )

    save_plot("01_outcome_distribution.png")


def plot_fuel_outcomes(df):
    table = pd.crosstab(
        df["Fuel"],
        df["Test end"],
    ).reindex(columns=OUTCOMES, fill_value=0)

    table.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
    )

    plt.title("Combustion Outcomes by Fuel")
    plt.xlabel("Fuel")
    plt.ylabel("Number of experiments")
    plt.xticks(rotation=0)
    plt.legend(title="Outcome")

    save_plot("02_fuel_outcomes.png")


def plot_oxygen_outcomes(df):
    oxygen = GAS_COLUMNS[0]

    groups = [
        df.loc[
            (df["Test end"] == outcome)
            & df[oxygen].notna(),
            oxygen,
        ].values
        for outcome in OUTCOMES
    ]

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        groups,
        tick_labels=OUTCOMES,
    )

    plt.title("Initial Oxygen Fraction by Combustion Outcome")
    plt.xlabel("Experimental outcome")
    plt.ylabel("Initial oxygen mole fraction")

    save_plot("03_oxygen_outcomes.png")


def plot_pressure_burntime(df):
    valid = df[
        (df["Ambient pressure; mmHg"] > 0)
        & (df["Burn time; s"] > 0)
    ].copy()

    plt.figure(figsize=(8, 5))

    for fuel in valid["Fuel"].unique():
        subset = valid[valid["Fuel"] == fuel]

        plt.scatter(
            subset["Ambient pressure; mmHg"],
            subset["Burn time; s"],
            label=fuel,
            alpha=0.75,
        )

    approximate = valid[valid["Burn time approximate"]]

    if not approximate.empty:
        plt.scatter(
            approximate["Ambient pressure; mmHg"],
            approximate["Burn time; s"],
            marker="x",
            s=100,
            label="Approximate burn time",
        )

    plt.title("Ambient Pressure vs Burn Time")
    plt.xlabel("Ambient pressure (mmHg)")
    plt.ylabel("Burn time (seconds)")
    plt.legend()

    save_plot("04_pressure_burntime.png")


def plot_droplet_behavior(df):
    valid = df[
        (df["Droplet initial diameter; mm"] > 0)
        & (df["Visible flame extinction diameter; mm"] > 0)
    ]

    plt.figure(figsize=(8, 5))

    for fuel in valid["Fuel"].unique():
        subset = valid[valid["Fuel"] == fuel]

        plt.scatter(
            subset["Droplet initial diameter; mm"],
            subset["Visible flame extinction diameter; mm"],
            label=fuel,
            alpha=0.75,
        )

    plt.title("Initial Droplet Diameter vs Extinction Diameter")
    plt.xlabel("Initial droplet diameter (mm)")
    plt.ylabel("Visible flame extinction diameter (mm)")
    plt.legend()

    save_plot("05_droplet_behavior.png")


def plot_gas_correlations(df):
    labels = ["Oxygen", "Nitrogen", "Carbon monoxide", "Helium"]

    gas_data = df[GAS_COLUMNS].copy()
    gas_data.columns = labels

    correlation = gas_data.corr()

    plt.figure(figsize=(7, 6))

    image = plt.imshow(
        correlation,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    plt.colorbar(image, label="Pearson correlation")

    plt.xticks(range(4), labels, rotation=30, ha="right")
    plt.yticks(range(4), labels)

    for i in range(4):
        for j in range(4):
            value = correlation.iloc[i, j]

            if pd.notna(value):
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                )

    plt.title("Ambient Gas Composition Correlation")

    save_plot("06_gas_correlation.png")


def plot_burning_rate(df):
    valid = df[
        (df["Burning rate; mm"] > 0)
        & df["Test end"].isin(OUTCOMES)
    ]

    groups = [
        valid.loc[
            valid["Test end"] == outcome,
            "Burning rate; mm",
        ].values
        for outcome in OUTCOMES
    ]

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        groups,
        tick_labels=OUTCOMES,
    )

    plt.title("Burning Rate by Experimental Outcome")
    plt.xlabel("Experimental outcome")
    plt.ylabel("Burning rate (mm/s, as recorded)")

    save_plot("07_burning_rate.png")


def generate_summary(df):
    summary = {
        "dataset": "NASA PSI-69 FLEX",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "outcomes": {
            str(k): int(v)
            for k, v in df["Test end"].value_counts().items()
        },
        "fuels": {
            str(k): int(v)
            for k, v in df["Fuel"].value_counts().items()
        },
        "approximate_burn_times": int(
            df["Burn time approximate"].sum()
        ),
        "zero_pressure_records": int(
            (df["Ambient pressure; mmHg"] == 0).sum()
        ),
        "missing_measurements": {
            col: int(df[col].isna().sum())
            for col in NUMERIC_COLUMNS
        },
        "composition_totals": {
            "mean": float(df["Composition total"].mean()),
            "minimum": float(df["Composition total"].min()),
            "maximum": float(df["Composition total"].max()),
            "records_below_1": int(
                (df["Composition total"] < 1.0 - 1e-9).sum()
            ),
        },
        "interpretation_warning": (
            "Exploratory associations only. "
            "These results do not establish causation."
        ),
    }

    with open(
        OUTPUT / "eda_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    return summary


def main():
    print("\nNASA FLEX SCIENTIFIC EDA")
    print("=" * 55)

    df = load_data()

    print("Dataset loaded:", len(df), "experiments")

    print("\nGenerating scientific visualizations...")

    plot_outcome_distribution(df)
    plot_fuel_outcomes(df)
    plot_oxygen_outcomes(df)
    plot_pressure_burntime(df)
    plot_droplet_behavior(df)
    plot_gas_correlations(df)
    plot_burning_rate(df)

    summary = generate_summary(df)

    print("\nEDA SUMMARY")
    print("=" * 55)

    print("Experiments:", summary["rows"])
    print("Approximate burn times:", summary["approximate_burn_times"])
    print("Zero-pressure records:", summary["zero_pressure_records"])

    print("\nComposition totals:")
    print(summary["composition_totals"])

    print("\nVISUALIZATION FILES")
    print("=" * 55)

    for file in sorted(OUTPUT.glob("*.png")):
        print(file.name)

    print("\nEDA COMPLETED SUCCESSFULLY")
    print("Output directory:", OUTPUT)


if __name__ == "__main__":
    main()
