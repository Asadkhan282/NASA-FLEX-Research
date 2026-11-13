
from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "data" / "raw" / "PSI-69_Experimental table_FLEX.csv"
OUTPUT = ROOT / "reports" / "oxygen_analysis"

OUTPUT.mkdir(parents=True, exist_ok=True)

OXYGEN = "O initial ambient composition; mole fraction"

OUTCOMES = ["Extinction", "Disruption", "Completion"]


def main():

    df = pd.read_csv(
        INPUT,
        encoding="cp1252",
        na_values=["-", "–", "—", ""],
    )

    df[OXYGEN] = pd.to_numeric(
        df[OXYGEN],
        errors="coerce",
    )

    groups = {}

    print("\nNASA FLEX OXYGEN ANALYSIS")
    print("=" * 55)

    for outcome in OUTCOMES:

        values = df.loc[
            df["Test end"] == outcome,
            OXYGEN,
        ].dropna()

        groups[outcome] = values.values

        print(f"\n{outcome}")
        print("Experiments:", len(values))
        print("Mean oxygen:", round(values.mean(), 5))
        print("Median oxygen:", round(values.median(), 5))
        print("Standard deviation:", round(values.std(), 5))
        print("Minimum:", round(values.min(), 5))
        print("Maximum:", round(values.max(), 5))

    valid_groups = [
        values for values in groups.values()
        if len(values) > 0
    ]

    statistic, p_value = kruskal(*valid_groups)

    all_values = np.concatenate(valid_groups)

    n = len(all_values)
    k = len(valid_groups)

    # Approximate Kruskal-Wallis effect size.
    epsilon_squared = max(
        0.0,
        (statistic - k + 1) / (n - k)
    )

    print("\nKRUSKAL-WALLIS TEST")
    print("=" * 55)

    print("H statistic:", round(statistic, 5))
    print("P-value:", f"{p_value:.8f}")
    print("Epsilon-squared:", round(epsilon_squared, 5))

    if p_value < 0.05:
        interpretation = (
            "Oxygen distributions differ significantly "
            "across at least one outcome group."
        )
    else:
        interpretation = (
            "No statistically significant difference in "
            "oxygen distributions was detected."
        )

    print("\nInterpretation:")
    print(interpretation)

    summary = {
        "sample_size": int(n),
        "group_statistics": {},
        "kruskal_wallis_H": float(statistic),
        "p_value": float(p_value),
        "epsilon_squared": float(epsilon_squared),
        "interpretation": interpretation,
        "limitation": (
            "This is an unadjusted observational comparison. "
            "It does not establish causation or isolate the "
            "independent effect of oxygen."
        ),
    }

    for outcome, values in groups.items():

        summary["group_statistics"][outcome] = {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)),
        }

    with open(
        OUTPUT / "oxygen_statistical_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    # Box plot
    plt.figure(figsize=(9, 6))

    plt.boxplot(
        valid_groups,
        tick_labels=OUTCOMES,
        showmeans=True,
    )

    plt.title("NASA FLEX: Oxygen Concentration by Outcome")
    plt.xlabel("Experimental outcome")
    plt.ylabel("Initial oxygen mole fraction")

    plt.tight_layout()

    plt.savefig(
        OUTPUT / "oxygen_outcome_comparison.png",
        dpi=200,
    )

    plt.close()

    print("\nFiles saved:")
    print(OUTPUT / "oxygen_statistical_results.json")
    print(OUTPUT / "oxygen_outcome_comparison.png")

    print("\nOXYGEN ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
