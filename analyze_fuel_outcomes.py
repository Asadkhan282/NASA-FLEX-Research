
from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "data" / "raw" / "PSI-69_Experimental table_FLEX.csv"
OUTPUT = ROOT / "reports" / "fuel_analysis"

OUTPUT.mkdir(parents=True, exist_ok=True)

OUTCOMES = ["Extinction", "Disruption", "Completion"]


def main():

    df = pd.read_csv(INPUT, encoding="cp1252")

    table = pd.crosstab(
        df["Fuel"],
        df["Test end"]
    ).reindex(columns=OUTCOMES, fill_value=0)

    percentages = table.div(
        table.sum(axis=1),
        axis=0
    ) * 100

    chi2, p_value, degrees, expected = chi2_contingency(table)

    n = table.to_numpy().sum()
    rows, cols = table.shape

    cramers_v = np.sqrt(
        chi2 / (n * min(rows - 1, cols - 1))
    )

    expected_table = pd.DataFrame(
        expected,
        index=table.index,
        columns=table.columns
    )

    residuals = (
        table - expected_table
    ) / np.sqrt(expected_table)

    results = {
        "sample_size": int(n),
        "observed_counts": table.to_dict(),
        "within_fuel_percentages": percentages.round(3).to_dict(),
        "chi_square": float(chi2),
        "degrees_of_freedom": int(degrees),
        "p_value": float(p_value),
        "cramers_v": float(cramers_v),
        "expected_counts": expected_table.round(3).to_dict(),
        "standardized_residuals": residuals.round(3).to_dict(),
        "interpretation": (
            "Statistically significant association at 0.05 level"
            if p_value < 0.05
            else "No statistically significant association at 0.05 level"
        ),
        "limitation": (
            "Observational association does not establish causation. "
            "Experimental conditions may differ between fuel groups."
        )
    }

    print("\nNASA FLEX FUEL OUTCOME ANALYSIS")
    print("=" * 55)

    print("\nObserved counts:")
    print(table)

    print("\nWithin-fuel percentages:")
    print(percentages.round(2))

    print("\nChi-square:", round(chi2, 4))
    print("Degrees of freedom:", degrees)
    print("P-value:", f"{p_value:.8f}")
    print("Cramer's V:", round(cramers_v, 4))

    print("\nExpected counts:")
    print(expected_table.round(2))

    print("\nStandardized residuals:")
    print(residuals.round(2))

    print("\nInterpretation:")
    print(results["interpretation"])

    with open(
        OUTPUT / "fuel_statistical_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)

    percentages.plot(
        kind="bar",
        stacked=True,
        figsize=(9, 6),
    )

    plt.title("NASA FLEX: Outcome Proportions by Fuel")
    plt.xlabel("Fuel")
    plt.ylabel("Percentage of experiments")
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.legend(title="Outcome")
    plt.tight_layout()

    plt.savefig(
        OUTPUT / "fuel_outcome_percentages.png",
        dpi=200
    )

    plt.close()

    print("\nResults saved in:", OUTPUT)
    print("\nFUEL ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
