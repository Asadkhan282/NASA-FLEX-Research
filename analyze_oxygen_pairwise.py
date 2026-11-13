
from pathlib import Path
from itertools import combinations
import json

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "data" / "raw" / "PSI-69_Experimental table_FLEX.csv"
OUTPUT = ROOT / "reports" / "oxygen_analysis"

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

    groups = {
        outcome: df.loc[
            df["Test end"] == outcome,
            OXYGEN,
        ].dropna().values
        for outcome in OUTCOMES
    }

    comparisons = []

    print("\nNASA FLEX: PAIRWISE OXYGEN ANALYSIS")
    print("=" * 65)

    for first, second in combinations(OUTCOMES, 2):

        a = groups[first]
        b = groups[second]

        result = mannwhitneyu(
            a,
            b,
            alternative="two-sided",
            method="asymptotic",
        )

        comparisons.append({
            "group_1": first,
            "group_2": second,
            "n_1": len(a),
            "n_2": len(b),
            "median_1": float(np.median(a)),
            "median_2": float(np.median(b)),
            "median_difference": float(
                np.median(b) - np.median(a)
            ),
            "U_statistic": float(result.statistic),
            "raw_p_value": float(result.pvalue),
        })

    raw_p = [
        item["raw_p_value"]
        for item in comparisons
    ]

    reject, adjusted_p, _, _ = multipletests(
        raw_p,
        alpha=0.05,
        method="holm",
    )

    for i, item in enumerate(comparisons):

        item["holm_adjusted_p"] = float(adjusted_p[i])
        item["significant_after_correction"] = bool(reject[i])

        print(
            f"\n{item['group_1']} vs {item['group_2']}"
        )

        print("Sample sizes:", item["n_1"], item["n_2"])
        print("Medians:", item["median_1"], item["median_2"])
        print("Median difference:", item["median_difference"])
        print("U statistic:", item["U_statistic"])
        print("Raw p-value:", f"{item['raw_p_value']:.8g}")
        print("Holm-adjusted p:", f"{item['holm_adjusted_p']:.8g}")
        print("Significant:", item["significant_after_correction"])

    with open(
        OUTPUT / "oxygen_pairwise_results.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "method": "Pairwise Mann-Whitney U with Holm correction",
                "comparisons": comparisons,
            },
            file,
            indent=4,
        )

    print("\nResults saved:")
    print(OUTPUT / "oxygen_pairwise_results.json")

    print("\nPAIRWISE OXYGEN ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
