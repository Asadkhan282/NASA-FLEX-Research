
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import chi2, norm


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "data" / "processed" / "flex_cleaned.csv"
OUTPUT_DIR = ROOT / "reports" / "multivariable_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Test end"
FUEL = "Fuel"
OXYGEN = "O initial ambient composition; mole fraction"

REFERENCE_OUTCOME = "Extinction"
OUTCOME_ORDER = ["Extinction", "Disruption", "Completion"]

PREDICTORS = ["oxygen", "methanol"]

CONFIDENCE_LEVEL = 0.95
Z_VALUE = norm.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fit_multinomial(y, X):
    """
    Fit a multinomial logistic regression model.
    """

    X = sm.add_constant(X, has_constant="add")

    model = sm.MNLogit(y, X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(
            method="newton",
            maxiter=500,
            disp=False
        )

    return result


def likelihood_ratio_test(full_model, reduced_model):
    """
    Likelihood-ratio test for nested multinomial models.
    """

    statistic = 2 * (full_model.llf - reduced_model.llf)

    # Protect against tiny negative floating-point differences.
    statistic = max(0.0, float(statistic))

    degrees_of_freedom = int(
        round(full_model.df_model - reduced_model.df_model)
    )

    p_value = float(
        chi2.sf(statistic, degrees_of_freedom)
    )

    return {
        "likelihood_ratio_chi_square": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value
    }


def calculate_odds_ratio(coefficient, standard_error, scale=1.0):
    """
    Calculate odds ratio and Wald confidence interval.

    scale=1.0 for a one-unit predictor increase.
    scale=0.01 for a 0.01 increase in oxygen mole fraction.
    """

    scaled_coefficient = float(coefficient) * scale
    scaled_se = float(standard_error) * scale

    lower_log = scaled_coefficient - Z_VALUE * scaled_se
    upper_log = scaled_coefficient + Z_VALUE * scaled_se

    return {
        "odds_ratio": float(np.exp(scaled_coefficient)),
        "ci_lower": float(np.exp(lower_log)),
        "ci_upper": float(np.exp(upper_log))
    }


def safe_float(value):
    """
    Convert NumPy/Pandas values to JSON-safe Python floats.
    """

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("\nNASA FLEX MULTIVARIABLE ANALYSIS")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required_columns = [TARGET, FUEL, OXYGEN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # 2. PREPARE COMPLETE-CASE DATA
    # --------------------------------------------------------

    analysis = df[required_columns].copy()

    analysis[TARGET] = analysis[TARGET].astype(str).str.strip()
    analysis[FUEL] = analysis[FUEL].astype(str).str.strip()

    analysis[OXYGEN] = pd.to_numeric(
        analysis[OXYGEN],
        errors="coerce"
    )

    analysis = analysis[
        analysis[TARGET].isin(OUTCOME_ORDER)
        & analysis[FUEL].isin(["Methanol", "Heptane"])
    ].copy()

    analysis = analysis.dropna(
        subset=[TARGET, FUEL, OXYGEN]
    ).copy()

    if analysis.empty:
        raise ValueError("No complete-case observations remain.")

    # Explicit outcome coding:
    # Extinction = 0 (reference)
    # Disruption = 1
    # Completion = 2

    outcome_mapping = {
        "Extinction": 0,
        "Disruption": 1,
        "Completion": 2
    }

    analysis["outcome_code"] = (
        analysis[TARGET].map(outcome_mapping).astype(int)
    )

    # Heptane is the fuel reference category.
    # Methanol = 1, Heptane = 0.

    analysis["methanol"] = (
        analysis[FUEL] == "Methanol"
    ).astype(int)

    analysis["oxygen"] = analysis[OXYGEN]

    y = analysis["outcome_code"]

    X = analysis[PREDICTORS].copy()

    print(f"\nComplete-case experiments: {len(analysis)}")

    print("\nOutcome counts:")
    print(
        analysis[TARGET].value_counts()
        .reindex(OUTCOME_ORDER, fill_value=0)
    )

    print("\nFuel counts:")
    print(analysis[FUEL].value_counts())

    # --------------------------------------------------------
    # 3. FIT FULL MULTINOMIAL MODEL
    # --------------------------------------------------------

    full_model = fit_multinomial(y, X)

    print("\nFULL MULTINOMIAL MODEL")
    print("=" * 65)
    print(full_model.summary())

    print("\nModel converged:", full_model.mle_retvals.get(
        "converged", False
    ))

    if not full_model.mle_retvals.get("converged", False):
        raise RuntimeError(
            "The full multinomial model did not converge."
        )

    # Internal MNLogit columns are 0 and 1:
    # column 0 = Disruption vs Extinction
    # column 1 = Completion vs Extinction

    comparison_names = {
        0: "Disruption_vs_Extinction",
        1: "Completion_vs_Extinction"
    }

    # --------------------------------------------------------
    # 4. ODDS RATIOS AND 95% CONFIDENCE INTERVALS
    # --------------------------------------------------------

    print("\nODDS RATIOS AND 95% CONFIDENCE INTERVALS")
    print("=" * 65)

    odds_ratio_results = {}

    for outcome_column, comparison_name in comparison_names.items():

        print(f"\n{comparison_name}")

        odds_ratio_results[comparison_name] = {}

        for predictor in PREDICTORS:

            # IMPORTANT:
            # Use positional column access for the outcome equation.
            # This avoids version-dependent confidence-interval indexing.

            coefficient = full_model.params.loc[
                predictor
            ].iloc[outcome_column]

            standard_error = full_model.bse.loc[
                predictor
            ].iloc[outcome_column]

            # Report oxygen per 0.01 mole-fraction increase.
            # Report methanol relative to Heptane.

            if predictor == "oxygen":
                scale = 0.01
                display_name = "Oxygen per 0.01 increase"
            else:
                scale = 1.0
                display_name = "Methanol vs Heptane"

            result = calculate_odds_ratio(
                coefficient,
                standard_error,
                scale
            )

            p_value = full_model.pvalues.loc[
                predictor
            ].iloc[outcome_column]

            odds_ratio_results[comparison_name][predictor] = {
                "display_name": display_name,
                "coefficient": safe_float(coefficient),
                "standard_error": safe_float(standard_error),
                "odds_ratio": result["odds_ratio"],
                "ci_95_lower": result["ci_lower"],
                "ci_95_upper": result["ci_upper"],
                "p_value": safe_float(p_value),
                "scale": scale
            }

            print(f"\n  {display_name}")
            print(f"    Coefficient: {coefficient:.6f}")
            print(f"    Standard error: {standard_error:.6f}")
            print(f"    Odds ratio: {result['odds_ratio']:.4f}")
            print(
                "    95% CI: "
                f"[{result['ci_lower']:.4f}, "
                f"{result['ci_upper']:.4f}]"
            )
            print(f"    p-value: {p_value:.8g}")

    # --------------------------------------------------------
    # 5. LIKELIHOOD-RATIO TESTS
    # --------------------------------------------------------

    print("\nADJUSTED PREDICTOR CONTRIBUTION")
    print("=" * 65)

    lrt_results = {}

    # Full model versus oxygen-only model:
    # Tests the contribution of methanol after adjusting for oxygen.

    X_oxygen_only = analysis[["oxygen"]]

    oxygen_only_model = fit_multinomial(
        y,
        X_oxygen_only
    )

    methanol_test = likelihood_ratio_test(
        full_model,
        oxygen_only_model
    )

    lrt_results["methanol_adjusted_for_oxygen"] = methanol_test

    print("\nMethanol adjusted for oxygen:")
    print(methanol_test)

    # Full model versus methanol-only model:
    # Tests the contribution of oxygen after adjusting for fuel.

    X_methanol_only = analysis[["methanol"]]

    methanol_only_model = fit_multinomial(
        y,
        X_methanol_only
    )

    oxygen_test = likelihood_ratio_test(
        full_model,
        methanol_only_model
    )

    lrt_results["oxygen_adjusted_for_methanol"] = oxygen_test

    print("\nOxygen adjusted for methanol:")
    print(oxygen_test)

    # --------------------------------------------------------
    # 6. MODEL FIT STATISTICS
    # --------------------------------------------------------

    pseudo_r_squared = float(full_model.prsquared)

    model_statistics = {
        "observations": int(len(analysis)),
        "outcome_reference": REFERENCE_OUTCOME,
        "fuel_reference": "Heptane",
        "predictors": PREDICTORS,
        "converged": bool(
            full_model.mle_retvals.get("converged", False)
        ),
        "log_likelihood": safe_float(full_model.llf),
        "null_log_likelihood": safe_float(full_model.llnull),
        "pseudo_r_squared": pseudo_r_squared,
        "aic": safe_float(full_model.aic),
        "bic": safe_float(full_model.bic),
        "overall_likelihood_ratio_p_value": safe_float(
            full_model.llr_pvalue
        )
    }

    print("\nMODEL FIT STATISTICS")
    print("=" * 65)

    for key, value in model_statistics.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # 7. SAVE RESULTS
    # --------------------------------------------------------

    output = {
        "analysis": "NASA FLEX multinomial logistic regression",
        "model_statistics": model_statistics,
        "odds_ratios": odds_ratio_results,
        "adjusted_likelihood_ratio_tests": lrt_results,
        "outcome_counts": {
            str(key): int(value)
            for key, value in (
                analysis[TARGET].value_counts().items()
            )
        },
        "fuel_counts": {
            str(key): int(value)
            for key, value in analysis[FUEL].value_counts().items()
        },
        "method_notes": [
            "Extinction is the reference outcome.",
            "Heptane is the reference fuel.",
            "Oxygen odds ratios are reported per 0.01 mole-fraction increase.",
            "Confidence intervals use the 95% Wald method.",
            "Analyses use complete cases for the selected variables.",
            "Associations do not establish causation."
        ]
    }

    json_path = OUTPUT_DIR / "multivariable_results.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Save the complete-case analysis dataset too.
    analysis_path = OUTPUT_DIR / "multivariable_complete_cases.csv"

    analysis.to_csv(
        analysis_path,
        index=False
    )

    # Save a compact odds-ratio table.
    table_rows = []

    for comparison, predictors in odds_ratio_results.items():

        for predictor, values in predictors.items():

            table_rows.append({
                "Comparison": comparison,
                "Predictor": values["display_name"],
                "Odds_Ratio": values["odds_ratio"],
                "CI_95_Lower": values["ci_95_lower"],
                "CI_95_Upper": values["ci_95_upper"],
                "P_Value": values["p_value"]
            })

    odds_ratio_table = pd.DataFrame(table_rows)

    table_path = OUTPUT_DIR / "odds_ratio_table.csv"

    odds_ratio_table.to_csv(
        table_path,
        index=False
    )

    print("\nOUTPUT FILES")
    print("=" * 65)
    print(json_path)
    print(analysis_path)
    print(table_path)

    print("\nMULTIVARIABLE ANALYSIS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
