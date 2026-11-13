
from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# NASA FLEX RESEARCH DASHBOARD
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "processed" / "flex_cleaned.csv"

EDA_DIR = ROOT / "reports" / "eda"
PRED_DIR = ROOT / "reports" / "prediction"
MODEL_DIR = ROOT / "reports" / "model_analysis"
STATS_DIR = ROOT / "reports" / "multivariable_analysis"
DEPLOYMENT_MODEL_FILE = ROOT / "models" / "flex_random_forest.joblib"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NASA FLEX Research Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stMetric"] {
        background-color: #171d29;
        border: 1px solid #293449;
        padding: 16px;
        border-radius: 12px;
    }

    .research-note {
        background-color: #182438;
        border-left: 4px solid #4b9cff;
        padding: 14px;
        border-radius: 6px;
        margin: 12px 0;
    }

    .warning-note {
        background-color: #332719;
        border-left: 4px solid #e5a33d;
        padding: 14px;
        border-radius: 6px;
        margin: 12px 0;
    }

    .small-muted {
        color: #aab4c5;
        font-size: 0.9rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_deployment_model(path):
    return joblib.load(path)


def available_file(path):
    return Path(path).exists()


def show_missing_file(path):
    st.error(f"Required project file not found: {path}")


required_files = [
    DATA_FILE,
    DEPLOYMENT_MODEL_FILE,
    PRED_DIR / "model_comparison_summary.csv",
    PRED_DIR / "out_of_fold_predictions.csv",
    EDA_DIR / "eda_summary.json",
    MODEL_DIR / "model_error_analysis_summary.json",
    MODEL_DIR / "date_grouped_random_forest_classification_report.json",
    MODEL_DIR / "permutation_feature_importance.csv",
    STATS_DIR / "multivariable_results.json",
]

missing_files = [p for p in required_files if not available_file(p)]

if missing_files:
    st.error("Some required analysis files are missing.")

    for file in missing_files:
        st.write(f"- `{file.relative_to(ROOT)}`")

    st.stop()


df = load_csv(DATA_FILE)

eda = load_json(EDA_DIR / "eda_summary.json")

model_comparison = load_csv(
    PRED_DIR / "model_comparison_summary.csv"
)

oof = load_csv(
    PRED_DIR / "out_of_fold_predictions.csv"
)

error_summary = load_json(
    MODEL_DIR / "model_error_analysis_summary.json"
)

classification_report = load_json(
    MODEL_DIR / "date_grouped_random_forest_classification_report.json"
)

feature_importance = load_csv(
    MODEL_DIR / "permutation_feature_importance.csv"
)

multivariable = load_json(
    STATS_DIR / "multivariable_results.json"
)

deployment_bundle = load_deployment_model(
    str(DEPLOYMENT_MODEL_FILE)
)
deployment_model = deployment_bundle["model"]
deployment_metadata = deployment_bundle["metadata"]


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

TEST = "FLEX Test #"
IDENTIFIER = "FLEX Identifier"
DATE = "Test Date"
FUEL = "Fuel"
OUTCOME = "Test end"

PRESSURE = "Ambient pressure; mmHg"
OXYGEN = "O initial ambient composition; mole fraction"
NITROGEN = "N initial ambient composition; mole fraction"
CO = "CO initial ambient composition; mole fraction"
HELIUM = "He initial ambient composition; mole fraction"

DIAMETER = "Droplet initial diameter; mm"
EXTINCTION_DIAMETER = "Visible flame extinction diameter; mm"
BURNING_RATE = "Burning rate; mm"
BURN_TIME = "Burn time; s"


# Convert measurement columns safely.
numeric_columns = [
    PRESSURE,
    OXYGEN,
    NITROGEN,
    CO,
    HELIUM,
    DIAMETER,
    EXTINCTION_DIAMETER,
    BURNING_RATE,
    BURN_TIME,
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


df[DATE] = df[DATE].astype(str)
df[OUTCOME] = df[OUTCOME].astype(str)
df[FUEL] = df[FUEL].astype(str)


OUTCOME_ORDER = ["Extinction", "Disruption", "Completion"]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔥 NASA FLEX")

st.sidebar.caption(
    "Flame Extinguishment Experiment Research"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Research Navigation",
    [
        "Research Overview",
        "Experimental Data Explorer",
        "Exploratory Data Analysis",
        "Statistical Analysis",
        "Model Evaluation",
        "Explainability & Errors",
        "Data Quality & Limitations",
        "Prediction Lab",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### Project Information")

st.sidebar.write("Dataset: NASA PSI-69 FLEX")

st.sidebar.write("Records:", len(df))

st.sidebar.write(
    "Predictive modeling records:",
    error_summary["records_analyzed"],
)

st.sidebar.caption(
    "Research prototype | Experimental dataset analysis"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_row(items):
    columns = st.columns(len(items))

    for column, (label, value, help_text) in zip(columns, items):
        column.metric(
            label=label,
            value=value,
            help=help_text,
        )


def section_title(title, description=None):
    st.header(title)

    if description:
        st.caption(description)


def show_image(path, caption=None):
    if available_file(path):
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {path.name}")


def safe_pvalue(value):
    if value < 0.000001:
        return f"{value:.3e}"

    return f"{value:.6f}"


def clean_plot():
    plt.tight_layout()
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close()


# ============================================================
# PAGE 1: RESEARCH OVERVIEW
# ============================================================

if page == "Research Overview":

    st.title("NASA FLEX Research Dashboard")

    st.markdown(
        """
        **Scientific analysis and machine-learning investigation
        of flame extinguishment experiments.**

        This application analyzes experimental observations from
        NASA's Physical Sciences Informatics (PSI) FLEX investigation.
        """
    )

    st.divider()

    outcome_counts = df[OUTCOME].value_counts()

    metric_row([
        (
            "Experimental records",
            f"{len(df):,}",
            "Records in the cleaned dataset.",
        ),
        (
            "Experiment dates",
            f"{df[DATE].nunique():,}",
            "Distinct recorded experiment dates.",
        ),
        (
            "Fuel types",
            f"{df[FUEL].nunique():,}",
            "Methanol and Heptane.",
        ),
        (
            "Modeling records",
            f"{error_summary['records_analyzed']:,}",
            "Records retained for predictive modeling.",
        ),
    ])

    st.divider()

    st.subheader("Research objectives")

    st.markdown(
        """
        1. Investigate the relationship between experimental
           conditions and flame outcomes.

        2. Compare flame outcomes across fuel types and oxygen
           concentrations.

        3. Perform statistical analysis of selected experimental
           variables.

        4. Evaluate machine-learning models for classifying
           experimental outcomes.

        5. Investigate prediction errors and feature importance,
           particularly for the less frequent Completion class.
        """
    )

    st.divider()

    st.subheader("Experimental outcome distribution")

    left, right = st.columns([1.2, 1])

    with left:
        counts = (
            df[OUTCOME]
            .value_counts()
            .reindex(OUTCOME_ORDER)
            .fillna(0)
        )

        st.bar_chart(
            counts,
            y_label="Number of experiments",
        )

    with right:
        for outcome in OUTCOME_ORDER:
            count = int(counts[outcome])
            percentage = count / len(df) * 100

            st.metric(
                outcome,
                f"{count} ({percentage:.2f}%)",
            )

    st.divider()

    st.subheader("Research workflow")

    st.markdown(
        """
        **NASA FLEX data → Data validation → Exploratory analysis
        → Statistical analysis → Predictive modeling
        → Error analysis → Scientific interpretation**
        """
    )

    st.markdown(
        """
        <div class="research-note">
        <b>Research principle:</b> Experimental associations and
        machine-learning predictions are not evidence of causation.
        The model is an exploratory research tool, not a validated
        spacecraft fire-safety system.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 2: EXPERIMENTAL DATA EXPLORER
# ============================================================

elif page == "Experimental Data Explorer":

    section_title(
        "Experimental Data Explorer",
        "Explore NASA FLEX experimental records using interactive filters.",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_fuels = st.multiselect(
            "Fuel",
            sorted(df[FUEL].dropna().unique()),
            default=sorted(df[FUEL].dropna().unique()),
        )

    with col2:
        selected_outcomes = st.multiselect(
            "Experimental outcome",
            OUTCOME_ORDER,
            default=OUTCOME_ORDER,
        )

    with col3:
        dates = sorted(df[DATE].dropna().unique())

        selected_dates = st.multiselect(
            "Experiment date",
            dates,
            default=dates,
        )

    filtered = df[
        df[FUEL].isin(selected_fuels)
        & df[OUTCOME].isin(selected_outcomes)
        & df[DATE].isin(selected_dates)
    ].copy()

    st.divider()

    metric_row([
        ("Filtered experiments", len(filtered), "Current filter selection."),
        ("Fuel types", filtered[FUEL].nunique(), "Fuel types in selection."),
        ("Outcomes", filtered[OUTCOME].nunique(), "Outcomes in selection."),
    ])

    st.subheader("Filtered experimental records")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download filtered experiments (CSV)",
        data=csv_data,
        file_name="nasa_flex_filtered_experiments.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("Oxygen concentration distribution")

    if not filtered.empty:

        fig, ax = plt.subplots(figsize=(9, 4))

        for outcome in OUTCOME_ORDER:
            subset = filtered.loc[
                filtered[OUTCOME] == outcome,
                OXYGEN,
            ].dropna()

            if not subset.empty:
                ax.hist(
                    subset,
                    bins=15,
                    alpha=0.5,
                    label=outcome,
                )

        ax.set_xlabel("Initial oxygen mole fraction")
        ax.set_ylabel("Number of experiments")
        ax.set_title("Oxygen concentration by experimental outcome")
        ax.legend()

        clean_plot()

    else:
        st.info("No experiments match the selected filters.")


# ============================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "Exploratory Data Analysis":

    section_title(
        "Exploratory Data Analysis",
        "Visual exploration of experimental distributions and relationships.",
    )

    st.subheader("Outcome distribution")
    show_image(EDA_DIR / "01_outcome_distribution.png")

    st.divider()

    st.subheader("Fuel and experimental outcomes")
    show_image(EDA_DIR / "02_fuel_outcomes.png")

    st.divider()

    st.subheader("Oxygen concentration and outcomes")
    show_image(EDA_DIR / "03_oxygen_outcomes.png")

    st.divider()

    st.subheader("Pressure and burn time")
    show_image(EDA_DIR / "04_pressure_burntime.png")

    st.divider()

    st.subheader("Droplet behavior")
    show_image(EDA_DIR / "05_droplet_behavior.png")

    st.divider()

    st.subheader("Gas composition correlation")
    show_image(EDA_DIR / "06_gas_correlation.png")

    st.divider()

    st.subheader("Burning rate")
    show_image(EDA_DIR / "07_burning_rate.png")

    st.caption(
        "These plots are exploratory. Their relationships should not "
        "be interpreted as causal effects."
    )


# ============================================================
# PAGE 4: STATISTICAL ANALYSIS
# ============================================================

elif page == "Statistical Analysis":

    section_title(
        "Statistical Analysis",
        "Multivariable multinomial logistic regression and adjusted associations.",
    )

    stats = multivariable["model_statistics"]

    st.subheader("Multivariable model summary")

    metric_row([
        (
            "Observations",
            stats["observations"],
            "Complete cases in the multinomial model.",
        ),
        (
            "Pseudo R²",
            f"{stats['pseudo_r_squared']:.4f}",
            "McFadden-style pseudo R-squared.",
        ),
        (
            "AIC",
            f"{stats['aic']:.2f}",
            "Akaike information criterion.",
        ),
        (
            "Model converged",
            "Yes" if stats["converged"] else "No",
            "Optimization convergence status.",
        ),
    ])

    st.write(
        "Reference outcome:",
        stats["outcome_reference"],
    )

    st.write(
        "Reference fuel:",
        stats["fuel_reference"],
    )

    st.write(
        "Overall likelihood-ratio p-value:",
        safe_pvalue(stats["overall_likelihood_ratio_p_value"]),
    )

    st.divider()

    st.subheader("Adjusted odds ratios")

    odds_rows = []

    for comparison, variables in multivariable["odds_ratios"].items():

        for variable, result in variables.items():

            odds_rows.append({
                "Comparison": comparison.replace("_", " "),
                "Predictor": result["display_name"],
                "Odds ratio": result["odds_ratio"],
                "95% CI lower": result["ci_95_lower"],
                "95% CI upper": result["ci_95_upper"],
                "p-value": result["p_value"],
            })

    odds_df = pd.DataFrame(odds_rows)

    st.dataframe(
        odds_df.style.format({
            "Odds ratio": "{:.3f}",
            "95% CI lower": "{:.3f}",
            "95% CI upper": "{:.3f}",
            "p-value": "{:.3e}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Oxygen odds ratios represent a 0.01 mole-fraction increase. "
        "Methanol is compared with Heptane."
    )

    st.divider()

    st.subheader("Adjusted likelihood-ratio tests")

    tests = multivariable["adjusted_likelihood_ratio_tests"]

    test_rows = []

    for name, result in tests.items():

        test_rows.append({
            "Test": name.replace("_", " "),
            "Chi-square": result["likelihood_ratio_chi_square"],
            "Degrees of freedom": result["degrees_of_freedom"],
            "p-value": result["p_value"],
        })

    st.dataframe(
        pd.DataFrame(test_rows).style.format({
            "Chi-square": "{:.3f}",
            "p-value": "{:.3e}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Scientific interpretation")

    st.markdown(
        """
        - Oxygen concentration is statistically associated with
          the observed outcome in this fitted model.

        - Fuel type also contributes adjusted statistical
          information after accounting for oxygen concentration.

        - The model uses Extinction as its reference outcome
          and Heptane as its reference fuel.

        - These findings describe associations within the
          analyzed experimental records. They do not establish
          causality.
        """
    )


# ============================================================
# PAGE 5: MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    section_title(
        "Machine Learning Model Evaluation",
        "Comparison of baseline, Logistic Regression, and Random Forest models.",
    )

    st.subheader("Model comparison")

    st.dataframe(
        model_comparison.style.format({
            "Accuracy": "{:.4f}",
            "Balanced accuracy": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Mean fold Macro F1": "{:.4f}",
            "Std fold Macro F1": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Stratified random and date-grouped validation results are "
        "reported separately. Date-grouped validation holds out "
        "experiment dates rather than randomly splitting individual records."
    )

    st.divider()

    st.subheader("Select validation and model")

    validation_options = sorted(
        oof["Validation"].dropna().unique().tolist()
    )

    preferred_validation = next(
        (
            value for value in validation_options
            if "group" in value.lower() or "date" in value.lower()
        ),
        validation_options[0] if validation_options else None,
    )

    selected_validation = st.selectbox(
        "Validation strategy",
        validation_options,
        index=validation_options.index(preferred_validation)
        if preferred_validation is not None else 0,
    )

    model_options = sorted(
        oof.loc[
            oof["Validation"] == selected_validation,
            "Model",
        ].dropna().unique().tolist()
    )

    preferred_model = next(
        (
            value for value in model_options
            if "random forest" in value.lower()
        ),
        model_options[0] if model_options else None,
    )

    selected_model = st.selectbox(
        "Model",
        model_options,
        index=model_options.index(preferred_model)
        if preferred_model is not None else 0,
    )

    selected_predictions = oof[
        (oof["Validation"] == selected_validation)
        & (oof["Model"] == selected_model)
    ].copy()

    if not selected_predictions.empty:

        actual = selected_predictions["Actual outcome"]
        predicted = selected_predictions["Predicted outcome"]

        accuracy = (actual == predicted).mean()

        st.metric(
            "Selected model accuracy",
            f"{accuracy:.2%}",
        )

        st.subheader("Prediction records")

        st.dataframe(
            selected_predictions,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download selected predictions",
            selected_predictions.to_csv(index=False).encode("utf-8"),
            file_name="selected_flex_predictions.csv",
            mime="text/csv",
        )

    st.divider()

    st.subheader("Date-grouped Random Forest classification report")

    report_rows = []

    for label in OUTCOME_ORDER:

        if label in classification_report:

            result = classification_report[label]

            report_rows.append({
                "Outcome": label,
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1-score": result["f1-score"],
                "Support": int(result["support"]),
            })

    st.dataframe(
        pd.DataFrame(report_rows).style.format({
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-score": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Confusion matrix")

    matrix = np.array(
        error_summary["date_grouped_random_forest_confusion_matrix"]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    image = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))

    ax.set_xticklabels(OUTCOME_ORDER)
    ax.set_yticklabels(OUTCOME_ORDER)

    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual outcome")
    ax.set_title("Date-grouped Random Forest confusion matrix")

    for row in range(3):
        for column in range(3):

            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

    fig.colorbar(image, ax=ax)

    clean_plot()

    st.caption(
        "Rows represent actual outcomes. Columns represent predicted outcomes."
    )


# ============================================================
# PAGE 6: EXPLAINABILITY AND ERRORS
# ============================================================

elif page == "Explainability & Errors":

    section_title(
        "Model Explainability & Error Analysis",
        "Investigating which inputs influence model performance and where predictions fail.",
    )

    st.subheader("Permutation feature importance")

    importance_display = feature_importance.copy()

    importance_display = importance_display.sort_values(
        "Mean_importance",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        importance_display["Feature"],
        importance_display["Mean_importance"],
        xerr=importance_display["Std_across_folds"],
        capsize=4,
    )

    ax.axvline(0, linewidth=0.8)

    ax.set_xlabel("Mean decrease in macro F1 after permutation")
    ax.set_title("Feature importance across date-grouped folds")

    clean_plot()

    st.dataframe(
        feature_importance.style.format({
            "Mean_importance": "{:.4f}",
            "Std_across_folds": "{:.4f}",
            "Mean_permutation_std": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Error bars show standard deviation of mean importance across folds. "
        "Importance is model- and dataset-dependent."
    )

    st.divider()

    st.subheader("Overall prediction errors")

    metric_row([
        (
            "Correct predictions",
            error_summary["correct_predictions"],
            "Correct date-grouped Random Forest predictions.",
        ),
        (
            "Incorrect predictions",
            error_summary["incorrect_predictions"],
            "Incorrect date-grouped Random Forest predictions.",
        ),
        (
            "Completion recall",
            f"{error_summary['completion_recall']:.1%}",
            "Only 5 of 27 Completion cases were identified.",
        ),
    ])

    st.divider()

    st.subheader("Completion-class challenge")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Correctly identified Completion",
            error_summary["correct_completion"],
        )

    with col2:
        st.metric(
            "Missed Completion",
            error_summary["missed_completion"],
        )

    st.markdown(
        """
        <div class="warning-note">
        <b>Important model limitation:</b> The model identifies only
        a small proportion of Completion experiments. Its overall
        accuracy should not be interpreted as equally strong
        performance across all outcome classes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("Prediction error categories")

    error_file = MODEL_DIR / "date_grouped_random_forest_misclassified.csv"

    if available_file(error_file):

        errors = load_csv(error_file)

        if "Error type" in errors.columns:

            error_counts = errors["Error type"].value_counts()

            st.bar_chart(
                error_counts,
                y_label="Number of errors",
            )

        st.dataframe(
            errors,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download misclassified experiments",
            errors.to_csv(index=False).encode("utf-8"),
            file_name="flex_misclassified_experiments.csv",
            mime="text/csv",
        )

    else:
        st.info("Misclassified experiment file is unavailable.")


# ============================================================
# PAGE 7: DATA QUALITY AND LIMITATIONS
# ============================================================

elif page == "Data Quality & Limitations":

    section_title(
        "Data Quality, Provenance & Research Limitations",
        "Transparent reporting of the experimental dataset and analytical constraints.",
    )

    st.subheader("Dataset overview")

    metric_row([
        ("Rows", eda["rows"], "Cleaned dataset records."),
        ("Columns in EDA", eda["columns"], "EDA summary reports 17 columns."),
        ("Approximate burn times", eda["approximate_burn_times"], "Approximate values require caution."),
        ("Zero-pressure records", eda["zero_pressure_records"], "Excluded from predictive modeling."),
    ])

    st.divider()

    st.subheader("Missing measurements")

    missing = eda["missing_measurements"]

    missing_df = pd.DataFrame({
        "Measurement": list(missing.keys()),
        "Missing values": list(missing.values()),
    })

    missing_df["Missing percentage"] = (
        missing_df["Missing values"] / len(df) * 100
    )

    st.dataframe(
        missing_df.style.format({
            "Missing percentage": "{:.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Gas composition quality")

    composition = eda["composition_totals"]

    metric_row([
        ("Mean composition total", f"{composition['mean']:.4f}", "Sum of recorded gas fractions."),
        ("Minimum", f"{composition['minimum']:.2f}", "Observed minimum composition total."),
        ("Maximum", f"{composition['maximum']:.2f}", "Observed maximum composition total."),
        ("Below 1.0", composition["records_below_1"], "Records with total below 1."),
    ])

    st.divider()

    st.subheader("Analytical limitations")

    limitations = [
        "The dataset contains 274 experimental records, not a large independent validation campaign.",
        "Predictive modeling excludes one zero-pressure record, leaving 273 records.",
        "Completion is the least represented outcome, with only 27 records.",
        "The date-grouped evaluation uses internal cross-validation, not an independent external test campaign.",
        "Some measurements are missing and are handled through model preprocessing where applicable.",
        "Two burn-time values are approximate and should not be treated as exact measurements.",
        "The recorded gas fractions do not always sum exactly to one.",
        "Feature importance does not establish physical causation.",
        "Predictive performance should not be generalized directly to spacecraft fire-safety decisions.",
    ]

    for item in limitations:
        st.markdown(f"- {item}")

    st.divider()

    st.subheader("Research interpretation")

    st.markdown(
        """
        The project investigates patterns in NASA FLEX experimental
        observations and evaluates the feasibility of classifying
        experimental outcomes using selected pre-experiment variables.

        Results should be interpreted within the scope of this dataset.
        Further validation using independent experiments, additional
        measurements, and improved representation of the Completion
        class is needed before practical deployment.
        """
    )

    st.divider()

    st.subheader("Data export")

    st.download_button(
        "Download cleaned NASA FLEX dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="nasa_flex_cleaned_dataset.csv",
        mime="text/csv",
    )



# ============================================================
# PAGE 8: PREDICTION LAB
# ============================================================

elif page == "Prediction Lab":

    section_title(
        "NASA FLEX Prediction Lab",
        "Explore model outputs for selected pre-experiment conditions.",
    )

    st.markdown(
        """
        Enter experimental conditions to obtain an exploratory
        classification from the trained Random Forest pipeline.
        """
    )

    st.warning(
        "Research prototype only. This model was fitted on the full "
        "eligible NASA FLEX dataset. Its interactive outputs are not "
        "independent validation results, calibrated physical "
        "probabilities, or predictions for operational fire-safety use."
    )

    # Restrict input controls to observed eligible training ranges.
    prediction_data = df.copy()
    prediction_data[PRESSURE] = pd.to_numeric(
        prediction_data[PRESSURE], errors="coerce"
    )
    prediction_data = prediction_data[
        prediction_data[PRESSURE] > 0
    ].copy()

    def get_input_range(column):
        values = pd.to_numeric(
            prediction_data[column], errors="coerce"
        ).dropna()

        if values.empty:
            raise ValueError(f"No valid values for {column}")

        return (
            float(values.min()),
            float(values.max()),
            float(values.median()),
        )

    try:
        p_min, p_max, p_default = get_input_range(PRESSURE)
        o_min, o_max, o_default = get_input_range(OXYGEN)
        co_min, co_max, co_default = get_input_range(CO)
        he_min, he_max, he_default = get_input_range(HELIUM)
        d_min, d_max, d_default = get_input_range(DIAMETER)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.subheader("Experimental conditions")

    st.caption(
        "Enter values within the observed training-data ranges. "
        "The model uses the units shown below."
    )

    with st.form("flex_prediction_form"):

        left, right = st.columns(2)

        with left:
            selected_fuel = st.selectbox(
                "Fuel",
                sorted(prediction_data[FUEL].dropna().unique()),
            )

            selected_pressure = st.number_input(
                "Ambient pressure (mmHg)",
                min_value=p_min,
                max_value=p_max,
                value=p_default,
                format="%.2f",
            )

            selected_oxygen = st.number_input(
                "Initial oxygen mole fraction",
                min_value=o_min,
                max_value=o_max,
                value=o_default,
                format="%.4f",
            )

        with right:
            selected_co = st.number_input(
                "Initial CO mole fraction",
                min_value=co_min,
                max_value=co_max,
                value=co_default,
                format="%.4f",
            )

            selected_helium = st.number_input(
                "Initial helium mole fraction",
                min_value=he_min,
                max_value=he_max,
                value=he_default,
                format="%.4f",
            )

            selected_diameter = st.number_input(
                "Initial droplet diameter (mm)",
                min_value=d_min,
                max_value=d_max,
                value=d_default,
                format="%.4f",
            )

        submitted = st.form_submit_button(
            "Predict experimental outcome",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        input_record = pd.DataFrame([{
            PRESSURE: selected_pressure,
            OXYGEN: selected_oxygen,
            CO: selected_co,
            HELIUM: selected_helium,
            DIAMETER: selected_diameter,
            FUEL: selected_fuel,
        }])

        predicted_outcome = deployment_model.predict(
            input_record
        )[0]

        probabilities = deployment_model.predict_proba(
            input_record
        )[0]

        # Use the fitted model's actual class order.
        probability_table = pd.DataFrame({
            "Outcome": deployment_model.classes_,
            "Model score": probabilities,
        }).sort_values(
            "Model score",
            ascending=False,
        )

        st.divider()
        st.subheader("Prediction result")

        st.metric(
            "Predicted experimental outcome",
            str(predicted_outcome),
        )

        st.subheader("Model class scores")

        display_table = probability_table.copy()
        display_table["Model score"] = display_table[
            "Model score"
        ].map(lambda value: f"{value:.2%}")

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            probability_table.set_index("Outcome"),
            y="Model score",
            y_label="Random Forest output score",
        )

        st.caption(
            "These are model-estimated class scores. They are not "
            "calibrated probabilities of a real-world fire outcome."
        )

        with st.expander("View submitted input values"):
            st.dataframe(
                input_record,
                use_container_width=True,
                hide_index=True,
            )


    # ============================================================
    # FUEL COMPARISON: SAVED MODEL OUTPUTS
    # ============================================================

    st.divider()
    st.subheader("Fuel Comparison Experiment")

    st.markdown(
        """
        Compare the saved Random Forest outputs for Heptane and
        Methanol while keeping the other experimental inputs fixed.
        """
    )

    st.info(
        "This comparison uses previously generated model outputs. "
        "It is a comparison of model behavior, not a new physical "
        "experiment or evidence of a causal fuel effect."
    )

    from pathlib import Path as _Path
    import json as _json

    _comparison_root = (
        _Path(__file__).resolve().parent
        / "reports"
        / "prediction_lab"
    )

    _scores_file = (
        _comparison_root / "fuel_comparison_scores.csv"
    )

    _differences_file = (
        _comparison_root / "fuel_comparison_differences.csv"
    )

    _summary_file = (
        _comparison_root / "fuel_comparison_summary.json"
    )

    _required_files = [
        _scores_file,
        _differences_file,
        _summary_file,
    ]

    _missing_files = [
        str(path.name)
        for path in _required_files
        if not path.exists()
    ]

    if _missing_files:
        st.error(
            "Fuel comparison files are missing: "
            + ", ".join(_missing_files)
        )
    else:
        _fuel_scores = pd.read_csv(_scores_file)
        _fuel_differences = pd.read_csv(_differences_file)

        with open(
            _summary_file,
            "r",
            encoding="utf-8",
        ) as _file:
            _fuel_summary = _json.load(_file)

        st.markdown("**Fixed experimental input conditions**")

        st.dataframe(
            pd.DataFrame([
                {"Input": "Ambient pressure", "Value": "759.60 mmHg"},
                {"Input": "Initial oxygen mole fraction", "Value": "0.2000"},
                {"Input": "Initial CO mole fraction", "Value": "0.0000"},
                {"Input": "Initial helium mole fraction", "Value": "0.0000"},
                {"Input": "Initial droplet diameter", "Value": "3.0100 mm"},
                {"Input": "Fuel", "Value": "Heptane / Methanol"},
            ]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Model scores by fuel**")

        st.dataframe(
            _fuel_scores,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "**Difference in model scores "
            "(Methanol minus Heptane)**"
        )

        st.dataframe(
            _fuel_differences,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Positive differences indicate a higher model score "
            "for Methanol; negative differences indicate a lower "
            "model score for Methanol."
        )

        with st.expander("View saved comparison summary"):
            st.json(_fuel_summary)

        st.warning(
            "The displayed scores are not calibrated probabilities. "
            "This single-input comparison does not establish that "
            "changing fuel causes a particular experimental outcome."
        )

    st.divider()
    st.subheader("Model and validation context")

    validation = deployment_metadata["validation_reference"]

    metric_row([
        (
            "Training records",
            deployment_metadata["training_records"],
            "Eligible records used to fit the deployment model.",
        ),
        (
            "Experiment dates",
            deployment_metadata["unique_test_dates"],
            "Distinct training experiment dates.",
        ),
        (
            "Trees",
            deployment_metadata["n_estimators"],
            "Random Forest estimator count.",
        ),
    ])

    st.write("**Previously measured date-grouped cross-validation results**")

    metric_row([
        (
            "Out-of-fold macro F1",
            f"{validation['out_of_fold_macro_f1']:.4f}",
            "Five-fold date-grouped evaluation.",
        ),
        (
            "Out-of-fold balanced accuracy",
            f"{validation['out_of_fold_balanced_accuracy']:.4f}",
            "Five-fold date-grouped evaluation.",
        ),
        (
            "Out-of-fold accuracy",
            f"{validation['out_of_fold_accuracy']:.4f}",
            "Five-fold date-grouped evaluation.",
        ),
    ])

    st.caption(
        "These are previously obtained out-of-fold metrics, not "
        "independent external test-set performance."
    )

    st.subheader("Model input features")

    st.markdown(
        """
        - Fuel type
        - Ambient pressure
        - Initial oxygen mole fraction
        - Initial carbon monoxide mole fraction
        - Initial helium mole fraction
        - Initial droplet diameter

        Post-experiment measurements such as burn time, burning rate,
        and visible flame extinction diameter are not model inputs.
        """
    )

    st.subheader("Limitations")

    st.markdown(
        """
        - The deployment model was fitted on all 273 eligible records.
        - Only 27 Completion records are present in the dataset.
        - Model scores have not been independently calibrated.
        - Inputs outside the observed training ranges are not supported.
        - This research prototype is not suitable for operational
          fire-safety decisions.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "NASA PSI-69 FLEX | Research prototype"
)

st.sidebar.caption(
    "Interpret findings within the limitations of the dataset."
)
