
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_FILE = ROOT / "app.py"
BACKUP_FILE = ROOT / "app_before_prediction_lab.py"

source = APP_FILE.read_text(encoding="utf-8")


def replace_once(old, new, description):
    global source

    count = source.count(old)

    if count != 1:
        raise RuntimeError(
            f"Expected exactly one match for {description}; found {count}."
        )

    source = source.replace(old, new, 1)


# Prevent accidental duplicate insertion.
if 'elif page == "Prediction Lab":' in source:
    raise RuntimeError("Prediction Lab is already present. No changes made.")

# 1. Add import.
replace_once(
    "import json\n",
    "import json\nimport joblib\n",
    "joblib import",
)

# 2. Add model path.
replace_once(
    'STATS_DIR = ROOT / "reports" / "multivariable_analysis"',
    'STATS_DIR = ROOT / "reports" / "multivariable_analysis"\n'
    'DEPLOYMENT_MODEL_FILE = ROOT / "models" / "flex_random_forest.joblib"',
    "deployment model path",
)

# 3. Add model loader.
replace_once(
    '@st.cache_data\ndef load_json(path):\n'
    '    with open(path, "r", encoding="utf-8") as file:\n'
    '        return json.load(file)\n',
    '@st.cache_data\ndef load_json(path):\n'
    '    with open(path, "r", encoding="utf-8") as file:\n'
    '        return json.load(file)\n\n\n'
    '@st.cache_resource\ndef load_deployment_model(path):\n'
    '    return joblib.load(path)\n',
    "deployment model loader",
)

# 4. Require the model file.
replace_once(
    "required_files = [\n    DATA_FILE,",
    "required_files = [\n    DATA_FILE,\n    DEPLOYMENT_MODEL_FILE,",
    "required files",
)

# 5. Load the saved model.
replace_once(
    'multivariable = load_json(\n'
    '    STATS_DIR / "multivariable_results.json"\n'
    ')\n',
    'multivariable = load_json(\n'
    '    STATS_DIR / "multivariable_results.json"\n'
    ')\n\n'
    'deployment_bundle = load_deployment_model(\n'
    '    str(DEPLOYMENT_MODEL_FILE)\n'
    ')\n'
    'deployment_model = deployment_bundle["model"]\n'
    'deployment_metadata = deployment_bundle["metadata"]\n',
    "deployment model loading",
)

# 6. Add navigation entry.
replace_once(
    '        "Data Quality & Limitations",\n    ],',
    '        "Data Quality & Limitations",\n'
    '        "Prediction Lab",\n    ],',
    "sidebar navigation",
)

# 7. Prediction Lab page.
prediction_page = r'''
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


'''

# 8. Insert page immediately before the footer.
footer_marker = "# ============================================================\n# FOOTER"

if source.count(footer_marker) != 1:
    raise RuntimeError("Could not locate the unique dashboard footer.")

source = source.replace(
    footer_marker,
    prediction_page + footer_marker,
    1,
)

# 9. Save only after all checks and replacements succeed.
if not BACKUP_FILE.exists():
    BACKUP_FILE.write_text(
        APP_FILE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

APP_FILE.write_text(source, encoding="utf-8")

print("Prediction Lab added successfully.")
print(f"Backup: {BACKUP_FILE}")
print(f"Updated dashboard: {APP_FILE}")
