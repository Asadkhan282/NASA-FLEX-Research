
from pathlib import Path
from datetime import datetime
import shutil
import sys


ROOT = Path(__file__).resolve().parent
APP_FILE = ROOT / "app.py"

SCORES_FILE = (
    ROOT / "reports" / "prediction_lab"
    / "fuel_comparison_scores.csv"
)

DIFFERENCES_FILE = (
    ROOT / "reports" / "prediction_lab"
    / "fuel_comparison_differences.csv"
)

SUMMARY_FILE = (
    ROOT / "reports" / "prediction_lab"
    / "fuel_comparison_summary.json"
)

MARKER = "# FUEL COMPARISON: SAVED MODEL OUTPUTS"

ANCHOR = (
    '    st.divider()\n'
    '    st.subheader("Model and validation context")'
)

INSERTION = '''
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

'''


def main():
    if not APP_FILE.exists():
        print("ERROR: app.py was not found.")
        return 1

    source = APP_FILE.read_text(encoding="utf-8")

    if MARKER in source:
        print("The fuel comparison section is already present.")
        print("No changes were made.")
        return 0

    if source.count(ANCHOR) != 1:
        print(
            "ERROR: Could not identify exactly one insertion point."
        )
        print("No changes were made.")
        return 1

    missing = [
        path for path in (
            SCORES_FILE,
            DIFFERENCES_FILE,
            SUMMARY_FILE,
        )
        if not path.exists()
    ]

    if missing:
        print("ERROR: Required comparison files are missing:")
        for path in missing:
            print(f"  - {path}")
        print("No changes were made.")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ROOT / f"app.py.backup_{timestamp}"

    shutil.copy2(APP_FILE, backup)
    print(f"Backup created: {backup.name}")

    updated = source.replace(
        ANCHOR,
        INSERTION + ANCHOR,
        1,
    )

    APP_FILE.write_text(updated, encoding="utf-8")

    try:
        compile(updated, str(APP_FILE), "exec")
    except SyntaxError as error:
        shutil.copy2(backup, APP_FILE)
        print("ERROR: Syntax validation failed.")
        print(f"Line: {error.lineno}")
        print(f"Details: {error}")
        print("Original app.py has been restored.")
        return 1

    print("SUCCESS: Fuel comparison added to Prediction Lab.")
    print("Python syntax validation passed.")
    print("Existing model and other dashboard pages were not changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
