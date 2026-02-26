from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import pandas as pd
import numpy as np


# --------------------------------------------------
# Sensitivity / Robustness Analysis
# --------------------------------------------------

def sensitivity(df, y_var, x_vars):
    """
    Leave-one-variable-out sensitivity analysis.
    Measures coefficient stability index (CSI).
    """

    # -----------------------------
    # Column validation
    # -----------------------------
    required = [y_var] + x_vars
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(
            f"Missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # -----------------------------
    # Base model (Fixed Effects)
    # -----------------------------
    base_model = PanelOLS(
        df[y_var],
        sm.add_constant(df[x_vars]),
        entity_effects=True
    ).fit(cov_type="robust")

    base_coef = base_model.params

    stability_records = []

    # -----------------------------
    # Leave-one-variable-out loop
    # -----------------------------
    for var in x_vars:

        reduced_vars = [v for v in x_vars if v != var]

        if not reduced_vars:
            continue

        res = PanelOLS(
            df[y_var],
            sm.add_constant(df[reduced_vars]),
            entity_effects=True
        ).fit(cov_type="robust")

        for coef_var in base_coef.index:

            if coef_var in res.params and base_coef[coef_var] != 0:

                csi = abs(base_coef[coef_var] - res.params[coef_var]) / abs(base_coef[coef_var])

                stability_records.append({
                    "Dropped Variable": var,
                    "Affected Coefficient": coef_var,
                    "CSI": csi
                })

    stability_df = pd.DataFrame(stability_records)

    return stability_df


# --------------------------------------------------
# Interpret Robustness
# --------------------------------------------------

def interpret_robustness(stability_df):
    """
    Classifies robustness based on CSI thresholds.
    """

    if stability_df.empty:
        return "No sensitivity results available."

    max_csi = stability_df["CSI"].max()

    if max_csi < 0.10:
        return "Highly robust model (coefficients stable)."
    elif max_csi < 0.30:
        return "Moderately robust model."
    else:
        return "Low robustness — coefficients sensitive to specification changes."


# --------------------------------------------------
# Robustness Score (0–100)
# --------------------------------------------------

def robustness_score(stability_df):
    """
    Converts maximum CSI into a robustness score.
    Higher = more stable.
    """

    if stability_df.empty:
        return 100

    max_csi = stability_df["CSI"].max()

    score = max(0, 100 - (max_csi * 100))

    return round(score, 2)


# --------------------------------------------------
# AI-Ready Summary Packager
# --------------------------------------------------

def package_robustness_results(stability_df, interpretation, score):
    summary_text = f"""
    ===============================
    ROBUSTNESS / SENSITIVITY ANALYSIS
    ===============================

    Maximum Coefficient Stability Index (CSI): {stability_df["CSI"].max() if not stability_df.empty else "N/A"}

    Robustness Interpretation:
    {interpretation}

    Robustness Score (0–100):
    {score}

    ===============================
    DETAILED RESULTS
    ===============================
    {stability_df}
    """

    return summary_text
