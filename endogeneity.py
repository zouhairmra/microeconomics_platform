import numpy as np


# --------------------------------------------------
# Endogeneity Risk Score
# --------------------------------------------------

def endogeneity_score(hausman_p=None, max_vif=None, bp_p=None):
    """
    Computes structured endogeneity risk score.
    """

    score = 0
    details = []

    # -----------------------------
    # Hausman (FE vs RE)
    # -----------------------------
    if hausman_p is not None:
        if hausman_p < 0.05:
            score += 30
            details.append("Hausman significant → FE preferred (endogeneity likely).")
        else:
            details.append("Hausman not significant → RE acceptable.")

    # -----------------------------
    # Multicollinearity (VIF)
    # -----------------------------
    if max_vif is not None:
        if max_vif > 10:
            score += 20
            details.append("High multicollinearity (VIF > 10).")
        elif max_vif > 5:
            score += 10
            details.append("Moderate multicollinearity (VIF between 5 and 10).")
        else:
            details.append("No serious multicollinearity.")

    # -----------------------------
    # Heteroskedasticity (Breusch-Pagan)
    # -----------------------------
    if bp_p is not None:
        if bp_p < 0.05:
            score += 10
            details.append("Heteroskedasticity detected.")
        else:
            details.append("No strong heteroskedasticity evidence.")

    # -----------------------------
    # Classification
    # -----------------------------
    if score < 30:
        level = "Low Endogeneity Risk"
    elif score < 60:
        level = "Moderate Endogeneity Risk"
    else:
        level = "High Endogeneity Risk"

    return score, level, details


# --------------------------------------------------
# Instrument Suggestions
# --------------------------------------------------

def suggest_instruments(variable):
    """
    Suggests potential instruments for common macro variables.
    """

    suggestions = {
        "fdi": ["Lagged FDI", "Global FDI shocks", "Neighboring-country FDI"],
        "gdppc": ["Lagged GDP per capita", "Commodity price shocks"],
        "institutions": ["Legal origin", "Colonial history", "Settler mortality"],
        "inflation": ["Lagged inflation", "Monetary policy shocks"],
        "trade": ["Lagged trade openness", "Global demand shocks"]
    }

    return suggestions.get(variable.lower(), ["Lagged values", "External shocks"])


# --------------------------------------------------
# Diagnostic Explanation Helper
# --------------------------------------------------

def interpret_endogeneity(score, level):
    """
    Provides structured academic interpretation.
    """

    interpretation = f"""
    Endogeneity Risk Score: {score}/60

    Risk Level: {level}

    Interpretation:
    """

    if "Low" in level:
        interpretation += "Model appears relatively exogenous, though caution is always advised."
    elif "Moderate" in level:
        interpretation += "Some risk of omitted variable bias or simultaneity. Consider IV or dynamic methods."
    else:
        interpretation += "High probability of endogeneity. Instrumental variables or GMM strongly recommended."

    return interpretation


# --------------------------------------------------
# AI-Ready Summary Packager
# --------------------------------------------------

def package_endogeneity_results(score, level, details):
    summary_text = f"""
    ===============================
    ENDOGENEITY DIAGNOSTICS
    ===============================

    Risk Score: {score}
    Risk Level: {level}

    Detailed Diagnostics:
    {details}
    """

    return summary_text
