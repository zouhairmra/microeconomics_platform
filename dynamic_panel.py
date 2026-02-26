from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import pandas as pd
import numpy as np


# --------------------------------------------------
# Dynamic Panel Model (Arellano-Bond style approximation)
# --------------------------------------------------

def run_arellano_bond(df, entity_col, time_col, y_var, x_vars):
    """
    Dynamic panel estimation using lagged dependent variable
    with entity fixed effects (approximation to Arellano-Bond).
    """

    # Copy to avoid modifying original
    df = df.copy()

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    entity_col = entity_col.strip().lower()
    time_col = time_col.strip().lower()
    y_var = y_var.strip().lower()
    x_vars = [x.strip().lower() for x in x_vars]

    # -----------------------------
    # Column validation
    # -----------------------------
    required = [entity_col, time_col, y_var] + x_vars
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(
            f"Missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # -----------------------------
    # Set MultiIndex
    # -----------------------------
    df = df.set_index([entity_col, time_col])
    df = df.sort_index()

    # -----------------------------
    # Create lagged dependent variable
    # -----------------------------
    df["lag_y"] = df.groupby(level=0)[y_var].shift(1)

    # Drop rows with missing lag
    df = df.dropna()

    if df.empty:
        raise ValueError("Dataset empty after lagging. Not enough time periods.")

    # -----------------------------
    # Define variables
    # -----------------------------
    y = df[y_var]
    X = df[["lag_y"] + x_vars]
    X = sm.add_constant(X)

    # -----------------------------
    # Estimate model
    # -----------------------------
    model = PanelOLS(y, X, entity_effects=True)
    results = model.fit(cov_type="robust")

    return results


# --------------------------------------------------
# Dynamic Interpretation Helper
# --------------------------------------------------

def interpret_dynamic_results(results):
    """
    Provides automatic economic interpretation
    of lag coefficient.
    """

    if "lag_y" not in results.params.index:
        return "Lag variable not found in results."

    lag_coef = results.params["lag_y"]

    interpretation = ""

    if lag_coef > 0:
        interpretation += "Positive lag coefficient → Evidence of persistence. "
    else:
        interpretation += "Negative lag coefficient → Possible mean reversion. "

    if abs(lag_coef) >= 1:
        interpretation += "Warning: |lag coefficient| ≥ 1 suggests potential non-stationarity or explosive dynamics."
    else:
        interpretation += "System appears dynamically stable (|lag| < 1)."

    return interpretation


# --------------------------------------------------
# AI-Ready Summary Packager
# --------------------------------------------------

def package_dynamic_results(results):
    summary_text = f"""
    ===============================
    DYNAMIC PANEL MODEL RESULTS
    ===============================
    {results.summary}

    Key Coefficient:
    Lagged Dependent Variable (lag_y): {results.params.get("lag_y", "N/A")}
    """

    return summary_text
