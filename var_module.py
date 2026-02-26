from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np


# --------------------------------------------------
# Augmented Dickey-Fuller Test
# --------------------------------------------------

def adf_test(series):
    """
    Returns p-value of ADF test.
    """
    result = adfuller(series.dropna())
    return result[1]


def check_stationarity(df):
    """
    Runs ADF test on each column.
    """
    stationarity_results = {}

    for col in df.columns:
        pval = adf_test(df[col])
        if pval < 0.05:
            stationarity_results[col] = "Stationary"
        else:
            stationarity_results[col] = "Non-stationary"

    return stationarity_results


# --------------------------------------------------
# VAR Estimation
# --------------------------------------------------

def run_var(df, max_lags=5):
    """
    Estimate VAR model using AIC-selected lag length.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    if df.shape[1] < 2:
        raise ValueError("VAR requires at least two variables.")

    df = df.dropna()

    model = VAR(df)

    try:
        lag = model.select_order(max_lags).aic
        if lag is None:
            lag = 1
    except Exception:
        lag = 1

    results = model.fit(lag)

    return results, lag


# --------------------------------------------------
# Stability Check
# --------------------------------------------------

def check_var_stability(results):
    """
    Checks if VAR satisfies stability condition (all roots < 1).
    """
    roots = results.roots
    stable = np.all(np.abs(roots) < 1)

    if stable:
        interpretation = "VAR is stable (all roots inside unit circle)."
    else:
        interpretation = "VAR is unstable (at least one root outside unit circle)."

    return stable, interpretation


# --------------------------------------------------
# IRF Interpretation Helper
# --------------------------------------------------

def interpret_irf(results, steps=10):
    """
    Provides qualitative interpretation guidance for IRFs.
    """
    irf = results.irf(steps)
    irf_values = irf.irfs

    max_response = np.max(np.abs(irf_values))

    interpretation = ""

    if max_response < 0.1:
        interpretation += "Impulse responses are economically small. "
    else:
        interpretation += "Impulse responses are economically meaningful. "

    interpretation += "Check persistence and sign of responses for policy relevance."

    return interpretation


# --------------------------------------------------
# AI-Ready Summary Packager
# --------------------------------------------------

def package_var_results(results, lag, stability_interp):
    summary_text = f"""
    ===============================
    VECTOR AUTOREGRESSION (VAR)
    ===============================

    Selected Lag Length (AIC): {lag}

    ===============================
    MODEL SUMMARY
    ===============================
    {results.summary()}

    ===============================
    STABILITY
    ===============================
    {stability_interp}
    """

    return summary_text
