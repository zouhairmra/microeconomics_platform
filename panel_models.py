import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# --------------------------------------------------
# Fixed Effects Model
# --------------------------------------------------

def run_fe(df, y_var, x_vars):
    if y_var not in df.columns:
        raise KeyError(f"{y_var} not found in dataframe.")

    for var in x_vars:
        if var not in df.columns:
            raise KeyError(f"{var} not found in dataframe.")

    y = df[y_var]
    X = sm.add_constant(df[x_vars])

    model = PanelOLS(y, X, entity_effects=True)
    results = model.fit(cov_type="robust")

    return results


# --------------------------------------------------
# Random Effects Model
# --------------------------------------------------

def run_re(df, y_var, x_vars):
    if y_var not in df.columns:
        raise KeyError(f"{y_var} not found in dataframe.")

    for var in x_vars:
        if var not in df.columns:
            raise KeyError(f"{var} not found in dataframe.")

    y = df[y_var]
    X = sm.add_constant(df[x_vars])

    model = RandomEffects(y, X)
    results = model.fit()

    return results


# --------------------------------------------------
# Hausman Test (Robust Version)
# --------------------------------------------------

def hausman(fe_res, re_res):
    b_FE = fe_res.params
    b_RE = re_res.params

    common = b_FE.index.intersection(b_RE.index)

    diff = b_FE[common] - b_RE[common]
    cov_diff = fe_res.cov.loc[common, common] - re_res.cov.loc[common, common]

    try:
        inv_cov = np.linalg.inv(cov_diff)
    except np.linalg.LinAlgError:
        return None, None, "Hausman test failed (singular covariance matrix)."

    stat = np.dot(np.dot(diff.T, inv_cov), diff)
    pval = 1 - stats.chi2.cdf(stat, len(diff))

    # Automatic interpretation
    if pval < 0.05:
        interpretation = "Reject H0 → Fixed Effects preferred (endogeneity likely)."
    else:
        interpretation = "Fail to reject H0 → Random Effects acceptable."

    return stat, pval, interpretation


# --------------------------------------------------
# VIF Diagnostic
# --------------------------------------------------

def compute_vif(df, x_vars):
    X = sm.add_constant(df[x_vars])
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    return vif_data


def interpret_vif(vif_df):
    max_vif = vif_df["VIF"].max()

    if max_vif < 5:
        return "No serious multicollinearity detected."
    elif max_vif < 10:
        return "Moderate multicollinearity risk."
    else:
        return "High multicollinearity detected. Results may be unstable."


# --------------------------------------------------
# AI-Ready Summary Packager
# --------------------------------------------------

def package_panel_results(fe_res, re_res, hausman_stat, hausman_p, hausman_interp):
    summary_text = f"""
    ===============================
    FIXED EFFECTS RESULTS
    ===============================
    {fe_res.summary}

    ===============================
    RANDOM EFFECTS RESULTS
    ===============================
    {re_res.summary}

    ===============================
    HAUSMAN TEST
    ===============================
    Statistic: {hausman_stat}
    P-value: {hausman_p}
    Interpretation: {hausman_interp}
    """

    return summary_text
