import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _require_panel_index(df: pd.DataFrame):
    """Ensure df has a 2-level MultiIndex required by linearmodels PanelOLS/RE."""
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
        raise ValueError(
            "PanelOLS/RandomEffects require a 2-level MultiIndex (entity, time).\n"
            "In your app, set it as: df = df.set_index(['country','year']).sort_index()"
        )

def _to_numeric(df: pd.DataFrame, cols: list):
    """Convert specified columns to numeric (coerce errors to NaN)."""
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# --------------------------------------------------
# Fixed Effects Model
# --------------------------------------------------

def run_fe(df, y_var, x_vars, cov_type="clustered", cluster_entity=True):
    """
    Fixed Effects using PanelOLS with entity effects.

    Parameters:
      cov_type: 'robust' or 'clustered' (recommended)
      cluster_entity: if cov_type='clustered', cluster by entity
    """
    _require_panel_index(df)

    if y_var not in df.columns:
        raise KeyError(f"{y_var} not found in dataframe.")

    for var in x_vars:
        if var not in df.columns:
            raise KeyError(f"{var} not found in dataframe.")

    # Ensure numeric y and X
    y = pd.to_numeric(df[y_var], errors="coerce")
    X = _to_numeric(df, x_vars)

    # Add constant
    X = sm.add_constant(X, has_constant="add")

    # Drop missing values consistently
    combined = pd.concat([y.rename(y_var), X], axis=1).dropna()
    y = combined[y_var]
    X = combined.drop(columns=[y_var])

    # FE model (drop_absorbed removes collinear variables automatically)
    model = PanelOLS(y, X, entity_effects=True, drop_absorbed=True)

    # Fit with appropriate covariance
    if cov_type == "clustered":
        results = model.fit(cov_type="clustered", cluster_entity=cluster_entity)
    else:
        results = model.fit(cov_type="robust")

    return results


# --------------------------------------------------
# Random Effects Model
# --------------------------------------------------

def run_re(df, y_var, x_vars, cov_type="clustered", cluster_entity=True):
    """
    Random Effects using RandomEffects.

    Parameters:
      cov_type: 'unadjusted', 'robust', or 'clustered' (recommended)
      cluster_entity: if cov_type='clustered', cluster by entity
    """
    _require_panel_index(df)

    if y_var not in df.columns:
        raise KeyError(f"{y_var} not found in dataframe.")

    for var in x_vars:
        if var not in df.columns:
            raise KeyError(f"{var} not found in dataframe.")

    # Ensure numeric y and X
    y = pd.to_numeric(df[y_var], errors="coerce")
    X = _to_numeric(df, x_vars)

    # Add constant
    X = sm.add_constant(X, has_constant="add")

    # Drop missing values consistently
    combined = pd.concat([y.rename(y_var), X], axis=1).dropna()
    y = combined[y_var]
    X = combined.drop(columns=[y_var])

    model = RandomEffects(y, X)

    # Fit with covariance type
    if cov_type == "clustered":
        results = model.fit(cov_type="clustered", cluster_entity=cluster_entity)
    elif cov_type == "robust":
        results = model.fit(cov_type="robust")
    else:
        results = model.fit()

    return results


# --------------------------------------------------
# Hausman Test (Robust + Safe)
# --------------------------------------------------

def hausman(fe_res, re_res):
    """
    Hausman test comparing FE vs RE.
    Uses pseudo-inverse if covariance difference is singular.
    """
    b_FE = fe_res.params
    b_RE = re_res.params

    common = b_FE.index.intersection(b_RE.index)

    diff = (b_FE[common] - b_RE[common]).values

    cov_FE = fe_res.cov.loc[common, common].values
    cov_RE = re_res.cov.loc[common, common].values
    cov_diff = cov_FE - cov_RE

    # Use pseudo-inverse to avoid singular failures
    inv_cov = np.linalg.pinv(cov_diff)

    stat = float(diff.T @ inv_cov @ diff)
    df_h = len(common)
    pval = 1 - stats.chi2.cdf(stat, df_h)

    if pval < 0.05:
        interpretation = "Reject H0 → Fixed Effects preferred (endogeneity likely)."
    else:
        interpretation = "Fail to reject H0 → Random Effects acceptable."

    return stat, pval, interpretation


# --------------------------------------------------
# VIF Diagnostic (Numeric + Cleaner)
# --------------------------------------------------

def compute_vif(df, x_vars):
    """
    Computes VIF for regressors.
    Note: VIF for constant is typically not informative; we drop it from output.
    """
    X = _to_numeric(df, x_vars)
    X = X.dropna()

    # Add constant for stability (then exclude it from output)
    Xc = sm.add_constant(X, has_constant="add")

    vif_data = pd.DataFrame()
    vif_data["Variable"] = Xc.columns
    vif_data["VIF"] = [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]

    # Drop the constant row (its VIF is often huge/infinite and not meaningful)
    vif_data = vif_data[vif_data["Variable"] != "const"].reset_index(drop=True)

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
