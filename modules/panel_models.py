import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
import numpy as np
from scipy import stats

def run_fe(df, y_var, x_vars):
    y = df[y_var]
    X = sm.add_constant(df[x_vars])
    model = PanelOLS(y, X, entity_effects=True)
    return model.fit(cov_type="robust")

def run_re(df, y_var, x_vars):
    y = df[y_var]
    X = sm.add_constant(df[x_vars])
    model = RandomEffects(y, X)
    return model.fit()

def hausman(fe_res, re_res):
    b_FE = fe_res.params
    b_RE = re_res.params
    common = b_FE.index.intersection(b_RE.index)

    diff = b_FE[common] - b_RE[common]
    cov_diff = fe_res.cov.loc[common, common] - re_res.cov.loc[common, common]

    stat = np.dot(np.dot(diff.T, np.linalg.inv(cov_diff)), diff)
    pval = 1 - stats.chi2.cdf(stat, len(diff))

    return stat, pval
