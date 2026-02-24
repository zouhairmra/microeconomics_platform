from linearmodels.panel import PanelOLS
import statsmodels.api as sm

def run_arellano_bond(df, y_var, x_vars):
    y = df[y_var]
    X = df[x_vars]
    model = ArellanoBond(y, X)
    return model.fit()
