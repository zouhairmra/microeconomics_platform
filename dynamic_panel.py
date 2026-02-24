from linearmodels.panel import PanelOLS
import statsmodels.api as sm
def run_arellano_bond(df, entity, time, y_var, x_vars):
    df = df.set_index([entity, time])
    
    # Create lagged dependent variable
    df["lag_y"] = df.groupby(level=0)[y_var].shift(1)
    df = df.dropna()

    y = df[y_var]
    X = df[["lag_y"] + x_vars]
    X = sm.add_constant(X)

    model = PanelOLS(y, X, entity_effects=True)
    results = model.fit(cov_type="robust")

    return results
