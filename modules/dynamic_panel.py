from linearmodels.panel import ArellanoBond

def run_arellano_bond(df, y_var, x_vars):
    y = df[y_var]
    X = df[x_vars]
    model = ArellanoBond(y, X)
    return model.fit()
