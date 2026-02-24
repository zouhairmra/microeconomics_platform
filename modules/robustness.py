from linearmodels.panel import PanelOLS
import statsmodels.api as sm

def sensitivity(df, y_var, x_vars):
    base_model = PanelOLS(
        df[y_var],
        sm.add_constant(df[x_vars]),
        entity_effects=True
    ).fit()

    base_coef = base_model.params

    stability = {}

    for var in x_vars:
        reduced = [v for v in x_vars if v != var]
        res = PanelOLS(
            df[y_var],
            sm.add_constant(df[reduced]),
            entity_effects=True
        ).fit()

        for coef_var in base_coef.index:
            if coef_var in res.params:
                csi = abs(base_coef[coef_var] - res.params[coef_var]) / abs(base_coef[coef_var])
                stability[(var, coef_var)] = csi

    return stability
