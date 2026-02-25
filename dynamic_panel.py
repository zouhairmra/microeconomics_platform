from linearmodels.panel import PanelOLS
import statsmodels.api as sm

def run_arellano_bond(df, entity_col, time_col, y_var, x_vars):
    # Copy to avoid modifying original
    df = df.copy()

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    entity_col = entity_col.strip().lower()
    time_col = time_col.strip().lower()
    y_var = y_var.strip().lower()
    x_vars = [x.strip().lower() for x in x_vars]

    # Check that entity and time exist
    if entity_col not in df.columns or time_col not in df.columns:
        available = df.columns.tolist()
        raise KeyError(
            f"Entity or Time column not found. "
            f"Got: {entity_col}, {time_col}. "
            f"Available columns: {available}"
        )

    # Set MultiIndex
    df = df.set_index([entity_col, time_col])

    # Create lagged dependent variable
    df["lag_y"] = df.groupby(level=0)[y_var].shift(1)
    df = df.dropna()

    # Define dependent and independent variables
    y = df[y_var]
    X = df[["lag_y"] + x_vars]
    X = sm.add_constant(X)

    # Fit PanelOLS with entity effects
    model = PanelOLS(y, X, entity_effects=True)
    results = model.fit(cov_type="robust")

    return results
