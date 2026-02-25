from linearmodels.panel import PanelOLS
import statsmodels.api as sm

def run_arellano_bond(df, entity_col, time_col, y_var, x_vars):
    # Clean column names (remove leading/trailing spaces)
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Check if entity and time columns exist
    if entity_col not in df.columns or time_col not in df.columns:
        raise KeyError(
            f"Entity or Time column not found. "
            f"Got: {entity_col}, {time_col}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Set MultiIndex
    df = df.set_index([entity_col, time_col])

    # Create lagged dependent variable
    df["lag_y"] = df.groupby(level=0)[y_var].shift(1)

    # Drop missing values after lagging
    df = df.dropna()

    # Define dependent and independent variables
    y = df[y_var]
    X = df[["lag_y"] + x_vars]
    X = sm.add_constant(X)

    # Fit Arellano-Bond-style PanelOLS with entity effects
    model = PanelOLS(y, X, entity_effects=True)
    results = model.fit(cov_type="robust")

    return results
