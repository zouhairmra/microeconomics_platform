from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

def adf_test(series):
    return adfuller(series)[1]

def run_var(df):
    model = VAR(df)
    lag = model.select_order(5).aic
    results = model.fit(lag)
    return results, lag
