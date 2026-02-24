import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from panel_models import run_fe, run_re, hausman
from dynamic_panel import run_arellano_bond
from diagnostics import compute_vif, heteroskedasticity, serial_corr
from endogeneity import endogeneity_score, suggest_instruments
from robustness import sensitivity
from var_module import run_var
from llm_engine import query_phi3
st.set_page_config(layout="wide")
st.title("AI-Augmented Econometric Research Laboratory")

page = st.sidebar.selectbox(
    "Select Module",
    ["Panel Models", "Dynamic Panel", "VAR Analysis"]
)

uploaded = st.file_uploader("Upload CSV Data", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    entity = st.selectbox("Entity ID", df.columns)
    time = st.selectbox("Time ID", df.columns)
    df = df.set_index([entity, time])

    y_var = st.selectbox("Dependent Variable", df.columns)
    x_vars = st.multiselect("Independent Variables", df.columns)

    if page == "Panel Models":

        fe_res = run_fe(df, y_var, x_vars)
        re_res = run_re(df, y_var, x_vars)

        st.text(fe_res.summary)
        st.text(re_res.summary)

        h_stat, h_p = hausman(fe_res, re_res)
        st.write("Hausman p-value:", h_p)

        X = sm.add_constant(df[x_vars])
        vif = compute_vif(X)
        bp_p = heteroskedasticity(fe_res.resids, X)
        dw = serial_corr(fe_res.resids)

        score, level = endogeneity_score(h_p, vif["VIF"].max(), bp_p)

        st.write("Endogeneity Risk:", level)

        stability = sensitivity(df, y_var, x_vars)

        results_dict = {
            "hausman_p": h_p,
            "max_vif": float(vif["VIF"].max()),
            "heteroskedasticity_p": bp_p,
            "durbin_watson": dw,
            "endogeneity_risk": level
        }

        prompt = f"""
        Based on:
        {results_dict}

        Provide academic interpretation and policy discussion.
        Do not invent coefficients.
        """

        interpretation = query_phi3(prompt)

        st.subheader("AI Interpretation")
        st.write(interpretation)

    elif page == "Dynamic Panel":
        ab_res = run_arellano_bond(df, y_var, x_vars)
        st.text(ab_res.summary)

    elif page == "VAR Analysis":
        df_macro = df.reset_index().drop(columns=[entity, time])
        var_res, lag = run_var(df_macro)
        st.write("Optimal Lag:", lag)
        st.write(var_res.summary())
