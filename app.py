# app.py
import streamlit as st
st.set_page_config(page_title="AI-Augmented Econometrics Lab", layout="wide")

import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import os

sys.path.insert(0, os.path.abspath("."))

# Modules
from panel_models import run_fe, run_re, hausman
from dynamic_panel import run_arellano_bond
from diagnostics import compute_vif, heteroskedasticity, serial_corr
from endogeneity import endogeneity_score, suggest_instruments
from robustness import sensitivity
from var_module import run_var
from llm_engine import query_phi3

st.title("📊 AI-Augmented Econometric Research Laboratory")

# Sidebar: Module selection
page = st.sidebar.selectbox(
    "Select Module",
    [
        "Panel Models",
        "Dynamic Panel",
        "Endogeneity & Instruments",
        "Robustness & Sensitivity",
        "VAR & IRF Analysis",
        "AI Policy Interpreter"
    ]
)

# CSV Upload
uploaded = st.file_uploader("Upload CSV Data", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    # Entity and Time IDs for panel data
    entity = st.selectbox("Entity ID", df.columns)
    time = st.selectbox("Time ID", df.columns)
    df = df.set_index([entity, time])

    y_var = st.selectbox("Dependent Variable", df.columns)
    x_vars = st.multiselect("Independent Variables", df.columns)

    # ----------------------------
    # 1️⃣ Panel Models
    # ----------------------------
    if page == "Panel Models":
        st.subheader("Panel Models: FE / RE / Hausman")
        fe_res = run_fe(df, y_var, x_vars)
        re_res = run_re(df, y_var, x_vars)

        st.text(fe_res.summary)
        st.text(re_res.summary)

        h_stat, h_p = hausman(fe_res, re_res)
        st.write("Hausman p-value:", h_p)

    # ----------------------------
    # 2️⃣ Dynamic Panel
    # ----------------------------
    elif page == "Dynamic Panel":
        st.subheader("Dynamic Panel (Arellano-Bond)")
        ab_res = run_arellano_bond(df, entity, time, y_var, x_vars)
        st.text(ab_res.summary)

    # ----------------------------
    # 3️⃣ Endogeneity & Instruments
    # ----------------------------
    elif page == "Endogeneity & Instruments":
        st.subheader("Endogeneity Risk & Suggested Instruments")

        fe_res = run_fe(df, y_var, x_vars)
        re_res = run_re(df, y_var, x_vars)
        h_stat, h_p = hausman(fe_res, re_res)

        X = sm.add_constant(df[x_vars])
        vif = compute_vif(X)
        bp_p = heteroskedasticity(fe_res.resids, X)
        dw = serial_corr(fe_res.resids)

        score, level = endogeneity_score(h_p, vif["VIF"].max(), bp_p)
        st.write("Endogeneity Risk Level:", level)

        if level == "High":
            instruments = suggest_instruments(df, x_vars)
            st.write("Suggested Instruments:")
            st.write(instruments)

    # ----------------------------
    # 4️⃣ Robustness & Sensitivity
    # ----------------------------
    elif page == "Robustness & Sensitivity":
        st.subheader("Robustness & Sensitivity Analysis")
        stability = sensitivity(df, y_var, x_vars)
        st.write("Coefficient Stability Across Specifications")
        st.write(stability)

    # ----------------------------
    # 5️⃣ VAR & IRF
    # ----------------------------
    elif page == "VAR & IRF Analysis":
        st.subheader("VAR Macro Analysis & Impulse Response")
        df_macro = df.reset_index().drop(columns=[entity, time])
        var_res, lag = run_var(df_macro)
        st.write("Optimal Lag:", lag)
        st.write(var_res.summary())

        try:
            irf = var_res.irf(10)
            fig = irf.plot(orth=False)
            st.pyplot(fig)
        except Exception:
            st.info("IRF plotting not available for this VAR model.")

    # ----------------------------
    # 6️⃣ AI Policy Interpreter
    # ----------------------------
    elif page == "AI Policy Interpreter":
        st.subheader("AI Policy Interpreter")

        user_input = st.text_area(
            "Paste regression diagnostics or summary (FE/RE, Hausman, VIF, BP, DW, etc.)"
        )

        if st.button("Generate Policy Discussion"):
            if not user_input.strip():
                st.warning("Please provide regression output first.")
            else:
                interpretation = query_phi3(user_input)
                st.write(interpretation)

else:
    st.info("Upload a CSV file to begin analysis.")
