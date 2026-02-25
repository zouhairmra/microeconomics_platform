# app.py
import streamlit as st
st.set_page_config(page_title="AI-Augmented Econometrics Lab", layout="wide")

import pandas as pd
import numpy as np
import statsmodels.api as sm
import requests
import time
import sys
import os

sys.path.insert(0, os.path.abspath("."))

# Econometrics modules
from panel_models import run_fe, run_re, hausman
from dynamic_panel import run_arellano_bond
from diagnostics import compute_vif, heteroskedasticity, serial_corr
from endogeneity import endogeneity_score, suggest_instruments
from robustness import sensitivity
from var_module import run_var

st.title("📊 AI-Augmented Econometric Research Laboratory")

# ==========================
# SIDEBAR
# ==========================
page = st.sidebar.selectbox(
    "Select Module",
    [
        "Panel Models",
        "Dynamic Panel",
        "Endogeneity & Instruments",
        "Robustness & Sensitivity",
        "VAR & IRF Analysis",
        "AI Assistant"
    ]
)

# ==========================
# ECONOMETRICS MODULES
# ==========================
if page != "AI Assistant":

    uploaded = st.file_uploader("Upload CSV Data", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)

        entity = st.selectbox("Entity ID", df.columns)
        time_id = st.selectbox("Time ID", df.columns)
        df = df.set_index([entity, time_id])

        y_var = st.selectbox("Dependent Variable", df.columns)
        x_vars = st.multiselect("Independent Variables", df.columns)

        if page == "Panel Models":
            fe_res = run_fe(df, y_var, x_vars)
            re_res = run_re(df, y_var, x_vars)

            st.text(fe_res.summary)
            st.text(re_res.summary)

            h_stat, h_p = hausman(fe_res, re_res)
            st.write("Hausman p-value:", h_p)

        elif page == "Dynamic Panel":
            ab_res = run_arellano_bond(df, entity, time_id, y_var, x_vars)
            st.text(ab_res.summary)

        elif page == "Endogeneity & Instruments":
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

        elif page == "Robustness & Sensitivity":
            stability = sensitivity(df, y_var, x_vars)
            st.write(stability)

        elif page == "VAR & IRF Analysis":
            df_macro = df.reset_index().drop(columns=[entity, time_id])
            var_res, lag = run_var(df_macro)
            st.write("Optimal Lag:", lag)
            st.write(var_res.summary())

            try:
                irf = var_res.irf(10)
                fig = irf.plot(orth=False)
                st.pyplot(fig)
            except:
                st.info("IRF plotting unavailable.")

    else:
        st.info("Upload a CSV file to begin analysis.")

# ==========================
# AI ASSISTANT (POE API)
# ==========================
elif page == "AI Assistant":

    st.header("🤖 EconLab AI Assistant")

    POE_API_URL = "https://poe.com/api/keys"
    POE_API_KEY = st.secrets.get("POE_API_KEY", "")

    MODEL = st.selectbox("Select model", ["maztouriabot", "gpt-4o-mini", "claude-3-haiku"])

    uploaded_file = st.file_uploader("Upload PDF or CSV", type=["pdf", "csv"])
    uploaded_text = ""

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_ai = pd.read_csv(uploaded_file)
            st.dataframe(df_ai.head())
            uploaded_text = df_ai.to_string(index=False)
        else:
            st.warning("PDF extraction requires PyPDF2 in requirements.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                headers = {
                    "Authorization": f"Bearer {POE_API_KEY}",
                    "Content-Type": "application/json"
                }

                content = (
                    f"File content:\n{uploaded_text[:4000]}\n\nQuestion: {user_input}"
                    if uploaded_text else user_input
                )

                payload = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": content}]
                }

                res = requests.post(POE_API_URL, headers=headers, json=payload)
                res.raise_for_status()
                response_text = res.json()["choices"][0]["message"]["content"]

                for token in response_text.split():
                    full_response += token + " "
                    placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)

                placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"API Error: {e}")
                full_response = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": full_response})
