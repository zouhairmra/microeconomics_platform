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

   st.header("🤖 EconLab AI Assistant (Powered by Groq)")

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        st.error("GROQ_API_KEY not found in Streamlit Secrets.")
        st.stop()

    client = Groq(api_key=api_key)

    MODEL = st.selectbox(
        "Select Model",
        [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768"
        ]
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask an economics or econometrics question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages,
                    temperature=0.3
                )

                response_text = response.choices[0].message.content

                for token in response_text.split():
                    full_response += token + " "
                    placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"Groq API Error: {e}")
                full_response = f"Error: {e}"

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
