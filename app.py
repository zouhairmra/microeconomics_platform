# app.py
import streamlit as st
st.set_page_config(page_title="AI-Augmented Econometrics Lab", layout="wide")

import pandas as pd
import numpy as np
import statsmodels.api as sm
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
        df.columns = df.columns.str.strip().str.lower()  # normalize headers

        st.write("Columns detected in CSV:", df.columns.tolist())

        # Select entity and time columns
        entity = st.selectbox("Entity ID", df.columns)
        time_id = st.selectbox("Time ID", df.columns)

        if entity not in df.columns or time_id not in df.columns:
            st.warning(f"Selected columns not found in data: {entity}, {time_id}")
        else:
            df_indexed = df.copy()
            y_var = st.selectbox("Dependent Variable", df_indexed.columns)
            x_vars = st.multiselect("Independent Variables", df_indexed.columns)

            # Panel Models
            if page == "Panel Models":
                if y_var and x_vars:
                    fe_res = run_fe(df_indexed, y_var, x_vars)
                    re_res = run_re(df_indexed, y_var, x_vars)

                    st.subheader("Fixed Effects Results")
                    st.text(fe_res.summary)
                    st.subheader("Random Effects Results")
                    st.text(re_res.summary)

                    h_stat, h_p = hausman(fe_res, re_res)
                    st.write("Hausman p-value:", h_p)
                else:
                    st.info("Please select dependent and independent variables.")

            # Dynamic Panel
            elif page == "Dynamic Panel":
                if y_var and x_vars:
                    try:
                        ab_res = run_arellano_bond(df_indexed, entity, time_id, y_var, x_vars)
                        st.subheader("Arellano-Bond PanelOLS Results")
                        st.text(ab_res.summary)
                    except KeyError as e:
                        st.error(f"Column Error: {e}")
                    except Exception as e:
                        st.error(f"Error running Arellano-Bond: {e}")
                else:
                    st.info("Please select dependent and independent variables.")

            # Endogeneity & Instruments
            elif page == "Endogeneity & Instruments":
                if y_var and x_vars:
                    fe_res = run_fe(df_indexed, y_var, x_vars)
                    re_res = run_re(df_indexed, y_var, x_vars)
                    h_stat, h_p = hausman(fe_res, re_res)

                    X = sm.add_constant(df_indexed[x_vars])
                    vif = compute_vif(X)
                    bp_p = heteroskedasticity(fe_res.resids, X)
                    dw = serial_corr(fe_res.resids)

                    score, level = endogeneity_score(h_p, vif["VIF"].max(), bp_p)
                    st.write("Endogeneity Risk Level:", level)

                    if level == "High":
                        instruments = suggest_instruments(df_indexed, x_vars)
                        st.write("Suggested Instruments:")
                        st.write(instruments)
                else:
                    st.info("Please select dependent and independent variables.")

            # Robustness & Sensitivity
            elif page == "Robustness & Sensitivity":
                if y_var and x_vars:
                    stability = sensitivity(df_indexed, y_var, x_vars)
                    st.write(stability)
                else:
                    st.info("Please select dependent and independent variables.")

            # VAR & IRF Analysis
            elif page == "VAR & IRF Analysis":
                df_macro = df_indexed.reset_index().drop(columns=[entity, time_id])
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
# AI ASSISTANT (Groq)
# ==========================
elif page == "AI Assistant":

    st.header("🤖 EconLab AI Assistant (Powered by Groq)")

    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in Streamlit Secrets.")
        st.stop()

    from groq import Groq
    client = Groq(api_key=api_key)
    MODEL = "openai/gpt-oss-120b"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask an economics question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages,
                    temperature=0.3
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error contacting AI assistant: {e}")
