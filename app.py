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
from panel_models import run_fe, run_re, hausman, compute_vif, package_panel_results
from dynamic_panel import run_arellano_bond, interpret_dynamic_results, package_dynamic_results
from robustness import sensitivity, interpret_robustness, robustness_score, package_robustness_results
from var_module import run_var, check_var_stability, interpret_irf, package_var_results
from endogeneity import endogeneity_score, interpret_endogeneity, package_endogeneity_results, suggest_instruments
from diagnostics import heteroskedasticity, serial_corr

# ==========================
# AI INTERPRETER FUNCTION
# ==========================
def ai_interpret(results_text, context):
    api_key = st.secrets.get("GROQ_API_KEY")
    from groq import Groq
    client = Groq(api_key=api_key)
    MODEL = "openai/gpt-oss-120b"

    prompt = f"""
    You are a professional econometrician.

    Context: {context}

    Provide:
    1. Statistical interpretation
    2. Economic meaning
    3. Econometric concerns
    4. Policy implications (if relevant)

    Results:
    {results_text}
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


st.title("📊 AI-Augmented Econometric Research Laboratory")

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
# MAIN ECONOMETRIC MODULES
# ==========================
if page != "AI Assistant":

    uploaded = st.file_uploader("Upload CSV Data", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip().str.lower()

        st.write("Columns detected:", df.columns.tolist())

        entity = st.selectbox("Entity ID", df.columns)
        time_id = st.selectbox("Time ID", df.columns)
df_panel = df.set_index([entity, time_id]).sort_index()
        y_var = st.selectbox("Dependent Variable", df.columns)
        x_vars = st.multiselect("Independent Variables", df.columns)

        # ==========================
        # PANEL MODELS
        # ==========================
        if page == "Panel Models" and y_var and x_vars:

            fe_res = run_fe(df_panel, y_var, x_vars)
            re_res = run_re(df_panel, y_var, x_vars)
            h_stat, h_p, h_interp = hausman(fe_res, re_res)

            st.subheader("Fixed Effects")
            st.text(fe_res.summary)

            st.subheader("Random Effects")
            st.text(re_res.summary)

            st.write("Hausman p-value:", h_p)
            st.info(h_interp)

            vif_df = compute_vif(df, x_vars)
            st.subheader("VIF Diagnostics")
            st.write(vif_df)

            if st.button("AI Interpretation (Panel Model)"):
                summary = package_panel_results(fe_res, re_res, h_stat, h_p, h_interp)
                interpretation = ai_interpret(summary, "Panel Model with Hausman Test")
                st.markdown(interpretation)

        # ==========================
        # DYNAMIC PANEL
        # ==========================
        elif page == "Dynamic Panel" and y_var and x_vars:

            ab_res = run_arellano_bond(df, entity, time_id, y_var, x_vars)
            st.subheader("Dynamic Panel Results")
            st.text(ab_res.summary)

            dyn_interp = interpret_dynamic_results(ab_res)
            st.info(dyn_interp)

            if st.button("AI Interpretation (Dynamic Panel)"):
                summary = package_dynamic_results(ab_res)
                interpretation = ai_interpret(summary, "Dynamic Panel Model")
                st.markdown(interpretation)

        # ==========================
        # ENDOGENEITY
        # ==========================
        elif page == "Endogeneity & Instruments" and y_var and x_vars:

           fe_res = run_fe(df_panel, y_var, x_vars)
           re_res = run_re(df_panel, y_var, x_vars)
            h_stat, h_p, _ = hausman(fe_res, re_res)

            X = sm.add_constant(df[x_vars])
            vif_df = compute_vif(df, x_vars)
            bp_p = heteroskedasticity(fe_res.resids, X)

            score, level, details = endogeneity_score(h_p, vif_df["VIF"].max(), bp_p)

            st.write("Endogeneity Risk:", level)
            st.write(details)

            st.markdown(interpret_endogeneity(score, level))

            if level.startswith("High"):
                for var in x_vars:
                    st.write(f"Instruments for {var}: {suggest_instruments(var)}")

            if st.button("AI Interpretation (Endogeneity)"):
                summary = package_endogeneity_results(score, level, details)
                interpretation = ai_interpret(summary, "Endogeneity Diagnostics")
                st.markdown(interpretation)

        # ==========================
        # ROBUSTNESS
        # ==========================
        elif page == "Robustness & Sensitivity" and y_var and x_vars:

            stability_df = sensitivity(df_panel, y_var, x_vars)
            st.write(stability_df)

            interp = interpret_robustness(stability_df)
            score = robustness_score(stability_df)

            st.info(interp)
            st.write("Robustness Score (0-100):", score)

            if st.button("AI Interpretation (Robustness)"):
                summary = package_robustness_results(stability_df, interp, score)
                interpretation = ai_interpret(summary, "Robustness Analysis")
                st.markdown(interpretation)

        # ==========================
        # VAR
        # ==========================
        elif page == "VAR & IRF Analysis":

            df_macro = df.drop(columns=[entity, time_id], errors="ignore")
            var_res, lag = run_var(df_macro)

            st.write("Selected Lag:", lag)
            st.write(var_res.summary())

            stable, stability_interp = check_var_stability(var_res)
            st.info(stability_interp)

            try:
                irf = var_res.irf(10)
                fig = irf.plot(orth=False)
                st.pyplot(fig)

                irf_interp = interpret_irf(var_res)
                st.info(irf_interp)
            except:
                st.info("IRF plotting unavailable.")

            if st.button("AI Interpretation (VAR)"):
                summary = package_var_results(var_res, lag, stability_interp)
                interpretation = ai_interpret(summary, "VAR Model with IRF")
                st.markdown(interpretation)

    else:
        st.info("Upload a CSV file to begin.")

# ==========================
# AI CHAT ASSISTANT
# ==========================
elif page == "AI Assistant":

    st.header("🤖 EconLab AI Assistant")

    api_key = st.secrets.get("GROQ_API_KEY")
    from groq import Groq
    client = Groq(api_key=api_key)
    MODEL = "openai/gpt-oss-120b"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask an econometrics question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=MODEL,
            messages=st.session_state.messages,
            temperature=0.3
        )

        answer = response.choices[0].message.content
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
