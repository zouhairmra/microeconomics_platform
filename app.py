# app.py
import streamlit as st
st.set_page_config(page_title="AI-Augmented Econometrics Lab", layout="wide")

import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import os
import json

sys.path.insert(0, os.path.abspath("."))

# Econometrics modules
from panel_models import run_fe, run_re, hausman, compute_vif, package_panel_results
from dynamic_panel import run_arellano_bond, interpret_dynamic_results, package_dynamic_results
from robustness import sensitivity, interpret_robustness, robustness_score, package_robustness_results
from var_module import run_var, check_var_stability, interpret_irf, package_var_results
from endogeneity import endogeneity_score, interpret_endogeneity, package_endogeneity_results, suggest_instruments
from diagnostics import heteroskedasticity, serial_corr


# ==========================
# PANEL PREPARATION HELPERS
# ==========================
def prepare_panel_df(df: pd.DataFrame, entity_col: str, time_col: str) -> pd.DataFrame:
    """
    Returns a copy of df indexed as a 2-level MultiIndex (entity, time),
    cleaned, sorted, and de-duplicated (one row per entity-time).
    Required by linearmodels.PanelOLS (FE/RE).
    """
    d = df.copy()

    # Standardize entity
    d[entity_col] = d[entity_col].astype(str).str.strip()

    # Convert time: numeric first (good for year), else datetime
    time_num = pd.to_numeric(d[time_col], errors="coerce")
    if time_num.notna().mean() > 0.80:
        d[time_col] = time_num.astype("Int64")
    else:
        d[time_col] = pd.to_datetime(d[time_col], errors="coerce")

    # Drop rows with missing keys
    d = d.dropna(subset=[entity_col, time_col])

    # Set MultiIndex
    d = d.set_index([entity_col, time_col]).sort_index()

    # De-duplicate: one obs per entity-time
    if d.index.duplicated().any():
        d = d.groupby(level=[0, 1]).mean(numeric_only=True)

    return d


def safe_numeric_frame(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Convert selected columns to numeric.
    """
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


# ==========================
# (1) STORE LAST RUN (Dynamic memory)
# ==========================
def store_last_run(module_name, entity, time_id, y_var, x_vars, summary_text, extra=None):
    st.session_state["last_run"] = {
        "module": module_name,
        "entity": entity,
        "time": time_id,
        "y": y_var,
        "x": x_vars,
        "summary": summary_text,
        "extra": extra or {}
    }


# ==========================
# (2) ECONOMETRICS CONTEXT PACKET
# ==========================
def build_econ_context(df, module_name, entity, time_id, y_var, x_vars, notes=None):
    return {
        "module": module_name,
        "panel_keys": {"entity": entity, "time": time_id},
        "dependent": y_var,
        "regressors": x_vars,
        "n_rows": int(len(df)),
        "notes": notes or {}
    }


# ==========================
# (7) COMPACT RESULTS PACKET (better AI inputs)
# ==========================
def compact_results_packet(result_obj, model_name=""):
    packet = {"model": model_name}
    try:
        packet["params"] = {k: float(v) for k, v in result_obj.params.to_dict().items()}
    except:
        pass
    try:
        packet["std_errors"] = {k: float(v) for k, v in result_obj.std_errors.to_dict().items()}
    except:
        pass
    try:
        packet["pvalues"] = {k: float(v) for k, v in result_obj.pvalues.to_dict().items()}
    except:
        pass
    for attr in ["nobs", "ntotal", "observations"]:
        if hasattr(result_obj, attr):
            try:
                packet["nobs"] = int(getattr(result_obj, attr))
                break
            except:
                pass
    for attr in ["rsquared", "rsquared_within", "rsquared_overall"]:
        if hasattr(result_obj, attr):
            try:
                packet[attr] = float(getattr(result_obj, attr))
            except:
                pass
    return packet


# ==========================
# (3) AI INTERPRETER (Econometrics-aware, multi-mode)
# ==========================
def ai_interpret(results_text, context_dict, mode="Referee Report", audience="Graduate", temperature=0.2):
    api_key = st.secrets.get("GROQ_API_KEY")
    from groq import Groq
    client = Groq(api_key=api_key)
    MODEL = "openai/gpt-oss-120b"

    context_json = json.dumps(context_dict, ensure_ascii=False)

    system_prompt = """
You are a senior econometrician and applied economist.

Hard rules:
- Always distinguish correlation from causal interpretation.
- If causal language is used, state the identification assumptions explicitly.
- Diagnose threats: omitted variables, simultaneity, reverse causality, measurement error, selection, dynamics.
- Be model-specific:
  * Panel FE/RE: strict exogeneity, within variation, time-varying confounding, clustering, serial correlation.
  * Dynamic panel (Arellano–Bond): Nickell bias, AR(1)/AR(2), Hansen/Sargan, instrument proliferation.
  * VAR/IRF: stationarity, stability, lag selection, identification, interpreting IRFs carefully.
- Recommend targeted robustness and falsification tests (not generic).
- Provide policy implications ONLY if identification is credible; otherwise say what evidence/design is needed.

Output as clear Markdown with headings and bullet points.
"""

    user_prompt = f"""
MODE: {mode}
AUDIENCE: {audience}

CONTEXT (JSON):
{context_json}

RESULTS:
{results_text}

Deliverables (always):
1) What the model estimates (econometrics meaning)
2) Interpretation of key coefficients (sign, magnitude, units, economic meaning)
3) Identification & assumptions for causal interpretation
4) Diagnostics & econometric concerns (what might bias results)
5) Robustness menu (5–10 checks tailored to THIS model)
6) Next best actions (prioritized)

Mode additions:
- Referee Report: Contribution / Major concerns / Required revisions
- Policy Memo: Decision takeaways / Risks / Sensitivity
- Socratic Tutor: Ask 3–5 clarification questions + short teaching explanations
- Diagnostics Coach: Recommend specific tests and how to interpret them
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


# --------------------------
# Page title
# --------------------------
st.title("📊 AI-Augmented Econometric Research Laboratory")


# ==========================
# Sidebar: Module selector
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

# Initialize AI defaults (so module buttons can reuse them even outside AI Assistant)
if "ai_mode" not in st.session_state:
    st.session_state["ai_mode"] = "Referee Report"
if "ai_audience" not in st.session_state:
    st.session_state["ai_audience"] = "Graduate"
if "ai_temp" not in st.session_state:
    st.session_state["ai_temp"] = 0.2


# ==========================
# MAIN ECONOMETRIC MODULES
# ==========================
if page != "AI Assistant":

    uploaded = st.file_uploader("Upload CSV Data", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip().str.lower()

        st.write("Columns detected:", df.columns.tolist())

        entity = st.selectbox("Entity ID", df.columns,
                              index=df.columns.get_loc("country") if "country" in df.columns else 0)
        time_id = st.selectbox("Time ID", df.columns,
                               index=df.columns.get_loc("year") if "year" in df.columns else 0)

        # Exclude entity/time from y/x selection
        candidate_cols = [c for c in df.columns if c not in {entity, time_id}]
        y_var = st.selectbox("Dependent Variable", candidate_cols)
        x_vars = st.multiselect("Independent Variables", candidate_cols)

        # ==========================
        # PANEL MODELS
        # ==========================
        if page == "Panel Models" and y_var and x_vars:

            df_panel = prepare_panel_df(df, entity, time_id)
            st.caption(f"Panel index ready: {df_panel.index.names} | levels={df_panel.index.nlevels}")

            fe_res = run_fe(df_panel, y_var, x_vars)
            re_res = run_re(df_panel, y_var, x_vars)
            h_stat, h_p, h_interp = hausman(fe_res, re_res)

            st.subheader("Fixed Effects")
            st.text(fe_res.summary)

            st.subheader("Random Effects")
            st.text(re_res.summary)

            st.write("Hausman p-value:", h_p)
            st.info(h_interp)

            X_vif = safe_numeric_frame(df, x_vars).dropna()
            vif_df = compute_vif(X_vif, x_vars)
            st.subheader("VIF Diagnostics")
            st.write(vif_df)

            # (4) Improved AI interpretation button
            if st.button("AI Interpretation (Panel Model)"):
                summary = package_panel_results(fe_res, re_res, h_stat, h_p, h_interp)

                ctx = build_econ_context(
                    df=df,
                    module_name="Panel Models (FE/RE + Hausman)",
                    entity=entity, time_id=time_id,
                    y_var=y_var, x_vars=x_vars,
                    notes={
                        "hausman_p": float(h_p) if h_p is not None else None,
                        "fe_packet": compact_results_packet(fe_res, "FE"),
                        "re_packet": compact_results_packet(re_res, "RE"),
                        "se_note": "FE: robust/clustered depends on panel_models.py; RE: default unless specified"
                    }
                )

                store_last_run("Panel Models", entity, time_id, y_var, x_vars, summary, extra=ctx["notes"])

                interpretation = ai_interpret(
                    results_text=summary,
                    context_dict=ctx,
                    mode=st.session_state["ai_mode"],
                    audience=st.session_state["ai_audience"],
                    temperature=st.session_state["ai_temp"]
                )
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

                ctx = build_econ_context(
                    df=df,
                    module_name="Dynamic Panel (Arellano–Bond)",
                    entity=entity, time_id=time_id,
                    y_var=y_var, x_vars=x_vars,
                    notes={
                        "ab_packet": compact_results_packet(ab_res, "Arellano–Bond"),
                        "reminder": "Check AR(2) and Hansen/Sargan; watch instrument proliferation."
                    }
                )

                store_last_run("Dynamic Panel", entity, time_id, y_var, x_vars, summary, extra=ctx["notes"])

                interpretation = ai_interpret(
                    results_text=summary,
                    context_dict=ctx,
                    mode=st.session_state["ai_mode"],
                    audience=st.session_state["ai_audience"],
                    temperature=st.session_state["ai_temp"]
                )
                st.markdown(interpretation)

        # ==========================
        # ENDOGENEITY
        # ==========================
        elif page == "Endogeneity & Instruments" and y_var and x_vars:

            df_panel = prepare_panel_df(df, entity, time_id)

            fe_res = run_fe(df_panel, y_var, x_vars)
            re_res = run_re(df_panel, y_var, x_vars)
            h_stat, h_p, _ = hausman(fe_res, re_res)

            # Build X aligned with FE residual index
            X = safe_numeric_frame(df_panel.reset_index(), x_vars)
            X = X.set_index(df_panel.index)
            X = sm.add_constant(X, has_constant="add").dropna()

            resid = pd.Series(fe_res.resids, index=df_panel.index)
            common_idx = resid.index.intersection(X.index)
            resid = resid.loc[common_idx]
            X = X.loc[common_idx]

            vif_df = compute_vif(safe_numeric_frame(df, x_vars).dropna(), x_vars)
            bp_p = heteroskedasticity(resid, X)

            score, level, details = endogeneity_score(h_p, vif_df["VIF"].max(), bp_p)

            st.write("Endogeneity Risk:", level)
            st.write(details)
            st.markdown(interpret_endogeneity(score, level))

            if level.startswith("High"):
                for var in x_vars:
                    st.write(f"Instruments for {var}: {suggest_instruments(var)}")

            if st.button("AI Interpretation (Endogeneity)"):
                summary = package_endogeneity_results(score, level, details)

                ctx = build_econ_context(
                    df=df,
                    module_name="Endogeneity Diagnostics (Hausman + VIF + BP test)",
                    entity=entity, time_id=time_id,
                    y_var=y_var, x_vars=x_vars,
                    notes={
                        "hausman_p": float(h_p) if h_p is not None else None,
                        "max_vif": float(vif_df["VIF"].max()) if "VIF" in vif_df else None,
                        "bp_p": float(bp_p) if bp_p is not None else None,
                        "risk_level": level
                    }
                )

                store_last_run("Endogeneity & Instruments", entity, time_id, y_var, x_vars, summary, extra=ctx["notes"])

                interpretation = ai_interpret(
                    results_text=summary,
                    context_dict=ctx,
                    mode=st.session_state["ai_mode"],
                    audience=st.session_state["ai_audience"],
                    temperature=st.session_state["ai_temp"]
                )
                st.markdown(interpretation)

        # ==========================
        # ROBUSTNESS
        # ==========================
        elif page == "Robustness & Sensitivity" and y_var and x_vars:

            stability_df = sensitivity(df, y_var, x_vars)
            st.write(stability_df)

            interp = interpret_robustness(stability_df)
            score = robustness_score(stability_df)

            st.info(interp)
            st.write("Robustness Score (0-100):", score)

            if st.button("AI Interpretation (Robustness)"):
                summary = package_robustness_results(stability_df, interp, score)

                ctx = build_econ_context(
                    df=df,
                    module_name="Robustness & Sensitivity",
                    entity=entity, time_id=time_id,
                    y_var=y_var, x_vars=x_vars,
                    notes={"robustness_score": float(score)}
                )

                store_last_run("Robustness & Sensitivity", entity, time_id, y_var, x_vars, summary, extra=ctx["notes"])

                interpretation = ai_interpret(
                    results_text=summary,
                    context_dict=ctx,
                    mode=st.session_state["ai_mode"],
                    audience=st.session_state["ai_audience"],
                    temperature=st.session_state["ai_temp"]
                )
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

                ctx = build_econ_context(
                    df=df_macro,
                    module_name="VAR & IRF Analysis",
                    entity=entity, time_id=time_id,
                    y_var="(VAR system)",
                    x_vars=list(df_macro.columns),
                    notes={"selected_lag": int(lag), "stability": stability_interp}
                )

                store_last_run("VAR & IRF Analysis", entity, time_id, "(VAR system)", list(df_macro.columns), summary, extra=ctx["notes"])

                interpretation = ai_interpret(
                    results_text=summary,
                    context_dict=ctx,
                    mode=st.session_state["ai_mode"],
                    audience=st.session_state["ai_audience"],
                    temperature=st.session_state["ai_temp"]
                )
                st.markdown(interpretation)

    else:
        st.info("Upload a CSV file to begin.")


# ==========================
# (5)(6) AI ASSISTANT: Econometrics Copilot + Research Design Expander
# ==========================
elif page == "AI Assistant":

    st.header("🤖 EconLab AI Assistant (Econometrics Copilot)")

    # ---- Sidebar AI controls
    st.sidebar.subheader("AI Settings")
    st.session_state["ai_mode"] = st.sidebar.selectbox(
        "AI Mode",
        ["Referee Report", "Policy Memo", "Socratic Tutor", "Diagnostics Coach"],
        index=["Referee Report", "Policy Memo", "Socratic Tutor", "Diagnostics Coach"].index(st.session_state["ai_mode"])
        if st.session_state["ai_mode"] in ["Referee Report", "Policy Memo", "Socratic Tutor", "Diagnostics Coach"]
        else 0
    )
    st.session_state["ai_audience"] = st.sidebar.selectbox(
        "Audience",
        ["Undergraduate", "Graduate", "Researcher", "Policy Maker"],
        index=["Undergraduate", "Graduate", "Researcher", "Policy Maker"].index(st.session_state["ai_audience"])
        if st.session_state["ai_audience"] in ["Undergraduate", "Graduate", "Researcher", "Policy Maker"]
        else 1
    )
    st.session_state["ai_temp"] = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, float(st.session_state["ai_temp"]), 0.05)

    # ---- Quick prompts
    st.subheader("Quick Econometrics Prompts")
    quick = st.selectbox(
        "Choose a prompt",
        [
            "Interpret my last model like a referee report",
            "What identification assumptions am I making?",
            "List the top threats to causal inference here",
            "Suggest robustness checks specific to my model",
            "Write a paper-ready Results paragraph",
            "Propose falsification/placebo tests",
            "What additional data would strengthen identification?",
            "Explain the model assumptions to a student",
        ]
    )

    # ---- (6) Research design expander (improves Socratic Tutor answers)
    design = {}
    with st.expander("Research Design (optional, improves AI answers)"):
        design["goal"] = st.text_input("Goal (causal? predictive? descriptive?)", "")
        design["treatment"] = st.text_input("Key treatment/policy variable (if any)", "")
        design["identification"] = st.text_area("Identification strategy (FE, IV, DiD, shocks, etc.)", "")
        design["data_notes"] = st.text_area("Data notes (measurement, missingness, sample)", "")

    last = st.session_state.get("last_run", None)
    if last:
        st.success(f"Loaded last run: **{last['module']}** | y={last['y']} | X={', '.join(last['x'])}")
        with st.expander("Last run summary"):
            st.text(last["summary"])
    else:
        st.info("No model run saved yet. Run a module (Panel/Dynamic/VAR/etc.) and come back here.")

    api_key = st.secrets.get("GROQ_API_KEY")
    from groq import Groq
    client = Groq(api_key=api_key)
    MODEL = "openai/gpt-oss-120b"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Use Quick Prompt"):
            st.session_state.messages.append({"role": "user", "content": quick})
    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = []

    # show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask an econometrics question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.messages:
        mode = st.session_state["ai_mode"]
        audience = st.session_state["ai_audience"]
        temp = st.session_state["ai_temp"]

        # Inject last run context dynamically
        if last:
            last_context = {
                "module": last["module"],
                "panel_keys": {"entity": last["entity"], "time": last["time"]},
                "dependent": last["y"],
                "regressors": last["x"],
                "notes": last.get("extra", {})
            }
            injected_context = f"""
LAST RUN CONTEXT (JSON):
{json.dumps(last_context, ensure_ascii=False)}

LAST RUN RESULTS:
{last["summary"]}

USER DESIGN NOTES:
{json.dumps(design, ensure_ascii=False)}
"""
        else:
            injected_context = f"""
No last run results available.
Ask the user which model they are using, what their identification strategy is, and what their main variables are.

USER DESIGN NOTES:
{json.dumps(design, ensure_ascii=False)}
"""

        system_prompt = f"""
You are an econometrics copilot.
MODE: {mode}
AUDIENCE: {audience}

Requirements:
- Be rigorous: distinguish correlation vs causation.
- State identification assumptions if making causal claims.
- Recommend model-specific diagnostics and robustness checks.
- End with prioritized next steps (3–7 bullet points).
"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": injected_context},
                *st.session_state.messages
            ],
            temperature=temp
        )

        answer = response.choices[0].message.content
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
