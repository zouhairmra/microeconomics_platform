import streamlit as st
import requests
import json
import pandas as pd
import io
from typing import Tuple, Optional

# ==========================
# OPTIONAL LIBRARIES
# ==========================
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    import statsmodels.api as sm
except ImportError:
    sm = None


# ==========================
# PAGE SETUP
# ==========================
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 EconLab — AI Assistant")
st.write("Ask anything about economics, econometrics, or data analysis — or upload a file for AI insights.")


# ==========================
# POE API CONFIG
# ==========================
POE_API_URL = "https://api.poe.com/v1/chat/completions"
POE_API_KEY = st.secrets.get("POE_API_KEY", "")

MODEL = st.sidebar.selectbox("Select model", ["maztouriabot", "gpt-4o-mini", "claude-3-haiku"])
MAX_DOC_CHARS = st.sidebar.slider("Max document chars sent to AI", 500, 8000, 3000, 500)
HISTORY_TURNS = st.sidebar.slider("Chat memory (turns)", 2, 20, 8, 1)
TIMEOUT = st.sidebar.slider("API timeout (seconds)", 10, 120, 60, 5)

if not POE_API_KEY:
    st.sidebar.warning("⚠️ POE_API_KEY missing in Streamlit secrets. API calls will fail.")


# ==========================
# CACHING: FILE EXTRACTION
# ==========================
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes: bytes) -> str:
    if not PdfReader:
        return ""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text).strip()

@st.cache_data(show_spinner=False)
def extract_text_from_docx(file_bytes: bytes) -> str:
    if not Document:
        return ""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs).strip()


# ==========================
# FILE UPLOAD
# ==========================
st.markdown("### 📂 Upload a file for AI analysis")
uploaded_file = st.file_uploader("Upload PDF, CSV, or Word", type=["pdf", "csv", "docx"])

# Session storage to avoid rework
if "uploaded_text" not in st.session_state:
    st.session_state["uploaded_text"] = ""
if "df" not in st.session_state:
    st.session_state["df"] = None

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    file_bytes = uploaded_file.getvalue()

    uploaded_text = ""
    df = None

    if file_ext == "pdf":
        if PdfReader:
            with st.spinner("Extracting PDF text..."):
                uploaded_text = extract_text_from_pdf(file_bytes)
            st.success("✅ PDF text extracted (cached).")
        else:
            st.warning("⚠️ PyPDF2 not installed. PDF disabled.")

    elif file_ext == "docx":
        if Document:
            with st.spinner("Extracting Word text..."):
                uploaded_text = extract_text_from_docx(file_bytes)
            st.success("✅ Word text extracted (cached).")
        else:
            st.warning("⚠️ python-docx not installed. Word upload disabled.")

    elif file_ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        st.success("✅ CSV data loaded.")
        st.dataframe(df.head())

        # Efficient "text" representation for LLM: schema + head + describe (not full table)
        numeric = df.select_dtypes(include="number")
        uploaded_text = (
            "CSV SUMMARY\n"
            f"- Rows: {len(df)}\n"
            f"- Columns: {len(df.columns)}\n\n"
            "DTYPES:\n"
            f"{df.dtypes.to_string()}\n\n"
            "HEAD (first 8 rows):\n"
            f"{df.head(8).to_string(index=False)}\n\n"
        )
        if len(numeric.columns) > 0:
            uploaded_text += "DESCRIBE (numeric):\n" + numeric.describe().to_string() + "\n"

    st.session_state["uploaded_text"] = uploaded_text
    st.session_state["df"] = df

    with st.expander("📜 Preview Extracted / Prepared Text for AI"):
        preview = uploaded_text[:2000]
        st.text(preview + ("..." if len(uploaded_text) > 2000 else ""))


# ==========================
# DATA ANALYSIS TOOLS (CSV only)
# ==========================
df = st.session_state.get("df", None)

if df is not None:
    st.markdown("### 📊 Data Analysis Tools")

    colA, colB, colC = st.columns(3)

    # Save Parquet in session (your requested feature)
    with colA:
        if st.button("💾 Save dataset as Parquet in session"):
            buf = io.BytesIO()
            df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
            st.session_state["dataset_parquet"] = buf.getvalue()
            st.success("Saved as Parquet in session_state['dataset_parquet'].")

    with colB:
        if st.button("♻️ Restore dataset from session Parquet"):
            if "dataset_parquet" in st.session_state:
                restored = pd.read_parquet(io.BytesIO(st.session_state["dataset_parquet"]), engine="pyarrow")
                st.dataframe(restored.head())
            else:
                st.warning("No Parquet found in session.")

    with colC:
        st.download_button(
            "⬇️ Download Parquet",
            data=(st.session_state.get("dataset_parquet", b"") or b""),
            file_name="dataset.parquet",
            mime="application/octet-stream",
            disabled=("dataset_parquet" not in st.session_state)
        )

    # Pairplot
    if plt and sns:
        st.subheader("Pairplot (select columns)")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) >= 2:
            chosen = st.multiselect("Choose up to 6 numeric columns", num_cols, default=num_cols[:min(4, len(num_cols))])
            chosen = chosen[:6]
            if st.button("Plot Pairplot"):
                if len(chosen) < 2:
                    st.warning("Select at least 2 columns.")
                else:
                    with st.spinner("Generating pairplot..."):
                        g = sns.pairplot(df[chosen].dropna())
                        st.pyplot(g.fig)
        else:
            st.info("Not enough numeric columns for pairplot.")
    else:
        st.info("Pairplot requires matplotlib + seaborn.")

    # OLS Regression
    if sm:
        st.subheader("OLS Regression (Statsmodels)")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) >= 2:
            with st.form("ols_form"):
                y_col = st.selectbox("Dependent variable (Y)", numeric_cols)
                X_cols = st.multiselect("Independent variables (X)", [c for c in numeric_cols if c != y_col])
                run = st.form_submit_button("Run OLS")

            if run:
                if not X_cols:
                    st.warning("Select at least 1 independent variable.")
                else:
                    X = sm.add_constant(df[X_cols])
                    y = df[y_col]
                    model = sm.OLS(y, X, missing="drop").fit()
                    st.text(model.summary().as_text())
        else:
            st.info("Not enough numeric columns for regression.")
    else:
        st.info("Regression requires statsmodels.")


# ==========================
# CHAT MEMORY
# ==========================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show chat log
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ==========================
# UTILS: BUILD PAYLOAD WITH HISTORY + DOC CONTEXT
# ==========================
def build_messages(history, new_user_text: str, doc_text: str, max_doc_chars: int, max_turns: int):
    # keep last N turns of chat to reduce token usage
    trimmed = history[-(max_turns * 2):] if max_turns else []

    # Optionally attach doc context as a system-like prefix for the user turn
    if doc_text:
        doc_context = doc_text[:max_doc_chars]
        new_user_text = (
            "You have the following file context. Use it when relevant.\n"
            "----- FILE CONTEXT START -----\n"
            f"{doc_context}\n"
            "----- FILE CONTEXT END -----\n\n"
            f"User question: {new_user_text}"
        )

    msgs = trimmed + [{"role": "user", "content": new_user_text}]
    return msgs


def call_poe_api(model: str, messages, api_key: str, timeout: int) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}

    r = requests.post(POE_API_URL, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# ==========================
# INPUT + SUMMARIZE BUTTON (more reliable than default prompt)
# ==========================
uploaded_text = st.session_state.get("uploaded_text", "")

col1, col2 = st.columns([3, 1])
with col2:
    summarize = st.button("🧾 Summarize uploaded file", disabled=(not uploaded_text))

user_input = st.chat_input("Type your question…")

if summarize:
    user_input = "Summarize the uploaded file. Provide key points, limitations, and suggested next steps."

if user_input:
    # Add user message to local chat
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            if not POE_API_KEY:
                st.error("Missing POE_API_KEY. Add it to Streamlit secrets.")
            else:
                with st.spinner("Calling Poe API..."):
                    api_messages = build_messages(
                        history=st.session_state["messages"][:-1],  # exclude this user message (already added)
                        new_user_text=user_input,
                        doc_text=uploaded_text,
                        max_doc_chars=MAX_DOC_CHARS,
                        max_turns=HISTORY_TURNS
                    )
                    response_text = call_poe_api(MODEL, api_messages, POE_API_KEY, TIMEOUT)

                st.markdown(response_text)
                st.session_state["messages"].append({"role": "assistant", "content": response_text})

        except Exception as e:
            st.error(f"❌ Error fetching response: {e}")
            st.session_state["messages"].append({"role": "assistant", "content": f"Error: {e}"})


# ==========================
# EXPORT / CLEAR CHAT
# ==========================
st.markdown("---")
c1, c2, c3 = st.columns([1, 1, 2])

if c1.button("🧹 Clear Chat"):
    st.session_state["messages"] = []
    st.toast("Chat cleared!")

if c2.button("💾 Export Chat"):
    if st.session_state["messages"]:
        chat_df = pd.DataFrame(st.session_state["messages"])
        st.download_button("Download CSV", chat_df.to_csv(index=False), "econlab_chat.csv", "text/csv")
    else:
        st.warning("No chat to export!")

st.caption("💡 EconLab AI Assistant — Powered by Poe API and Streamlit.")
