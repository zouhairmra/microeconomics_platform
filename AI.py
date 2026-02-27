import streamlit as st
import requests
import pandas as pd
import io

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

# Sidebar controls
st.sidebar.header("AI Settings")
MODEL = st.sidebar.selectbox("Select model", ["maztouriabot", "gpt-4o-mini", "claude-3-haiku"])
HISTORY_TURNS = st.sidebar.slider("Chat memory (turns)", 2, 20, 8, 1)
TIMEOUT = st.sidebar.slider("API timeout (seconds)", 10, 120, 60, 5)

st.sidebar.divider()
MAX_DOC_CHARS = st.sidebar.slider("Max file context chars sent to AI", 500, 8000, 3000, 500)
INCLUDE_FILE_CONTEXT = st.sidebar.checkbox("Include file context in prompt", value=False)

if not POE_API_KEY:
    st.sidebar.warning("⚠️ POE_API_KEY missing in Streamlit secrets. API calls will fail.")

# ==========================
# SESSION STATE INIT (privacy-safe: per-session only)
# ==========================
st.session_state.setdefault("uploaded_text", "")
st.session_state.setdefault("df", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("dataset_parquet", None)
st.session_state.setdefault("uploaded_file_name", "")

# ==========================
# FILE UPLOAD (PUBLIC APP: do NOT cache extracted content)
# ==========================
st.markdown("### 📂 Upload a file for AI analysis")
uploaded_file = st.file_uploader("Upload PDF, CSV, or Word", type=["pdf", "csv", "docx"])

def extract_pdf_text(file_bytes: bytes) -> str:
    if not PdfReader:
        return ""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text).strip()

def extract_docx_text(file_bytes: bytes) -> str:
    if not Document:
        return ""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs).strip()

def build_csv_ai_context(df: pd.DataFrame, head_rows: int = 8, max_describe_rows: int = 200_000) -> str:
    """
    Efficient context: schema + head + describe.
    If df is huge, compute describe on a sample to keep it fast.
    """
    numeric = df.select_dtypes(include="number")

    parts = []
    parts.append("CSV SUMMARY")
    parts.append(f"- Rows: {len(df)}")
    parts.append(f"- Columns: {len(df.columns)}\n")

    parts.append("DTYPES:")
    parts.append(df.dtypes.to_string())
    parts.append("\nHEAD (first rows):")
    parts.append(df.head(head_rows).to_string(index=False))

    if len(numeric.columns) > 0:
        parts.append("\nDESCRIBE (numeric):")
        if len(df) > max_describe_rows:
            sample = df.sample(n=max_describe_rows, random_state=42)
            parts.append(sample.select_dtypes(include="number").describe().to_string())
            parts.append(f"\n(Note: describe computed on a random sample of {max_describe_rows:,} rows.)")
        else:
            parts.append(numeric.describe().to_string())

    return "\n".join(parts).strip()

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    file_bytes = uploaded_file.getvalue()

    st.session_state["uploaded_file_name"] = uploaded_file.name

    uploaded_text = ""
    df = None

    if file_ext == "pdf":
        if PdfReader:
            with st.spinner("Extracting PDF text..."):
                uploaded_text = extract_pdf_text(file_bytes)
            st.success("✅ PDF text extracted (session-only).")
        else:
            st.warning("⚠️ PyPDF2 not installed. PDF upload disabled.")

    elif file_ext == "docx":
        if Document:
            with st.spinner("Extracting Word text..."):
                uploaded_text = extract_docx_text(file_bytes)
            st.success("✅ Word text extracted (session-only).")
        else:
            st.warning("⚠️ python-docx not installed. Word upload disabled.")

    elif file_ext == "csv":
        with st.spinner("Loading CSV..."):
            df = pd.read_csv(io.BytesIO(file_bytes))
        st.success("✅ CSV loaded.")
        st.dataframe(df.head())

        # Build efficient context (schema + head + describe)
        with st.spinner("Preparing AI-friendly CSV context..."):
            uploaded_text = build_csv_ai_context(df)

    st.session_state["uploaded_text"] = uploaded_text
    st.session_state["df"] = df

    with st.expander("📜 Preview file context prepared for AI (first 2000 chars)"):
        st.text(uploaded_text[:2000] + ("..." if len(uploaded_text) > 2000 else ""))

# Clear upload context (public app: good privacy UX)
col_clear1, col_clear2 = st.columns([1, 3])
with col_clear1:
    if st.button("🧽 Clear uploaded context"):
        st.session_state["uploaded_text"] = ""
        st.session_state["df"] = None
        st.session_state["dataset_parquet"] = None
        st.session_state["uploaded_file_name"] = ""
        st.toast("Uploaded content cleared from this session.")

# ==========================
# DATA ANALYSIS TOOLS (CSV only)
# ==========================
df = st.session_state.get("df")

if df is not None:
    st.markdown("### 📊 Data Analysis Tools")

    # Parquet in session (Cloud-safe)
    cA, cB, cC = st.columns(3)
    with cA:
        if st.button("💾 Save dataset as Parquet in session"):
            try:
                buf = io.BytesIO()
                df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
                st.session_state["dataset_parquet"] = buf.getvalue()
                st.success("Saved Parquet in session_state['dataset_parquet'].")
            except Exception as e:
                st.error(f"Parquet save failed (check pyarrow in requirements): {e}")

    with cB:
        if st.button("♻️ Restore dataset from session Parquet"):
            if st.session_state.get("dataset_parquet"):
                restored = pd.read_parquet(io.BytesIO(st.session_state["dataset_parquet"]), engine="pyarrow")
                st.dataframe(restored.head())
                st.success("Dataset restored from Parquet bytes.")
            else:
                st.warning("No Parquet found in session.")

    with cC:
        st.download_button(
            "⬇️ Download Parquet",
            data=st.session_state.get("dataset_parquet") or b"",
            file_name="dataset.parquet",
            mime="application/octet-stream",
            disabled=not bool(st.session_state.get("dataset_parquet"))
        )

    # Pairplot (guard rails for big CSVs)
    if plt and sns:
        st.subheader("Pairplot (select columns)")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) >= 2:
            chosen = st.multiselect("Choose up to 6 numeric columns", num_cols, default=num_cols[:min(4, len(num_cols))])
            chosen = chosen[:6]
            if st.button("Plot Pairplot"):
                if len(chosen) < 2:
                    st.warning("Select at least 2 numeric columns.")
                else:
                    # For large datasets, sample to speed up plot
                    plot_df = df[chosen].dropna()
                    if len(plot_df) > 5000:
                        plot_df = plot_df.sample(5000, random_state=42)
                        st.info("Pairplot uses a sample of 5,000 rows for speed.")
                    with st.spinner("Generating pairplot..."):
                        g = sns.pairplot(plot_df)
                        st.pyplot(g.fig)
        else:
            st.info("Not enough numeric columns for pairplot.")
    else:
        st.info("Pairplot requires matplotlib + seaborn.")

    # OLS Regression (use form to reduce reruns)
    if sm:
        st.subheader("OLS Regression (Statsmodels)")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) >= 2:
            with st.form("ols_form"):
                y_col = st.selectbox("Dependent variable (Y)", numeric_cols)
                X_cols = st.multiselect("Independent variables (X)", [c for c in numeric_cols if c != y_col])
                run_ols = st.form_submit_button("Run OLS")

            if run_ols:
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
# CHAT DISPLAY
# ==========================
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================
# BUILD API MESSAGES (efficient history + optional file context)
# ==========================
def get_last_turns(messages, turns: int):
    return messages[-(turns * 2):] if turns else []

def call_poe(model: str, messages, api_key: str, timeout: int) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages}
    r = requests.post(POE_API_URL, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ==========================
# INPUT + SUMMARIZE BUTTON (explicit, avoids accidental calls)
# ==========================
uploaded_text = st.session_state.get("uploaded_text", "")

col1, col2 = st.columns([3, 1])
with col2:
    summarize = st.button("🧾 Summarize uploaded file", disabled=(not uploaded_text))

user_input = st.chat_input("Type your question…")

if summarize:
    user_input = (
        "Summarize the uploaded file. Provide: (1) key points, (2) limitations or missing info, "
        "(3) suggested next steps. Keep it concise."
    )

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if not POE_API_KEY:
            st.error("Missing POE_API_KEY. Add it to Streamlit secrets.")
        else:
            try:
                with st.spinner("Calling Poe API..."):
                    history = get_last_turns(st.session_state["messages"][:-1], HISTORY_TURNS)
                    content = user_input

                    # Attach file context only if user chooses
                    if INCLUDE_FILE_CONTEXT and uploaded_text:
                        content = (
                            "Use the following file context when relevant.\n"
                            "----- FILE CONTEXT START -----\n"
                            f"{uploaded_text[:MAX_DOC_CHARS]}\n"
                            "----- FILE CONTEXT END -----\n\n"
                            f"User question: {user_input}"
                        )

                    api_messages = history + [{"role": "user", "content": content}]
                    response_text = call_poe(MODEL, api_messages, POE_API_KEY, TIMEOUT)

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
