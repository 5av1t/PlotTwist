import os, json, ast, traceback
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import google.generativeai as genai
import requests  # used only to fetch the template CSV from GitHub

# -------------------- CONFIG --------------------
st.set_page_config(page_title="PlotTwist — Data Copilot", page_icon="📊", layout="wide")
MODEL_NAME = "gemini-1.5-flash"

# 🔴 TODO: replace this with YOUR GitHub raw CSV URL after you upload sales_template.csv
# Example: https://raw.githubusercontent.com/<user>/<repo>/main/sales_template.csv
RAW_CSV_URL = "https://raw.githubusercontent.com/<user>/<repo>/main/sales_template.csv"

# API key: Streamlit Cloud -> Settings -> Secrets
API_KEY = os.getenv("GOOGLE_API_KEY")
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
if API_KEY:
    genai.configure(api_key=API_KEY)

# -------------------- LLM PROMPT --------------------
SYSTEM_PREAMBLE = """
You are a Python data assistant. Output ONLY a Python code snippet (no backticks, no prose).
Constraints:
- A pandas DataFrame named `df` is already loaded with the user's data.
- Allowed libraries: pandas as pd, numpy as np, matplotlib.pyplot as plt.
- Forbidden: imports, file I/O, network, os/sys/subprocess/pathlib/socket/pickle, input(), exec/eval.
- If you produce a table, assign it to variable `result_df`.
- If you produce a chart, assign the figure to variable `fig` (e.g., fig = plt.gcf()).
- Do not modify global state. Keep code under 120 lines.
"""

def llm_generate_code(user_instruction: str, df: pd.DataFrame) -> str:
    if not API_KEY:
        return "# ERROR: Missing GOOGLE_API_KEY.\nresult_df = df.head(5)"
    schema = {
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "sample_rows": df.head(3).to_dict(orient="records")
    }
    prompt = (
        SYSTEM_PREAMBLE
        + "\nDataFrame schema (JSON):\n"
        + json.dumps(schema, indent=2)
        + "\n\nUser request:\n"
        + user_instruction
        + "\n\n# Python code only below:\n"
    )
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```") and not ln.strip().startswith("python")]
            text = "\n".join(lines).strip()
        return text
    except Exception as e:
        return f"# ERROR calling Google AI: {e}\nresult_df = df.head(5)"

# -------------------- SANDBOX --------------------
FORBIDDEN_CALLS = {"open","exec","eval","compile","__import__","input","system","popen","spawn","fork","kill"}
FORBIDDEN_ATTR_ROOTS = {"os","sys","subprocess","pathlib","socket","requests","shutil","pickle","dill"}
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.With, ast.Global, ast.Nonlocal, ast.Try, ast.AsyncFunctionDef)

def validate_snippet(snippet: str):
    try:
        tree = ast.parse(snippet)
    except Exception as e:
        return False, f"Code not parsable: {e}"
    class Guard(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"Forbidden call: {node.func.id}()")
            self.generic_visit(node)
        def visit_Attribute(self, node: ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in FORBIDDEN_ATTR_ROOTS:
                raise ValueError(f"Forbidden module usage: {base.id}")
            self.generic_visit(node)
        def generic_visit(self, node):
            if isinstance(node, FORBIDDEN_NODES):
                raise ValueError(f"Forbidden node: {type(node).__name__}")
            super().generic_visit(node)
    Guard().visit(tree)
    return True, None

def run_snippet(snippet: str, df: pd.DataFrame):
    ok, err = validate_snippet(snippet)
    if not ok:
        return None, None, f"Validation failed:\n{err}"
    safe_globals = {"pd": pd, "np": np, "plt": plt}
    safe_locals = {"df": df.copy()}
    code = snippet.strip()
    if code.startswith("```"):
        code = "\n".join(
            ln for ln in code.splitlines()
            if not ln.strip().startswith("```") and not ln.strip().startswith("python")
        )
    try:
        exec(code, safe_globals, safe_locals)  # sandboxed globals/locals
        result_df = safe_locals.get("result_df")
        fig = safe_locals.get("fig", plt.gcf())
        if fig and not fig.axes:
            fig = None
        return result_df, fig, "Executed successfully."
    except Exception as e:
        return None, None, "Execution error:\n" + "".join(traceback.format_exception_only(type(e), e))

# -------------------- SIDEBAR: DOWNLOAD TEMPLATE --------------------
with st.sidebar:
    st.header("⬇️ Download Sales Template (CSV)")
    st.caption("Clean, one-year dataset for testing. Upload it back below or use your own.")
    # Try to fetch the raw CSV from GitHub; fall back to a tiny built-in sample if URL not set/can't fetch
    csv_bytes, fetch_err = None, None
    if RAW_CSV_URL.startswith("http"):
        try:
            r = requests.get(RAW_CSV_URL, timeout=10)
            r.raise_for_status()
            csv_bytes = r.content
        except Exception as e:
            fetch_err = str(e)
    if csv_bytes:
        st.download_button(
            label="Download sales_template.csv",
            data=csv_bytes,
            file_name="sales_template.csv",
            mime="text/csv"
        )
    else:
        st.warning("Template URL not reachable. Set a valid RAW GitHub URL in app.py.")
        fallback = (
            "OrderID,OrderDate,Week,Customer,Product,Category,Region,Quantity,UnitPrice,Revenue\n"
            "20001,2024-01-05,1,Acme Corp,Widget A,Widgets,North,12,15,180\n"
            "20002,2024-02-05,6,Beta LLC,Gizmo X,Gizmos,East,8,45,360\n"
        )
        st.download_button("Download minimal_template.csv", data=fallback, file_name="minimal_template.csv", mime="text/csv")

# -------------------- MAIN UI --------------------
st.title("📊 PlotTwist — Data Copilot (Gemini)")
st.caption("Upload CSV/XLSX → Ask in plain English → Gemini writes pandas/matplotlib → Safe execute → Table/Chart.")

# Upload + preview
file = st.file_uploader("Upload your sales CSV/XLSX", type=["csv","xlsx"])
df = None
if file:
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            import openpyxl  # ensure installed; used by pandas for .xlsx
            df = pd.read_excel(file)
        st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        st.dataframe(df.head(50), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# Prompt → code → run
if df is not None:
    st.markdown("### Ask for a summary or chart")
    user_prompt = st.text_area(
        "Example: “Summarize monthly revenue and draw a line chart by month. Also show top 5 customers by total revenue.”",
        height=90
    )
    if st.button("Generate & Run", type="primary"):
        with st.spinner("Asking Gemini for code…"):
            code = llm_generate_code(user_prompt, df)
        st.subheader("Generated code")
        st.code(code or "# (empty response)", language="python")
        if code and not code.strip().startswith("# ERROR"):
            with st.spinner("Executing code safely…"):
                result_df, fig, logs = run_snippet(code, df)
            st.markdown(f"**Logs:** {logs}")
            if isinstance(result_df, pd.DataFrame):
                st.subheader("Result table")
                st.dataframe(result_df, use_container_width=True)
            if fig is not None:
                st.subheader("Chart")
                st.pyplot(fig, clear_figure=True)
else:
    st.info("Tip: download the template from the left sidebar, then upload it here to try prompts immediately.")
