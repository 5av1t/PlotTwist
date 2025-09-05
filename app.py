import os
import io
import json
import ast
import traceback
from typing import Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Google AI (Gemini) ---
# Uses API key from st.secrets["GOOGLE_API_KEY"] or env GOOGLE_API_KEY
import google.generativeai as genai

API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
if API_KEY:
    genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-flash"  # fast & inexpensive

st.set_page_config(page_title="Planner Copilot", page_icon="📊", layout="wide")

# ---------------- LLM: generate python snippet ----------------
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
        # strip code fences if present
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```") and not ln.strip().startswith("python")]
            text = "\n".join(lines).strip()
        return text
    except Exception as e:
        return f"# ERROR calling Google AI: {e}\nresult_df = df.head(5)"

# ---------------- sandbox / validator ----------------
FORBIDDEN_CALLS = {"open","exec","eval","compile","__import__","input","system","popen","spawn","fork","kill"}
FORBIDDEN_ATTR_ROOTS = {"os","sys","subprocess","pathlib","socket","requests","shutil","pickle","dill"}
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.With, ast.Global, ast.Nonlocal, ast.Try, ast.AsyncFunctionDef)

def validate_snippet(snippet: str) -> Tuple[bool, Optional[str]]:
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
            # block os.*, sys.*, etc.
            base = node
            names = []
            while isinstance(base, ast.Attribute):
                names.append(base.attr)
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
    """
    Execute code with restricted globals. Returns (result_df, fig, logs)
    """
    ok, err = validate_snippet(snippet)
    if not ok:
        return None, None, f"Validation failed:\n{err}"

    safe_globals = {
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    safe_locals = {"df": df.copy()}
    code = snippet.strip()

    # remove lingering fences if any
    if code.startswith("```"):
        code = "\n".join(
            ln for ln in code.splitlines()
            if not ln.strip().startswith("```") and not ln.strip().startswith("python")
        )

    try:
        # Execute
        exec(code, safe_globals, safe_locals)  # noqa: S102 (intentional in restricted env)
        result_df = safe_locals.get("result_df")
        fig = safe_locals.get("fig", plt.gcf())

        # If nothing was drawn, avoid showing an empty figure
        if fig and not fig.axes:
            fig = None

        return result_df, fig, "Executed successfully."
    except Exception as e:
        return None, None, "Execution error:\n" + "".join(traceback.format_exception_only(type(e), e))

# ---------------- UI ----------------
st.title("📊 Planner Copilot — Google AI + Streamlit")
st.caption("Upload CSV/XLSX → Ask in plain English → App generates & safely runs pandas/matplotlib code.")

with st.sidebar:
    st.header("Settings")
    st.write("Model:", MODEL_NAME)
    st.write("API Key present:", "✅" if API_KEY else "❌")
    st.markdown(
        "Tips:\n- Example: *summarize sales by month and plot a weekly line chart.*\n"
        "- Table → assign to `result_df`. Chart → set `fig = plt.gcf()`."
    )

file = st.file_uploader("Upload a CSV or Excel file", type=["csv","xlsx"])
df: Optional[pd.DataFrame] = None

if file:
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

if df is not None:
    st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    with st.expander("Preview data"):
        st.dataframe(df.head(20), use_container_width=True)

    user_prompt = st.text_area(
        "What do you want?",
        value="Summarize sales by month and plot a weekly line chart of sales volume.",
        help="Describe the transformation or chart you need.",
    )

    if st.button("Generate & Run", type="primary"):
        with st.spinner("Asking Google AI to write code…"):
            code = llm_generate_code(user_prompt, df)

        st.subheader("Generated code")
        st.code(code or "# (empty response)", language="python")

        if not code or code.strip().startswith("# ERROR"):
            st.stop()

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
    st.info("Upload a CSV or Excel file to begin.")
