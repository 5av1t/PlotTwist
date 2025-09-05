import os
import json
import traceback
import ast

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai

# ----------- API KEY -----------
API_KEY = os.getenv("GOOGLE_API_KEY")
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in Streamlit secrets.")
else:
    genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-flash"

st.set_page_config(page_title="PlotTwist", page_icon="📊", layout="wide")
st.title("📊 PlotTwist — Data Copilot")
st.caption("Upload CSV/XLSX → Ask in plain English → Get tables & charts.")

# ----------- LLM ----------- 
SYSTEM_PROMPT = """
You are a Python data assistant. Output ONLY a Python snippet (no prose).
- DataFrame is already loaded as `df`
- Allowed: pandas, numpy, matplotlib.pyplot
- Forbidden: imports, file I/O, os/sys/subprocess/network
- If table: assign to `result_df`
- If chart: assign figure to `fig = plt.gcf()`
"""

def llm_generate_code(prompt, df):
    schema = {
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "sample": df.head(3).to_dict(orient="records"),
    }
    full_prompt = SYSTEM_PROMPT + "\nSchema:\n" + json.dumps(schema, indent=2) + "\nUser request:\n" + prompt

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(full_prompt)
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = "\n".join([ln for ln in text.splitlines() if not ln.startswith("```") and not ln.startswith("python")])
        return text
    except Exception as e:
        return f"# ERROR: {e}\nresult_df = df.head()"

# ----------- Sandbox (light) -----------
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.With)
FORBIDDEN_CALLS = {"open","exec","eval","__import__","input"}

def validate_snippet(snippet):
    try:
        tree = ast.parse(snippet)
        for node in ast.walk(tree):
            if isinstance(node, FORBIDDEN_NODES):
                return False, f"Forbidden node: {type(node).__name__}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return False, f"Forbidden call: {node.func.id}"
        return True, None
    except Exception as e:
        return False, str(e)

def run_snippet(code, df):
    ok, err = validate_snippet(code)
    if not ok:
        return None, None, f"Validation failed: {err}"

    safe_globals = {"pd": pd, "np": np, "plt": plt}
    safe_locals = {"df": df.copy()}

    try:
        exec(code, safe_globals, safe_locals)
        return safe_locals.get("result_df"), safe_locals.get("fig", plt.gcf()), "Success"
    except Exception as e:
        return None, None, "Error: " + "".join(traceback.format_exception_only(type(e), e))

# ----------- UI -----------
file = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])
if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    st.dataframe(df.head(), use_container_width=True)

    prompt = st.text_area("Ask me anything:", "Summarize sales by month and plot a line chart")
    if st.button("Run"):
        code = llm_generate_code(prompt, df)
        st.subheader("Generated Code")
        st.code(code, language="python")

        result_df, fig, logs = run_snippet(code, df)
        st.write("**Logs:**", logs)

        if isinstance(result_df, pd.DataFrame):
            st.subheader("Result Table")
            st.dataframe(result_df, use_container_width=True)

        if fig:
            st.subheader("Chart")
            st.pyplot(fig)
else:
    st.info("Upload a file to get started.")

