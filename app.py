import os, io, json, ast, traceback, re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import google.generativeai as genai
import requests

# Forecasting / analytics helpers (preloaded, so no imports needed in LLM code)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import datetime

# ================== APP CONFIG ==================
st.set_page_config(page_title="PlotTwist — Sales Analytics Copilot", page_icon="📊", layout="wide")
MODEL_NAME = "gemini-1.5-flash"


RAW_XLSX_URL = "https://github.com/5av1t/PlotTwist/blob/c773fb15163ea3ec251819672eff5ba54cef4f03/sales_template.xlsx"

# API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
if API_KEY:
    genai.configure(api_key=API_KEY)

# ================== LLM PROMPT ==================
SYSTEM_PREAMBLE = """
You are a Python data assistant. Output ONLY a Python code snippet (no backticks, no prose).

Constraints:
- DataFrame is preloaded as `df`.
- Allowed: pandas as pd, numpy as np, matplotlib.pyplot as plt.
- Forbidden: imports, file I/O, network, os/sys/subprocess/pathlib/socket/pickle/tempfile, input(), exec/eval/compile, __import__.
- For forecasting/analytics, you already have access to:
  • ExponentialSmoothing (from statsmodels.tsa.holtwinters)
  • LinearRegression (from sklearn.linear_model)
  • datetime
- Do NOT import these, just use them directly.
- If you produce a table, assign to `result_df`.
- If you produce a chart, assign to `fig = plt.gcf()`.
- Keep code under 120 lines.
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

# ================== CODE SANITIZER & SANDBOX ==================
# 1) Strip any import statements the LLM may add anyway
IMPORT_LINE_RE = re.compile(r'^\s*(?:from\s+\S+\s+import\s+.*|import\s+.+)$')
CONTINUATION_RE = re.compile(r'.*\\\s*$')  # handle "from x import y, \\" continued lines

def sanitize_code(snippet: str) -> str:
    if not isinstance(snippet, str):
        return ""
    # Remove fenced code markers
    code = snippet.strip()
    if code.startswith("```"):
        code = "\n".join(
            ln for ln in code.splitlines()
            if not ln.strip().startswith("```") and not ln.strip().startswith("python")
        )
    # First pass: drop all pure import lines
    cleaned_lines = []
    skip_cont = False
    for ln in code.splitlines():
        if skip_cont:
            # keep skipping continuation lines of an import block
            if CONTINUATION_RE.match(ln):
                continue
            else:
                skip_cont = False
                continue
        if IMPORT_LINE_RE.match(ln):
            # if it ends with "\" then subsequent line continues the import statement
            if CONTINUATION_RE.match(ln):
                skip_cont = True
            continue
        cleaned_lines.append(ln)
    cleaned = "\n".join(cleaned_lines)
    return cleaned

# 2) Guardrails (still block dangerous ops / modules / dunders)
FORBIDDEN_CALLS = {"open","exec","eval","compile","__import__","input","system"}
FORBIDDEN_ATTR_ROOTS = {"os","sys","subprocess","pathlib","socket","requests","shutil","pickle","dill","tempfile","builtins","importlib"}
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom)  # should be gone after sanitize, but double-check
DISALLOWED_ATTR_NAMES = {"__dict__","__class__","__mro__","__subclasses__","__globals__","__getattribute__","__getattr__"}

def validate_snippet(snippet: str):
    try:
        tree = ast.parse(snippet)
    except Exception as e:
        return False, f"Code not parsable: {e}"

    class Guard(ast.NodeVisitor):
        def visit(self, node):
            if isinstance(node, FORBIDDEN_NODES):
                raise ValueError(f"Forbidden node: {type(node).__name__}")
            return super().visit(node)
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"Forbidden call: {node.func.id}()")
            self.generic_visit(node)
        def visit_Attribute(self, node: ast.Attribute):
            if node.attr in DISALLOWED_ATTR_NAMES:
                raise ValueError(f"Forbidden attribute access: {node.attr}")
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in FORBIDDEN_ATTR_ROOTS:
                raise ValueError(f"Forbidden module usage: {base.id}.*")
            self.generic_visit(node)
    try:
        Guard().visit(tree)
    except ValueError as ve:
        return False, str(ve)
    return True, None

def run_snippet(snippet: str, df: pd.DataFrame):
    # sanitize first (remove imports), then validate & execute
    cleaned = sanitize_code(snippet)
    ok, err = validate_snippet(cleaned)
    if not ok:
        return None, None, f"Validation failed: {err}"
    safe_globals = {
        "pd": pd, "np": np, "plt": plt,
        "ExponentialSmoothing": ExponentialSmoothing,
        "LinearRegression": LinearRegression,
        "datetime": datetime
    }
    safe_locals = {"df": df.copy()}
    try:
        exec(cleaned, safe_globals, safe_locals)
        result_df = safe_locals.get("result_df")
        fig = safe_locals.get("fig", plt.gcf())
        if fig and not fig.axes:
            fig = None
        return result_df, fig, "Executed successfully (imports stripped)."
    except Exception as e:
        return None, None, "Execution error:\n" + "".join(traceback.format_exception_only(type(e), e))

# ================== TEMPLATE (FALLBACK) ==================
TEMPLATE_COLUMNS = ["OrderID","OrderDate","Week","Customer","Product","Category","Region","Quantity","UnitPrice","Revenue"]

def build_fallback_template_df() -> pd.DataFrame:
    rows, order_id = [], 40001
    customers = ["Acme Corp","Beta LLC","Delta Inc","Echo Ltd"]
    products  = [("Widget A","Widgets",15),("Widget B","Widgets",19),("Gizmo X","Gizmos",45),("Gizmo Y","Gizmos",60)]
    regions   = ["North","South","East","West"]
    for m in range(1,13):
        for i in range(3):
            cust = customers[(m+i)%len(customers)]
            prod, cat, price = products[(m+i)%len(products)]
            reg  = regions[(m+i)%len(regions)]
            qty  = 5+((m+i)%20)
            date = pd.Timestamp(year=2024,month=m,day=min(5+i*7,28))
            week = int(pd.Timestamp(date).isocalendar().week)
            revenue = qty*price
            rows.append([order_id,date.date().isoformat(),week,cust,prod,cat,reg,qty,float(price),float(revenue)])
            order_id+=1
    return pd.DataFrame(rows,columns=TEMPLATE_COLUMNS)

def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name="Sales") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as writer:
        df.to_excel(writer,index=False,sheet_name=sheet_name)
    buf.seek(0)
    return buf.read()

# ================== SIDEBAR: DOWNLOAD TEMPLATE ==================
with st.sidebar:
    st.header("⬇️ Download Sales Template (Excel)")
    st.caption("Dummy dataset (12 months). Upload it back or use your own.")
    xlsx_bytes=None
    if RAW_XLSX_URL.startswith("http"):
        try:
            r=requests.get(RAW_XLSX_URL,timeout=12); r.raise_for_status(); xlsx_bytes=r.content
        except: pass
    if xlsx_bytes:
        st.download_button("Download sales_template.xlsx",data=xlsx_bytes,file_name="sales_template.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        fallback=dataframe_to_excel_bytes(build_fallback_template_df())
        st.download_button("Download fallback_template.xlsx",data=fallback,file_name="fallback_template.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ================== MAIN UI ==================
st.title("📊 PlotTwist — Sales Analytics Copilot")
st.caption("Upload Excel → Ask → Gemini generates pandas/matplotlib/analytics code → Safe run → Table/Chart.")

file=st.file_uploader("Upload sales file (.xlsx preferred, .csv accepted)",type=["xlsx","csv"])
df=None
if file:
    try:
        if file.name.lower().endswith(".xlsx"):
            df=pd.read_excel(file,engine="openpyxl")
        else:
            df=pd.read_csv(file,on_bad_lines="skip")
        st.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
        st.dataframe(df.head(50),use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

if df is not None:
    st.markdown("### Ask a question")
    user_prompt=st.text_area(
        "Examples:\n- Monthly revenue trend\n- Forecast next 6 months of revenue with ExponentialSmoothing\n- Top 10 customers by revenue (bar chart)\n- Regression: predict Revenue from Quantity",
        height=110
    )
    if st.button("Generate & Run",type="primary"):
        with st.spinner("Asking Gemini…"):
            code=llm_generate_code(user_prompt,df)
        st.subheader("Sanitized code (imports auto-removed)")
        st.code(sanitize_code(code) if code else "# empty",language="python")
        if code and not code.strip().startswith("# ERROR"):
            with st.spinner("Executing safely…"):
                result_df,fig,logs=run_snippet(code,df)
            st.markdown(f"**Logs:** {logs}")
            if isinstance(result_df,pd.DataFrame):
                st.subheader("Result table"); st.dataframe(result_df,use_container_width=True)
            if fig is not None:
                st.subheader("Chart"); st.pyplot(fig,clear_figure=True)
else:
    st.info("💡 Tip: download the Excel template from the sidebar and try prompts right away.")
