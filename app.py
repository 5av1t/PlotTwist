import os, io, json, ast, traceback, re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import google.generativeai as genai
import requests

# Forecasting / analytics helpers (preloaded for LLM code)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import datetime

# ================== APP CONFIG ==================
st.set_page_config(page_title="PlotTwist — Sales Analytics Copilot", page_icon="📊", layout="wide")
MODEL_NAME = "gemini-1.5-flash"
RAW_XLSX_URL = "https://github.com/5av1t/PlotTwist/blob/b10bf12045e5a2e29db17b55c3d9cb6499b3727f/sales_template.xlsx"  # <-- set your raw GitHub URL

API_KEY = os.getenv("GOOGLE_API_KEY")
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
if API_KEY:
    genai.configure(api_key=API_KEY)

# ---------- JSON SAFE SERIALIZER ----------
def _to_jsonable(x):
    if isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date)):
        return x.isoformat()
    if isinstance(x, (np.integer,)):   return int(x)
    if isinstance(x, (np.floating,)):  return float(x)
    if isinstance(x, (np.bool_,)):     return bool(x)
    return x

def _sample_rows_json(df: pd.DataFrame, n: int = 3):
    if df.empty: return []
    return df.head(n).applymap(_to_jsonable).to_dict(orient="records")

# ================== LLM PROMPT ==================
SYSTEM_PREAMBLE = """
You are a Python data assistant. Output ONLY a Python code snippet (no backticks, no prose).

Constraints:
- DataFrame is preloaded as `df`.
- Allowed: pandas as pd, numpy as np, matplotlib.pyplot as plt.
- Forbidden: imports, file I/O, network, os/sys/subprocess/pathlib/socket/pickle/tempfile, input(), exec/eval/compile, __import__.
- For forecasting/analytics, you already have:
  • ExponentialSmoothing (from statsmodels.tsa.holtwinters)
  • LinearRegression (from sklearn.linear_model)
  • datetime
- Do NOT import these, just use them directly.
- If you produce a table, assign to `result_df`.
- If you produce a chart, assign to `fig = plt.gcf()`.
- Prefer trend-only ExponentialSmoothing unless dataset has ≥24 months for seasonal models.
- Keep code under 120 lines.
"""

def llm_generate_code(user_instruction: str, df: pd.DataFrame) -> str:
    if not API_KEY:
        return "# ERROR: Missing GOOGLE_API_KEY.\nresult_df = df.head(5)"
    schema = {
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "sample_rows": _sample_rows_json(df, 3),
    }
    prompt = (
        SYSTEM_PREAMBLE
        + "\nDataFrame schema (JSON):\n"
        + json.dumps(schema, indent=2)
        + "\n\nUser request:\n"
        + (user_instruction or "")
        + "\n\n# Python code only below:\n"
    )
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", None) or "").strip()
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```") and not ln.strip().startswith("python")]
            text = "\n".join(lines).strip()
        return text
    except Exception as e:
        return f"# ERROR calling Google AI: {e}\nresult_df = df.head(5)"

# ================== CODE SANITIZER & SANDBOX ==================
IMPORT_LINE_RE = re.compile(r'^\s*(?:from\s+\S+\s+import\s+.*|import\s+.+)$')
CONTINUATION_RE = re.compile(r'.*\\\s*$')

def sanitize_code(snippet: str) -> str:
    if not isinstance(snippet, str): return ""
    code = snippet.strip()
    if code.startswith("```"):
        code = "\n".join(ln for ln in code.splitlines() if not ln.strip().startswith("```") and not ln.strip().startswith("python"))
    cleaned_lines, skip_cont = [], False
    for ln in code.splitlines():
        if skip_cont:
            if CONTINUATION_RE.match(ln): continue
            else: skip_cont = False; continue
        if IMPORT_LINE_RE.match(ln):
            if CONTINUATION_RE.match(ln): skip_cont = True
            continue
        cleaned_lines.append(ln)
    return "\n".join(cleaned_lines)

FORBIDDEN_CALLS = {"open","exec","eval","compile","__import__","input","system"}
FORBIDDEN_ATTR_ROOTS = {"os","sys","subprocess","pathlib","socket","requests","shutil","pickle","dill","tempfile","builtins","importlib"}
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom)
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
            while isinstance(base, ast.Attribute): base = base.value
            if isinstance(base, ast.Name) and base.id in FORBIDDEN_ATTR_ROOTS:
                raise ValueError(f"Forbidden module usage: {base.id}.*")
            self.generic_visit(node)
    try:
        Guard().visit(tree)
    except ValueError as ve:
        return False, str(ve)
    return True, None

def run_snippet(snippet: str, df: pd.DataFrame):
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
        if fig and not fig.axes: fig = None
        return result_df, fig, "Executed successfully (imports stripped)."
    except Exception as e:
        msg = "".join(traceback.format_exception_only(type(e), e))
        if "initial seasonals" in msg:
            try:
                alt = cleaned.replace("seasonal=\"add\"", "").replace("seasonal='add'", "")
                exec(alt, safe_globals, safe_locals)
                result_df = safe_locals.get("result_df")
                fig = safe_locals.get("fig", plt.gcf())
                if fig and not fig.axes: fig = None
                return result_df, fig, "Executed with fallback: removed seasonal component."
            except Exception as e2:
                return None, None, "Execution error after fallback:\n" + "".join(traceback.format_exception_only(type(e2), e2))
        return None, None, "Execution error:\n" + msg

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

# ================== HELPERS: AUTO INSIGHTS ==================
def ensure_dates_and_revenue(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "OrderDate" in out.columns:
        out["OrderDate"] = pd.to_datetime(out["OrderDate"], errors="coerce")
    elif "Date" in out.columns:
        out["OrderDate"] = pd.to_datetime(out["Date"], errors="coerce")
    else:
        # try best-effort: find first datetime-like column
        for c in out.columns:
            try:
                cand = pd.to_datetime(out[c], errors="raise")
                out["OrderDate"] = cand
                break
            except Exception:
                continue
        if "OrderDate" not in out.columns:
            out["OrderDate"] = pd.NaT
    # Revenue
    if "Revenue" not in out.columns:
        if {"Quantity","UnitPrice"}.issubset(out.columns):
            out["Revenue"] = pd.to_numeric(out["Quantity"], errors="coerce") * pd.to_numeric(out["UnitPrice"], errors="coerce")
        else:
            out["Revenue"] = np.nan
    return out

def kpis(df: pd.DataFrame):
    total_rev = pd.to_numeric(df["Revenue"], errors="coerce").sum(skipna=True)
    orders = len(df)
    aov = (total_rev / orders) if orders else 0.0
    # Top customer/product
    top_cust = df.groupby("Customer")["Revenue"].sum().sort_values(ascending=False).head(1)
    top_prod = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(1)
    return {
        "total_revenue": float(total_rev) if pd.notna(total_rev) else 0.0,
        "aov": float(aov) if pd.notna(aov) else 0.0,
        "top_customer": (top_cust.index[0], float(top_cust.iloc[0])) if len(top_cust) else ("—", 0.0),
        "top_product": (top_prod.index[0], float(top_prod.iloc[0])) if len(top_prod) else ("—", 0.0),
    }

def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    s = df.dropna(subset=["OrderDate"]).set_index("OrderDate")["Revenue"]
    return s.resample("MS").sum().rename("Revenue").to_frame()

def yoy_growth(mrev: pd.DataFrame) -> float | None:
    if len(mrev) < 24: return None
    last12  = mrev["Revenue"].iloc[-12:].sum()
    prev12  = mrev["Revenue"].iloc[-24:-12].sum()
    if prev12 == 0: return None
    return float((last12 - prev12) / prev12 * 100.0)

def quick_forecast_plot(mrev: pd.DataFrame, ax) -> str:
    # If enough history, try seasonal; else trend-only
    y = mrev["Revenue"].astype(float)
    label = ""
    if len(mrev) >= 24:
        try:
            model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12).fit()
            label = "ETS Additive (trend+seasonal)"
        except Exception:
            model = ExponentialSmoothing(y, trend="add").fit()
            label = "ETS Additive (trend-only fallback)"
    else:
        model = ExponentialSmoothing(y, trend="add").fit()
        label = "ETS Additive (trend-only)"
    fcast = model.forecast(6)
    mrev["Revenue"].plot(ax=ax)
    fcast.index = pd.date_range(start=mrev.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS")
    fcast.plot(ax=ax, linestyle="--")
    ax.set_title("Revenue Forecast (next 6 months)")
    ax.set_ylabel("Revenue")
    ax.legend(["History", "Forecast"])
    return label

# ================== MAIN UI ==================
st.title("📊 PlotTwist — Sales Analytics Copilot")
st.caption("Upload Excel → Instant insights + charts → Click a prompt chip or ask your own → Gemini generates code → Safe run.")

file = st.file_uploader("Upload sales file (.xlsx preferred, .csv accepted)", type=["xlsx","csv"])
df = None
if file:
    try:
        if file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(file, engine="openpyxl")
        else:
            df = pd.read_csv(file, on_bad_lines="skip")
        st.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
        st.dataframe(df.head(50), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# ======= WOW MOMENT: Auto KPIs + Charts + Forecast preview =======
if df is not None:
    df2 = ensure_dates_and_revenue(df)
    mrev = monthly_revenue(df2)

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    k = kpis(df2)
    c1.metric("Total Revenue", f"{k['total_revenue']:,.0f}")
    c2.metric("Avg Order Value", f"{k['aov']:,.2f}")
    c3.metric("Top Customer", f"{k['top_customer'][0]}", f"{k['top_customer'][1]:,.0f}")
    c4.metric("Top Product", f"{k['top_product'][0]}", f"{k['top_product'][1]:,.0f}")
    yg = yoy_growth(mrev)
    c5.metric("YoY Growth (L12 vs P12)", f"{yg:.1f}%" if yg is not None else "—")

    st.markdown("### Quick Charts")
    colA, colB = st.columns(2)
    with colA:
        fig1, ax1 = plt.subplots()
        if not mrev.empty:
            mrev.plot(ax=ax1, legend=False)
            ax1.set_title("Monthly Revenue")
            ax1.set_ylabel("Revenue")
        st.pyplot(fig1, clear_figure=True)

    with colB:
        fig2, ax2 = plt.subplots()
        if "Customer" in df2.columns:
            top_c = df2.groupby("Customer")["Revenue"].sum().sort_values(ascending=False).head(5)
            top_c.plot.bar(ax=ax2)
            ax2.set_title("Top 5 Customers by Revenue")
            ax2.set_ylabel("Revenue")
            ax2.set_xlabel("")
        st.pyplot(fig2, clear_figure=True)

    colC, _ = st.columns([2,1])
    with colC:
        fig3, ax3 = plt.subplots()
        if "Category" in df2.columns:
            mix = df2.groupby("Category")["Revenue"].sum()
            if len(mix) > 0:
                ax3.pie(mix.values, labels=mix.index, autopct="%1.0f%%")
                ax3.set_title("Product Mix by Revenue")
        st.pyplot(fig3, clear_figure=True)

    # Forecast preview if we have history
    if len(mrev) >= 6:
        st.markdown("### 🔮 Forecast Preview")
        fig4, ax4 = plt.subplots()
        label = quick_forecast_plot(mrev, ax4)
        st.pyplot(fig4, clear_figure=True)
        st.caption(f"*Model used: {label}*")

    st.markdown("---")

# ======= PROMPT CHIPS + GEMINI RUN =======
if df is not None:
    st.markdown("### Ask a question or try a suggestion")
    suggestions = [
        "Show monthly revenue trend with a clean line chart.",
        "Top 10 customers by total revenue as a bar chart.",
        "Revenue by region as a sorted bar chart.",
        "Forecast next 12 months of revenue using ExponentialSmoothing.",
        "Scatterplot of Quantity vs Revenue with a regression line.",
        "Category-wise monthly revenue stacked area chart."
    ]
    chip_cols = st.columns(len(suggestions))
    chosen = None
    for i, s in enumerate(suggestions):
        if chip_cols[i].button(s, use_container_width=True):
            chosen = s

    default_text = chosen or st.session_state.get("last_prompt", "")
    user_prompt = st.text_area("Your prompt", value=default_text, height=100, key="prompt_box")

    run_now = st.button("Generate & Run", type="primary") or (chosen is not None)
    if run_now:
        st.session_state["last_prompt"] = user_prompt
        with st.spinner("Asking Gemini…"):
            code = llm_generate_code(user_prompt, df2)
        st.subheader("Sanitized code (imports auto-removed)")
        st.code(sanitize_code(code) if code else "# empty", language="python")
        if code and not code.strip().startswith("# ERROR"):
            with st.spinner("Executing safely…"):
                result_df, fig, logs = run_snippet(code, df2)
            st.markdown(f"**Logs:** {logs}")
            if isinstance(result_df, pd.DataFrame):
                st.subheader("Result table"); st.dataframe(result_df, use_container_width=True)
            if fig is not None:
                st.subheader("Chart"); st.pyplot(fig, clear_figure=True)
else:
    st.info("💡 Tip: download the Excel template from the sidebar and try prompts right away.")
