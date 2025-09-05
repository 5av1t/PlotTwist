import os, io, json, ast, traceback, re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import google.generativeai as genai
import requests
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import datetime
from datetime import timedelta

# ================== CONFIG ==================
st.set_page_config(page_title="PlotTwist — Sales Analytics Copilot", page_icon="📊", layout="wide")
MODEL_NAME = "gemini-1.5-flash"
FIG_W, FIG_H = 3.5, 2.4

# Replace with your GitHub RAW URLs after upload
RAW_XLSX_URL = "https://raw.githubusercontent.com/5av1t/plottwist/main/sales_template.xlsx"
RAW_CSV_URL = "https://raw.githubusercontent.com/5av1t/plottwist/main/sales_template.csv"

API_KEY = os.getenv("GOOGLE_API_KEY")
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
if API_KEY:
    genai.configure(api_key=API_KEY)

# ================== POWER MODE (safe-but-flexible) ==================
DEFAULT_ALLOWED_PREFIXES = [
    "pandas", "numpy", "matplotlib", "sklearn.linear_model", "statsmodels.tsa.holtwinters", "scipy", "scipy.stats"
]
SEABORN_PREFIX = "seaborn"  # optional

# ---------- Format helpers ----------
def fmt_currency(x):
    try: x = float(x)
    except Exception: return "—"
    if abs(x) >= 1_000_000: return f"${x/1_000_000:.1f}M"
    if abs(x) >= 1_000:     return f"${x/1_000:.1f}K"
    return f"${x:,.0f}"

def fmt_percent(x):
    try: return f"{float(x):.1f}%"
    except Exception: return "—"

# ---------- JSON-safe serializer ----------
def _to_jsonable(x):
    if isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date)):
        return x.isoformat()
    if isinstance(x, (np.integer,)):   return int(x)
    if isinstance(x, (np.floating,)):  return float(x)
    if isinstance(x, (np.bool_,)):     return bool(x)
    return x

def _sample_rows_json(df, n=3):
    if df.empty: return []
    return df.head(n).applymap(_to_jsonable).to_dict(orient="records")

# ================== DATE-SAFE PLOTTING HELPERS ==================
def plot_datetime(ax, x_like, y_vals, **kwargs):
    """Plot with a guaranteed date x-axis to avoid categorical UnitData issues."""
    xd = pd.to_datetime(x_like, errors="coerce")
    xnum = mdates.date2num(pd.DatetimeIndex(xd).to_pydatetime())
    line = ax.plot(xnum, np.asarray(y_vals, dtype=float), **kwargs)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    return line

def fill_between_datetime(ax, x_like, y1, y2, **kwargs):
    xd = pd.to_datetime(x_like, errors="coerce")
    xnum = mdates.date2num(pd.DatetimeIndex(xd).to_pydatetime())
    ax.fill_between(xnum, np.asarray(y1, dtype=float), np.asarray(y2, dtype=float), **kwargs)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

# ================== LLM PROMPT (dynamic by mode) ==================
BASE_PREAMBLE = """
You are a Python data assistant. Output ONLY a Python code snippet (no backticks, no prose).

Data:
- A pandas DataFrame is preloaded as `df`.

Charting:
- Use pandas/numpy/matplotlib.
- For any date x-axis, ALWAYS use the helpers:
  • plot_datetime(ax, x, y)
  • fill_between_datetime(ax, x, y1, y2)
- Assign tables to `result_df`, charts to `fig = plt.gcf()`.

Forecasting:
- You already have ExponentialSmoothing, LinearRegression, datetime.
- Prefer trend-only ExponentialSmoothing unless dataset has ≥24 months for seasonality.

Limits:
- No network calls. Keep code under 120 lines.
"""

def build_preamble(pro_mode: bool, allow_seaborn: bool) -> str:
    if pro_mode:
        allowed = DEFAULT_ALLOWED_PREFIXES.copy()
        if allow_seaborn:
            allowed.append(SEABORN_PREFIX)
        allowed_str = ", ".join(allowed)
        return BASE_PREAMBLE + f"""

Imports (Pro Mode):
- You MAY import ONLY from these packages/prefixes: {allowed_str}.
- Do NOT import anything else (no os/sys/subprocess/requests/pathlib/importlib/etc).
"""
    else:
        return BASE_PREAMBLE + """
Imports (Standard Mode):
- Do NOT import anything; required libs are preloaded for you.
"""

def llm_generate_code(user_instruction, df, pro_mode: bool, allow_seaborn: bool):
    if not API_KEY:
        return "# ERROR: Missing GOOGLE_API_KEY.\nresult_df = df.head(5)"
    schema = {
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "sample_rows": _sample_rows_json(df, 3),
    }
    preamble = build_preamble(pro_mode, allow_seaborn)
    prompt = preamble + "\nSchema:\n" + json.dumps(schema, indent=2) \
             + "\n\nUser request:\n" + (user_instruction or "") + "\n\n# Python code only:\n"
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

# ================== UPLOAD READER (robust for mobile) ==================
def read_any_table(uploaded):
    if uploaded is None:
        raise ValueError("No file provided")
    uploaded.seek(0)
    raw = uploaded.read()
    uploaded.seek(0)
    name = (uploaded.name or "").lower()

    def try_csv(b):
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try: return pd.read_csv(io.StringIO(b.decode(enc)), on_bad_lines="skip")
            except Exception: continue
        return pd.read_csv(io.BytesIO(b), on_bad_lines="skip")

    if name.endswith(".xlsx") and raw[:2] == b"PK":
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    if name.endswith(".xlsx") and raw[:2] != b"PK":
        try: return try_csv(raw)
        except Exception: pass
        raise ValueError("This .xlsx isn’t a valid Excel file. Try CSV instead.")
    return try_csv(raw)

# ================== SANDBOX (with optional whitelisted imports) ==================
IMPORT_LINE_RE = re.compile(r'^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import\s+.*|import\s+([A-Za-z0-9_\. ,]+))\s*$')
CONTINUATION_RE = re.compile(r'.*\\\s*$')

FORBIDDEN_CALLS = {"open","exec","eval","compile","__import__","input","system"}
FORBIDDEN_ATTR_ROOTS = {"os","sys","subprocess","pathlib","socket","requests","shutil","pickle","dill","tempfile","builtins","importlib"}
DISALLOWED_ATTR_NAMES = {"__dict__","__class__","__mro__","__subclasses__","__globals__","__getattribute__","__getattr__"}

def sanitize_code(snippet: str, pro_mode: bool) -> str:
    if not isinstance(snippet, str): return ""
    code = snippet.strip()
    if code.startswith("```"):
        code = "\n".join(ln for ln in code.splitlines() if not ln.strip().startswith("```") and not ln.strip().startswith("python"))
    if pro_mode:
        return code  # keep imports; validate later
    # Standard: strip all imports entirely
    cleaned_lines, skip = [], False
    for ln in code.splitlines():
        if skip:
            if CONTINUATION_RE.match(ln): continue
            else: skip = False; continue
        if IMPORT_LINE_RE.match(ln):
            if CONTINUATION_RE.match(ln): skip = True
            continue
        cleaned_lines.append(ln)
    return "\n".join(cleaned_lines)

def _module_prefix_allowed(mod: str, allowed_prefixes: list[str]) -> bool:
    return any(mod == p or mod.startswith(p + ".") for p in allowed_prefixes)

def validate_snippet(snippet: str, pro_mode: bool, allow_seaborn: bool):
    try:
        tree = ast.parse(snippet)
    except Exception as e:
        return False, f"Code not parsable: {e}"

    allowed_prefixes = DEFAULT_ALLOWED_PREFIXES.copy()
    if allow_seaborn:
        allowed_prefixes.append(SEABORN_PREFIX)

    class Guard(ast.NodeVisitor):
        def visit(self, node):
            # allow/validate imports in pro mode only
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if not pro_mode:
                    raise ValueError("Imports are not allowed in Standard Mode.")
                mod = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod = alias.name
                        if not _module_prefix_allowed(mod, allowed_prefixes):
                            raise ValueError(f"Disallowed import: {mod}")
                else:
                    mod = node.module or ""
                    if not _module_prefix_allowed(mod, allowed_prefixes):
                        raise ValueError(f"Disallowed import-from: {mod}")
                return
            return super().visit(node)

        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"Forbidden call: {node.func.id}()")
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute):
            if node.attr in DISALLOWED_ATTR_NAMES:
                raise ValueError(f"Forbidden attribute: {node.attr}")
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

def make_safe_import(allowed_prefixes: list[str]):
    real_import = __import__
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if not _module_prefix_allowed(name, allowed_prefixes):
            raise ImportError(f"Blocked import: {name}")
        return real_import(name, globals, locals, fromlist, level)
    return _safe_import

def run_snippet(snippet: str, df: pd.DataFrame, pro_mode: bool, allow_seaborn: bool):
    cleaned = sanitize_code(snippet, pro_mode)
    ok, err = validate_snippet(cleaned, pro_mode, allow_seaborn)
    if not ok:
        return None, None, f"Validation failed: {err}"

    allowed_prefixes = DEFAULT_ALLOWED_PREFIXES.copy()
    if allow_seaborn:
        allowed_prefixes.append(SEABORN_PREFIX)

    # Build safe __builtins__ consistently across runtimes (module vs dict)
    builtins_obj = __builtins__
    builtins_dict = builtins_obj if isinstance(builtins_obj, dict) else builtins_obj.__dict__
    safe_builtins = builtins_dict.copy()
    if pro_mode:
        safe_builtins["__import__"] = make_safe_import(allowed_prefixes)

    # Safe globals exposed to LLM code
    safe_globals = {
        "pd": pd, "np": np, "plt": plt, "mdates": mdates,
        "ExponentialSmoothing": ExponentialSmoothing,
        "LinearRegression": LinearRegression,
        "datetime": datetime,
        "plot_datetime": plot_datetime,
        "fill_between_datetime": fill_between_datetime,
        "__builtins__": safe_builtins,
    }
    safe_locals = {"df": df.copy()}

    try:
        exec(cleaned, safe_globals, safe_locals)
        result_df = safe_locals.get("result_df")
        fig = safe_locals.get("fig", plt.gcf())
        if fig and not fig.axes: fig = None
        return result_df, fig, "Executed OK"
    except Exception as e:
        return None, None, "Execution error:\n" + "".join(traceback.format_exception_only(type(e), e))

# ================== AUTO-INSIGHTS HELPERS ==================
def ensure_dates_and_revenue(df):
    out = df.copy()
    if "OrderDate" in out.columns:
        out["OrderDate"] = pd.to_datetime(out["OrderDate"], errors="coerce")
    elif "Date" in out.columns:
        out["OrderDate"] = pd.to_datetime(out["Date"], errors="coerce")
    else:
        for c in out.columns:
            try:
                out["OrderDate"] = pd.to_datetime(out[c], errors="raise"); break
            except Exception:
                continue
    if "Revenue" not in out.columns:
        if {"Quantity","UnitPrice"}.issubset(out.columns):
            out["Revenue"] = pd.to_numeric(out["Quantity"], errors="coerce") * pd.to_numeric(out["UnitPrice"], errors="coerce")
        else:
            out["Revenue"] = np.nan
    return out

def monthly_revenue(df):
    s = pd.to_datetime(df["OrderDate"], errors="coerce")
    m = pd.Series(pd.to_numeric(df["Revenue"], errors="coerce"), index=s)
    out = m.dropna().resample("MS").sum().to_frame(name="Revenue")
    out.index = pd.DatetimeIndex(out.index)
    return out

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ Mode & Templates")
    pro_mode = st.toggle("Pro mode: allow whitelisted imports", value=False,
                         help="Allows imports from pandas, numpy, matplotlib, sklearn.linear_model, statsmodels.tsa.holtwinters, scipy.stats (and optionally seaborn).")
    allow_seaborn = st.toggle("Also allow seaborn (if installed)", value=False)

    st.caption("Use RAW links for these files in your repo.")
    try:
        rx = requests.get(RAW_XLSX_URL, timeout=8); rx.raise_for_status()
        st.download_button("Excel (.xlsx) template", data=rx.content, file_name="sales_template.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.warning("Excel template not available")

    try:
        rc = requests.get(RAW_CSV_URL, timeout=8); rc.raise_for_status()
        st.download_button("CSV (.csv) template", data=rc.content, file_name="sales_template.csv", mime="text/csv")
    except Exception:
        st.warning("CSV template not available")

# ================== MAIN UI ==================
st.title("📊 PlotTwist — Sales Analytics Copilot (Super App)")
st.caption("Upload Excel/CSV → Gemini generates insights → Compact KPIs + charts. Pro mode allows safe imports.")

file = st.file_uploader("Upload sales file (.xlsx or .csv)", type=["xlsx","csv"])
df2 = None
if file:
    try:
        df2 = ensure_dates_and_revenue(read_any_table(file))
        st.success(f"Loaded {len(df2):,} rows × {len(df2.columns)} cols")
        st.dataframe(df2.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# ====== GEMINI FIRST ======
if df2 is not None:
    st.markdown("### 🤖 Gemini — Automated Analysis")
    default_prompt = (
        "Summarize the dataset with 1) monthly revenue trend (compact line), "
        "2) top 5 customers (bar), 3) total revenue + avg order value table, "
        "and if >=24 months, forecast next 6 months with ExponentialSmoothing. "
        "For any date x-axis, use plot_datetime(ax, x, y) and fill_between_datetime(ax, x, y1, y2)."
    )
    if "ran_auto" not in st.session_state: st.session_state["ran_auto"] = False
    user_prompt = st.text_area("Your prompt", value=st.session_state.get("last_prompt", default_prompt), height=90)
    run_now = st.button("Generate & Run", type="primary") or (not st.session_state["ran_auto"])
    if run_now:
        st.session_state["last_prompt"] = user_prompt
        with st.spinner("Asking Gemini…"):
            code = llm_generate_code(user_prompt, df2, pro_mode, allow_seaborn)
        st.subheader("Sanitized/validated code")
        st.code(code if code else "# empty", language="python")
        if code and not code.strip().startswith("# ERROR"):
            with st.spinner("Executing safely…"):
                result_df, fig, logs = run_snippet(code, df2, pro_mode, allow_seaborn)
            st.markdown(f"**Logs:** {logs}")
            if isinstance(result_df, pd.DataFrame): st.dataframe(result_df, use_container_width=True)
            if fig is not None: st.pyplot(fig, use_container_width=False, clear_figure=True)
        st.session_state["ran_auto"] = True

    # ===== Optional compact forecast preview (date-safe) =====
    mrev = monthly_revenue(df2)
    if len(mrev) >= 12:
        st.markdown("### 🔮 Forecast Preview")
        y = mrev["Revenue"].astype(float)
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
        resid = y - model.fittedvalues.reindex(y.index).bfill()
        resid_std = float(np.nanstd(resid))
        fcast.index = pd.date_range(start=mrev.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS")

        figF, axF = plt.subplots(figsize=(FIG_W+0.7, FIG_H))
        plot_datetime(axF, mrev.index, y.values, label="History")
        plot_datetime(axF, fcast.index, fcast.values, linestyle="--", label="Forecast")
        fill_between_datetime(axF, fcast.index, (fcast - 1.96*resid_std).values, (fcast + 1.96*resid_std).values, alpha=0.2)
        axF.set_title("Revenue Forecast (6 mo)", fontsize=10, pad=6)
        axF.set_ylabel("Revenue"); axF.grid(alpha=0.2); axF.legend(fontsize=8)
        figF.tight_layout()
        st.pyplot(figF, use_container_width=False, clear_figure=True)
        st.caption(f"*Model used: {label}*")

else:
    st.info("💡 Tip: Download a template from the sidebar and try again.")
