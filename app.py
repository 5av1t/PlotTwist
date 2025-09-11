import os, io, json, traceback, re
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
from datetime import timedelta  # exposed for LLM code

# ================== CONFIG ==================
st.set_page_config(page_title="PlotTwist — Data-Aware & Self-Healing", page_icon="📊", layout="wide")
MODEL_NAME = "gemini-1.5-flash"
FIG_W, FIG_H = 3.5, 2.4

# 👉 UPDATE THIS to the raw CSV in your GitHub repo (keeps upload optional)
RAW_CSV_URL  = "https://raw.githubusercontent.com/5av1t/test1/main/sales_template.csv"
RAW_XLSX_URL = "https://raw.githubusercontent.com/5av1t/test1/main/sales_template.xlsx"  # optional template

# Gemini key (optional; app still works without it for non-LLM flows)
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
if API_KEY:
    genai.configure(api_key=API_KEY)

# ================== SHIMS / HELPERS ==================
def register_matplotlib_converters():
    """No-op shim for legacy code snippets that call this."""
    return None

def fmt_currency(x):
    try: x = float(x)
    except Exception: return "—"
    if abs(x) >= 1_000_000: return f"${x/1_000_000:.1f}M"
    if abs(x) >= 1_000:     return f"${x/1_000:.1f}K"
    return f"${x:,.0f}"

def fmt_percent(x):
    try: return f"{float(x):.1f}%"
    except Exception: return "—"

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

# ----- Date-safe plotting + tick helpers -----
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

def set_tick_label_alignment(ax, axis="x", rotation=0, ha="center"):
    """Safely set rotation + horizontal alignment on tick labels."""
    if axis in ("x", "both"):
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(rotation)
            lbl.set_horizontalalignment(ha)
    if axis in ("y", "both"):
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(rotation)
            lbl.set_horizontalalignment(ha)

# ================== DATA LOADING ==================
def read_csv_from_repo(raw_url: str) -> pd.DataFrame:
    # Try remote raw URL first
    if raw_url:
        try:
            r = requests.get(raw_url, timeout=10)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception:
            pass
    # Fallback to local file in repo, if present
    for candidate in ["sales_template.csv", "data.csv", "sales.csv"]:
        if os.path.exists(candidate):
            return pd.read_csv(candidate)
    raise FileNotFoundError("Could not load CSV from repo. Update RAW_CSV_URL or add 'sales_template.csv' to repo root.")

def read_any_table(uploaded):
    """Mobile-safe reader for optional user upload (xlsx or csv)."""
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

def ensure_dates_and_revenue(df):
    out = df.copy()
    # Date column
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
    # Revenue column
    if "Revenue" not in out.columns:
        if {"Quantity","UnitPrice"}.issubset(out.columns):
            out["Revenue"] = pd.to_numeric(out["Quantity"], errors="coerce") * pd.to_numeric(out["UnitPrice"], errors="coerce")
        else:
            amt = next((c for c in out.columns if "amount" in c.lower() or "revenue" in c.lower() or "sales" in c.lower()), None)
            if amt is not None:
                out["Revenue"] = pd.to_numeric(out[amt], errors="coerce")
            else:
                out["Revenue"] = np.nan
    return out

def monthly_revenue(df):
    s = pd.to_datetime(df["OrderDate"], errors="coerce")
    m = pd.Series(pd.to_numeric(df["Revenue"], errors="coerce"), index=s)
    out = m.dropna().resample("MS").sum().to_frame(name="Revenue")
    out.index = pd.DatetimeIndex(out.index)
    return out

# ================== DATA SNAPSHOT (what Gemini sees) ==================
def build_data_snapshot(df: pd.DataFrame, *, head_rows: int = 20, max_chars: int = 50000) -> dict:
    """Compact summary of actual data (schema, stats, top values, head) to guide Gemini's code."""
    snap = {}
    nulls = df.isna().sum().to_dict()
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    rows, cols = df.shape
    snap["shape"] = {"rows": int(rows), "cols": int(cols)}
    snap["columns"] = list(df.columns)
    snap["dtypes"] = dtypes
    snap["null_counts"] = {k: int(v) for k, v in nulls.items()}

    # Date ranges
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower()]
    date_info = {}
    for c in date_cols[:3]:
        try:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                date_info[c] = {"min": s.min().isoformat() if pd.notna(s.min()) else None,
                                "max": s.max().isoformat() if pd.notna(s.max()) else None}
        except Exception:
            pass
    if date_info: snap["date_ranges"] = date_info

    # Numeric stats
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        desc = df[num_cols].describe().round(3).to_dict()
        snap["numeric_describe"] = {k: {kk: _to_jsonable(vv) for kk, vv in v.items()} for k, v in desc.items()}

    # Categorical top values
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c])]
    cat_counts = {}
    for c in cat_cols[:6]:
        try:
            vc = df[c].astype(str).value_counts().head(8)
            cat_counts[c] = {str(k): int(v) for k, v in vc.items()}
        except Exception:
            continue
    if cat_counts: snap["top_values"] = cat_counts

    # CSV head (respect char budget)
    try:
        head_csv = df.head(head_rows).to_csv(index=False)
        if len(head_csv) > max_chars:
            head_csv = head_csv[:max_chars]
        snap["csv_head"] = head_csv
    except Exception:
        snap["csv_head"] = ""

    return snap

# ================== LLM PROMPTS ==================
BASE_RULES = """
You are a Python data assistant. Output ONLY a Python code snippet (no backticks, no prose).

Data:
- A pandas DataFrame is preloaded as `df`.
- You are also given a DATA SNAPSHOT extracted from the real file: schema, dtypes, nulls, value counts, numeric stats, and the CSV head.
- Your code MUST be consistent with the snapshot (column names, types, shapes).

Charting:
- Use pandas/numpy/matplotlib.
- For any date x-axis, ALWAYS use:
  • plot_datetime(ax, x, y)
  • fill_between_datetime(ax, x, y1, y2)
- To rotate/align tick labels, use set_tick_label_alignment(ax, axis="x", rotation=45, ha="right").
  Do NOT pass 'ha' to tick_params (it will error).

Results:
- Assign tables to `result_df` (pd.DataFrame).
- Assign charts to `fig = plt.gcf()`.

Forecasting:
- You can use ExponentialSmoothing, LinearRegression, datetime, timedelta.
- Prefer trend-only ExponentialSmoothing unless dataset has ≥24 months for seasonality.

Limits:
- No network calls. Keep code under 120 lines. Do not touch files.
- register_matplotlib_converters() exists but is a no-op; do not rely on it.
"""

def _remove_md_fences(text: str) -> str:
    if not isinstance(text, str): return ""
    if text.strip().startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```") and not ln.strip().startswith("python")]
        return "\n".join(lines).strip()
    return text.strip()

def llm_generate_code(user_instruction, df, snapshot: dict):
    if not API_KEY:
        return "# ERROR: Missing GOOGLE_API_KEY.\nresult_df = df.head(5)"
    schema = {
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "sample_rows": _sample_rows_json(df, 3),
    }
    prompt = (
        BASE_RULES
        + "\n\n# DATA SNAPSHOT (JSON)\n" + json.dumps(snapshot, ensure_ascii=False)[:90000]
        + "\n\n# LIGHT SCHEMA\n" + json.dumps(schema, ensure_ascii=False)
        + "\n\n# USER REQUEST\n" + (user_instruction or "")
        + "\n\n# PYTHON CODE ONLY BELOW\n"
    )
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt)
    return _remove_md_fences(getattr(resp, "text", "") or "")

def llm_repair_code(prev_code: str, error_text: str, df: pd.DataFrame, snapshot: dict):
    """Ask Gemini to repair the previous code using the error + snapshot."""
    if not API_KEY:
        return None
    schema = {
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "sample_rows": _sample_rows_json(df, 3),
    }
    repair_prompt = (
        BASE_RULES
        + "\nYou previously produced code that errored. Repair it using the DATA SNAPSHOT."
        + "\nReturn only corrected Python code (no backticks, no prose)."
        + "\n\n# PREVIOUS CODE\n" + prev_code
        + "\n\n# ERROR\n" + error_text
        + "\n\n# DATA SNAPSHOT (JSON)\n" + json.dumps(snapshot, ensure_ascii=False)[:90000]
        + "\n\n# LIGHT SCHEMA\n" + json.dumps(schema, ensure_ascii=False)
        + "\n\n# CORRECTED PYTHON CODE ONLY BELOW\n"
    )
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(repair_prompt)
    return _remove_md_fences(getattr(resp, "text", "") or "")

# ================== SANDBOX (execute LLM code) ==================
def run_snippet(snippet: str, df: pd.DataFrame):
    # Hard cleanup for a few common issues
    if "tick_params(" in snippet and "ha=" in snippet:
        snippet = re.sub(r"(tick_params\([^)]*)ha\s*=\s*['\"][^'\"]+['\"]\s*,?\s*", r"\1", snippet)

    builtins_obj = __builtins__
    builtins_dict = builtins_obj if isinstance(builtins_obj, dict) else builtins_obj.__dict__
    safe_builtins = builtins_dict.copy()

    safe_globals = {
        "pd": pd, "np": np, "plt": plt, "mdates": mdates,
        "ExponentialSmoothing": ExponentialSmoothing,
        "LinearRegression": LinearRegression,
        "datetime": datetime, "timedelta": timedelta,
        "plot_datetime": plot_datetime,
        "fill_between_datetime": fill_between_datetime,
        "set_tick_label_alignment": set_tick_label_alignment,
        "register_matplotlib_converters": register_matplotlib_converters,
        "__builtins__": safe_builtins,
    }
    safe_locals = {"df": df.copy()}
    try:
        exec(snippet, safe_globals, safe_locals)
        result_df = safe_locals.get("result_df")
        fig = safe_locals.get("fig", plt.gcf())
        if fig and not fig.axes: fig = None
        return result_df, fig, "Executed OK"
    except Exception as e:
        return None, None, "Execution error:\n" + "".join(traceback.format_exception_only(type(e), e))

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("⬇️ Templates")
    st.caption("These links pull from your GitHub repo raw URLs.")
    if RAW_XLSX_URL:
        try:
            rx = requests.get(RAW_XLSX_URL, timeout=8); rx.raise_for_status()
            st.download_button("Excel template (.xlsx)", data=rx.content, file_name="sales_template.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            st.warning("Excel template not available")
    if RAW_CSV_URL:
        try:
            rc = requests.get(RAW_CSV_URL, timeout=8); rc.raise_for_status()
            st.download_button("CSV template (.csv)", data=rc.content, file_name="sales_template.csv", mime="text/csv")
        except Exception:
            st.warning("CSV template not available")

    st.divider()
    st.header("📦 Data Source")
    use_repo = st.toggle("Use repository CSV (recommended)", value=True)
    share_snapshot = st.checkbox("Share a snapshot of the data with Gemini", value=True,
                                 help="Includes schema, stats, and CSV head in the LLM prompt.")
    head_rows = st.number_input("Rows for CSV head", 5, 200, 30, step=5)
    max_chars = st.slider("Max snapshot characters", 5_000, 120_000, 40_000, step=5_000)

    st.caption("Or use your own file (optional):")
    uploaded_file = st.file_uploader("Upload Excel/CSV (optional)", type=["xlsx","csv"])

# ================== MAIN UI ==================
st.title("📊 PlotTwist — Sales Analytics Copilot (Data-Aware & Self-Healing)")
st.caption("Same app as before. Now defaults to your repo CSV so users don’t need to upload. Upload remains optional.")

# Load data per your preference
df2 = None
load_err = None
try:
    if use_repo:
        # Use repo CSV by default
        df_raw = read_csv_from_repo(RAW_CSV_URL)
    elif uploaded_file is not None:
        # Use uploaded file if user opted out of repo CSV
        if uploaded_file.name.lower().endswith(".xlsx"):
            df_raw = read_any_table(uploaded_file)
        else:
            df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = read_csv_from_repo(RAW_CSV_URL)  # fallback to repo anyway
    df2 = ensure_dates_and_revenue(df_raw)
    st.success(f"Loaded {len(df2):,} rows × {len(df2.columns)} cols")
    st.dataframe(df2.head(20), use_container_width=True)
except Exception as e:
    load_err = str(e)
    st.error(f"Failed to load data: {load_err}")

# ===== LLM-driven analysis (unchanged UX) =====
if df2 is not None:
    st.markdown("### 🤖 Ask Gemini")
    default_prompt = (
        "Summarize the dataset with 1) monthly revenue trend (compact line), "
        "2) top 5 customers (bar), 3) total revenue + avg order value table, "
        "and if ≥24 months, forecast next 6 months (same chart)."
    )
    auto_repair = st.checkbox("Auto-repair with Gemini on error", value=True)
    user_prompt = st.text_area("Your prompt", value=default_prompt, height=100)

    if st.button("Generate & Run", type="primary"):
        snapshot = build_data_snapshot(df2, head_rows=int(head_rows), max_chars=int(max_chars)) if share_snapshot else {
            "shape": {"rows": int(df2.shape[0]), "cols": int(df2.shape[1])},
            "columns": list(df2.columns)
        }

        # 1) First attempt
        with st.spinner("Asking Gemini…"):
            code_v1 = llm_generate_code(user_prompt, df2, snapshot)
        st.subheader("Generated code (v1)")
        st.code(code_v1 or "# empty", language="python")

        with st.spinner("Executing v1…"):
            result_df, fig, logs = run_snippet(code_v1, df2)
        st.markdown(f"**Logs (v1):** {logs}")
        if isinstance(result_df, pd.DataFrame):
            st.dataframe(result_df, use_container_width=True)
        if fig is not None:
            st.pyplot(fig, use_container_width=False, clear_figure=True)

        # 2) Auto-repair if failed
        failed = logs.startswith("Execution error:")
        if auto_repair and failed:
            with st.spinner("Repairing with Gemini…"):
                code_v2 = llm_repair_code(code_v1, logs, df2, snapshot) or ""
            st.subheader("Repaired code (v2)")
            st.code(code_v2 or "# empty", language="python")
            with st.spinner("Executing v2…"):
                result_df2, fig2, logs2 = run_snippet(code_v2, df2)
            st.markdown(f"**Logs (v2):** {logs2}")
            if isinstance(result_df2, pd.DataFrame):
                st.dataframe(result_df2, use_container_width=True)
            if fig2 is not None:
                st.pyplot(fig2, use_container_width=False, clear_figure=True)

    # ===== Compact built-in forecast preview (unchanged) =====
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
        set_tick_label_alignment(axF, axis="x", rotation=45, ha="right")
        axF.set_title("Revenue Forecast (6 mo)", fontsize=10, pad=6)
        axF.set_ylabel("Revenue"); axF.grid(alpha=0.2); axF.legend(fontsize=8)
        figF.tight_layout()
        st.pyplot(figF, use_container_width=False, clear_figure=True)
        st.caption(f"*Model used: {label}*")
else:
    if load_err is None:
        st.info("Upload a file or enable 'Use repository CSV' in the sidebar to proceed.")
