import os, io, json, re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import datetime
from datetime import timedelta
import google.generativeai as genai

# ================== CONFIG ==================
st.set_page_config(page_title="PlotTwist — One-Click Monthly Review", page_icon="📊", layout="wide")
FIG_W, FIG_H = 3.6, 2.5

# 👉 Update this to your repo’s raw CSV once committed
RAW_CSV_URL = "https://raw.githubusercontent.com/5av1t/PlotTwist/test1/sales_template.csv"

# Gemini key (optional; app still works without it)
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
if API_KEY:
    genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-1.5-flash"

# ================== DATE-SAFE PLOTTING HELPERS ==================
def plot_datetime(ax, x_like, y_vals, **kwargs):
    xd = pd.to_datetime(x_like, errors="coerce")
    xnum = mdates.date2num(pd.DatetimeIndex(xd).to_pydatetime())
    line = ax.plot(xnum, np.asarray(y_vals, dtype=float), **kwargs)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    return line

def set_tick_label_alignment(ax, axis="x", rotation=0, ha="center"):
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
    raise FileNotFoundError("Could not load CSV. Update RAW_CSV_URL or add 'sales_template.csv' to the repo root.")

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Best-effort date column detection
    date_col = None
    for c in out.columns:
        lc = c.lower()
        if lc in ("orderdate", "date", "order_date", "invoice_date"):
            date_col = c; break
    if date_col is None:
        # heuristic: first column that parses
        for c in out.columns:
            try:
                pd.to_datetime(out[c], errors="raise")
                date_col = c
                break
            except Exception:
                continue
    if date_col is None:
        raise ValueError("No date-like column found. Include an 'OrderDate' or 'Date' column.")

    out["OrderDate"] = pd.to_datetime(out[date_col], errors="coerce")
    # Revenue detection or derivation
    if "Revenue" not in out.columns:
        q = next((c for c in out.columns if c.lower() in ("qty","quantity","units")), None)
        p = next((c for c in out.columns if c.lower() in ("price","unitprice","unit_price","selling_price")), None)
        if q and p:
            out["Revenue"] = pd.to_numeric(out[q], errors="coerce") * pd.to_numeric(out[p], errors="coerce")
        else:
            # fallback: try an amount column
            amt = next((c for c in out.columns if "amount" in c.lower() or "revenue" in c.lower() or "sales" in c.lower()), None)
            if amt:
                out["Revenue"] = pd.to_numeric(out[amt], errors="coerce")
            else:
                raise ValueError("No revenue column found and cannot derive it (need Quantity & UnitPrice or an Amount column).")
    out["Revenue"] = pd.to_numeric(out["Revenue"], errors="coerce")

    # Optional convenience columns
    if "Customer" not in out.columns:
        # guess a customer-like column
        cust = next((c for c in out.columns if "customer" in c.lower() or "client" in c.lower() or "buyer" in c.lower()), None)
        if cust: out["Customer"] = out[cust].astype(str)
        else: out["Customer"] = "Unknown"
    if "Product" not in out.columns:
        prod = next((c for c in out.columns if "product" in c.lower() or "item" in c.lower() or "sku" in c.lower()), None)
        if prod: out["Product"] = out[prod].astype(str)
        else: out["Product"] = "Unknown"

    return out.dropna(subset=["OrderDate", "Revenue"])

def last_month_window(tz="Asia/Kolkata"):
    today = pd.Timestamp.now(tz=tz).normalize()
    first_this_month = today.replace(day=1)
    last_month_end = first_this_month - pd.Timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    return last_month_start.tz_localize(None), last_month_end.tz_localize(None)

# ================== KPI COMPUTATION ==================
def compute_last_month_kpis(df: pd.DataFrame):
    start, end = last_month_window()
    mask_last = (df["OrderDate"] >= start) & (df["OrderDate"] <= end)
    last_df = df.loc[mask_last].copy()

    # previous month
    prev_end = start - pd.Timedelta(days=1)
    prev_start = prev_end.replace(day=1)
    mask_prev = (df["OrderDate"] >= prev_start) & (df["OrderDate"] <= prev_end)
    prev_df = df.loc[mask_prev].copy()

    rev = float(last_df["Revenue"].sum()) if not last_df.empty else 0.0
    orders = int(last_df.shape[0])
    aov = float(rev / orders) if orders > 0 else 0.0

    prev_rev = float(prev_df["Revenue"].sum()) if not prev_df.empty else 0.0
    growth = ((rev - prev_rev) / prev_rev * 100.0) if prev_rev > 0 else np.nan

    top_customer = last_df.groupby("Customer")["Revenue"].sum().sort_values(ascending=False).head(1)
    top_product = last_df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(1)

    best_customer = (top_customer.index[0], float(top_customer.iloc[0])) if not top_customer.empty else ("—", 0.0)
    best_product  = (top_product.index[0], float(top_product.iloc[0])) if not top_product.empty else ("—", 0.0)

    # Daily series for chart
    if not last_df.empty:
        daily = last_df.set_index("OrderDate").resample("D")["Revenue"].sum().reset_index()
    else:
        daily = pd.DataFrame({"OrderDate": pd.date_range(start, end, freq="D"), "Revenue": 0.0})

    kpis = {
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "revenue": rev,
        "orders": orders,
        "aov": aov,
        "prev_revenue": prev_rev,
        "growth_pct": growth,
        "best_customer": {"name": best_customer[0], "revenue": best_customer[1]},
        "best_product": {"name": best_product[0], "revenue": best_product[1]},
    }
    return kpis, daily, last_df

def fmt_money(x):
    if abs(x) >= 1_000_000: return f"${x/1_000_000:.1f}M"
    if abs(x) >= 1_000: return f"${x/1_000:.1f}K"
    return f"${x:,.0f}"

# ================== GENAI NARRATIVE ==================
def gemini_summary(kpis: dict, samples: pd.DataFrame):
    if not API_KEY:
        return "_Gemini summary unavailable (no GOOGLE_API_KEY set)._"
    model = genai.GenerativeModel(MODEL_NAME)
    brief = {
        "period": kpis["period"],
        "revenue": kpis["revenue"],
        "orders": kpis["orders"],
        "aov": kpis["aov"],
        "prev_revenue": kpis["prev_revenue"],
        "growth_pct": kpis["growth_pct"],
        "best_customer": kpis["best_customer"],
        "best_product": kpis["best_product"],
        "sample_rows": samples.head(10).to_dict(orient="records"),
    }
    prompt = (
        "Write a crisp executive summary for the monthly sales performance below. "
        "Explain KPIs in plain language, call out growth/decline, top customer/product, "
        "and suggest 2–3 actions. Keep it under 120 words.\n\n"
        + json.dumps(brief, ensure_ascii=False)
    )
    try:
        resp = model.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        return f"_Gemini error: {e}_"

# ================== UI ==================
st.title("📊 PlotTwist — One-Click Monthly Review")
st.caption("No uploads. We’ll use the CSV in your repo. Click once → KPIs, charts, and a GenAI summary.")

with st.sidebar:
    st.header("Data source")
    st.write("Using CSV from your repo:")
    st.code(RAW_CSV_URL or "sales_template.csv", language="text")
    use_custom = st.toggle("Use my own file (optional)", value=False)
    uploaded = None
    if use_custom:
        uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

# Load data
try:
    if use_custom and uploaded is not None:
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = read_csv_from_repo(RAW_CSV_URL)
    df = normalize_df(df_raw)
    st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Run analysis
run = st.button("▶️ Run Last Month Analysis", type="primary")

if run:
    kpis, daily, last_df = compute_last_month_kpis(df)

    # — KPIs —
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue (last month)", fmt_money(kpis["revenue"]))
    c2.metric("Orders", f"{kpis['orders']:,}")
    if np.isnan(kpis["growth_pct"]):
        c3.metric("Growth vs prev.", "—")
    else:
        c3.metric("Growth vs prev.", f"{kpis['growth_pct']:.1f}%")
    c4.metric("Avg Order Value", fmt_money(kpis["aov"]))
    st.caption(f"Period: {kpis['period']['start']} → {kpis['period']['end']}")

    # — Charts (compact & mobile-friendly) —
    st.markdown("#### Revenue by Day (last month)")
    fig1, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))
    plot_datetime(ax1, daily["OrderDate"], daily["Revenue"], marker="o", linewidth=1.5)
    set_tick_label_alignment(ax1, "x", rotation=45, ha="right")
    ax1.set_ylabel("Revenue")
    ax1.grid(alpha=0.2)
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=False, clear_figure=True)

    # Top customer/product bars
    st.markdown("#### Top Customer & Top Product (last month)")
    colA, colB = st.columns(2)
    if not last_df.empty:
        top_cust = last_df.groupby("Customer")["Revenue"].sum().sort_values(ascending=False).head(5)
        top_prod = last_df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(5)

        with colA:
            fig2, ax2 = plt.subplots(figsize=(FIG_W, FIG_H))
            ax2.bar(top_cust.index.astype(str), top_cust.values)
            ax2.set_title("Top Customers")
            ax2.set_ylabel("Revenue"); ax2.grid(axis="y", alpha=0.2)
            ax2.tick_params(axis="x", labelrotation=30)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=False, clear_figure=True)

        with colB:
            fig3, ax3 = plt.subplots(figsize=(FIG_W, FIG_H))
            ax3.bar(top_prod.index.astype(str), top_prod.values)
            ax3.set_title("Top Products")
            ax3.set_ylabel("Revenue"); ax3.grid(axis="y", alpha=0.2)
            ax3.tick_params(axis="x", labelrotation=30)
            fig3.tight_layout()
            st.pyplot(fig3, use_container_width=False, clear_figure=True)
    else:
        st.info("No transactions last month.")

    # — Highlights table —
    st.markdown("#### Highlights")
    st.table(pd.DataFrame({
        "Metric": ["Revenue", "Orders", "Avg Order Value", "Prev. Revenue", "Growth vs Prev.", "Best Customer", "Best Product"],
        "Value": [
            fmt_money(kpis["revenue"]),
            f"{kpis['orders']:,}",
            fmt_money(kpis["aov"]),
            fmt_money(kpis["prev_revenue"]),
            "—" if np.isnan(kpis["growth_pct"]) else f"{kpis['growth_pct']:.1f}%",
            f"{kpis['best_customer']['name']} ({fmt_money(kpis['best_customer']['revenue'])})",
            f"{kpis['best_product']['name']} ({fmt_money(kpis['best_product']['revenue'])})",
        ]
    }))

    # — Gemini summary —
    st.markdown("#### 🤖 Executive Summary (GenAI)")
    summary = gemini_summary(kpis, last_df)
    st.markdown(summary)

else:
    st.info("Click **Run Last Month Analysis** to compute KPIs, charts, and the GenAI summary.")
