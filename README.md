# 📊 PlotTwist — Sales Analytics Copilot (Data-Aware & Self-Healing)

PlotTwist is a **Streamlit + Gemini-powered super app** that turns your uploaded sales data into instant analytics, charts, and forecasts.  

Upload a file → Gemini reads a **data snapshot** of it → generates Python code → app runs it safely.  
If the code errors? PlotTwist automatically asks Gemini to **repair** and reruns — self-healing analytics ✨.

---

## 🚀 Features

- **Excel & CSV upload** (mobile-safe parsing)
- **Gemini sees your file** via a smart **data snapshot** (schema, dtypes, top values, numeric stats, CSV head)
- **LLM-generated analysis code** — no manual scripting
- **Self-healing loop**: auto-repair with Gemini on errors
- **Compact charts** with date-safe plotting helpers (`plot_datetime`, `fill_between_datetime`)
- **Executive-ready KPIs** (total revenue, AOV, top customers)
- **Forecast preview** using Holt-Winters (trend-only or seasonal if enough data)
- **Template downloads** (Excel + CSV) to get started fast
- **Pro mode** (optional) for advanced imports like seaborn, sklearn, statsmodels

---

## 📂 Project Structure

├── app.py # Main Streamlit app
├── sales_template.xlsx # Example Excel template
├── sales_template.csv # Example CSV template
├── requirements.txt # Python dependencies
└── README.md # Project documentation
🧑‍💻 Usage

Upload your sales data file (.xlsx or .csv).

Ask Gemini what you want, e.g.:

“Show monthly revenue trend, top 5 customers, and forecast the next 6 months.”

View results: tables + charts generated on the fly.

If something breaks, the app auto-repairs the code and re-runs it.

🛡️ Safety Features

Sandboxed execution — Gemini code runs in a restricted environment

Custom helpers to avoid Matplotlib crashes

No file/network access for generated code

Self-healing loop to fix errors automatically

📊 Example Insights

Monthly revenue trend line

Top N customers by revenue

Category/region breakdowns

Total revenue & average order value

Forecasted sales for the next 6 months

🌟 Roadmap

 Option to share entire CSV if file is small

 Export results as PDF/PowerPoint

 More chart templates (heatmaps, stacked bars, etc.)

📜 License

MIT License — feel free to use and adapt.
