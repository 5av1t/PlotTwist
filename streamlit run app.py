
Open the URL shown, upload a CSV/XLSX, and type prompts like:

- "Summarize total sales by month; show a bar chart sorted descending."
- "Weekly line chart of sales volume for FY24 only."
- "Top 10 customers by revenue and a table with their MoM growth."

## Safety Notes

- The LLM is instructed to use only `pandas`, `numpy`, and `matplotlib.pyplot`.
- The app **blocks imports, file/network access, and dangerous builtins** via AST checks.
- Code executes in a restricted namespace on your machine/session only.

> If you need stricter controls later (timeouts, extra guards), we can harden the sandbox and/or move to a containerized runtime.
