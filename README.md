# Smart Market Basket Analysis & Product Recommendation System

A local, beginner-friendly Data Mining project built with Python, Streamlit, Pandas, mlxtend, and Plotly. It cleans retail transaction rows, creates a transaction/product basket, mines frequent itemsets with Apriori, generates association rules, and uses matching rules to recommend products.

## 1. Features

- Overview KPIs calculated from the active dataset
- CSV upload and required-column validation
- Preprocessing report for missing identifiers/products, invalid quantities, cancelled invoices, and duplicate invoice-product pairs
- Boolean invoice-by-product matrix
- Apriori frequent itemsets with configurable support and maximum length
- Association rules filtered by confidence and lift
- Rule-based product recommendations for one or multiple selected products
- Interactive Plotly charts and filterable tables
- CSV exports for cleaned data, basket matrix, itemsets, rules, and recommendations
- Built-in deterministic demo data

## 2. Requirements

- Python 3.10, 3.11, or 3.12
- Windows 10/11, macOS, or Linux
- Internet access only to install packages (the app itself runs locally)

## 3. Windows PowerShell setup

Open PowerShell in the `market_basket_mining` project folder.

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If you have Python 3.11 or 3.10, replace `-3.12` with `-3.11` or `-3.10`.

If PowerShell blocks activation, use this for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Or skip activation and call the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser. No API key, database, or cloud account is needed.

## 4. Quick demonstration

1. Launch the app. It loads `data/sample_transactions.csv` automatically.
2. The Overview shows calculated transaction/product/itemset/rule counts.
3. Use the sidebar to adjust minimum support, maximum itemset length, confidence, and lift.
4. Click **Run analysis** after changing settings.
5. Open **Product Recommendations**, select products such as `BREAD` and `MILK`, and review the recommendations.
6. Open **Data Explorer** to view cleaned data, cleaning statistics, and a small basket-matrix preview.
7. Use the download buttons to export results as CSV.

The included dataset is intentionally small and patterned for a quick classroom demonstration. Its results are illustrative and are not representative of a real retailer.

## 5. Input CSV schema

Required columns:

| Column | Meaning | Example |
|---|---|---|
| `InvoiceNo` | Transaction/invoice identifier | `10001` |
| `Description` | Product name | `MILK` |
| `Quantity` | Units purchased; must be positive | `2` |

Optional columns: `InvoiceDate`, `UnitPrice`, and `CustomerID`. Extra columns are retained. Header whitespace is trimmed, and product descriptions are trimmed, whitespace-normalized, and uppercased.

### Cleaning policy

- Rows with blank invoice numbers or product descriptions are removed.
- Cancelled invoices whose invoice number begins with `C` (case-insensitive) are removed.
- Non-numeric, zero, and negative quantities are removed.
- Duplicate `(InvoiceNo, Description)` rows are reduced to the first valid row. This project does **not** sum duplicate quantities; this is a deliberate simple policy and is shown in the UI.
- Optional dates/prices are parsed where possible; invalid values become missing.
- Missing `CustomerID` does not remove a transaction because it is not required for basket analysis.

## 6. Optional UCI Online Retail dataset

Dataset page: https://archive.ics.uci.edu/dataset/352/online+retail

The dataset is commonly distributed as an Excel workbook. Download the Excel file from UCI and convert it to CSV with Pandas:

```python
import pandas as pd

df = pd.read_excel("Online Retail.xlsx")
df.to_csv("online_retail.csv", index=False)
```

Then upload `online_retail.csv` through **Data Upload**. The original UCI column names commonly include `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, and `CustomerID`. This app requires `InvoiceNo`, `Description`, and `Quantity`; those names are already present in the original dataset.

### Recommended manageable subset

For a first run, take the first 5,000–20,000 rows after conversion, or filter to a single month and keep a manageable product set. Example:

```python
import pandas as pd

df = pd.read_csv("online_retail.csv", nrows=10000)
df.to_csv("online_retail_subset.csv", index=False)
```

The application has safeguards of 20,000 unique invoices and 1,000 unique products. These are practical limits, not universal performance guarantees. Apriori can grow rapidly when support is low and many distinct products exist. Start with support `0.05`–`0.10` and maximum itemset length `2` or `3`; increase complexity gradually.

## 7. Algorithms and metrics

- **Transaction matrix:** each row is an invoice, each column is a product, and `True` means the product occurs in that invoice.
- **Apriori:** finds itemsets whose support is at least the chosen minimum support.
- **Support:** proportion of all transactions containing an itemset.
- **Confidence:** proportion of transactions containing antecedent A that also contain consequent B.
- **Lift:** confidence(A → B) divided by support(B). Lift above 1 indicates a positive association relative to the consequent's baseline frequency; it does not prove causation.
- **Recommendation:** select rules whose entire antecedent is contained in the shopper's selected basket. Recommend consequents not already selected. When multiple rules suggest the same product, keep the strongest confidence, then lift, then support.

## 8. Project report (short version)

### Abstract
This project presents a local retail analytics application that applies association-rule mining to transaction data. It preprocesses sales records, builds a boolean basket matrix, identifies frequent product combinations using Apriori, and generates recommendations from association rules. A Streamlit dashboard provides interactive metrics, charts, tables, and CSV downloads.

### Problem statement
Retailers collect many transaction records but may not know which products are frequently purchased together. Manual inspection is slow and does not scale well. A data mining workflow can identify recurring combinations and provide evidence-based cross-selling suggestions.

### Objectives
1. Clean and validate retail transaction data.
2. Convert invoice lines into a transaction/product matrix.
3. Find frequent itemsets with Apriori.
4. Generate and filter association rules using support, confidence, and lift.
5. Recommend products based on a selected basket.
6. Present results in a usable dashboard and allow CSV export.

### Methodology
CSV input → schema validation → cleaning → boolean basket matrix → Apriori itemsets → association rules → rule-based recommendations → charts, tables, and downloads.

### Algorithms
Apriori prunes candidate combinations whose subsets are not frequent. Association rules are evaluated using support, confidence, and lift. Recommendation logic matches the full antecedent against the selected basket and excludes already selected products.

### Architecture
- `app.py`: Streamlit pages, controls, charts, tables, downloads, and session state.
- `src/preprocessing.py`: schema validation, cleaning statistics, basket matrix.
- `src/mining.py`: Apriori, rule generation, CSV-friendly conversions.
- `src/recommendations.py`: ranking recommendations from rules.
- `data/sample_transactions.csv`: built-in demonstration data.

### Implementation
The app is a local Python project and needs no external API or database. Streamlit handles interaction, Pandas/NumPy handle tabular processing, mlxtend provides Apriori and association rules, and Plotly renders interactive charts.

### Results
The app calculates metrics from the currently active dataset and selected thresholds. The synthetic sample contains repeated shopping patterns (for example, bread/milk/eggs, tea/biscuits/milk, and pasta/sauce/cheese), so the dashboard can demonstrate itemsets and rules immediately. Exact results vary with thresholds and are computed at runtime.

### Limitations
- Association is not causation.
- Apriori can be expensive on large datasets or at low support.
- The simple duplicate policy retains the first invoice-product row rather than summing quantities.
- Recommendations are only as useful as the historical data and mined rules.
- The sample dataset is small and synthetic.

### Future scope
Add date-range filters, category hierarchies, customer segmentation, FP-Growth comparison, weighted recommendations, user authentication, and automated report export after the core app is stable.

### Conclusion
The project demonstrates a complete introductory data mining pipeline from transaction cleaning to product recommendations. It is achievable as a 2–3 day individual college project and provides clear opportunities for experimentation with thresholds and data quality.

## 9. Viva questions and answers

1. **What is data mining?**  
   Data mining is the process of discovering useful patterns, relationships, or trends from data.

2. **What is market basket analysis?**  
   It studies which products tend to occur together in the same transaction.

3. **What is a transaction?**  
   A group of products purchased under one invoice or basket identifier.

4. **What is an itemset?**  
   A set of one or more products, such as `{BREAD, MILK}`.

5. **What is Apriori?**  
   An algorithm that finds frequent itemsets using the property that every subset of a frequent itemset must also be frequent.

6. **What is support?**  
   The fraction of all transactions that contain an itemset. If 20 of 100 invoices contain milk, its support is 0.20.

7. **What is confidence?**  
   For rule A → B, confidence is support(A ∪ B) divided by support(A). It estimates how often B appears when A appears.

8. **What is lift?**  
   Lift is confidence(A → B) divided by support(B). A lift above 1 indicates B appears with A more often than its overall frequency would suggest.

9. **Does lift prove causation?**  
   No. It measures association, not cause and effect.

10. **What is an association rule?**  
    A directional relationship such as `{TEA} → {BISCUITS}` derived from transaction patterns.

11. **Why preprocess the data?**  
    Missing identifiers, cancelled invoices, invalid quantities, and inconsistent product names can distort the discovered patterns.

12. **Why remove cancelled invoices beginning with C?**  
    In many retail datasets, these identifiers represent cancellations or credit notes rather than completed purchases.

13. **Why convert products to uppercase?**  
    To avoid treating values such as `Milk`, `MILK`, and ` milk ` as different products.

14. **Why create a boolean basket matrix?**  
    Apriori needs to know whether each product is present in each transaction; quantities are not required for basic presence/absence mining.

15. **Why use minimum support?**  
    It filters out combinations that occur too rarely and reduces the search space.

16. **What happens if support is too high?**  
    Useful but less common itemsets may be missed, and no rules may be produced.

17. **What happens if support is too low?**  
    Many itemsets may be generated, increasing runtime and memory usage and potentially producing noisy rules.

18. **Why filter by confidence and lift?**  
    Confidence controls rule reliability conditional on the antecedent; lift helps compare the rule against the consequent's baseline frequency.

19. **How are recommendations generated?**  
    The selected basket must contain every item in a rule's antecedent. Products in its consequent that are not already selected become candidates.

20. **How are duplicate recommendations handled?**  
    Keep the strongest rule for each product, prioritizing confidence, then lift, then support.

21. **Why use Streamlit?**  
    It makes it possible to build an interactive Python dashboard without writing a separate frontend application.

22. **Why use Plotly?**  
    Plotly provides interactive charts with hover details and zooming.

23. **What is a limitation of this project?**  
    Results depend on data quality and thresholds; patterns do not prove causation, and Apriori can be slow on large datasets.

24. **How could the project be improved?**  
    Compare Apriori with FP-Growth, add time-based trends, and evaluate recommendation usefulness on held-out transactions.

## 10. Troubleshooting

### `py` or Python command is not found
Install Python 3.10–3.12 from python.org and enable the option to add Python to PATH. Reopen PowerShell and run `py --version`.

### PowerShell says script execution is disabled
Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate the virtual environment again. This changes policy only for the current terminal process.

### `ModuleNotFoundError`
Activate `.venv`, then run `python -m pip install -r requirements.txt`. Ensure VS Code is using `.venv` as its Python interpreter.

### `streamlit` is not recognized
Run `python -m streamlit run app.py` instead of `streamlit run app.py`.

### `No module named mlxtend`
Run `python -m pip install mlxtend` inside the activated virtual environment.

### CSV says required columns are missing
Check spelling: `InvoiceNo`, `Description`, `Quantity`. Remove accidental spaces in the header or use a CSV editor to rename columns.

### No itemsets or rules appear
Lower minimum support, use more transactions, or check whether product descriptions have become too fragmented. Try the sample dataset to verify the basic workflow.

### App feels slow or memory usage is high
Raise support, reduce maximum itemset length to 2 or 3, or upload a smaller dataset. Apriori may generate many combinations on wide baskets.

### `UnicodeDecodeError` or CSV parsing error
Export the file as UTF-8 CSV from Excel. If the file is an Excel workbook, convert it to CSV using the Pandas snippet in the UCI section.

### Port 8501 is busy
Run `python -m streamlit run app.py --server.port 8502`.

## 11. Project structure

```text
market_basket_mining/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── mining.py
│   └── recommendations.py
├── data/
│   └── sample_transactions.csv
└── assets/
```
