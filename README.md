# Data Mining and Warehousing Project

This project builds a complete analytics pipeline using the Online Retail dataset:

1. Validate raw data
2. Clean and enrich transactions
3. Build a star schema data warehouse
4. Generate visual outputs for reporting

The final warehouse tables are ready to connect to BI tools such as Power BI.

## Project Objective

Transform raw e-commerce transaction data into a structured, analysis-ready warehouse with:

- A central fact table for sales
- Four dimensions for date, customer, product, and region
- Supporting visuals for exploratory analysis

## Tech Stack

- Python 3.11
- pandas
- numpy
- matplotlib
- seaborn

## Project Structure

```text
dmdw_project/
|-- data/
|   |-- raw/
|   |   `-- online_retail.csv
|   |-- processed/
|   |   `-- cleaned_retail.csv
|   `-- warehouse/
|       |-- dim_customer.csv
|       |-- dim_date.csv
|       |-- dim_product.csv
|       |-- dim_region.csv
|       `-- fact_sales.csv
|-- notebooks/
|   |-- 01_preprocessing.py
|   |-- 02_data_warehouse.py
|   `-- 02_error_fix_fullDate.py
|-- outputs/
|   `-- figures/
|       |-- 01_monthly_revenue.png
|       |-- 02_top_countries.png
|       |-- 03_top_products.png
|       |-- 04_orders_by_day.png
|       `-- 05_price_distribution.png
|-- verify.py
`-- requirements.txt
```

## Data Source

- Input file: `data/raw/online_retail.csv`
- Encoding used: `ISO-8859-1`

The dataset includes invoice-level transactions, product information, customer IDs, and country-level geography.

## Setup

### 1) Create and activate a virtual environment (Windows PowerShell)

```powershell
py -m venv dmdw_env
& .\dmdw_env\Scripts\Activate.ps1
```

### 2) Install dependencies

`requirements.txt` is currently empty, so install the required libraries directly:

```powershell
pip install pandas numpy matplotlib seaborn
```

Optionally, lock them:

```powershell
pip freeze > requirements.txt
```

## How to Run

Run scripts in this order from the project root:

### 1) Verify the raw dataset

```powershell
py verify.py
```

What it does:

- Loads the raw CSV
- Prints shape, dtypes, missing values, unique counts, and date range

### 2) Clean and preprocess data

```powershell
py notebooks/01_preprocessing.py
```

What it does:

- Converts `InvoiceDate` to datetime
- Converts `CustomerID` to string
- Removes rows with missing customer IDs
- Removes cancelled invoices (`InvoiceNo` starts with `C`)
- Removes non-positive quantity or price rows
- Creates engineered columns including `TotalPrice`, `Year`, `Month`, `Quarter`, and `DayOfWeek`
- Saves cleaned output to `data/processed/cleaned_retail.csv`
- Saves EDA figures to `outputs/figures/`

### 3) Build the star schema warehouse

```powershell
py notebooks/02_data_warehouse.py
```

What it does:

- Creates `dim_date`
- Creates `dim_customer` (with RFM features)
- Creates `dim_product`
- Creates `dim_region` (country-to-continent mapping)
- Creates `fact_sales` with foreign keys
- Saves all tables to `data/warehouse/`

### 4) Fix the Date dimension for continuous dates (if needed)

```powershell
py notebooks/02_error_fix_fullDate.py
```

Use this when your BI tool requires a fully continuous date table (no missing days).

## Star Schema Overview

The warehouse follows a classic star schema:

- `fact_sales`
  - Measures: `Quantity`, `UnitPrice`, `TotalPrice`
  - Keys: `CustomerID`, `StockCode`, `DateKey`, `RegionKey`

- `dim_date`
  - Time hierarchy attributes (`Day`, `Month`, `Quarter`, `Year`, `WeekNumber`, `IsWeekend`)

- `dim_customer`
  - Customer attributes plus RFM metrics (`Recency`, `Frequency`, `Monetary`)

- `dim_product`
  - Product description, average price, total sold, and price tier

- `dim_region`
  - Country and continent mapping for geographic analysis

## Outputs

### Processed Data

- `data/processed/cleaned_retail.csv`

### Warehouse Tables

- `data/warehouse/fact_sales.csv`
- `data/warehouse/dim_date.csv`
- `data/warehouse/dim_customer.csv`
- `data/warehouse/dim_product.csv`
- `data/warehouse/dim_region.csv`

### Figures

- Monthly revenue trend
- Top countries by revenue
- Top products by revenue
- Orders by day of week
- Transaction value distribution (log scale)

## Common Issues

- `UnicodeDecodeError` when reading raw CSV:
  - Ensure `encoding='ISO-8859-1'` is used.

- Power BI cannot mark `dim_date` as a Date Table:
  - Run `notebooks/02_error_fix_fullDate.py` to regenerate `dim_date` with every day included.

- `FileNotFoundError` for input data:
  - Run scripts from the project root to preserve relative paths.

## Suggested Next Improvements

1. Add pinned package versions to `requirements.txt`.
2. Add a `scripts/run_pipeline.ps1` file to execute all steps in sequence.
3. Add validation checks (row counts, null checks, key integrity) after warehouse generation.
4. Add a data dictionary section for every warehouse column.

## Author Notes

This repository is organized for coursework in Data Mining and Warehousing and is structured to support:

- ETL reproducibility
- Star schema modeling best practices
- Smooth handoff to Power BI for OLAP-style analysis
