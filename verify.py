# verify.py
# PURPOSE: Confirm the dataset loaded correctly and check its structure
# Run this before starting any notebooks

import pandas as pd

# ── Load the dataset ─────────────────────────────────────────────────────────
df = pd.read_csv('data/raw/online_retail.csv', encoding='ISO-8859-1')
# NOTE: encoding='ISO-8859-1' is required — this file contains
# special characters (£ signs, accented letters) that break
# the default UTF-8 reader

# ── Basic shape ──────────────────────────────────────────────────────────────
print("=" * 50)
print(f"✅ Rows    : {df.shape[0]:,}")
print(f"✅ Columns : {df.shape[1]}")

# ── Column names and types ───────────────────────────────────────────────────
print("\n--- Columns & Data Types ---")
print(df.dtypes)

# ── First 5 rows ─────────────────────────────────────────────────────────────
print("\n--- First 5 Rows ---")
print(df.head())

# ── Missing values ───────────────────────────────────────────────────────────
print("\n--- Missing Values ---")
print(df.isnull().sum())

# ── Unique counts (important for warehouse design) ───────────────────────────
print("\n--- Unique Value Counts ---")
print(f"Unique Invoices   : {df['InvoiceNo'].nunique():,}")
print(f"Unique Products   : {df['StockCode'].nunique():,}")
print(f"Unique Customers  : {df['CustomerID'].nunique():,}")
print(f"Unique Countries  : {df['Country'].nunique():,}")
print(f"Date Range        : {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")