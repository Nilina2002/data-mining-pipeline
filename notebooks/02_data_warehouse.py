# 02_data_warehouse.py
# PURPOSE : Transform the cleaned retail dataset into a Star Schema
#           Data Warehouse consisting of 1 Fact table + 4 Dimension tables.
#           These CSV files connect directly into Power BI.
#
# STAR SCHEMA DESIGN:
#
#   dim_date ──────────┐
#   dim_customer ──────┤
#                      ├──► fact_sales
#   dim_product ───────┤
#   dim_region ────────┘

import pandas as pd
import os

os.makedirs('../data/warehouse', exist_ok=True)

# ════════════════════════════════════════════════════════════
# LOAD CLEANED DATA
# ════════════════════════════════════════════════════════════
print("▶ Loading cleaned dataset...")
df = pd.read_csv('../data/processed/cleaned_retail.csv', 
                 parse_dates=['InvoiceDate'])
print(f"   Loaded: {len(df):,} rows")

# ════════════════════════════════════════════════════════════
# DIMENSION TABLE 1 — dim_date
# PURPOSE: Enables OLAP time-intelligence operations such as
#          Roll-Up (day→month→quarter→year) and Drill-Down.
# ════════════════════════════════════════════════════════════
print("\n▶ Building dim_date...")

# Extract every unique date from the transaction data
dates = df['InvoiceDate'].dt.normalize().drop_duplicates().reset_index(drop=True)

dim_date = pd.DataFrame({
    'DateKey'    : dates.dt.strftime('%Y%m%d').astype(int),  # surrogate key e.g. 20110105
    'FullDate'   : dates,
    'Day'        : dates.dt.day,
    'Month'      : dates.dt.month,
    'MonthName'  : dates.dt.strftime('%B'),
    'Quarter'    : dates.dt.quarter,
    'Year'       : dates.dt.year,
    'DayOfWeek'  : dates.dt.day_name(),
    'WeekNumber' : dates.dt.isocalendar().week.astype(int),
    'IsWeekend'  : dates.dt.dayofweek >= 5
})

dim_date.to_csv('../data/warehouse/dim_date.csv', index=False)
print(f"   ✅ dim_date: {len(dim_date):,} rows")
print(f"      Columns: {list(dim_date.columns)}")

# ════════════════════════════════════════════════════════════
# DIMENSION TABLE 2 — dim_customer
# PURPOSE: Stores one row per customer for customer-level
#          analysis. RFM scores added here for clustering.
# ════════════════════════════════════════════════════════════
print("\n▶ Building dim_customer...")

# Snapshot date: the day AFTER the last transaction
# Used to calculate Recency in RFM analysis
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg(
    LastPurchaseDate = ('InvoiceDate', 'max'),
    Frequency        = ('InvoiceNo',  'nunique'),   # number of orders
    Monetary         = ('TotalPrice', 'sum')        # total spend
).reset_index()

# Recency = days since last purchase (lower = better customer)
rfm['Recency'] = (snapshot_date - rfm['LastPurchaseDate']).dt.days

# Get the primary country per customer (mode of their transactions)
customer_country = (df.groupby('CustomerID')['Country']
                      .agg(lambda x: x.mode()[0])
                      .reset_index()
                      .rename(columns={'Country': 'PrimaryCountry'}))

dim_customer = rfm.merge(customer_country, on='CustomerID')
dim_customer = dim_customer[[
    'CustomerID', 'PrimaryCountry',
    'Recency', 'Frequency', 'Monetary', 'LastPurchaseDate'
]]

dim_customer.to_csv('../data/warehouse/dim_customer.csv', index=False)
print(f"   ✅ dim_customer: {len(dim_customer):,} rows")
print(f"      Columns: {list(dim_customer.columns)}")

# ════════════════════════════════════════════════════════════
# DIMENSION TABLE 3 — dim_product
# PURPOSE: One row per unique product with stock code,
#          description and pricing tier for OLAP drill-down.
# ════════════════════════════════════════════════════════════
print("\n▶ Building dim_product...")

dim_product = (df.groupby('StockCode')
                 .agg(
                     Description  = ('Description', lambda x: x.mode()[0]),
                     AvgUnitPrice = ('UnitPrice',   'mean'),
                     TotalSold    = ('Quantity',    'sum')
                  )
                 .reset_index())

# Create a price tier for OLAP slice/dice operations
dim_product['PriceTier'] = pd.cut(
    dim_product['AvgUnitPrice'],
    bins=[0, 1, 5, 15, 50, float('inf')],
    labels=['<£1', '£1-5', '£5-15', '£15-50', '£50+']
)

dim_product.to_csv('../data/warehouse/dim_product.csv', index=False)
print(f"   ✅ dim_product: {len(dim_product):,} rows")
print(f"      Columns: {list(dim_product.columns)}")

# ════════════════════════════════════════════════════════════
# DIMENSION TABLE 4 — dim_region
# PURPOSE: Geographic dimension for country-level OLAP
#          analysis (Slice by country, Roll-up to continent)
# ════════════════════════════════════════════════════════════
print("\n▶ Building dim_region...")

# Manual continent mapping for OLAP Roll-Up capability
continent_map = {
    'United Kingdom':'Europe','Germany':'Europe','France':'Europe',
    'Spain':'Europe','Netherlands':'Europe','Belgium':'Europe',
    'Switzerland':'Europe','Portugal':'Europe','Italy':'Europe',
    'Finland':'Europe','Norway':'Europe','Denmark':'Europe',
    'Sweden':'Europe','Austria':'Europe','Poland':'Europe',
    'Greece':'Europe','Cyprus':'Europe','Malta':'Europe',
    'Lithuania':'Europe','Iceland':'Europe','Channel Islands':'Europe',
    'EIRE':'Europe','Czech Republic':'Europe','RSA':'Africa',
    'Nigeria':'Africa','USA':'North America','Canada':'North America',
    'Australia':'Oceania','Japan':'Asia','Singapore':'Asia',
    'Hong Kong':'Asia','Bahrain':'Asia','Lebanon':'Asia',
    'Israel':'Asia','Saudi Arabia':'Asia','United Arab Emirates':'Asia',
    'Brazil':'South America','European Community':'Europe',
    'Unspecified':'Unknown'
}

countries = df['Country'].unique()
dim_region = pd.DataFrame({
    'Country'   : countries,
    'Continent' : [continent_map.get(c, 'Unknown') for c in countries],
    'RegionKey' : range(1, len(countries) + 1)
})

dim_region.to_csv('../data/warehouse/dim_region.csv', index=False)
print(f"   ✅ dim_region: {len(dim_region):,} rows")
print(f"      Columns: {list(dim_region.columns)}")

# ════════════════════════════════════════════════════════════
# FACT TABLE — fact_sales
# PURPOSE: Central table storing every transaction line item.
#          Contains foreign keys linking to all 4 dimensions.
#          Stores measurable facts: Quantity, UnitPrice, TotalPrice.
# ════════════════════════════════════════════════════════════
print("\n▶ Building fact_sales...")

# Build DateKey foreign key to link with dim_date
df['DateKey'] = df['InvoiceDate'].dt.strftime('%Y%m%d').astype(int)

# Merge region key from dim_region
df = df.merge(dim_region[['Country','RegionKey']], on='Country', how='left')

fact_sales = df[[
    'InvoiceNo',    # transaction identifier
    'CustomerID',   # FK → dim_customer
    'StockCode',    # FK → dim_product
    'DateKey',      # FK → dim_date
    'RegionKey',    # FK → dim_region
    'Quantity',     # MEASURE
    'UnitPrice',    # MEASURE
    'TotalPrice'    # MEASURE (pre-calculated for performance)
]].copy()

fact_sales.to_csv('../data/warehouse/fact_sales.csv', index=False)
print(f"   ✅ fact_sales: {len(fact_sales):,} rows")
print(f"      Columns: {list(fact_sales.columns)}")
print("\n" + "=" * 50)
print("🏛️  DATA WAREHOUSE SUMMARY")
print("=" * 50)

import os
for f in ['fact_sales','dim_date','dim_customer','dim_product','dim_region']:
    path = f'../data/warehouse/{f}.csv'
    size = os.path.getsize(path) / 1024
    rows = sum(1 for _ in open(path)) - 1
    print(f"  {f:<20} {rows:>7,} rows   {size:>8.1f} KB")

print("=" * 50)
print("\n✅ Star schema complete. Ready to connect to Power BI.")
print("   Folder: data/warehouse/")