import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("▶ Loading raw data...")
df = pd.read_csv('../data/raw/online_retail.csv', encoding='ISO-8859-1')
print(f"   Raw shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

print("\n▶ Fixing data types...")

# Convert InvoiceDate from string → proper datetime
# This enables time-based OLAP operations later
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# CustomerID should be a string (it's an ID, not a number to calculate with)
df['CustomerID'] = df['CustomerID'].astype(str)

print("   InvoiceDate → datetime ✅")
print("   CustomerID  → string   ✅")

print("\n▶ Cleaning data...")
original_len = len(df)

df = df[df['CustomerID'] != 'nan']
print(f"   Removed {original_len - len(df):,} rows with missing CustomerID")

cancelled_mask = df['InvoiceNo'].astype(str).str.startswith('C')
df = df[~cancelled_mask]
print(f"   Removed {cancelled_mask.sum():,} cancelled transactions")

df = df[df['Quantity'] > 0]

df = df[df['UnitPrice'] > 0]

print(f"   Final clean shape: {df.shape[0]:,} rows")

print("\n▶ Engineering new features...")

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

df['Year']        = df['InvoiceDate'].dt.year
df['Month']       = df['InvoiceDate'].dt.month
df['MonthName']   = df['InvoiceDate'].dt.strftime('%B')
df['Quarter']     = df['InvoiceDate'].dt.quarter
df['DayOfWeek']   = df['InvoiceDate'].dt.day_name()
df['Hour']        = df['InvoiceDate'].dt.hour

print("   TotalPrice, Year, Month, Quarter, DayOfWeek, Hour ✅")

print("\n▶ Running EDA and saving figures...")

sns.set_style("whitegrid")
sns.set_palette("husl")

monthly = df.groupby(['Year', 'Month'])['TotalPrice'].sum().reset_index()
monthly['Period'] = monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2)

plt.figure(figsize=(14, 5))
plt.plot(monthly['Period'], monthly['TotalPrice'], marker='o', linewidth=2, color='steelblue')
plt.xticks(rotation=45, ha='right')
plt.title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Total Revenue (£)')
plt.tight_layout()
plt.savefig('../outputs/figures/01_monthly_revenue.png', dpi=150)
plt.close()
print("   ✅ Saved: 01_monthly_revenue.png")

top_countries = (df.groupby('Country')['TotalPrice']
                   .sum()
                   .sort_values(ascending=False)
                   .head(10))

plt.figure(figsize=(12, 5))
sns.barplot(x=top_countries.index, y=top_countries.values, palette='Blues_d')
plt.title('Top 10 Countries by Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Country')
plt.ylabel('Total Revenue (£)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('../outputs/figures/02_top_countries.png', dpi=150)
plt.close()
print("   ✅ Saved: 02_top_countries.png")

top_products = (df.groupby('Description')['TotalPrice']
                  .sum()
                  .sort_values(ascending=False)
                  .head(10))

plt.figure(figsize=(12, 5))
sns.barplot(x=top_products.values, y=top_products.index, palette='Greens_d')
plt.title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Total Revenue (£)')
plt.ylabel('Product')
plt.tight_layout()
plt.savefig('../outputs/figures/03_top_products.png', dpi=150)
plt.close()
print("   ✅ Saved: 03_top_products.png")

day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
orders_by_day = df.groupby('DayOfWeek')['InvoiceNo'].nunique().reindex(day_order)

plt.figure(figsize=(10, 4))
sns.barplot(x=orders_by_day.index, y=orders_by_day.values, palette='Purples_d')
plt.title('Number of Orders by Day of Week', fontsize=14, fontweight='bold')
plt.xlabel('Day')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.savefig('../outputs/figures/04_orders_by_day.png', dpi=150)
plt.close()
print("   ✅ Saved: 04_orders_by_day.png")

plt.figure(figsize=(10, 4))
df['TotalPrice'].apply(np.log1p).hist(bins=50, color='coral', edgecolor='white')
plt.title('Distribution of Transaction Value (log scale)', fontsize=14, fontweight='bold')
plt.xlabel('log(TotalPrice + 1)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('../outputs/figures/05_price_distribution.png', dpi=150)
plt.close()
print("   ✅ Saved: 05_price_distribution.png")

print("\n" + "=" * 50)
print("📊 DATASET SUMMARY (use these in your report)")
print("=" * 50)
print(f"  Total transactions  : {len(df):,}")
print(f"  Unique invoices     : {df['InvoiceNo'].nunique():,}")
print(f"  Unique customers    : {df['CustomerID'].nunique():,}")
print(f"  Unique products     : {df['StockCode'].nunique():,}")
print(f"  Unique countries    : {df['Country'].nunique():,}")
print(f"  Date range          : {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
print(f"  Total revenue       : £{df['TotalPrice'].sum():,.2f}")
print(f"  Avg order value     : £{df.groupby('InvoiceNo')['TotalPrice'].sum().mean():,.2f}")
print("=" * 50)

df.to_csv('../data/processed/cleaned_retail.csv', index=False)
print(f"\n✅ Cleaned dataset saved → data/processed/cleaned_retail.csv")
print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")