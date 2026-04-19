import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder



print(" Loading cleaned data...")
df = pd.read_csv('../data/processed/cleaned_retail.csv',
                 parse_dates=['InvoiceDate'])
print(f"   Loaded: {len(df):,} rows")


df_uk = df[df['Country'] == 'United Kingdom'].copy()
print(f"   UK transactions: {len(df_uk):,} rows")
print(f"   UK invoices    : {df_uk['InvoiceNo'].nunique():,}")


print("\n Building basket matrix...")

basket = (df_uk.groupby(['InvoiceNo', 'Description'])['Quantity']
               .sum()
               .unstack(fill_value=0))

print(f"   Basket matrix shape: {basket.shape[0]:,} invoices × {basket.shape[1]:,} products")

def encode_binary(x):
    return x.apply(lambda val: 1 if val > 0 else 0)

basket_binary = basket.apply(encode_binary)
print(f"   Binary encoding complete ")


print("\n▶ Running Apriori algorithm...")
print("   (This may take 1-2 minutes for a dataset this size)")

frequent_itemsets = apriori(
    basket_binary,
    min_support=0.02,        # minimum 2% of transactions
    use_colnames=True,       # use product names not column indices
    max_len=3                # find 1-itemsets, 2-itemsets, 3-itemsets
)

# Add a readable itemset length column
frequent_itemsets['itemset_length'] = frequent_itemsets['itemsets'].apply(len)

print(f"    Frequent itemsets found: {len(frequent_itemsets):,}")
print(f"\n   Breakdown by itemset size:")
print(frequent_itemsets['itemset_length'].value_counts().sort_index()
      .rename({1:'1-itemsets (single products)',
               2:'2-itemsets (pairs)',
               3:'3-itemsets (triplets)'}).to_string())


print("\n Generating association rules...")

rules = association_rules(
    frequent_itemsets,
    metric='lift',
    min_threshold=1.0        
)

# Add readable columns for reporting
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# Sort by lift (strongest associations first)
rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

print(f"    Total rules generated : {len(rules):,}")
print(f"   Rules with lift > 2     : {len(rules[rules['lift'] > 2]):,}")
print(f"   Rules with lift > 3     : {len(rules[rules['lift'] > 3]):,}")
print(f"   Rules with confidence>70%: {len(rules[rules['confidence'] > 0.7]):,}")


print("\n TOP 10 RULES BY LIFT (use these in your report):")


top10 = rules[['antecedents_str','consequents_str',
               'support','confidence','lift']].head(10)

for i, row in top10.iterrows():
    print(f"\n  Rule #{i+1}")
    print(f"  IF customer buys : {row['antecedents_str']}")
    print(f"  THEN buys        : {row['consequents_str']}")
    print(f"  Support          : {row['support']:.4f}  ({row['support']*100:.2f}% of transactions)")
    print(f"  Confidence       : {row['confidence']:.4f} ({row['confidence']*100:.2f}% of the time)")
    print(f"  Lift             : {row['lift']:.4f}")

print("\n Generating visualisations...")
sns.set_style("whitegrid")

top20 = rules.head(20).copy()
top20['rule_label'] = top20['antecedents_str'] + '  →  ' + top20['consequents_str']
# Truncate long labels for readability
top20['rule_label'] = top20['rule_label'].str[:60]

plt.figure(figsize=(14, 8))
bars = plt.barh(range(len(top20)), top20['lift'], color='steelblue', edgecolor='white')
plt.yticks(range(len(top20)), top20['rule_label'], fontsize=8)
plt.xlabel('Lift', fontsize=12)
plt.title('Top 20 Association Rules by Lift', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../outputs/figures/06_top_rules_lift.png', dpi=150)
plt.close()
print("    Saved: 06_top_rules_lift.png")

plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    rules['support'],
    rules['confidence'],
    c=rules['lift'],
    cmap='RdYlGn',
    alpha=0.6,
    s=30,
    edgecolors='none'
)
plt.colorbar(scatter, label='Lift')
plt.xlabel('Support',    fontsize=12)
plt.ylabel('Confidence', fontsize=12)
plt.title('Association Rules — Support vs Confidence (colour = Lift)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/07_support_confidence_scatter.png', dpi=150)
plt.close()
print("    Saved: 07_support_confidence_scatter.png")

single_items = (frequent_itemsets[frequent_itemsets['itemset_length'] == 1]
                .copy()
                .sort_values('support', ascending=False)
                .head(15))
single_items['item'] = single_items['itemsets'].apply(lambda x: list(x)[0])
single_items['item'] = single_items['item'].str[:40]  # truncate

plt.figure(figsize=(12, 6))
sns.barplot(x='support', y='item', data=single_items, palette='Blues_d')
plt.xlabel('Support (% of transactions)', fontsize=11)
plt.ylabel('Product', fontsize=11)
plt.title('Top 15 Most Frequently Purchased Items', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/08_top_frequent_items.png', dpi=150)
plt.close()
print("    Saved: 08_top_frequent_items.png")

rules_export = rules[['antecedents_str', 'consequents_str',
                       'support', 'confidence', 'lift',
                       'leverage', 'conviction']].copy()
rules_export.columns = ['Antecedent', 'Consequent',
                        'Support', 'Confidence', 'Lift',
                        'Leverage', 'Conviction']
rules_export.to_csv('../outputs/association_rules.csv', index=False)
print(f"\n Rules saved → outputs/association_rules.csv")

print("ACADEMIC SUMMARY")

print(f"  Algorithm          : Apriori (Agrawal & Srikant, 1994)")
print(f"  Dataset scope      : UK transactions only")
print(f"  Transactions used  : {df_uk['InvoiceNo'].nunique():,}")
print(f"  Products in matrix : {basket_binary.shape[1]:,}")
print(f"  Min Support        : 2% (0.02)")
print(f"  Min Lift threshold : 1.0")
print(f"  Frequent itemsets  : {len(frequent_itemsets):,}")
print(f"  Rules generated    : {len(rules):,}")
print(f"  Strongest lift     : {rules['lift'].max():.4f}")
print(f"  Avg confidence     : {rules['confidence'].mean():.4f}")
