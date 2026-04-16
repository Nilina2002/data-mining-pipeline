import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# adding k-medoids 
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import warnings


warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ════════════════════════════════════════════════════════════
# SECTION 1 — BUILD RFM TABLE FROM CLEANED DATA
# ════════════════════════════════════════════════════════════
print("▶ Building RFM table...")

df = pd.read_csv('../data/processed/cleaned_retail.csv',
                 parse_dates=['InvoiceDate'])

# Snapshot date: one day after the last transaction
# All recency values are calculated relative to this date
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg(
    Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency = ('InvoiceNo',   'nunique'),
    Monetary  = ('TotalPrice',  'sum')
).reset_index()

print(f"   RFM table: {len(rfm):,} customers")
print(f"\n   RFM Statistics:")
print(rfm[['Recency','Frequency','Monetary']].describe().round(2).to_string())

# ════════════════════════════════════════════════════════════
# SECTION 2 — HANDLE OUTLIERS & SCALE FEATURES
# ════════════════════════════════════════════════════════════
print("\n▶ Handling outliers and scaling...")

# Cap extreme outliers at the 99th percentile
# Reason: a handful of very large B2B customers would otherwise
# dominate the clustering and produce meaningless segments
for col in ['Recency', 'Frequency', 'Monetary']:
    upper = rfm[col].quantile(0.99)
    rfm[f'{col}_capped'] = rfm[col].clip(upper=upper)
    print(f"   {col}: capped at {upper:.1f} (99th percentile)")

# Apply log transformation to Frequency and Monetary
# Reason: both are right-skewed — log transform normalises the distribution
# making K-Means distance calculations more meaningful
rfm['Frequency_log'] = np.log1p(rfm['Frequency_capped'])
rfm['Monetary_log']  = np.log1p(rfm['Monetary_capped'])

# Standardise all features to mean=0, std=1
# Reason: K-Means uses Euclidean distance — features on different scales
# (days vs £ spent) would bias the algorithm toward high-magnitude features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(
    rfm[['Recency_capped', 'Frequency_log', 'Monetary_log']]
)

print(f"   StandardScaler applied — mean=0, std=1 ✅")

# ════════════════════════════════════════════════════════════
# SECTION 3 — FIND OPTIMAL K (ELBOW + SILHOUETTE)
# ════════════════════════════════════════════════════════════
# The Elbow Method plots inertia (within-cluster sum of squares)
# against K. The "elbow" — where adding more clusters gives
# diminishing returns — is the optimal K.
#
# Silhouette Score measures how well each point fits its own
# cluster vs neighbouring clusters. Score closer to 1 = better.
# ════════════════════════════════════════════════════════════
print("\n▶ Finding optimal K (Elbow + Silhouette)...")

inertias    = []
silhouettes = []
K_range     = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10,
                random_state=42)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(rfm_scaled, km.labels_))
    print(f"   K={k}  Inertia={km.inertia_:,.0f}  Silhouette={silhouettes[-1]:.4f}")

# ── Elbow + Silhouette Plot ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow curve
ax1.plot(K_range, inertias, marker='o', linewidth=2, color='steelblue')
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
ax1.set_title('Elbow Method — Optimal K', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Silhouette scores
ax2.plot(K_range, silhouettes, marker='s', linewidth=2, color='coral')
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score — Optimal K', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/figures/09_elbow_silhouette.png', dpi=150)
plt.close()
print("\n   ✅ Saved: 09_elbow_silhouette.png")

# ════════════════════════════════════════════════════════════
# SECTION 4 — FIT FINAL K-MEANS MODEL
# ════════════════════════════════════════════════════════════
# Based on elbow + silhouette, K=4 is typically optimal for
# RFM customer segmentation — producing 4 actionable segments:
# Champions, Loyal, At-Risk, Lost/Inactive
# Adjust K below if your plots suggest otherwise
# ════════════════════════════════════════════════════════════

# Find the best K automatically from silhouette scores
best_k = K_range[silhouettes.index(max(silhouettes))]
print(f"\n▶ Best K by silhouette score: {best_k}")
print(f"   (You can override this if the elbow plot suggests differently)")

# Fit final model
print(f"\n▶ Fitting final K-Means model with K={best_k}...")
kmeans_final = KMeans(
    n_clusters=best_k,
    init='k-means++',   # smarter initialisation than random
    n_init=10,          # run 10 times, keep best result
    random_state=42     # reproducibility
)
rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)

print(f"   ✅ Customers per cluster:")
print(rfm['Cluster'].value_counts().sort_index().to_string())

# ════════════════════════════════════════════════════════════
# SECTION 4B — FIT K-MEDOIDS MODEL (FOR COMPARISON)
# ════════════════════════════════════════════════════════════
print("\n▶ Fitting K-Medoids model for comparison...")

kmedoids = KMedoids(
    n_clusters=best_k,
    metric='manhattan',   # better for RFM
    init='k-medoids++',
    random_state=42
)

rfm['Cluster_KMedoids'] = kmedoids.fit_predict(rfm_scaled)

print("   ✅ K-Medoids clustering complete")

print("\n▶ Comparing K-Means vs K-Medoids...")

# K-Means score
kmeans_score = silhouette_score(rfm_scaled, rfm['Cluster'])

# K-Medoids score
kmedoids_score = silhouette_score(rfm_scaled, rfm['Cluster_KMedoids'])

print(f"\n📊 Silhouette Scores:")
print(f"   K-Means   : {kmeans_score:.4f}")
print(f"   K-Medoids : {kmedoids_score:.4f}")

print("\n📊 Cluster Distribution Comparison:")

print("\nK-Means:")
print(rfm['Cluster'].value_counts().sort_index())

print("\nK-Medoids:")
print(rfm['Cluster_KMedoids'].value_counts().sort_index())

print("\n📊 K-Means Centers:")
print(kmeans_final.cluster_centers_)

print("\n📊 K-Medoids (actual data points):")
print(kmedoids.medoid_indices_)

# ════════════════════════════════════════════════════════════
# SECTION 5 — LABEL CLUSTERS (BOTH K-MEANS & K-MEDOIDS)
# ════════════════════════════════════════════════════════════
print("\n▶ Analysing cluster profiles...")

# ─────────────────────────────────────────────
# K-MEANS CLUSTER LABELING
# ─────────────────────────────────────────────
cluster_summary_kmeans = rfm.groupby('Cluster').agg(
    Count     = ('CustomerID', 'count'),
    Recency   = ('Recency',    'mean'),
    Frequency = ('Frequency',  'mean'),
    Monetary  = ('Monetary',   'mean')
).round(2)

print("\n📋 K-MEANS CLUSTER PROFILES:")
print(cluster_summary_kmeans.to_string())


# Labeling logic (same for both)
def label_cluster(row):
    r, f, m = row['Recency'], row['Frequency'], row['Monetary']
    if r < 30 and f > 5 and m > 1000:
        return 'Champions'
    elif r < 60 and f >= 3:
        return 'Loyal Customers'
    elif r > 150 and f <= 2:
        return 'Lost / Inactive'
    elif r > 90:
        return 'At-Risk'
    else:
        return 'Potential Loyalists'


cluster_summary_kmeans['Segment'] = cluster_summary_kmeans.apply(label_cluster, axis=1)

print("\n📋 K-MEANS CLUSTER LABELS:")
print(cluster_summary_kmeans[['Count','Segment']].to_string())

# Map back
label_map_kmeans = cluster_summary_kmeans['Segment'].to_dict()
rfm['Segment'] = rfm['Cluster'].map(label_map_kmeans)


# ─────────────────────────────────────────────
# K-MEDOIDS CLUSTER LABELING (🔥 IMPORTANT)
# ─────────────────────────────────────────────
cluster_summary_kmedoids = rfm.groupby('Cluster_KMedoids').agg(
    Count     = ('CustomerID', 'count'),
    Recency   = ('Recency',    'mean'),
    Frequency = ('Frequency',  'mean'),
    Monetary  = ('Monetary',   'mean')
).round(2)

print("\n📋 K-MEDOIDS CLUSTER PROFILES:")
print(cluster_summary_kmedoids.to_string())

cluster_summary_kmedoids['Segment'] = cluster_summary_kmedoids.apply(label_cluster, axis=1)

print("\n📋 K-MEDOIDS CLUSTER LABELS:")
print(cluster_summary_kmedoids[['Count','Segment']].to_string())

# 🔥 THIS IS THE CRITICAL LINE YOU NEEDED
label_map_kmedoids = cluster_summary_kmedoids['Segment'].to_dict()
rfm['Segment_KMedoids'] = rfm['Cluster_KMedoids'].map(label_map_kmedoids)

# ════════════════════════════════════════════════════════════
# SECTION 6 — VISUALISATIONS
# ════════════════════════════════════════════════════════════
print("\n▶ Generating visualisations...")

cluster_colors = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0']
palette = {seg: cluster_colors[i]
           for i, seg in enumerate(rfm['Segment'].unique())}

# ── Plot 1: 3D RFM Scatter ────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection='3d')

for seg, grp in rfm.groupby('Segment'):
    ax.scatter(grp['Recency'], grp['Frequency'], grp['Monetary'],
               label=seg, alpha=0.6, s=20)

ax.set_xlabel('Recency (days)',   fontsize=10)
ax.set_ylabel('Frequency',        fontsize=10)
ax.set_zlabel('Monetary (£)',     fontsize=10)
ax.set_title('Customer Segments — RFM 3D View',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('../outputs/figures/10_cluster_3d.png', dpi=150)
plt.close()
print("   ✅ Saved: 10_cluster_3d.png")

# ── Plot 2: Cluster Profile Bar Charts ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['Recency', 'Frequency', 'Monetary']
titles  = ['Avg Recency (days)\n(lower = better)',
           'Avg Frequency (orders)',
           'Avg Monetary Value (£)']
colors  = ['#FF9800', '#2196F3', '#4CAF50']

for ax, metric, title, color in zip(axes, metrics, titles, colors):
    data = rfm.groupby('Segment')[metric].mean().sort_values()
    ax.barh(data.index, data.values, color=color, edgecolor='white')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(metric)

plt.suptitle('Cluster Profiles by RFM Dimension',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../outputs/figures/11_cluster_profiles.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 11_cluster_profiles.png")

# ── Plot 3: RFM Boxplots per Segment ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, metric in zip(axes, metrics):
    rfm_plot = rfm[rfm[metric] <= rfm[metric].quantile(0.95)]
    order    = rfm_plot.groupby('Segment')[metric].median().sort_values().index
    sns.boxplot(data=rfm_plot, x='Segment', y=metric,
                order=order, ax=ax, palette='Set2')
    ax.set_title(f'{metric} by Segment', fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=20)

plt.suptitle('RFM Distribution per Customer Segment',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../outputs/figures/12_rfm_boxplots.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 12_rfm_boxplots.png")

# ── Plot 4: Segment Size Pie Chart ───────────────────────────────────────────
seg_counts = rfm['Segment'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(seg_counts.values,
        labels=seg_counts.index,
        autopct='%1.1f%%',
        colors=cluster_colors[:len(seg_counts)],
        startangle=140,
        wedgeprops={'edgecolor':'white', 'linewidth':2})
plt.title('Customer Segment Distribution',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/13_segment_pie.png', dpi=150)
plt.close()
print("   ✅ Saved: 13_segment_pie.png")

# ════════════════════════════════════════════════════════════
# SECTION 7 — SAVE OUTPUTS
# ════════════════════════════════════════════════════════════
rfm.to_csv('../data/processed/customer_segments.csv', index=False)
print(f"\n✅ Segmented customers saved → ../data/processed/customer_segments.csv")

# ════════════════════════════════════════════════════════════
# SECTION 8 — ACADEMIC SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📚 ACADEMIC SUMMARY (use in your report)")
print("=" * 60)
print(f"  Algorithm        : K-Means (k-means++ initialisation)")
print(f"  Features used    : Recency, Frequency, Monetary (RFM)")
print(f"  Preprocessing    : Log transform + StandardScaler")
print(f"  Optimal K        : {best_k} (Elbow + Silhouette method)")
print(f"  Silhouette Score : {max(silhouettes):.4f}")
print(f"  Total customers  : {len(rfm):,}")
print()
for seg, grp in rfm.groupby('Segment'):
    pct = len(grp) / len(rfm) * 100
    print(f"  Segment: {seg:<25} "
          f"{len(grp):>5,} customers ({pct:.1f}%)")
print("=" * 60)