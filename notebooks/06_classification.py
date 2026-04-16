# 06_classification.py
# PURPOSE : Build and evaluate classification models to predict
#           customer segments based on RFM features.
#
# KDD CONTEXT: Classification is SUPERVISED learning. We use the
#              cluster labels from Step 5 as ground-truth targets
#              and train models to predict segment membership.
#              This enables real-time classification of NEW customers.
#
# MODELS TRAINED:
#   1. Decision Tree       — interpretable, good for report/viva
#   2. Random Forest       — ensemble, typically best accuracy
#   3. K-Nearest Neighbour — instance-based, good comparison
#
# OUTPUT:
#   outputs/classification_report.csv
#   outputs/figures/14_confusion_matrices.png
#   outputs/figures/15_feature_importance.png
#   outputs/figures/16_roc_curves.png
#   outputs/figures/17_decision_tree.png

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score,
                             RocCurveDisplay)
from sklearn.preprocessing import label_binarize
import warnings


warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

# ════════════════════════════════════════════════════════════
# SECTION 1 — LOAD SEGMENTED CUSTOMER DATA
# ════════════════════════════════════════════════════════════
print("▶ Loading customer segments...")
rfm = pd.read_csv('../data/processed/customer_segments.csv')
print(f"   Loaded: {len(rfm):,} customers")
print(f"   Segments: {rfm['Segment'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════
# SECTION 2 — PREPARE FEATURES & TARGET
# ════════════════════════════════════════════════════════════
print("\n▶ Preparing features and target variable...")

# Features: raw RFM values (not scaled — tree models don't need scaling,
# and we want interpretable thresholds in the Decision Tree)
X = rfm[['Recency', 'Frequency', 'Monetary']].copy()

# Target: segment label (encoded as integer for sklearn)
le = LabelEncoder()
y  = le.fit_transform(rfm['Segment'])

print(f"   Features : {list(X.columns)}")
print(f"   Target   : Segment ({len(le.classes_)} classes)")
print(f"   Classes  : {list(le.classes_)}")

# ── Train/Test Split ──────────────────────────────────────────────────────────
# 80% training, 20% testing — standard split for datasets of this size
# stratify=y ensures class proportions are preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n   Train set : {len(X_train):,} customers (80%)")
print(f"   Test set  : {len(X_test):,} customers (20%)")

# Scale for KNN (KNN IS sensitive to feature scale)
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ════════════════════════════════════════════════════════════
# SECTION 3 — TRAIN ALL THREE MODELS
# ════════════════════════════════════════════════════════════
print("\n▶ Training classification models...")

models = {
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5,          # limit depth to prevent overfitting
        min_samples_split=10, # require at least 10 samples to split
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,     # 100 decision trees in the ensemble
        max_depth=10,
        random_state=42,
        n_jobs=-1             # use all CPU cores
    ),
    'K-Nearest Neighbour': KNeighborsClassifier(
        n_neighbors=5,        # use 5 nearest neighbours (standard default)
        metric='euclidean'
    )
}

# Store results for comparison
results     = {}
cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n   Training: {name}...")

    # Use scaled data for KNN, raw for tree-based models
    Xtr = X_train_scaled if name == 'K-Nearest Neighbour' else X_train
    Xte = X_test_scaled  if name == 'K-Nearest Neighbour' else X_test
    Xs  = X_train_scaled if name == 'K-Nearest Neighbour' else X_train

    # Fit model
    model.fit(Xtr, y_train)

    # Predictions
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)

    # Metrics
    acc    = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, Xs, y_train,
                                cv=cv, scoring='accuracy')

    results[name] = {
        'model'    : model,
        'y_pred'   : y_pred,
        'y_prob'   : y_prob,
        'accuracy' : acc,
        'cv_mean'  : cv_scores.mean(),
        'cv_std'   : cv_scores.std(),
        'report'   : classification_report(y_test, y_pred,
                                           target_names=le.classes_,
                                           output_dict=True)
    }

    print(f"   Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"   CV Accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ════════════════════════════════════════════════════════════
# SECTION 4 — MODEL COMPARISON TABLE
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 MODEL COMPARISON")
print("=" * 60)
print(f"  {'Model':<25} {'Test Acc':>10} {'CV Acc':>10} {'CV Std':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
for name, res in results.items():
    print(f"  {name:<25} {res['accuracy']:>10.4f} "
          f"{res['cv_mean']:>10.4f} {res['cv_std']:>10.4f}")
print("=" * 60)

# Identify best model
best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
print(f"\n  🏆 Best model: {best_model_name} "
      f"(CV Accuracy: {results[best_model_name]['cv_mean']:.4f})")

# ════════════════════════════════════════════════════════════
# SECTION 5 — DETAILED CLASSIFICATION REPORT (Best Model)
# ════════════════════════════════════════════════════════════
best = results[best_model_name]
print(f"\n📋 DETAILED REPORT — {best_model_name}:")
print(classification_report(y_test, best['y_pred'],
                             target_names=le.classes_))

# ════════════════════════════════════════════════════════════
# SECTION 6 — VISUALISATIONS
# ════════════════════════════════════════════════════════════
print("\n▶ Generating visualisations...")

# ── Plot 1: Confusion Matrices (all 3 models) ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                ax=ax, linewidths=0.5)
    ax.set_title(f'{name}\nAccuracy: {res["accuracy"]:.4f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual',    fontsize=10)
    ax.tick_params(axis='x', rotation=20)
    ax.tick_params(axis='y', rotation=0)

plt.suptitle('Confusion Matrices — All Models',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/14_confusion_matrices.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 14_confusion_matrices.png")

# ── Plot 2: Feature Importance (Random Forest) ───────────────────────────────
rf_model     = results['Random Forest']['model']
importances  = pd.Series(rf_model.feature_importances_,
                          index=['Recency','Frequency','Monetary'])
importances  = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 4))
colors = ['#FF9800' if v == importances.max() else '#2196F3'
          for v in importances.values]
plt.barh(importances.index, importances.values,
         color=colors, edgecolor='white')
plt.xlabel('Feature Importance Score', fontsize=11)
plt.title('Random Forest — Feature Importance',
          fontsize=13, fontweight='bold')
for i, v in enumerate(importances.values):
    plt.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('../outputs/figures/15_feature_importance.png', dpi=150)
plt.close()
print("   ✅ Saved: 15_feature_importance.png")

# ── Plot 3: Decision Tree Visualisation ───────────────────────────────────────
dt_model = results['Decision Tree']['model']

plt.figure(figsize=(20, 8))
plot_tree(dt_model,
          feature_names=['Recency','Frequency','Monetary'],
          class_names=le.classes_,
          filled=True,
          rounded=True,
          fontsize=9,
          impurity=False,
          proportion=False)
plt.title('Decision Tree — Customer Segment Classification',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/16_decision_tree.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: 16_decision_tree.png")

# ── Plot 4: Cross-Validation Score Comparison ────────────────────────────────
model_names = list(results.keys())
cv_means    = [results[m]['cv_mean'] for m in model_names]
cv_stds     = [results[m]['cv_std']  for m in model_names]
colors      = ['#4CAF50' if m == best_model_name
               else '#2196F3' for m in model_names]

plt.figure(figsize=(9, 5))
bars = plt.bar(model_names, cv_means, yerr=cv_stds,
               color=colors, edgecolor='white',
               capsize=8, width=0.5)
plt.ylabel('Cross-Validation Accuracy', fontsize=11)
plt.title('Model Comparison — 5-Fold Cross-Validation',
          fontsize=13, fontweight='bold')
plt.ylim(0, 1.05)
for bar, mean, std in zip(bars, cv_means, cv_stds):
    plt.text(bar.get_x() + bar.get_width()/2,
             mean + std + 0.01,
             f'{mean:.4f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/figures/17_model_comparison.png', dpi=150)
plt.close()
print("   ✅ Saved: 17_model_comparison.png")

# ════════════════════════════════════════════════════════════
# SECTION 7 — SAVE FULL RESULTS TO CSV
# ════════════════════════════════════════════════════════════
rows = []
for name, res in results.items():
    rep = res['report']
    for segment in le.classes_:
        rows.append({
            'Model'    : name,
            'Segment'  : segment,
            'Precision': rep[segment]['precision'],
            'Recall'   : rep[segment]['recall'],
            'F1-Score' : rep[segment]['f1-score'],
            'Support'  : rep[segment]['support'],
            'CV_Accuracy': res['cv_mean'],
            'Test_Accuracy': res['accuracy']
        })

pd.DataFrame(rows).to_csv('../outputs/classification_report.csv', index=False)
print(f"\n✅ Full report saved → outputs/classification_report.csv")

# ════════════════════════════════════════════════════════════
# SECTION 8 — ACADEMIC SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📚 ACADEMIC SUMMARY (use in your report)")
print("=" * 60)
print(f"  Task              : Multi-class classification (3 segments)")
print(f"  Features          : Recency, Frequency, Monetary")
print(f"  Train/Test Split  : 80% / 20% (stratified)")
print(f"  Evaluation Method : 5-fold stratified cross-validation")
print()
for name, res in results.items():
    marker = ' 🏆' if name == best_model_name else ''
    print(f"  {name:<25} "
          f"Acc={res['accuracy']:.4f}  "
          f"CV={res['cv_mean']:.4f} ± {res['cv_std']:.4f}{marker}")
print("=" * 60)