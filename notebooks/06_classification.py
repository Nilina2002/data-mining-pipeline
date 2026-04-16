# 06_classification.py
# PURPOSE : Build and evaluate classification models to predict
#           customer segments based on RFM features.
#
# NOTE:
#   This version uses K-MEDOIDS-based segmentation (Segment_KMedoids)
#   as the ground truth target instead of K-Means.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ════════════════════════════════════════════════════════════
# SECTION 1 — LOAD SEGMENTED CUSTOMER DATA
# ════════════════════════════════════════════════════════════
print("▶ Loading customer segments...")
rfm = pd.read_csv('../data/processed/customer_segments.csv')

# ⚠️ Ensure K-Medoids labels exist
if 'Segment_KMedoids' not in rfm.columns:
    raise ValueError("❌ 'Segment_KMedoids' not found. Update clustering script first.")

print(f"   Loaded: {len(rfm):,} customers")
print(f"   Segments (K-Medoids): {rfm['Segment_KMedoids'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════
# SECTION 2 — PREPARE FEATURES & TARGET
# ════════════════════════════════════════════════════════════
print("\n▶ Preparing features and target variable...")

# Features
X = rfm[['Recency', 'Frequency', 'Monetary']].copy()

# ✅ TARGET = K-MEDOIDS SEGMENTS
le = LabelEncoder()
y  = le.fit_transform(rfm['Segment_KMedoids'])

print(f"   Features : {list(X.columns)}")
print(f"   Target   : Segment_KMedoids ({len(le.classes_)} classes)")
print(f"   Classes  : {list(le.classes_)}")

# ── Train/Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n   Train set : {len(X_train):,} customers (80%)")
print(f"   Test set  : {len(X_test):,} customers (20%)")

# Scale for KNN
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ════════════════════════════════════════════════════════════
# SECTION 3 — TRAIN MODELS
# ════════════════════════════════════════════════════════════
print("\n▶ Training classification models...")

models = {
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    'K-Nearest Neighbour': KNeighborsClassifier(
        n_neighbors=5,
        metric='euclidean'
    )
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n   Training: {name}...")

    Xtr = X_train_scaled if name == 'K-Nearest Neighbour' else X_train
    Xte = X_test_scaled  if name == 'K-Nearest Neighbour' else X_test
    Xs  = X_train_scaled if name == 'K-Nearest Neighbour' else X_train

    model.fit(Xtr, y_train)

    y_pred = model.predict(Xte)

    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, Xs, y_train,
                                cv=cv, scoring='accuracy')

    results[name] = {
        'model'    : model,
        'y_pred'   : y_pred,
        'accuracy' : acc,
        'cv_mean'  : cv_scores.mean(),
        'cv_std'   : cv_scores.std(),
        'report'   : classification_report(
                        y_test, y_pred,
                        target_names=le.classes_,
                        output_dict=True)
    }

    print(f"   Test Accuracy  : {acc:.4f}")
    print(f"   CV Accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ════════════════════════════════════════════════════════════
# SECTION 4 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 MODEL COMPARISON")
print("=" * 60)

for name, res in results.items():
    print(f"{name:<25} Acc={res['accuracy']:.4f}  CV={res['cv_mean']:.4f}")

best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
print(f"\n🏆 Best model: {best_model_name}")

# ════════════════════════════════════════════════════════════
# SECTION 5 — CONFUSION MATRICES
# ════════════════════════════════════════════════════════════
print("\n▶ Generating confusion matrices...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                ax=ax)
    ax.set_title(name)

plt.tight_layout()
plt.savefig('../outputs/figures/14_confusion_matrices.png')
plt.close()

# ════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════
rf_model = results['Random Forest']['model']

importances = pd.Series(
    rf_model.feature_importances_,
    index=['Recency','Frequency','Monetary']
).sort_values()

plt.figure(figsize=(8, 4))
importances.plot(kind='barh')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('../outputs/figures/15_feature_importance.png')
plt.close()

# ════════════════════════════════════════════════════════════
# SECTION 7 — DECISION TREE VISUAL
# ════════════════════════════════════════════════════════════
dt_model = results['Decision Tree']['model']

plt.figure(figsize=(18, 8))
plot_tree(dt_model,
          feature_names=['Recency','Frequency','Monetary'],
          class_names=le.classes_,
          filled=True)
plt.tight_layout()
plt.savefig('../outputs/figures/16_decision_tree.png')
plt.close()

# ════════════════════════════════════════════════════════════
# SECTION 8 — SAVE REPORT
# ════════════════════════════════════════════════════════════
rows = []
for name, res in results.items():
    for segment in le.classes_:
        rows.append({
            'Model': name,
            'Segment': segment,
            'F1': res['report'][segment]['f1-score'],
            'Accuracy': res['accuracy']
        })

pd.DataFrame(rows).to_csv('../outputs/classification_report.csv', index=False)

print("\n✅ Classification complete and saved!")

# ════════════════════════════════════════════════════════════
# SECTION 9 — SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📚 ACADEMIC SUMMARY")
print("=" * 60)
print(f"Task        : Multi-class classification ({len(le.classes_)} segments)")
print(f"Target      : K-Medoids Segments")
print(f"Best Model  : {best_model_name}")
print("=" * 60)