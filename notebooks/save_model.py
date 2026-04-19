# save_model.py
# PURPOSE: Serialize the trained Random Forest model and scaler
# so they can be loaded by the deployment portal

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print(" Loading customer segments...")
rfm = pd.read_csv('../data/processed/customer_segments.csv')

# ── Prepare features & target (same as 06_classification.py) ─────────────────
X = rfm[['Recency', 'Frequency', 'Monetary']].copy()
le = LabelEncoder()
y  = le.fit_transform(rfm['Segment_KMedoids'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train final Random Forest on full training set ────────────────────────────
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print(f"    Model trained")

# ── Compute dataset stats for the portal's reference ranges ──────────────────
stats = {
    'recency_min'  : int(rfm['Recency'].min()),
    'recency_max'  : int(rfm['Recency'].quantile(0.99)),
    'frequency_min': int(rfm['Frequency'].min()),
    'frequency_max': int(rfm['Frequency'].quantile(0.99)),
    'monetary_min' : float(rfm['Monetary'].min()),
    'monetary_max' : float(rfm['Monetary'].quantile(0.99)),
    'classes'      : list(le.classes_),
    'total_customers': len(rfm),
    'segment_counts': rfm['Segment_KMedoids'].value_counts().to_dict()
}

# ── Save everything ───────────────────────────────────────────────────────────
import os, json
os.makedirs('../outputs/model', exist_ok=True)

with open('../outputs/model/rf_model.pkl',    'wb') as f: pickle.dump(rf, f)
with open('../outputs/model/label_encoder.pkl','wb') as f: pickle.dump(le, f)
with open('../outputs/model/dataset_stats.json','w') as f: json.dump(stats, f, indent=2)

print(" Saved:")
print("   ../outputs/model/rf_model.pkl")
print("   ../outputs/model/label_encoder.pkl")
print("   ../outputs/model/dataset_stats.json")
print(f"\n   Classes : {list(le.classes_)}")
print(f"   Features: Recency, Frequency, Monetary")