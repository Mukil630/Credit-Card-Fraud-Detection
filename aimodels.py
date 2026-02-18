# fraud_detection_pipeline.py
"""
Credit-card fraud detection — end-to-end example.
Requires: pandas, numpy, scikit-learn, imbalanced-learn, xgboost, joblib
pip install pandas numpy scikit-learn imbalanced-learn xgboost joblib
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

# 1) Load data
df = pd.read_csv("creditcard.csv")
# Quick sanity
print("Rows, cols:", df.shape)
print(df['Class'].value_counts(normalize=True))

# 2) Split features & target
X = df.drop(columns=['Class'])
y = df['Class']

# 3) Train-test split (stratify to preserve rare class)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Preprocessing: scale numeric features (Time and Amount often need scaling)
scaler = StandardScaler()
# fit only on train
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5) Handle class imbalance with SMOTE on training data
sm = SMOTE(random_state=42, n_jobs=-1)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
print("After SMOTE, class distribution:", np.bincount(y_res))

# 6) Modeling: try three models and pick best (logic: start simple, escalate)
models = {
    "logreg": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "xgb": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
}

# Optional quick training & evaluation loop
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    print(f"\n>>> TRAINING {name}")
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    print("Confusion matrix:\n", confusion_matrix(y_te, preds))
    print("Classification report:\n", classification_report(y_te, preds, digits=4))
    print("ROC AUC:", roc_auc_score(y_te, probs))
    # Precision-Recall AUC
    p, r, _ = precision_recall_curve(y_te, probs)
    print("PR AUC:", auc(r, p))
    return model, probs

results = {}
for n, m in models.items():
    trained, probs = evaluate_model(n, m, X_res, y_res, X_test_scaled, y_test)
    results[n] = (trained, probs)

# 7) Threshold tuning example for best model (choose by ROC AUC or business metric)
# Pick the best by ROC AUC
best_name = max(results.keys(), key=lambda k: roc_auc_score(y_test, results[k][1]))
best_model = results[best_name][0]
print("\nSelected best model:", best_name)

# Compute precision/recall curve to choose threshold that prioritizes recall (catch fraud)
probs_best = best_model.predict_proba(X_test_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs_best)

# Example: choose threshold with recall >= 0.8 (you may choose as per business requirements)
target_recall = 0.8
idx = np.argmax(recall >= target_recall)  # first index reaching the target recall
if idx < len(thresholds):
    chosen_threshold = thresholds[idx]
else:
    chosen_threshold = 0.5  # fallback

print(f"Chosen threshold for recall >= {target_recall}: {chosen_threshold:.4f}")
final_preds = (probs_best >= chosen_threshold).astype(int)
print("Final Confusion matrix (thresholded):\n", confusion_matrix(y_test, final_preds))
print("Final classification report:\n", classification_report(y_test, final_preds, digits=4))

# 8) Save model + scaler
joblib.dump({"model": best_model, "scaler": scaler, "threshold": chosen_threshold}, "fraud_detector_v1.joblib")
print("Saved model to fraud_detector_v1.joblib")

# 9) Helper inference function for a single transaction
def predict_transaction(transaction_df):
    """
    transaction_df : pd.DataFrame with same columns as training X (single-row DataFrame)
    returns : dict with probability and label (0/1) using chosen threshold
    """
    sc = joblib.load("fraud_detector_v1.joblib")
    scaler_loaded = sc['scaler']
    model_loaded = sc['model']
    threshold_loaded = sc['threshold']
    Xs = scaler_loaded.transform(transaction_df)
    prob = model_loaded.predict_proba(Xs)[:, 1][0]
    label = int(prob >= threshold_loaded)
    return {"probability_fraud": float(prob), "label": label, "threshold": float(threshold_loaded)}

# Example usage of predict_transaction
# single_tx = X_test.iloc[[0]]   # uncomment to test
# print(predict_transaction(single_tx))
