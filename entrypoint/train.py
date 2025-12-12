# entrypoint/train.py
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# -------------------------
# CONFIG / PATHS
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
FEAT_PATH = ROOT / "data" / "03-features" / "FD001_features.csv"
OUT_MODEL_DIR = ROOT / "models"
OUT_PRED_DIR = ROOT / "data" / "04-predictions"
OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_PRED_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load features
# -------------------------
print("Loading features:", FEAT_PATH)
if not FEAT_PATH.exists():
    raise FileNotFoundError(f"Features file not found: {FEAT_PATH}")

df = pd.read_csv(FEAT_PATH)
print("Raw features shape:", df.shape)

# Ensure last-cycle snapshot
if "time_cycle" in df.columns:
    last_rows = df.loc[df.groupby("unit")["time_cycle"].idxmax()].reset_index(drop=True)
else:
    last_rows = df.reset_index(drop=True)

# Target presence check
if "failure_within_30" not in last_rows.columns:
    raise ValueError("Target column 'failure_within_30' not found. Run preprocessing to create the target.")

y = last_rows["failure_within_30"].values
groups = last_rows["unit"].values

# Select feature columns (drop known non-features)
drop_cols = {"unit", "time_cycle", "max_cycle", "RUL", "failure_within_30"}
features = [c for c in last_rows.columns if c not in drop_cols]
if len(features) == 0:
    raise ValueError("No feature columns found after dropping non-feature columns.")

X = last_rows[features].copy()
print("Feature matrix shape:", X.shape)

# -------------------------
# Impute + Scale
# -------------------------
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Save preprocessing objects
joblib.dump(imputer, OUT_MODEL_DIR / "imputer_joblib.pkl")
joblib.dump(scaler, OUT_MODEL_DIR / "scaler_joblib.pkl")

# -------------------------
# Group split (no leakage)
# -------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(gss.split(X_scaled, y, groups=groups))

X_train, X_val = X_scaled[train_idx], X_scaled[valid_idx]
y_train, y_val = y[train_idx], y[valid_idx]

print("Train size:", X_train.shape, "Val size:", X_val.shape)
print("Unique classes in train:", np.unique(y_train), "unique in val:", np.unique(y_val))

# -------------------------
# Train model
# -------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced"
)

clf.fit(X_train, y_train)

# Save model
model_path = OUT_MODEL_DIR / "rf_failure_joblib.pkl"
joblib.dump(clf, model_path)
print("Model saved to:", model_path)

# -------------------------
# Evaluate
# -------------------------
y_pred = clf.predict(X_val)

# robust predict_proba handling
y_score = None
if hasattr(clf, "predict_proba"):
    proba = clf.predict_proba(X_val)
    # if shape[1] == 2 -> binary probs for class 0 and 1
    if proba.ndim == 2 and proba.shape[1] >= 2:
        y_score = proba[:, 1]
    elif proba.ndim == 2 and proba.shape[1] == 1:
        # Only one column -> probability for the single present class.
        # If that single column corresponds to class 1, use it; otherwise map it.
        # We will fallback to using predictions as scores if unsure.
        try:
            classes = clf.classes_
            if len(classes) == 1:
                # only one class present in training -> use constant scores
                y_score = np.full_like(y_pred, fill_value=float(classes[0]), dtype=float)
            else:
                # defensive: if classes[1] exists, pick column 0 as that class prob
                y_score = proba[:, 0]
        except Exception:
            y_score = None
else:
    # Some classifiers provide decision_function instead
    if hasattr(clf, "decision_function"):
        try:
            dfun = clf.decision_function(X_val)
            # if binary, decision_function returns shape (n,) or (n,1)
            if dfun.ndim == 1:
                y_score = dfun
            else:
                y_score = dfun[:, 0]
        except Exception:
            y_score = None

# If still None, fallback to using predicted labels (0/1) as score
if y_score is None:
    print("Warning: predict_proba/decision_function unavailable or ambiguous. Falling back to predicted labels as score.")
    y_score = y_pred.astype(float)

# Compute metrics safely (roc_auc requires at least two classes in y_true)
metrics = {}
metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
metrics["precision"] = float(precision_score(y_val, y_pred, zero_division=0))
metrics["recall"] = float(recall_score(y_val, y_pred, zero_division=0))
metrics["f1"] = float(f1_score(y_val, y_pred, zero_division=0))

try:
    metrics["roc_auc"] = float(roc_auc_score(y_val, y_score))
except Exception as e:
    metrics["roc_auc"] = None
    print("Could not compute ROC AUC:", e)

print("Validation metrics:", json.dumps(metrics, indent=2))

# Save metrics & predictions
(OUT_PRED_DIR / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

pred_df = pd.DataFrame({
    "unit": last_rows.iloc[valid_idx]["unit"].values,
    "y_true": y_val,
    "y_pred": y_pred,
    "y_score": y_score
})
pred_df.to_csv(OUT_PRED_DIR / "eval_report.csv", index=False)

# Save feature importances
try:
    importances = clf.feature_importances_
    fi = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
    fi.to_csv(OUT_PRED_DIR / "feature_importances.csv", index=False)
except Exception as e:
    print("Could not save feature importances:", e)

print("Done.")
