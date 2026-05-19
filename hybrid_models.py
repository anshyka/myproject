"""
=============================================================================
  Parkinson's Disease Detection — Hybrid Model Ensemble Pipeline
  Combines XGBoost, SVM, and Random Forest into hybrid architectures
  to maximize clinical Recall and overall Accuracy.
=============================================================================
"""

import sys
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import time
import json

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OUTPUT_DIR = 'artifacts/hybrid_comparison'
MODEL_DIR = 'artifacts/models'

np.random.seed(RANDOM_STATE)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("=" * 70, flush=True)
print("  PARKINSON'S DISEASE — HYBRID MODEL PIPELINE", flush=True)
print("=" * 70, flush=True)

print("\n[1/7] Loading dataset...", flush=True)

df = pd.read_csv('data/raw/pd_speech_features.csv')
print(f"  Dataset shape: {df.shape}")

# Detect target column
target_col = None
for col in ['class', 'status']:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError("No target column found. Expected 'class' or 'status'.")

# Drop non-feature columns
drop_cols = [target_col]
for col in ['id', 'name']:
    if col in df.columns:
        drop_cols.append(col)

X = df.drop(columns=drop_cols)
y = df[target_col]
feature_names = X.columns.tolist()

print(f"  Target: '{target_col}' | Features: {X.shape[1]}")
print(f"  Class distribution: {dict(y.value_counts())}")

# ─────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Splitting data (80/20 stratified)...", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
)
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Pre-scale for estimators that need it (SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# 3. TUNE INDIVIDUAL BASE MODELS FIRST
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Tuning individual base models with RandomizedSearchCV...", flush=True)

from sklearn.model_selection import RandomizedSearchCV

cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Class imbalance ratio for XGBoost
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos = neg_count / pos_count

# --- Tune XGBoost ---
print("  >> Tuning XGBoost...", end=" ", flush=True)
xgb_pipe = Pipeline([('scaler', StandardScaler()), ('model', XGBClassifier(
    eval_metric='logloss', use_label_encoder=False, random_state=RANDOM_STATE, verbosity=0,
))])
xgb_search = RandomizedSearchCV(
    xgb_pipe,
    param_distributions={
        'model__n_estimators': [100, 200, 300, 400],
        'model__max_depth': [3, 5, 7, 10],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'model__min_child_weight': [1, 3, 5],
        'model__gamma': [0, 0.1, 0.3],
        'model__reg_lambda': [0.5, 1.0, 2.0],
        'model__scale_pos_weight': [1, scale_pos, scale_pos * 1.5],
    },
    n_iter=30, scoring='f1', cv=cv_strategy, random_state=RANDOM_STATE,
    n_jobs=-1, refit=True,
)
xgb_search.fit(X_train, y_train)
xgb_cv_f1 = xgb_search.best_score_
print(f"CV F1={xgb_cv_f1:.4f}", flush=True)

# --- Tune SVM ---
print("  >> Tuning SVM...", end=" ", flush=True)
svm_pipe = Pipeline([('scaler', StandardScaler()), ('model', SVC(
    probability=True, random_state=RANDOM_STATE,
))])
svm_search = RandomizedSearchCV(
    svm_pipe,
    param_distributions={
        'model__C': [0.5, 1, 5, 10, 50, 100],
        'model__kernel': ['rbf'],
        'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'model__class_weight': ['balanced'],
    },
    n_iter=20, scoring='f1', cv=cv_strategy, random_state=RANDOM_STATE,
    n_jobs=-1, refit=True,
)
svm_search.fit(X_train, y_train)
svm_cv_f1 = svm_search.best_score_
print(f"CV F1={svm_cv_f1:.4f}", flush=True)

# --- Tune Random Forest ---
print("  >> Tuning Random Forest...", end=" ", flush=True)
rf_pipe = Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier(
    random_state=RANDOM_STATE, n_jobs=-1,
))])
rf_search = RandomizedSearchCV(
    rf_pipe,
    param_distributions={
        'model__n_estimators': [200, 300, 400, 500],
        'model__max_depth': [10, 15, 20, None],
        'model__min_samples_split': [2, 3, 5],
        'model__min_samples_leaf': [1, 2],
        'model__max_features': ['sqrt', 'log2', 0.3, 0.5],
        'model__class_weight': ['balanced', 'balanced_subsample'],
        'model__criterion': ['gini', 'entropy'],
    },
    n_iter=30, scoring='f1', cv=cv_strategy, random_state=RANDOM_STATE,
    n_jobs=-1, refit=True,
)
rf_search.fit(X_train, y_train)
rf_cv_f1 = rf_search.best_score_
print(f"CV F1={rf_cv_f1:.4f}", flush=True)

# Extract TUNED best estimators (the inner model from each pipeline)
xgb_tuned = xgb_search.best_estimator_.named_steps['model']
svm_tuned = svm_search.best_estimator_.named_steps['model']
rf_tuned = rf_search.best_estimator_.named_steps['model']

print(f"\n  Best XGBoost params: {xgb_search.best_params_}")
print(f"  Best SVM params:    {svm_search.best_params_}")
print(f"  Best RF params:     {rf_search.best_params_}")

# ─────────────────────────────────────────────────────────────
# 4. BUILD HYBRID / ENSEMBLE MODELS FROM TUNED BASE LEARNERS
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Constructing hybrid ensembles from tuned base models...", flush=True)

# All models dict: name -> (model, needs_scaled_data)
all_models = {}

# --- Individual baselines (using already-tuned pipelines) ---
all_models['XGBoost'] = (xgb_search.best_estimator_, False)
all_models['SVM'] = (svm_search.best_estimator_, False)
all_models['Random Forest'] = (rf_search.best_estimator_, False)

# --- Compute performance-based weights for soft voting ---
# Weight each model proportionally to its CV F1 score
total_f1 = xgb_cv_f1 + svm_cv_f1 + rf_cv_f1
w_xgb = round(xgb_cv_f1 / total_f1 * 10, 1)
w_svm = round(svm_cv_f1 / total_f1 * 10, 1)
w_rf  = round(rf_cv_f1 / total_f1 * 10, 1)
print(f"  Soft voting weights (from CV F1): XGB={w_xgb}, SVM={w_svm}, RF={w_rf}")

# Clone tuned hyperparams into fresh estimator instances for the VotingClassifier
import sklearn.base as skbase
xgb_clone = skbase.clone(xgb_tuned)
svm_clone = skbase.clone(svm_tuned)
rf_clone  = skbase.clone(rf_tuned)
xgb_clone2 = skbase.clone(xgb_tuned)
svm_clone2 = skbase.clone(svm_tuned)
rf_clone2  = skbase.clone(rf_tuned)

# --- Hybrid 1: Soft Voting (performance-weighted, tuned estimators) ---
all_models['Hybrid: Soft Voting'] = (
    Pipeline([
        ('scaler', StandardScaler()),
        ('model', VotingClassifier(
            estimators=[
                ('xgb', xgb_clone),
                ('svm', svm_clone),
                ('rf', rf_clone),
            ],
            voting='soft',
            weights=[w_xgb, w_svm, w_rf],
            n_jobs=-1,
        )),
    ]),
    False
)

# --- Hybrid 2: Hard Voting (tuned estimators) ---
all_models['Hybrid: Hard Voting'] = (
    Pipeline([
        ('scaler', StandardScaler()),
        ('model', VotingClassifier(
            estimators=[
                ('xgb', xgb_clone2),
                ('svm', svm_clone2),
                ('rf', rf_clone2),
            ],
            voting='hard',
            n_jobs=-1,
        )),
    ]),
    False
)



print(f"  Total models to evaluate: {len(all_models)}")
for name in all_models:
    tag = "🔀 HYBRID" if "Hybrid" in name else "📊 BASE"
    print(f"    {tag}  {name}")

# ─────────────────────────────────────────────────────────────
# 5. TRAIN & EVALUATE ALL MODELS
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Training and evaluating all models...", flush=True)
print("-" * 70, flush=True)

results = {}

for name, (model, _) in all_models.items():
    print(f"\n  >> {name}...", end=" ", flush=True)
    start = time.time()

    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)

    # Not all models support predict_proba natively via pipeline
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        y_proba = None
        auc = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec_0 = recall_score(y_test, y_pred, pos_label=0)

    # Cross-val score for robustness
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='recall')

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'recall_class0': rec_0,
        'f1': f1,
        'auc': auc,
        'cv_recall_mean': cv_scores.mean(),
        'cv_recall_std': cv_scores.std(),
        'train_time': elapsed,
    }

    auc_str = f"{auc:.4f}" if auc else "N/A"
    print(f"Acc={acc:.4f} | Rec={rec:.4f} | F1={f1:.4f} | AUC={auc_str} | {elapsed:.1f}s", flush=True)

# ─────────────────────────────────────────────────────────────
# 6. GENERATE COMPARISON PLOTS
# ─────────────────────────────────────────────────────────────
print("\n\n[6/7] Generating comparison plots...", flush=True)

COLORS = {
    'XGBoost': '#2196F3',
    'SVM': '#FF5722',
    'Random Forest': '#4CAF50',
    'Hybrid: Soft Voting': '#9C27B0',
    'Hybrid: Hard Voting': '#E91E63',
}

# ── 6a. ROC Curves ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
for name, res in results.items():
    if res['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.4f})",
                color=COLORS.get(name, '#333'), linewidth=2.2,
                linestyle='--' if 'Hybrid' not in name else '-')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves — Base vs Hybrid Models', fontsize=16, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/hybrid_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ ROC curves saved")

# ── 6b. Precision-Recall Curves ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
for name, res in results.items():
    if res['y_proba'] is not None:
        p_vals, r_vals, _ = precision_recall_curve(y_test, res['y_proba'])
        ap = average_precision_score(y_test, res['y_proba'])
        ax.plot(r_vals, p_vals, label=f"{name} (AP={ap:.4f})",
                color=COLORS.get(name, '#333'), linewidth=2.2,
                linestyle='--' if 'Hybrid' not in name else '-')

ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision-Recall Curves — Base vs Hybrid', fontsize=16, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/hybrid_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Precision-Recall curves saved")

# ── 6c. Confusion Matrices ──────────────────────────────────
n_models = len(results)
cols = 4
rows = (n_models + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
axes = axes.flatten() if n_models > 1 else [axes]

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Healthy', "PD"])
    disp.plot(ax=axes[idx], cmap='Blues', values_format='d', colorbar=False)
    short = name.replace('Hybrid: ', '')
    axes[idx].set_title(short, fontsize=11, fontweight='bold')

# Hide unused subplots
for idx in range(n_models, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('Confusion Matrices — Base vs Hybrid Models', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/hybrid_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Confusion matrices saved")

# ── 6d. Metrics Bar Chart ───────────────────────────────────
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metric_keys = ['accuracy', 'precision', 'recall', 'f1']

fig, ax = plt.subplots(figsize=(18, 8))
x = np.arange(len(metric_names))
n = len(results)
bar_w = 0.8 / n

for idx, (name, res) in enumerate(results.items()):
    vals = [res[k] for k in metric_keys]
    offset = (idx - n / 2 + 0.5) * bar_w
    bars = ax.bar(x + offset, vals, bar_w, label=name,
                  color=COLORS.get(name, '#333'), alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
                rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=13)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Performance Comparison — Base vs Hybrid Models', fontsize=16, fontweight='bold')
ax.set_ylim(0, 1.18)
ax.legend(fontsize=9, loc='upper left', ncol=2)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/hybrid_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Metrics comparison chart saved")

# ── 6e. CV Recall Stability Chart ───────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
names_list = list(results.keys())
cv_means = [results[n]['cv_recall_mean'] for n in names_list]
cv_stds = [results[n]['cv_recall_std'] for n in names_list]
x_pos = np.arange(len(names_list))
colors_list = [COLORS.get(n, '#333') for n in names_list]

bars = ax.bar(x_pos, cv_means, yerr=cv_stds, capsize=5,
              color=colors_list, alpha=0.85, edgecolor='white')
for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std + 0.01,
            f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels([n.replace('Hybrid: ', 'H: ') for n in names_list],
                   fontsize=10, rotation=25, ha='right')
ax.set_ylabel('Cross-Validated Recall', fontsize=13)
ax.set_title('CV Recall Stability (mean ± std)', fontsize=16, fontweight='bold')
ax.set_ylim(0, 1.2)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/hybrid_cv_recall_stability.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ CV Recall stability chart saved")

# ─────────────────────────────────────────────────────────────
# 7. RESULTS SUMMARY & WINNER DETERMINATION
# ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  FULL RESULTS — BASE vs HYBRID MODELS")
print("=" * 70)

summary_data = []
for name, res in results.items():
    auc_str = f"{res['auc']:.4f}" if res['auc'] else "N/A"
    summary_data.append({
        'Model': name,
        'Type': 'Hybrid' if 'Hybrid' in name else 'Base',
        'Accuracy': f"{res['accuracy']:.4f}",
        'Precision': f"{res['precision']:.4f}",
        'Recall (PD)': f"{res['recall']:.4f}",
        'Recall (Healthy)': f"{res['recall_class0']:.4f}",
        'F1-Score': f"{res['f1']:.4f}",
        'AUC-ROC': auc_str,
        'CV Recall': f"{res['cv_recall_mean']:.4f}±{res['cv_recall_std']:.4f}",
        'Time (s)': f"{res['train_time']:.1f}",
    })

summary_df = pd.DataFrame(summary_data)
print(f"\n{summary_df.to_string(index=False)}\n")

# ── Winner Determination ────────────────────────────────────
print("=" * 70)
print("  🏆 OPTIMAL MODEL DETERMINATION")
print("=" * 70)
print("\n  Scoring: 0.50 × Recall + 0.30 × Accuracy + 0.20 × AUC")

scores = {}
for name, res in results.items():
    auc_val = res['auc'] if res['auc'] is not None else 0.5
    composite = 0.50 * res['recall'] + 0.30 * res['accuracy'] + 0.20 * auc_val
    scores[name] = composite
    tag = "🔀" if "Hybrid" in name else "  "
    print(f"  {tag} {name:35s}: {composite:.4f}")

winner = max(scores, key=scores.get)
winner_res = results[winner]

print(f"\n  {'🏆' * 5}")
print(f"  OPTIMAL MODEL: {winner}")
print(f"  Composite Score: {scores[winner]:.4f}")
print(f"  Accuracy:  {winner_res['accuracy']:.4f}")
print(f"  Recall:    {winner_res['recall']:.4f}")
print(f"  F1-Score:  {winner_res['f1']:.4f}")
auc_w = f"{winner_res['auc']:.4f}" if winner_res['auc'] else "N/A"
print(f"  AUC-ROC:   {auc_w}")
print(f"  {'🏆' * 5}")

# ── Save all model artifacts ───────────────────────────────
print("\n  Saving model artifacts...", flush=True)

# Save the optimal (winner) model
winner_artifact = {
    'model_pipeline': winner_res['model'],
    'feature_names': feature_names,
    'model_name': winner,
    'model_type': 'hybrid' if 'Hybrid' in winner else 'base',
    'metrics': {
        'accuracy': winner_res['accuracy'],
        'precision': winner_res['precision'],
        'recall': winner_res['recall'],
        'f1': winner_res['f1'],
        'auc': winner_res['auc'],
        'cv_recall_mean': winner_res['cv_recall_mean'],
    },
}

optimal_path = f'{MODEL_DIR}/optimal_hybrid_model.joblib'
joblib.dump(winner_artifact, optimal_path)
print(f"  ✅ Optimal model saved: {optimal_path}")

# Save each individual model
for name, res in results.items():
    safe_name = name.lower().replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
    model_path = f'{MODEL_DIR}/{safe_name}.joblib'
    artifact = {
        'model_pipeline': res['model'],
        'feature_names': feature_names,
        'model_name': name,
        'model_type': 'hybrid' if 'Hybrid' in name else 'base',
        'metrics': {
            'accuracy': res['accuracy'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1': res['f1'],
            'auc': res['auc'],
        },
    }
    joblib.dump(artifact, model_path)
    print(f"  ✅ {name} → {model_path}")

# ── Save summary artifacts ─────────────────────────────────
summary_df.to_csv(f'{OUTPUT_DIR}/hybrid_comparison_summary.csv', index=False)
print(f"\n  ✅ Summary CSV: {OUTPUT_DIR}/hybrid_comparison_summary.csv")

# Save a JSON report too
json_report = {
    'optimal_model': winner,
    'optimal_composite_score': round(scores[winner], 4),
    'all_scores': {k: round(v, 4) for k, v in scores.items()},
    'summary': summary_data,
}
with open(f'{OUTPUT_DIR}/hybrid_report.json', 'w') as f:
    json.dump(json_report, f, indent=2)
print(f"  ✅ JSON report: {OUTPUT_DIR}/hybrid_report.json")

# ── Final summary ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  ALL ARTIFACTS SAVED IN:")
print("=" * 70)
print(f"  📁 Models:  {MODEL_DIR}/")
print(f"  📁 Plots:   {OUTPUT_DIR}/")
print(f"  📈 hybrid_roc_curves.png")
print(f"  📈 hybrid_pr_curves.png")
print(f"  📈 hybrid_confusion_matrices.png")
print(f"  📈 hybrid_metrics_comparison.png")
print(f"  📈 hybrid_cv_recall_stability.png")
print(f"  📊 hybrid_comparison_summary.csv")
print(f"  📊 hybrid_report.json")
print("=" * 70)
print("\n✅ Hybrid pipeline complete!")
