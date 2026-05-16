"""
=============================================================================
  Parkinson's Disease Detection — 3-Model Comparison & Fine-Tuning
  Models: XGBoost, SVM (RBF), Random Forest
  Objective: Maximize Recall & Accuracy
=============================================================================
"""

import sys
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
N_ITER_SEARCH = 20        # Number of random search iterations per model
SCORING_METRIC = 'f1'     # F1 balances recall + precision (prevents all-positive bias)
OUTPUT_DIR = 'artifacts/comparison'

np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────
print("=" * 70, flush=True)
print("  PARKINSON'S DISEASE — 3-MODEL COMPARISON PIPELINE", flush=True)
print("=" * 70, flush=True)

print("\n[1/6] Loading dataset...", flush=True)

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

print(f"  Target column: '{target_col}'")
print(f"  Feature count: {X.shape[1]}")
print(f"  Class distribution: {dict(y.value_counts())}")

# ─────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Splitting data (80/20 stratified)...", flush=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"  Train set: {X_train.shape[0]} samples")
print(f"  Test set:  {X_test.shape[0]} samples")

# ─────────────────────────────────────────────────────────────
# 3. DEFINE MODELS & HYPERPARAMETER SEARCH SPACES
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Defining models and hyperparameter search spaces...", flush=True)

cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

models = {
    'XGBoost': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBClassifier(
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                verbosity=0,
            )),
        ]),
        'params': {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7, 10],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__subsample': [0.7, 0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0],
            'model__min_child_weight': [1, 3, 5],
            'model__gamma': [0, 0.1, 0.3],
            'model__reg_lambda': [0.5, 1.0, 2.0],
        },
    },
    'SVM': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
        'params': {
            'model__C': [0.1, 1, 5, 10, 50],
            'model__kernel': ['rbf', 'poly'],
            'model__gamma': ['scale', 'auto', 0.001, 0.01],
            'model__degree': [2, 3],  # Only for poly kernel
            'model__class_weight': [None, 'balanced'],
        },
    },
    'Random Forest': {
        'pipeline': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        'params': {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 10, 15, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2', 0.5],
            'model__bootstrap': [True, False],
            'model__criterion': ['gini', 'entropy'],
            'model__class_weight': [None, 'balanced'],
        },
    },
}

# ─────────────────────────────────────────────────────────────
# 4. FINE-TUNE EACH MODEL
# ─────────────────────────────────────────────────────────────
print("\n[4/6] Fine-tuning models with RandomizedSearchCV...", flush=True)
print(f"  Optimization metric: {SCORING_METRIC}", flush=True)
print(f"  CV folds: {CV_FOLDS}", flush=True)
print(f"  Random search iterations per model: {N_ITER_SEARCH}", flush=True)
print("-" * 70, flush=True)

results = {}

for name, config in models.items():
    print(f"\n  >> Tuning {name}...", flush=True)
    start_time = time.time()

    search = RandomizedSearchCV(
        estimator=config['pipeline'],
        param_distributions=config['params'],
        n_iter=N_ITER_SEARCH,
        scoring=SCORING_METRIC,
        cv=cv_strategy,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        refit=True,
        return_train_score=True,
    )

    search.fit(X_train, y_train)
    elapsed = time.time() - start_time

    # Get predictions
    y_pred = search.best_estimator_.predict(X_test)
    y_proba = search.best_estimator_.predict_proba(X_test)[:, 1]

    # Compute all metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Also compute class-0 recall (healthy detection)
    rec_0 = recall_score(y_test, y_pred, pos_label=0)

    results[name] = {
        'best_estimator': search.best_estimator_,
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'recall_class0': rec_0,
        'f1': f1,
        'auc': auc,
        'train_time': elapsed,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
    }

    print(f"     Best CV {SCORING_METRIC}: {search.best_score_:.4f}", flush=True)
    print(f"     Test Accuracy:  {acc:.4f}", flush=True)
    print(f"     Test Recall:    {rec:.4f}", flush=True)
    print(f"     Test F1:        {f1:.4f}", flush=True)
    print(f"     Test AUC:       {auc:.4f}", flush=True)
    print(f"     Time: {elapsed:.1f}s", flush=True)
    sys.stdout.flush()

# ─────────────────────────────────────────────────────────────
# 5. GENERATE ALL PLOTS
# ─────────────────────────────────────────────────────────────
print("\n\n[5/6] Generating comparison plots...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'XGBoost': '#2196F3',
    'SVM': '#FF5722',
    'Random Forest': '#4CAF50',
}

# ── 5a. ROC Curves ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.4f})",
            color=COLORS[name], linewidth=2.5)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve Comparison', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ ROC curves saved")

# ── 5b. Precision-Recall Curves ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

for name, res in results.items():
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, res['y_proba'])
    ap = average_precision_score(y_test, res['y_proba'])
    ax.plot(rec_vals, prec_vals, label=f"{name} (AP = {ap:.4f})",
            color=COLORS[name], linewidth=2.5)

ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision-Recall Curve Comparison', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/precision_recall_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Precision-Recall curves saved")

# ── 5c. Confusion Matrices (side by side) ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Healthy (0)', "Parkinson's (1)"])
    disp.plot(ax=axes[idx], cmap='Blues', values_format='d', colorbar=False)
    axes[idx].set_title(f'{name}', fontsize=14, fontweight='bold')

fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Confusion matrices saved")

# ── 5d. Metrics Bar Chart Comparison ────────────────────────
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(metric_names))
bar_width = 0.25
offsets = [-bar_width, 0, bar_width]

for idx, (name, res) in enumerate(results.items()):
    values = [res[k] for k in metric_keys]
    bars = ax.bar(x + offsets[idx], values, bar_width,
                  label=name, color=COLORS[name], alpha=0.85, edgecolor='white')
    # Add value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=13)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
ax.set_ylim(0, 1.12)
ax.legend(fontsize=12, loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Metrics comparison chart saved")

# ── 5e. Class-wise Recall Comparison ────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(2)
bar_width = 0.25
offsets = [-bar_width, 0, bar_width]
class_labels = ['Healthy (Class 0)', "Parkinson's (Class 1)"]

for idx, (name, res) in enumerate(results.items()):
    r0 = res['recall_class0']
    r1 = res['recall']
    bars = ax.bar(x + offsets[idx], [r0, r1], bar_width,
                  label=name, color=COLORS[name], alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, [r0, r1]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(class_labels, fontsize=13)
ax.set_ylabel('Recall', fontsize=13)
ax.set_title('Per-Class Recall Comparison (Critical for Clinical Use)', fontsize=15, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/class_recall_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Class-wise recall chart saved")

# ─────────────────────────────────────────────────────────────
# 6. PRINT FULL RESULTS & DETERMINE WINNER
# ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  FULL RESULTS")
print("=" * 70)

for name, res in results.items():
    print(f"\n{'─' * 60}")
    print(f"  📊 {name}")
    print(f"{'─' * 60}")
    print(f"  Best CV Recall:         {res['best_cv_score']:.4f}")
    print(f"  Training Time:          {res['train_time']:.1f}s")
    print(f"\n  Best Hyperparameters:")
    for param, value in sorted(res['best_params'].items()):
        print(f"    {param}: {value}")
    print(f"\n  Test Set Classification Report:")
    print(classification_report(y_test, res['y_pred'],
          target_names=['Healthy (0)', "Parkinson's (1)"]))

# ── Summary Table ───────────────────────────────────────────
print("\n" + "=" * 70)
print("  COMPARISON SUMMARY TABLE")
print("=" * 70)

summary_data = []
for name, res in results.items():
    summary_data.append({
        'Model': name,
        'Accuracy': f"{res['accuracy']:.4f}",
        'Precision': f"{res['precision']:.4f}",
        'Recall (PD)': f"{res['recall']:.4f}",
        'Recall (Healthy)': f"{res['recall_class0']:.4f}",
        'F1-Score': f"{res['f1']:.4f}",
        'AUC-ROC': f"{res['auc']:.4f}",
        'Time (s)': f"{res['train_time']:.1f}",
    })

summary_df = pd.DataFrame(summary_data)
print(f"\n{summary_df.to_string(index=False)}\n")

# ── Determine Winner ────────────────────────────────────────
# Scoring: weighted combination of Recall (50%) + Accuracy (30%) + AUC (20%)
print("=" * 70)
print("  🏆 WINNER DETERMINATION")
print("=" * 70)
print("\n  Scoring formula: 0.50 × Recall + 0.30 × Accuracy + 0.20 × AUC")

scores = {}
for name, res in results.items():
    composite = 0.50 * res['recall'] + 0.30 * res['accuracy'] + 0.20 * res['auc']
    scores[name] = composite
    print(f"  {name:15s}: {composite:.4f}")

winner = max(scores, key=scores.get)
print(f"\n  {'🏆' * 5}")
print(f"  WINNER: {winner}")
print(f"  Composite Score: {scores[winner]:.4f}")
print(f"  Accuracy:  {results[winner]['accuracy']:.4f}")
print(f"  Recall:    {results[winner]['recall']:.4f}")
print(f"  F1-Score:  {results[winner]['f1']:.4f}")
print(f"  AUC-ROC:   {results[winner]['auc']:.4f}")
print(f"  {'🏆' * 5}")

# ── Save the winner model ──────────────────────────────────
os.makedirs('artifacts/models', exist_ok=True)

winner_artifact = {
    'model_pipeline': results[winner]['best_estimator'],
    'feature_names': X.columns.tolist(),
    'model_name': winner,
    'metrics': {
        'accuracy': results[winner]['accuracy'],
        'recall': results[winner]['recall'],
        'f1': results[winner]['f1'],
        'auc': results[winner]['auc'],
    },
    'best_params': results[winner]['best_params'],
}

winner_path = 'artifacts/models/best_model_comparison.joblib'
joblib.dump(winner_artifact, winner_path)
print(f"\n  Winner model saved to: {winner_path}")

# Save all 3 models
for name, res in results.items():
    model_filename = f"artifacts/models/{name.lower().replace(' ', '_')}_tuned.joblib"
    artifact = {
        'model_pipeline': res['best_estimator'],
        'feature_names': X.columns.tolist(),
        'model_name': name,
        'metrics': {
            'accuracy': res['accuracy'],
            'recall': res['recall'],
            'f1': res['f1'],
            'auc': res['auc'],
        },
        'best_params': res['best_params'],
    }
    joblib.dump(artifact, model_filename)
    print(f"  {name} model saved to: {model_filename}")

# ── Save summary CSV ───────────────────────────────────────
summary_df.to_csv(f'{OUTPUT_DIR}/comparison_summary.csv', index=False)
print(f"\n  Summary CSV saved to: {OUTPUT_DIR}/comparison_summary.csv")

print("\n" + "=" * 70)
print("  ALL PLOTS SAVED IN: artifacts/comparison/")
print("=" * 70)
print("  📈 roc_curves.png")
print("  📈 precision_recall_curves.png")
print("  📈 confusion_matrices.png")
print("  📈 metrics_comparison.png")
print("  📈 class_recall_comparison.png")
print("  📊 comparison_summary.csv")
print("=" * 70)
print("\n✅ Pipeline complete!")
