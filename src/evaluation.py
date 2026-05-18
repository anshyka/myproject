import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from pycaret.classification import predict_model, get_config, pull

# Apply professional publication plot styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

def save_custom_confusion_matrix(y_true, y_pred, title, filename):
    """
    Generates and saves a high-resolution confusion matrix with percentages.
    """
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentage cell distribution
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    labels = np.asarray([f"{val}\n({pct:.1f}%)" for val, pct in zip(cm.flatten(), cm_percent.flatten())]).reshape(2,2)
    
    sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', cbar=True,
                xticklabels=['Healthy (0)', 'Parkinson\'s (1)'],
                yticklabels=['Healthy (0)', 'Parkinson\'s (1)'],
                annot_kws={"size": 11, "weight": "bold"})
    
    plt.title(f'Confusion Matrix: {title}', pad=15, weight='bold')
    plt.ylabel('Actual Clinical Ground Truth', labelpad=10)
    plt.xlabel('Predicted Medical Diagnosis', labelpad=10)
    plt.tight_layout()
    
    os.makedirs('artifacts/plots', exist_ok=True)
    plt.savefig(f'artifacts/plots/{filename}', dpi=300)
    plt.close()

def plot_model_comparison(metrics_dict):
    """
    Generates a professional side-by-side performance comparison bar chart.
    """
    df_metrics = pd.DataFrame(metrics_dict)
    df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="muted")
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.4f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 5),
                        textcoords='offset points',
                        fontsize=9, weight='bold')
            
    plt.title("Performance Comparison Summary: Standalone vs. Combined Ensemble Team", pad=20, weight='bold')
    plt.ylim(0, 1.1)
    plt.ylabel("Evaluation Metrics Value", labelpad=10)
    plt.xlabel("Tested Model Configurations", labelpad=10)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig('artifacts/plots/model_performance_comparison.png', dpi=300)
    plt.close()

def plot_cross_validation_stability(cv_df):
    """
    Tracks and plots performance curves across GroupKFold iterations 
    to visually demonstrate training stability and verification optimization.
    """
    folds_df = cv_df.loc[~cv_df.index.isin(['Mean', 'Std'])]
    folds_df = folds_df.reset_index().rename(columns={'index': 'Fold'})
    folds_df['Fold'] = folds_df['Fold'].astype(str)
    
    plt.figure(figsize=(10, 5))
    plt.plot(folds_df['Fold'], folds_df['Accuracy'], marker='o', linewidth=2, label='Training/Val Accuracy', color='#1f77b4')
    plt.plot(folds_df['Fold'], folds_df['F1'], marker='s', linewidth=2, label='Training/Val F1-Score', color='#ff7f0e')
    plt.plot(folds_df['Fold'], folds_df['Recall'], marker='^', linewidth=2, linestyle='--', label='Training/Val Recall', color='#2ca02c')
    
    plt.title("Cross-Validation Convergence Stability Curve (10 Patient Folds)", pad=15, weight='bold')
    plt.xlabel("GroupKFold Validation Iteration", labelpad=10)
    plt.ylabel("Performance Scaling Metric", labelpad=10)
    plt.ylim(0.5, 1.05)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig('artifacts/plots/cv_stability_trajectory.png', dpi=300)
    plt.close()

def plot_roc_curves(base_experts, expert_names, ensemble_model, X_test, y_test):
    """
    Generates a consolidated ROC/AUC curve for all individual sub-models 
    and the final voting ensemble to serve as the project's optimization proof.
    """
    plt.figure(figsize=(9, 7))
    
    # Plot curves for each individual sub-expert algorithm
    for name, expert in zip(expert_names, base_experts):
        if hasattr(expert, "predict_proba"):
            y_probs = expert.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', alpha=0.7)

    # Plot curve for the final integrated voting ensemble model team
    if hasattr(ensemble_model, "predict_proba"):
        ensemble_probs = ensemble_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Combined Ensemble Team (AUC = {roc_auc:.4f})', linewidth=3, color='black')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing Reference')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Carbon Copy Rate (1 - Specificity)', labelpad=10)
    plt.ylabel('True Positive Rate (Clinical Sensitivity / Recall)', labelpad=10)
    plt.title('Receiver Operating Characteristic (ROC) Curvature Analysis', pad=15, weight='bold')
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig('artifacts/plots/model_roc_curves.png', dpi=300)
    plt.close()

def evaluate_and_plot(ensemble_model):
    """
    Executes an enterprise-level clinical validation sequence, computes exact performance 
    matrices for sub-components, and exports comprehensive evaluation diagrams.
    """
    print("\n" + "="*75)
    print("      ENTERPRISE GRADE CLINICAL PIPELINE DIAGNOSTIC SUITE       ")
    print("="*75)

    # 1. Gather configured verification arrays directly from PyCaret configuration registries
    X_test_transformed = get_config('X_test_transformed')
    y_test = get_config('y_test')

    base_experts = ensemble_model.estimators_
    expert_names = [type(expert).__name__ for expert in base_experts]

    # Structuring data matrix maps for our comparative bar charts
    metrics_summary = {"Model": [], "Testing Accuracy": [], "Macro F1-Score": []}

    # 2. PHASE 1: Standalone Sub-Model Evaluations WITH Full Classification Reports
    print("\n>>> PHASE 1: INDIVIDUAL BASE MODEL EVALUATIONS")
    for name, expert in zip(expert_names, base_experts):
        y_pred_individual = expert.predict(X_test_transformed)
        acc_ind = accuracy_score(y_test, y_pred_individual)
        f1_ind = f1_score(y_test, y_pred_individual, average='macro')
        
        metrics_summary["Model"].append(name.replace('Classifier', ''))
        metrics_summary["Testing Accuracy"].append(acc_ind)
        metrics_summary["Macro F1-Score"].append(f1_ind)
        
        print(f"\n📊 Performance Profile: {name}")
        print(f"  └─ Independent Test Accuracy : {acc_ind:.4f}")
        print(f"  └─ Independent Test F1-Score : {f1_ind:.4f}")
        print("\nDetailed Base Classification Matrix:")
        print(classification_report(y_test, y_pred_individual, target_names=['Healthy', 'Parkinson\'s']))
        print("-" * 45)
        
        file_safe_name = name.lower().replace('classifier', '').strip()
        save_custom_confusion_matrix(y_test, y_pred_individual, name, f'confusion_matrix_{file_safe_name}.png')

    # 3. PHASE 2: Cross-Validation Trajectory Evaluation (Tracks Training/Validation Matrix)
    print("\n>>> PHASE 2: CROSS-VALIDATION ACCURACY & SELECTION HISTORY")
    cv_leaderboard = pull()
    print(cv_leaderboard.to_string())
    
    try:
        plot_cross_validation_stability(cv_leaderboard)
        print("  └─ [SUCCESS] Validation trajectory trend plot exported.")
    except Exception as e:
        print(f"  └─ Plotting Notification: Cross-validation trajectory plotting skipped: {e}")

    # 4. PHASE 3: Complete Unified Voting Ensemble Evaluation
    print("\n>>> PHASE 3: COMBINED VOTING ENSEMBLE SYSTEM VERIFICATION")
    ensemble_predictions = predict_model(ensemble_model, verbose=False)
    
    y_true_combined = ensemble_predictions['class'].values
    y_pred_combined = ensemble_predictions['prediction_label'].values

    testing_accuracy = accuracy_score(y_true_combined, y_pred_combined)
    combined_f1 = f1_score(y_true_combined, y_pred_combined, average='macro')

    metrics_summary["Model"].append("Voting Ensemble (Team)")
    metrics_summary["Testing Accuracy"].append(testing_accuracy)
    metrics_summary["Macro F1-Score"].append(combined_f1)

    print("\n" + "-"*60)
    print("             FINAL SYSTEM HOLDOUT REPORT (UNSEEN DATA)        ")
    print("-"*60)
    print(f" Combined Ensemble System Testing Accuracy : {testing_accuracy:.4f}")
    print(f" Combined Ensemble System Macro F1-Score   : {combined_f1:.4f}")
    print("\nFinal Joint Clinical Classification Matrix Summary:")
    print(classification_report(y_true_combined, y_pred_combined, target_names=['Healthy', 'Parkinson\'s']))
    print("-"*60)

    # 5. PHASE 4: Export Comparative System Visuals & ROC Analytics
    print("\n>>> PHASE 4: EXPORTING VISUALIZATION GRAPH ARTIFACTS")
    save_custom_confusion_matrix(y_true_combined, y_pred_combined, 'Combined Ensemble Team', 'confusion_matrix_combined.png')
    plot_model_comparison(metrics_summary)
    plot_roc_curves(base_experts, expert_names, ensemble_model, X_test_transformed, y_test)
    
    print("\n[INFO] Comprehensive project report diagrams saved to 'artifacts/plots/' folder.")
    print("="*75 + "\n")