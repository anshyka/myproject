# Technical Specification: Parkinson's Disease Audio Detection ML Pipeline

## 1. Project Overview
This project aims to build a robust, medical-grade machine learning training pipeline to detect Parkinson's Disease (PD) using pre-extracted acoustic features. The pipeline emphasizes automated benchmarking, strict data leakage prevention through patient-level grouping, and hyperparameter optimization geared specifically toward maximizing clinical recall. 

**Scope:** Model training, evaluation, and serialization only (no frontend/dashboard deployment in this phase).

## 2. Technology Stack & Architecture
* **Language:** Python 3.9+
* **Core Libraries:** `pandas`, `numpy`, `scikit-learn`
* **AutoML Framework:** `pycaret` (for automated model benchmarking)
* **Hyperparameter Tuning:** `optuna` (or PyCaret's native Optuna integration)
* **Testing:** `pytest`
* **Artifact Serialization:** `joblib` or `pickle`

## 3. Dataset Requirements & Handling
* **Source:** Oxford Parkinson's Disease Dataset (Tabular/CSV format).
* **Data Type:** Pre-extracted vocal biomarkers (e.g., MDVP:Fo(Hz), MDVP:Jitter(%), MDVP:Shimmer, HNR, PPE).
* **Target Variable:** `status` (1 = Parkinson's, 0 = Healthy).
* **Grouping Variable:** `name` or `subject_id` (Critical for cross-validation).
* **Data Preprocessing:**
    * Drop non-predictive metadata (retain the Patient ID column strictly for grouping during cross-validation; it must be excluded from the actual feature set `X` during training).
    * Apply standard scaling (`StandardScaler`) to normalize the wide variance in acoustic measurements (e.g., fundamental frequency vs. micro-jitter percentages).

## 4. Pipeline Execution Phases

### Phase 1: Data Ingestion & Preprocessing
1. **Load Data:** Ingest the CSV dataset using Pandas.
2. **Sanity Checks:** Verify the existence of the target column (`status`) and the grouping column (`name`/`subject_id`).
3. **Scaling:** Apply feature scaling to all numeric columns excluding the target and ID columns.

### Phase 2: AutoML Benchmarking
1. **Framework Setup:** Initialize the PyCaret classification environment using `setup()`.
2. **Cross-Validation Strategy:** Implement **Group K-Fold Cross-Validation** (e.g., `k=5` or `k=10`). 
    * **Strict Constraint:** The pipeline MUST group by the Patient ID column. No single patient's recordings can exist in both the training fold and the validation fold simultaneously. This is mandatory to prevent data leakage.
3. **Model Comparison:** Execute `compare_models()` to train and evaluate baseline algorithms. The search space must explicitly include:
    * Random Forest
    * XGBoost
    * Support Vector Machines (SVM)
    * Logistic Regression
4. **Selection:** Automatically extract the top-performing model from the leaderboard for the tuning phase.

### Phase 3: Advanced Hyperparameter Tuning
1. **Optimization Engine:** Pass the winning model from Phase 2 into a deep hyperparameter tuning sweep using Optuna.
2. **Objective Function:** The tuner MUST be configured to optimize strictly for **Recall** (minimizing False Negatives). 
3. **Search Space Definition:** Define appropriate parameter grids based on the winning algorithm (e.g., `max_depth`, `learning_rate`, `n_estimators` for tree-based models; `C` and `gamma` for SVMs). Allow the tuner to run for a defined number of trials (e.g., `n_trials=100`).

### Phase 4: Evaluation & Export
1. **Final Evaluation:** Generate a comprehensive classification report on the hold-out test set using the final tuned model. Output metrics must include:
    * Recall (Primary metric)
    * F1-Score
    * Precision
    * Overall Accuracy
    * Confusion Matrix
2. **Interpretability:** Generate a Feature Importance plot (or SHAP values) to highlight which specific acoustic biomarkers are driving the predictions.
3. **Artifact Export:** Serialize the final tuned model, the scaler object, and the feature column names to a `.pkl` or `.joblib` file for future inference deployment.

## 5. Error Handling Strategies
The pipeline must include `try/except` blocks and logging for the following critical failure points:
* **File I/O Errors:** Catch `FileNotFoundError` if the Oxford dataset CSV is missing from the specified directory and log a clear terminal warning.
* **Missing Columns:** Raise a custom `ValueError` if the `status` or `name`/`subject_id` columns are missing from the ingested dataframe before training begins.
* **Convergence Warnings:** Suppress or log `ConvergenceWarning` (common with Logistic Regression/SVMs on unscaled data), ensuring the script gracefully continues execution.
* **Tuning Timeouts:** Implement a timeout threshold for the Optuna study to prevent infinite loops if the search space is misconfigured.

## 6. Testing Plan
The developer must include a `tests/` directory with `pytest` scripts verifying the following:
* **Data Leakage Test (Critical):** A unit test that intercepts the Group K-Fold split and asserts that the intersection of Patient IDs in `train_folds` and `val_folds` is exactly zero (`len(set(train_ids).intersection(set(val_ids))) == 0`).
* **Shape Validation Test:** Assert that the number of features entering the model exactly matches the number of features outputted by the scaler.
* **Target Distribution Test:** Assert that both class 0 and class 1 are present in the training data to prevent training a completely skewed/dead model.
* **Export Verification Test:** Assert that the `.pkl` artifact is successfully created on disk after a mock training run.
*