# Parkinson's Disease Audio Detection: ML Pipeline Blueprint

This document outlines a step-by-step, test-driven approach to building a medical-grade ML pipeline for Parkinson's Disease detection. The process is broken down into five iterative phases, each with a specific prompt designed for a code-generation LLM.

---

## 🏗️ Project Architecture
The pipeline follows a modular structure to ensure maintainability and testability:
* **Data Layer:** Schema validation and ID-based splitting.
* **Validation Layer:** Strict Group K-Fold to prevent patient-level data leakage.
* **AutoML Layer:** Rapid benchmarking using PyCaret.
* **Optimization Layer:** Optuna-driven hyperparameter tuning focusing on **Recall**.
* **Export Layer:** Model and preprocessing artifact serialization.

---

## 🚀 Iterative Implementation Prompts

### Prompt 1: Project Setup & Data Ingestion
**Goal:** Initialize the environment and create a robust, validated data loading mechanism.

> `PROMPT:`
> "Act as a Senior ML Engineer. We are building a Parkinson's Disease detection pipeline based on the Oxford dataset. 
> 
> **Task 1:** Create the project structure:
> - `src/preprocessing.py`, `src/__init__.py`
> - `tests/test_preprocessing.py`
> - `requirements.txt` (include: pandas, scikit-learn, pytest)
> 
> **Task 2:** In `src/preprocessing.py`, implement `ParkinsonsDataLoader`:
> - `load_data(path)`: Loads the CSV.
> - `validate_schema(df)`: Ensures 'status' (target) and 'name' (subject ID) exist. Raise `ValueError` if missing.
> - `get_feature_arrays(df)`: Separates X and y. Critically, ensure the 'name' column is extracted as a `groups` array but is **removed** from the feature set `X`.
> 
> **Task 3:** In `tests/test_preprocessing.py`, write tests for:
> - Handling missing columns.
> - Ensuring the 'name' column never ends up in the feature matrix `X`.
> 
> Follow PEP8 and use type hinting."

---

### Prompt 2: Leakage-Proof Cross-Validation
**Goal:** Implement the primary clinical constraint—Group K-Fold validation.

> `PROMPT:`
> "Building on the previous code, we need to implement the data leakage prevention logic.
> 
> **Task 1:** In `src/preprocessing.py`, implement a `DataSplitter` class that uses `sklearn.model_selection.GroupKFold`. 
> 
> **Task 2:** Create a critical unit test in `tests/test_preprocessing.py` called `test_patient_leakage`. 
> - Create a synthetic dataset where one 'name' (Patient ID) has 5 different rows.
> - Run the `GroupKFold` split.
> - Assert that for every fold, the set of 'name' IDs in the training set and the set of 'name' IDs in the validation set have **zero** intersection.
> 
> **Task 3:** Add a `PreProcessor` class that wraps `StandardScaler`. It must be able to fit on training data and transform validation data without leakage."

---

### Prompt 3: AutoML Benchmarking (PyCaret)
**Goal:** Use automated tools to find the best baseline model while respecting patient groups.

> `PROMPT:`
> "We are now adding the AutoML benchmarking phase using PyCaret.
> 
> **Task 1:** Update `requirements.txt` to include `pycaret`.
> 
> **Task 2:** Create `src/benchmarking.py`. Implement `run_benchmarking(df)`:
> - Initialize PyCaret `setup()`.
> - **Mandatory:** Set `groups='name'` and `fold_strategy='groupkfold'` in the setup.
> - Target is 'status'.
> 
> **Task 3:** Implement `compare_models_clinical()`:
> - Benchmarks: Random Forest, XGBoost, SVM, and Logistic Regression.
> - Sort the leaderboard by **Recall** (clinical priority).
> - Return the top-performing model object.
> 
> **Task 4:** Add a test to verify that the PyCaret setup correctly identifies the 'name' column as the grouping variable."

---

### Prompt 4: Optuna Hyperparameter Tuning
**Goal:** Optimize the champion model specifically to minimize False Negatives.

> `PROMPT:`
> "We will now fine-tune the winning model from the previous step.
> 
> **Task 1:** In `src/tuner.py`, implement `fine_tune_recall(model)`:
> - Use PyCaret's `tune_model` with `search_library='optuna'`.
> - The objective function must be optimized strictly for **Recall**.
> - Set `n_iter=50` and include a `timeout` of 300 seconds.
> 
> **Task 2:** Implement an evaluation function in `src/evaluation.py`:
> - Generate a `classification_report`.
> - Generate and save a Feature Importance plot (or SHAP) to show which vocal biomarkers (e.g., PPE, Jitter) are most predictive.
> 
> **Task 3:** Add a test that ensures the tuned model's recall is not lower than the baseline model's recall."

---

### Prompt 5: Pipeline Integration & Artifact Export
**Goal:** Wire the components into a single execution script and serialize the output.

> `PROMPT:`
> "Final Step: Integrate all components into a production-ready pipeline.
> 
> **Task 1:** Create `src/main.py` as the entry point. It should:
> 1. Load and Validate data.
> 2. Perform scaling.
> 3. Run the GroupKFold benchmarking.
> 4. Tune the best model for Recall.
> 5. Log the final clinical metrics.
> 
> **Task 2:** Artifact Export:
> - Use `joblib` to save a dictionary containing: the tuned model, the fitted scaler, and the list of feature names.
> - Name the file `parkinsons_model_v1.joblib`.
> 
> **Task 3:** Add an integration test in `tests/test_integration.py` that runs the full pipeline on a dummy dataset and checks if the `.joblib` file was created and is loadable."

---

## ✅ Final Checklist for Implementation
- [ ] **Recall > 0.90:** Priority over accuracy.
- [ ] **No Leakage:** Patient IDs never cross-pollinate folds.
- [ ] **Interpretability:** Feature importance plots generated.
- [ ] **Standardization:** All acoustic features scaled.