from pycaret.classification import tune_model


def fine_tune_model(model):
    print("Tuning model for MCC... (This may take a moment)")
    tuned_model = tune_model(
        estimator=model,
        optimize='MCC',
        search_library='optuna',
        n_iter=50,
        choose_better=True,
        verbose=True  # <--- Change this to True
    )
    return tuned_model