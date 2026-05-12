from pycaret.classification import tune_model


def fine_tune_recall(model):
    print("Tuning model for F1... (This may take a moment)")
    tuned_model = tune_model(
        estimator=model,
        optimize='F1',
        search_library='optuna',
        n_iter=50,
        choose_better=True,
        verbose=True  # <--- Change this to True
    )
    return tuned_model