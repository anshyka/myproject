from pycaret.classification import tune_model


def fine_tune_recall(model):
    print("Tuning model for F1... (This may take a moment)")
    tuned_model = tune_model(
        estimator=model,
        optimize='F1',  # Changed from 'Recall'
        search_library='optuna',
        n_iter=50,
        choose_better=True,
        verbose=False
    )
    return tuned_model