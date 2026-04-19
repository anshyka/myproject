from pycaret.classification import tune_model


def fine_tune_recall(model):
    """
    Fine-tunes the baseline model using Optuna.
    Optimizes strictly for Recall to minimize False Negatives.
    """
    print("Tuning model for Recall... (This may take a moment)")

    tuned_model = tune_model(
        estimator=model,
        optimize='Recall',
        search_library='optuna',
        n_iter=50,  # This safely bounds the time on its own
        choose_better=True,
        verbose=False
    )

    return tuned_model