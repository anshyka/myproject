import os
import pandas as pd
from sklearn.metrics import classification_report
from pycaret.classification import predict_model, plot_model


def evaluate_and_plot(model):
    """
    Evaluates the model on the holdout set and saves a feature importance plot.
    """
    print("\nEvaluating model on Holdout Set...")

    # predict_model without data automatically scores the PyCaret holdout set
    predictions = predict_model(model, verbose=False)

    print("\nClassification Report (Holdout Set):")
    print(classification_report(predictions['status'], predictions['prediction_label']))

    # Create artifacts directory if it doesn't exist
    os.makedirs('artifacts/plots', exist_ok=True)

    # Generate Feature Importance Plot
    print("\nGenerating Feature Importance Plot...")
    try:
        # PyCaret saves the plot in the current working directory by default
        plot_model(model, plot='feature', save=True)

        # Move it to our artifacts folder
        if os.path.exists('Feature Importance.png'):
            os.replace('Feature Importance.png', 'artifacts/plots/feature_importance.png')
            print("Plot saved to artifacts/plots/feature_importance.png")
    except Exception as e:
        print(f"Could not generate feature importance plot for this model type: {e}")