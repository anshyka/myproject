import joblib
import pandas as pd
from pycaret.classification import predict_model


def run_inference(patient_data: dict, model_path: str = 'artifacts/models/parkinsons_model_v1.joblib'):
    """
    Loads the serialized model and makes a live clinical prediction.
    """
    print("Loading clinical model artifact...")

    # Load the artifact (which contains both the model and the expected feature names)
    try:
        artifact = joblib.load(model_path)
        model_pipeline = artifact['model_pipeline']
        feature_names = artifact['feature_names']
    except FileNotFoundError:
        print(f"Error: Model artifact not found at {model_path}. Run main.py first.")
        return

    # Convert the single patient dictionary into a DataFrame using the exact schema
    # the model expects, dropping any extraneous data.
    df_live = pd.DataFrame([patient_data])[feature_names]

    print("Analyzing vocal biomarkers...\n")

    # PyCaret handles the internal scaling and transformation automatically
    prediction = predict_model(model_pipeline, data=df_live, verbose=False)

    # PyCaret 3.x outputs predictions in these specific columns
    result_class = prediction['prediction_label'].iloc[0]
    confidence = prediction['prediction_score'].iloc[0] * 100

    diagnosis = "Parkinson's Disease Detected" if result_class == 1 else "Healthy (No Parkinson's Detected)"

    print("-" * 50)
    print(f"DIAGNOSIS:  {diagnosis}")
    print(f"CONFIDENCE: {confidence:.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    # This is a sample array representing a new patient's vocal analysis from a clinic.
    # (These specific values actually correspond to a positive PD case in the Oxford dataset).
    new_patient_recording = {
        'MDVP:Fo(Hz)': 119.992,
        'MDVP:Fhi(Hz)': 157.302,
        'MDVP:Flo(Hz)': 74.997,
        'MDVP:Jitter(%)': 0.00784,
        'MDVP:Jitter(Abs)': 0.00007,
        'MDVP:RAP': 0.0037,
        'MDVP:PPQ': 0.00554,
        'Jitter:DDP': 0.01109,
        'MDVP:Shimmer': 0.04374,
        'MDVP:Shimmer(dB)': 0.426,
        'Shimmer:APQ3': 0.02182,
        'Shimmer:APQ5': 0.0313,
        'MDVP:APQ': 0.02971,
        'Shimmer:DDA': 0.06545,
        'NHR': 0.02211,
        'HNR': 21.033,
        'RPDE': 0.414783,
        'DFA': 0.815285,
        'spread1': -4.813031,
        'spread2': 0.266482,
        'D2': 2.301442,
        'PPE': 0.284654
    }

    run_inference(new_patient_recording)