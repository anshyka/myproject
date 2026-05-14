import os
import logging
import joblib
from src.preprocessing import ParkinsonsDataLoader
from src.benchmarking import run_benchmarking, compare_models_clinical
from src.tuner import fine_tune_model
from src.evaluation import evaluate_and_plot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_pipeline(data_path: str) -> None:
    logger.info("Initializing Parkinson's Disease ML pipeline execution.")

    try:
        # 1. Load and Validate Data
        loader = ParkinsonsDataLoader()
        df = loader.load_data(data_path)
        loader.validate_schema(df)
        logger.info(f"Data ingested successfully. Shape: {df.shape}")

        # 2. Benchmarking (Initializes PyCaret & Feature Selection)
        run_benchmarking(df)

        # 3. Get Top 3 Baseline Models (Excluding 'dummy')
        top_3_baselines = compare_models_clinical()
        logger.info(f"Top 3 candidates selected. Starting individual tuning.")

        tuned_models = []

        # 4. Loop through each model to tune them individually
        for i, model in enumerate(top_3_baselines):
            model_name = type(model).__name__
            logger.info(f"Tuning Model {i+1}/3: {model_name}")
            
            try:
                tuned_candidate = fine_tune_model(model)
                tuned_models.append(tuned_candidate)
            except Exception as tune_err:
                logger.warning(f"Failed to tune {model_name}: {tune_err}")

        # 5. Pick the ultimate winner
        if not tuned_models:
            raise ValueError("No models were successfully tuned.")
        
        final_winner = tuned_models[0]
        logger.info(f"Champion model selected: {type(final_winner).__name__}")

        # 6. Evaluate and Plot Results
        evaluate_and_plot(final_winner)

        # 7. Artifact Export
        os.makedirs('artifacts/models', exist_ok=True)
        export_path = 'artifacts/models/parkinsons_model_v1.joblib'

        # Metadata for clinical tracking
        feature_names = df.drop(columns=['class', 'id'], errors='ignore').columns.tolist()
        artifact = {
            'model_pipeline': final_winner,
            'feature_names': feature_names
        }

        joblib.dump(artifact, export_path)
        logger.info(f"Pipeline complete. Artifact saved to: {export_path}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_pipeline('data/raw/pd_speech_features.csv')