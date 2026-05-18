import os
import logging
import joblib
from src.preprocessing import ParkinsonsDataLoader
from src.benchmarking import run_benchmarking, compare_models_clinical
from src.evaluation import evaluate_and_plot
from pycaret.classification import blend_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_pipeline(data_path: str) -> None:
    logger.info("Initializing Parkinson's Disease Ensemble ML pipeline execution.")

    try:
        # 1. Load and Validate Data
        loader = ParkinsonsDataLoader()
        df = loader.load_data(data_path)
        loader.validate_schema(df)
        logger.info(f"Data ingested successfully. Shape: {df.shape}")

        # 2. Benchmarking (Initializes PyCaret & Feature Selection)
        run_benchmarking(df)

        # 3. Get Top 3 Probability-Capable Models sorted by F1
        top_3_baselines = compare_models_clinical()
        logger.info("Top 3 F1-optimized models selected successfully.")

        # 4. Implement Model Ensembling via Blending optimized for F1
        logger.info("Blending top 3 models into a Soft-Voting Ensemble Classifier...")
        ensemble_winner = blend_models(
            estimator_list=top_3_baselines, 
            method='soft', 
            optimize='F1'  # Dynamically tunes voting weights based on F1-Score performance
        )
        logger.info(f"Ensemble pipeline compiled successfully: {type(ensemble_winner).__name__}")

        # 5. Evaluate and Plot Results on Holdout Set
        logger.info("Evaluating ensemble model on holdout dataset...")
        evaluate_and_plot(ensemble_winner)

        # 6. Artifact Export
        os.makedirs('artifacts/models', exist_ok=True)
        export_path = 'artifacts/models/parkinsons_ensemble_model_v1.joblib'

        feature_names = df.drop(columns=['class', 'id'], errors='ignore').columns.tolist()
        artifact = {
            'model_pipeline': ensemble_winner,
            'feature_names': feature_names
        }

        joblib.dump(artifact, export_path)
        logger.info(f"Pipeline execution complete. Artifact saved to: {export_path}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_pipeline('data/raw/pd_speech_features.csv')