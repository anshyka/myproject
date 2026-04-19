import os
import logging
import joblib
from src.preprocessing import ParkinsonsDataLoader
from src.benchmarking import run_benchmarking, compare_models_clinical
from src.tuner import fine_tune_recall
from src.evaluation import evaluate_and_plot

# Configure logging for production-grade output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_pipeline(data_path: str) -> None:
    """
    Executes the end-to-end Parkinson's detection machine learning pipeline.
    Includes data ingestion, schema validation, AutoML benchmarking, 
    hyperparameter tuning, evaluation, and artifact serialization.
    """
    logger.info("Initializing Parkinson's Disease ML pipeline execution.")

    try:
        # 1. Load and Validate Data
        logger.info(f"Loading dataset from: {data_path}")
        loader = ParkinsonsDataLoader()
        df = loader.load_data(data_path)
        loader.validate_schema(df)
        logger.info(f"Data ingested and schema validated successfully. Shape: {df.shape}")

        # 2. Benchmarking
        logger.info("Starting AutoML benchmarking with strict GroupKFold constraints.")
        run_benchmarking(df)

        # 3. Find Best Baseline
        best_baseline = compare_models_clinical()
        logger.info(f"Baseline model selection complete. Champion model: {type(best_baseline).__name__}")

        # 4. Fine-Tune for Clinical Recall
        logger.info("Initiating Optuna hyperparameter tuning optimizing for Recall.")
        tuned_model = fine_tune_recall(best_baseline)
        logger.info("Hyperparameter tuning completed successfully.")

        # 5. Evaluate and Generate Interpretability Plots
        logger.info("Evaluating tuned model on the holdout set.")
        evaluate_and_plot(tuned_model)

        # 6. Artifact Export
        os.makedirs('artifacts/models', exist_ok=True)
        export_path = 'artifacts/models/parkinsons_model_v1.joblib'

        feature_names = df.drop(columns=['status', 'name']).columns.tolist()

        artifact = {
            'model_pipeline': tuned_model,
            'feature_names': feature_names
        }

        joblib.dump(artifact, export_path)
        logger.info(f"Pipeline execution complete. Artifact serialized to: {export_path}")

    except Exception as e:
        logger.error(f"Pipeline execution failed due to an error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Target dataset path
    run_pipeline('data/raw/parkinsons_classification_data.csv')