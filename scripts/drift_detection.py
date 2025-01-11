import pandas as pd
import numpy as np
import mlflow

def calculate_psi(expected, actual, bins=10):
    expected_percents, bins = np.histogram(expected, bins=bins)
    actual_percents, _ = np.histogram(actual, bins=bins)

    expected_percents = expected_percents / sum(expected_percents)
    actual_percents = actual_percents / sum(actual_percents)

    psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi

def drift_detection(input_path, train_path, test_path):
    # Load data
    baseline_data = pd.read_parquet("/opt/airflow/data/baseline_data.parquet")
    new_data = pd.read_parquet("/opt/airflow/data/processed_data.parquet")

    # Calculate PSI
    psi = calculate_psi(baseline_data['features'], new_data['features'])

    # Log PSI to MLflow
    mlflow.set_experiment("Customer Churn Prediction")
    with mlflow.start_run():
        mlflow.log_metric("psi", psi)

    # Trigger retraining if PSI exceeds threshold
    if psi > 0.2:
        print("Drift detected. Triggering model retraining...")
    else:
        print("No significant drift detected.")

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1]
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    drift_detection(input_path, train_path, test_path)