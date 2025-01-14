# Customer Churn Prediction MLOps Pipeline 🤓

An end-to-end MLOps pipeline for customer churn prediction, designed for scalability and reproducibility. This project is developed as part of the requirements for **X-Ops IF4054** 👽. The solution leverages **Apache Spark** for big data processing, **Apache Airflow** for workflow orchestration, and **MLflow** for experiment tracking and model management. The project includes workflows for data preprocessing, drift detection, retraining, and deployment tracking.

---

## Overview

A scalable, reproducible, and automated pipeline to predict customer churn for a telecom dataset, addressing key MLOps/DataOps challenges like data drift monitoring and model retraining.

### Key Features
1. **Data Cleanup Pipeline**:
   - Handles nulls and empty strings.
   - Performs feature engineering using Apache Spark.

2. **Drift Simulation and Monitoring**:
   - Simulates data drift using custom workflows.
   - Utilizes **Population Stability Index (PSI)** for drift detection.

3. **Automated Model Retraining**:
   - Monitors drift and triggers retraining workflows in Apache Airflow.
   - Tracks model performance and lifecycle using MLflow.

4. **Deployment and Monitoring**:
   - Models deployed and served with Docker.
   - Includes CI/CD workflows via GitLab for streamlined deployment.

---

## Dataset

**Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/code)

- **Target Variable**: `Churn`


---

## Tools and Technologies

- **Apache Spark**: Data preprocessing and feature engineering.
- **Apache Airflow**: Workflow orchestration and scheduling.
- **MLflow**: Experiment tracking, model registry, and deployment.
- **Docker**: Containerization.
- **GitLab CI/CD**: Continuous integration and deployment.
- **HDFS**: Distributed storage for datasets and models.
- **FastAPI**: REST API for model serving.
- **Prometheus & Grafana**: Monitoring and visualization for deployed models in ML Server.

---

## Installation and Setup

### Prerequisites

Ensure the following are installed on your system:
- Docker

### Running with Docker

1. Build and run the services (Might take 1 hrs):
   ```bash
    docker-compose up --build -d
    ```
2. Access the Airflow UI at `http://localhost:8080` and MLflow UI at `http://localhost:5001`.
3. Run the Airflow DAGs at Airflow UI or configure DAG on mounted `ml-pipeline/dags` folder.
4. Hit the ML Server at `http://localhost:8000` for model serving.

---

## Workflow Details

### 1. Data Cleanup Pipeline
- **Purpose**: Handle missing values, empty strings, and data preprocessing.
- **Implementation**: 
  - Apache Spark is used for data transformations and cleaning.
  - The pipeline ensures all features are appropriately scaled and encoded for modeling.

### 2. Drift Simulation
- **Purpose**: Simulate and monitor data drift.
- **Implementation**: 
  - Custom workflows simulate drifted datasets to test model robustness.
  - **Population Stability Index (PSI)** is calculated to detect significant shifts in data distribution.

### 3. Drift Monitoring and Automated Retraining
- **Purpose**: Detect drift in real-time and trigger retraining workflows.
- **Implementation**:
  - Apache Airflow monitors data and model performance.
  - When PSI exceeds a threshold, an automated retraining process is triggered.
  - The retrained models are versioned and logged in MLflow.

### 4. Deployment and Monitoring
- **Purpose**: Serve models and track performance.
- **Implementation**:
  - Models are containerized using Docker.
  - Monitoring systems track key performance metrics and ensure model health.
  - CI/CD workflows with GitLab manage seamless deployments and updates.

---

## CI/CD Pipeline

The CI/CD pipeline automates:
- Testing of the codebase especially the ML pipeline to ensure quality.
- ML Model training and deployment.

---

## Future Enhancements

- Use Kubernetes for container orchestration.
- Add support for real-time data streaming for near-instant drift detection.
- Building and pushing Docker images to a container registry for CD.

---

## Contributors

| Name                        | NIM      | Contributions                                                      |
|-----------------------------|----------|--------------------------------------------------------------------|
| Rizky Abdillah Rasyid       | 13521109 | MLFlow setup, ML models registry, Data drift simulation and detection.                            |
| Muhammad Abdul Aziz Ghazali | 13521128 | Apache Airflow Deployment, Container environment and orchestration, ML Model serving and Continous Delivery |
| Muhammad Zaki Amanullah     | 13521128 | Experimentation of machine learning model, codebase CI/CD pipeline.             |