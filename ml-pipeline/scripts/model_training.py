import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def model_training(train_path, test_path, model_path):
    spark = SparkSession.builder.appName("ModelTraining").getOrCreate()

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("churn-customer")

    # Load train and test data
    train_data = spark.read.parquet(train_path)
    test_data = spark.read.parquet(test_path)

    mlflow.spark.autolog(log_input_examples=True, log_model_signatures=True, silent=True)

    with mlflow.start_run(run_name=f'LogReg'):

        # Train logistic regression model
        lr = LogisticRegression(labelCol="Churn", featuresCol="features")
        model = lr.fit(train_data)

        feature_df = train_data.select("features").limit(5).toPandas()
        prediction_df = predictions.select("prediction").limit(5).toPandas()
        signature = infer_signature(feature_df, prediction_df)

        # Evaluate the model
        evaluator = BinaryClassificationEvaluator(labelCol="Churn", metricName="areaUnderROC")
        predictions = model.transform(test_data)
        auc = evaluator.evaluate(predictions)
        print(f"AUC: {auc}")
        mlflow.log_metric("auc", auc)

        mlflow.spark.log_model(
            artifact_path='Model',
            spark_model=model,
            signature=signature,
            input_example=train_data[:5],
            code_paths=['model_training'],
            registered_model_name="churn_log_reg"
        )

    # Save the model to artifact TODO
    # model.save(model_path)
    
    spark.stop()


if __name__ == "_main_":
    import sys
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    model_path = sys.argv[3]
    model_training(train_path, test_path, model_path)