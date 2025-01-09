from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def model_training(train_path, test_path, model_path):
    spark = SparkSession.builder.appName("ModelTraining").getOrCreate()

    # Load train and test data
    train_data = spark.read.parquet(train_path)
    test_data = spark.read.parquet(test_path)

    # Train logistic regression model
    lr = LogisticRegression(labelCol="Churn", featuresCol="features")
    model = lr.fit(train_data)

    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(labelCol="Churn", metricName="areaUnderROC")
    predictions = model.transform(test_data)
    auc = evaluator.evaluate(predictions)
    print(f"AUC: {auc}")

    # Save the model
    model.save(model_path)
    
    spark.stop()


if __name__ == "__main__":
    import sys
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    model_path = sys.argv[3]
    model_training(train_path, test_path, model_path)
