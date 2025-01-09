from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("TelcoCustomerChurnExperimentation") \
    .getOrCreate()

# Load the dataset
data_path = "dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Data preprocessing
# Handle missing values
df = df.na.fill({"TotalCharges": 0})

# Convert categorical variables to numeric using StringIndexer
categorical_columns = [col for col in df.columns if df.select(col).dtypes[0][1] == 'string' and col != "Churn"]
for column in categorical_columns:
    indexer = StringIndexer(inputCol=column, outputCol=f"{column}_indexed", handleInvalid="skip")
    df = indexer.fit(df).transform(df)

# Target encoding
label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
df = label_indexer.fit(df).transform(df)

# Feature assembly
feature_columns = [f"{col}_indexed" if col in categorical_columns else col for col in df.columns if col not in ["Churn", "label"]]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(df)

# Train-test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Experimentation with multiple models
# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

# Hyperparameter tuning using ParamGridBuilder and CrossValidator
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=3)

# Train the model
cv_model = crossval.fit(train_data)

# Best model and evaluation
best_model = cv_model.bestModel
predictions = best_model.transform(test_data)
auc = evaluator.evaluate(predictions)

print(f"Best Model AUC: {auc}")
print(f"Best Model Parameters: regParam={best_model._java_obj.getRegParam()}, "
      f"elasticNetParam={best_model._java_obj.getElasticNetParam()}")

# Additional experimentation: Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)
rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)
rf_auc = evaluator.evaluate(rf_predictions)

print(f"Random Forest AUC: {rf_auc}")

# Stop Spark session
spark.stop()
