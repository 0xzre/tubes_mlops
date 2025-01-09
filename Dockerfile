# Use Apache Airflow official image
FROM apache/airflow:2.10.4-python3.9

# Install necessary libraries
USER root
RUN apt-get update && apt-get install -y gcc libpq-dev

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/opt/airflow

# Copy DAGs and scripts
COPY dags/ $AIRFLOW_HOME/dags/
COPY scripts/ $AIRFLOW_HOME/scripts/

# Set permissions
RUN chown -R airflow: $AIRFLOW_HOME/dags/ $AIRFLOW_HOME/scripts/

# Switch to Airflow user
USER airflow

RUN pip install --upgrade pip && \
pip install pandas numpy mlflow pyspark scikit-learn boto3 apache-airflow apache-airflow-providers-apache-spark 

# Set the entrypoint
# ENTRYPOINT ["airflow"]
# CMD ["webserver"]
WORKDIR $AIRFLOW_HOME