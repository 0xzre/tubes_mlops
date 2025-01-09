# Use Apache Airflow official image
FROM apache/airflow:2.10.4-python3.9

# Install necessary libraries
USER root
RUN apt-get update && apt-get install -y gcc libpq-dev procps default-jre

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/opt/airflow
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

ARG HADOOP_VERSION=2.10.1
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=/etc/hadoop
ENV MULTIHOMED_NETWORK=1
ENV USER=root

RUN HADOOP_URL="https://archive.apache.org/dist/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz" \
    && curl 'https://dist.apache.org/repos/dist/release/hadoop/common/KEYS' | gpg --import - \
    && curl -fSL "$HADOOP_URL" -o /tmp/hadoop.tar.gz \
    && curl -fSL "$HADOOP_URL.asc" -o /tmp/hadoop.tar.gz.asc \
    && gpg --verify /tmp/hadoop.tar.gz.asc \
    && mkdir -p "${HADOOP_HOME}" \
    && tar -xvf /tmp/hadoop.tar.gz -C "${HADOOP_HOME}" --strip-components=1 \
    && rm /tmp/hadoop.tar.gz /tmp/hadoop.tar.gz.asc \
    && ln -s "${HADOOP_HOME}/etc/hadoop" /etc/hadoop \
    && mkdir "${HADOOP_HOME}/logs" \
    && mkdir /hadoop-data

ENV PATH="$HADOOP_HOME/bin/:$PATH"

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