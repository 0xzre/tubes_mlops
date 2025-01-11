#!/bin/bash
docker cp dataset namenode:/tmp
docker exec -it namenode bash -c "hdfs dfs -mkdir -p /dataset"
docker exec -it namenode bash -c "hdfs dfs -put /tmp/dataset dataset"