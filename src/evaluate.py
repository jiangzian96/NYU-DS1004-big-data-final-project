#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' evaluate
Usage:
    $ spark-submit evaluate.py hdfs:/user/zj444 10 0.1 0.05
'''

import pyspark.sql.functions as F
from pyspark.ml.recommendation import ALS, ALSModel
import sys
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics


def main(spark, rank, regParam, path, fraction):
    TEMP_PATH = "/models/ALS_{}_{}_{}".format(rank, regParam, fraction)
    ALS_PATH = TEMP_PATH + "/als"
    MODEL_PATH = TEMP_PATH + "/als_model"
    print("Loading model...")
    als = ALS.load(path + ALS_PATH)
    model = ALSModel.load(path + MODEL_PATH)
    print("Loading data...")
    validation = spark.read.parquet("{}/data/processed/validation_{}.parquet".format(path, fraction))
    validation.createOrReplaceTempView("validation")

    # RMSE
    predictions = model.transform(validation)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("RSME:", rmse)
    '''
    predictions = model.recommendForAllUsers(500)
    predictions.createOrReplaceTempView("predictions")
    groundtruth = validation.groupby("user_id").agg(F.collect_set("book_id").alias('groundtruth'))
    groundtruth.createOrReplaceTempView("groundtruth")
    total = spark.sql("SELECT g.user_id, g.groundtruth AS groundtruth, p.recommendations AS predictions FROM groundtruth g JOIN predictions p ON g.user_id = p.user_id")
    total.createOrReplaceTempView("total")

    data = total.selectExpr("predictions.book_id", "groundtruth")
    print("df to rdd...")
    rdd = data.rdd.map(tuple)
    print("creating metrics...")
    metrics = RankingMetrics(rdd)
    print("meanAveragePrecision:", metrics.meanAveragePrecision)
    print("precision at 500:", metrics.precisionAt(500))
    print("ndcgAt 500:", metrics.ndcgAt(500))
    '''


if __name__ == "__main__":
    memory = "8g"
    spark = SparkSession.builder.appName("evaluate").config("spark.executor.memory", memory).config("spark.driver.memory", memory).getOrCreate()
    path = sys.argv[1]
    rank = int(sys.argv[2])
    regParam = float(sys.argv[3])
    fraction = sys.argv[4]
    main(spark, rank, regParam, path, fraction)
