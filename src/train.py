#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' train
Usage:
    $ spark-submit train.py hdfs:/user/zj444 10 0.1 0.05
'''

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import sys


def main(spark, rank, regParam, path, fraction):
    train = spark.read.parquet("{}/data/processed/train_{}.parquet".format(path, fraction))
    als = ALS(rank=rank, maxIter=5, seed=42, regParam=regParam,
              userCol='user_id', itemCol='book_id', ratingCol='rating',
              coldStartStrategy="drop")
    print("Training ALS model with rank {} and regularization {} with {} of data...".format(rank, regParam, fraction))
    model = als.fit(train)
    temp_path = "/ALS_{}_{}_{}".format(rank, regParam, fraction)
    als_path = temp_path + "/als"
    print("Saving model...")
    als.save(path + "/models" + als_path)
    model_path = temp_path + "/als_model"
    model.save(path + "/models" + model_path)


if __name__ == "__main__":
    memory = "8g"
    spark = SparkSession.builder.appName("train").config("spark.executor.memory", memory).config("spark.driver.memory", memory).getOrCreate()
    path = sys.argv[1]
    rank = int(sys.argv[2])
    regParam = float(sys.argv[3])
    fraction = sys.argv[4]
    main(spark, rank, regParam, path, fraction)
