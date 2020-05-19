#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' save data to parquet
Usage:
    $ spark-submit to_parquet.py hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv hdfs:/user/zj444
'''

import sys

from pyspark.sql import SparkSession


def main(spark, data_path, output_dir):
    df = spark.read.csv(data_path, header=True, schema="user_id INT, book_id INT, is_read INT, rating INT, is_reviewed INT")
    df.write.parquet("{}/data/raw/interactions.parquet".format(output_dir), mode="overwrite")


if __name__ == "__main__":
    spark = SparkSession.builder.appName("save csv to parquet").getOrCreate()
    data_path = sys.argv[1]
    output_dir = sys.argv[2]
    main(spark, data_path, output_dir)
