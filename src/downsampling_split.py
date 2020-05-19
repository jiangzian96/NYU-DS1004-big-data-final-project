#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' downsampling and train/val/test split
Usage:
    $ spark-submit downsampling_split.py 0.05 hdfs:/user/zj444
'''

import sys
from pyspark.sql import SparkSession


def main(spark, fraction, path):
    interactions_df = spark.read.parquet("{}/data/raw/interactions.parquet".format(path))
    interactions_df.createOrReplaceTempView("interactions")

    # select 0.1% of users and take all interactions
    print("Sampling {}%% users......".format(fraction*100))
    all_users = spark.sql("SELECT DISTINCT user_id FROM interactions")
    sampled_users = all_users.sample(False, fraction=fraction, seed=42)
    sampled_users.createOrReplaceTempView("sampled_users")

    sampled_interactions = spark.sql("SELECT * FROM interactions WHERE user_id in \
                               (SELECT * FROM sampled_users)")
    sampled_interactions.createOrReplaceTempView("sampled_interactions")

    # filter out users with less than 10 interactions
    print("Filtering out users with less than 10 interactions......")
    count_interactions = spark.sql("SELECT user_id, COUNT(user_id) as count FROM sampled_interactions \
        GROUP BY user_id")
    count_interactions.createOrReplaceTempView("count_interactions")
    filtered_interactions = spark.sql("SELECT interactions.* FROM interactions \
        JOIN count_interactions \
            ON count_interactions.user_id = interactions.user_id \
        WHERE count >= 10")
    filtered_interactions.createOrReplaceTempView("filtered_interactions")

    # train/val/test split
    print("train val test split......")
    all_users = spark.sql("SELECT DISTINCT user_id FROM filtered_interactions")
    train_users = all_users.sample(False, fraction=0.6, seed=42)
    train_users.createOrReplaceTempView("train_users")
    train_interactions = spark.sql("SELECT * FROM filtered_interactions WHERE user_id in \
                                   (SELECT * FROM train_users)")
    train_interactions.createOrReplaceTempView("train_interactions")
    val_test_users = all_users.select('user_id').subtract(train_users.select('user_id'))
    val_users = val_test_users.sample(False, fraction=0.5, seed=42)
    val_users.createOrReplaceTempView("val_users")

    test_users = val_test_users.select('user_id').subtract(val_users.select('user_id'))
    test_users.createOrReplaceTempView("test_users")
    val_interactions = spark.sql("SELECT * FROM filtered_interactions WHERE user_id in \
                                   (SELECT * FROM val_users)")
    val_interactions.createOrReplaceTempView("val_interactions")

    test_interactions = spark.sql("SELECT * FROM filtered_interactions WHERE user_id in \
                                   (SELECT * FROM test_users)")
    test_interactions.createOrReplaceTempView("test_interactions")
    val_users_list = [row["user_id"] for row in val_users.collect()]

    test_users_list = [row["user_id"] for row in test_users.collect()]
    val_fractions = dict(zip(val_users_list, [0.5 for _ in range(len(val_users_list))]))
    test_fractions = dict(zip(test_users_list, [0.5 for _ in range(len(test_users_list))]))
    val_training = val_interactions.sampleBy("user_id", fractions=val_fractions, seed=42)
    val_validation = val_interactions.subtract(val_training)

    test_training = test_interactions.sampleBy("user_id", fractions=test_fractions, seed=42)
    test_validation = test_interactions.subtract(test_training)
    train_interactions = train_interactions.union(val_training)
    train_interactions = train_interactions.union(test_training)

    # save
    print("Saving as parquet files......")
    train_interactions.write.parquet("{}/data/processed/train_{}.parquet".format(path, fraction), mode="overwrite")
    val_validation.write.parquet("{}/data/processed/validation_{}.parquet".format(path, fraction), mode="overwrite")
    test_validation.write.parquet("{}/data/processed/testing_{}.parquet".format(path, fraction), mode="overwrite")


if __name__ == "__main__":
    memory = "8g"
    spark = SparkSession.builder.appName("downsampling and split").config("spark.executor.memory", memory).config("spark.driver.memory", memory).getOrCreate()
    fraction = float(sys.argv[1])
    path = sys.argv[2]
    main(spark, fraction, path)
