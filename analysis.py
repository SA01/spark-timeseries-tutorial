import time

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from common import create_spark_session


def join_zones_lookup_data(spark: SparkSession) -> DataFrame:
    trips = (
        spark
        .read
        .parquet("data/trips/*/*.parquet")
        .filter(
            (F.col("tpep_pickup_datetime") >= '2021-01-01 00:00:00') &
            (F.col("tpep_pickup_datetime") < '2023-01-01 00:00:00')
        )
    )
    zones = spark.read.option("header", "true").csv("data/taxi+_zone_lookup.csv")

    pickup_zones = (
        zones
        .withColumnRenamed("LocationID", "PULocationID")
        .withColumnRenamed("Borough","PU_Borough")
        .withColumnRenamed("Zone", "PU_Zone")
        .withColumnRenamed("service_zone", "PU_service_zone")
    )

    dropoff_zones = (
        zones
        .withColumnRenamed("LocationID", "DOLocationID")
        .withColumnRenamed("Borough","DO_Borough")
        .withColumnRenamed("Zone", "DO_Zone")
        .withColumnRenamed("service_zone", "DO_service_zone")
    )

    joined_trips = (
        trips
        .join(pickup_zones, ["PULocationID"], "inner")
        .join(dropoff_zones, ["DOLocationID"], "inner")
    )

    joined_trips.show(truncate=False)
    joined_trips.printSchema()

    return joined_trips


def intro_windows(data: DataFrame):
    window_rows_between = Window.partitionBy("DOLocationID").orderBy("tpep_dropoff_datetime").rowsBetween(-10, 0)

    avg_dist_last_10 = data.withColumn("avg_dist_last_10", F.avg("trip_distance").over(window_rows_between))

    avg_dist_last_10.show(truncate=False)


def analysis_number_of_rides_per_segment_of_day(data: DataFrame):
    day_segment_window = Window.partitionBy("PULocationID", "pu_date", "day_segment")

    data_with_cols = (
        data
        .withColumn("trip_id", F.monotonically_increasing_id())
        .withColumn("pu_date", F.to_date(F.col("tpep_pickup_datetime")))
        .withColumn("day_segment", F.date_part(F.lit("Hour"), F.col("tpep_pickup_datetime")))
        .withColumn("day_segment", F.floor((F.col("day_segment") / 3) + 1))
        .withColumn("dow", F.date_part(F.lit("DAYOFWEEK"), F.col("tpep_pickup_datetime")))
    )

    res_data = (
        data_with_cols
        .withColumn("pickups_per_day_per_segment", F.count("trip_id").over(day_segment_window))
        .withColumn("avg_total_amount", F.avg("total_amount").over(day_segment_window))
    )

    (
        res_data
        .filter(F.col("PULocationID") == 250)
        .sort("tpep_pickup_datetime")
        .select(
            "trip_id", "PULocationID", "DOLocationID", "tpep_pickup_datetime", "pu_date", "day_segment", "dow",
            "tpep_dropoff_datetime", "total_amount", "pickups_per_day_per_segment", "avg_total_amount"
        )
        .show(truncate=False, n=20)
    )

    (
        res_data
        .groupby("PULocationID", "pu_date", "day_segment")
        .agg(
            F.count("trip_id").alias("pickups_per_day_per_segment"),
            F.avg("total_amount").alias("avg_total_amount")
        )
        .filter(F.col("PULocationID") == 250)
        .sort("PULocationID", "pu_date", "day_segment")
        .show(truncate=False)
    )

    # Output the data to get complete query plan for the whole data
    # res_data.write.parquet("data/outputs/test")


def analysis_number_of_rides_per_segment_of_day_with_error(data: DataFrame):
    day_segment_pre_window = Window.partitionBy("PULocationID", "pu_date", "day_segment")
    day_segment_pu_loc_window = Window.partitionBy("PULocationID", "dow", "day_segment")

    data_with_cols = (
        data
        .withColumn("trip_id", F.monotonically_increasing_id())
        .withColumn("pu_date", F.to_date(F.col("tpep_pickup_datetime")))
        .withColumn("day_segment", F.date_part(F.lit("Hour"), F.col("tpep_pickup_datetime")))
        .withColumn("day_segment", F.floor((F.col("day_segment") / 3) + 1))
        .withColumn("dow", F.date_part(F.lit("DAYOFWEEK"), F.col("tpep_pickup_datetime")))
    )

    res_data = (
        data_with_cols
        .withColumn("pickups_per_day_per_segment", F.count("trip_id").over(day_segment_pre_window))
        .withColumn("avg_pickups_at_pu_loc", F.avg("pickups_per_day_per_segment").over(day_segment_pu_loc_window))
    )

    (
        res_data
        .filter((F.col("PULocationID") == 250) & (F.col("day_segment") == 4) & (F.col("dow") == 6))
        .sort("tpep_pickup_datetime")
        .select(
            "trip_id", "PULocationID", "DOLocationID", "tpep_pickup_datetime", "pu_date", "day_segment", "dow",
            "tpep_dropoff_datetime", "total_amount", "pickups_per_day_per_segment", "avg_pickups_at_pu_loc"
        )
        .show(truncate=False, n=100)
    )

    (
        res_data
        .filter((F.col("PULocationID") == 250) & (F.col("day_segment") == 4) & (F.col("dow") == 6))
        .groupby()
        .agg(
            F.avg("pickups_per_day_per_segment").alias("avg_pickups_at_pu_loc"),
            F.sum("pickups_per_day_per_segment").alias("sum_pickups_at_pu_loc"),
            F.count(F.lit(1)).alias("num_rows")
        )
        .show(truncate=False, n=200)
    )

    (
        res_data
        .groupby("PULocationID", "dow", "pu_date", "day_segment")
        .agg(F.countDistinct("trip_id").alias("pickups_per_day_per_segment"))
        .sort("pu_date", "day_segment")
        .groupby("PULocationID", "dow", "day_segment")
        .agg(
            F.avg("pickups_per_day_per_segment").alias("avg_pickups_at_pu_loc"),
            F.sum("pickups_per_day_per_segment").alias("sum_pickups_at_pu_loc"),
            F.count(F.lit(1)).alias("num_rows")
        )
        .filter(F.col("PULocationID") == 250)
        .filter((F.col("PULocationID") == 250) & (F.col("day_segment") == 4) & (F.col("dow") == 6))
        .sort("dow", "day_segment")
        .show(truncate=False, n=200)
    )


def window_last_15_min_of_dropoff_example(data: DataFrame):
    window_def = Window.partitionBy("DOLocationID").orderBy("tpep_dropoff_datetime_numeric").rangeBetween(-(180 * 60), -1)

    res_data = (
        data
        .withColumn("tpep_dropoff_datetime_numeric", F.to_unix_timestamp("tpep_dropoff_datetime"))
        .withColumn("avg_trip_distance", F.avg("trip_distance").over(window_def))
        .withColumn("min_dropoff_time", F.min("tpep_dropoff_datetime").over(window_def))
        .withColumn("max_dropoff_time", F.max("tpep_dropoff_datetime").over(window_def))
        .withColumn("num_rows", F.count(F.lit(1)).over(window_def))
    )

    (
        res_data
        .filter(F.col("DOLocationID") == 20)
        .sort("tpep_dropoff_datetime")
        .select(
            "PULocationID", "DOLocationID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "tpep_dropoff_datetime_numeric", "trip_distance", "avg_trip_distance", "min_dropoff_time",
            "max_dropoff_time", "num_rows"
        )
        .show(truncate=False)
    )


def gap_identification_last_pickup(data: DataFrame):
    window_def = Window.partitionBy("PULocationID").orderBy("tpep_pickup_datetime").rowsBetween(-1, -1)

    res_data = (
        data
        .withColumn("previous_pickup_time",F.first("tpep_pickup_datetime").over(window_def))
        .withColumn(
            "time_difference_minutes",
            F.round((F.unix_timestamp("tpep_pickup_datetime") - F.unix_timestamp("previous_pickup_time")) / 60)
        )
    )

    (
        res_data
        .select(
            "PULocationID", "DOLocationID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "time_difference_minutes"
        )
        .show(truncate=False)
    )


def gap_identification_last_pickup_with_outliers(data: DataFrame):
    window_def = Window.partitionBy("PULocationID", "tpep_pickup_date").orderBy("tpep_pickup_datetime").rowsBetween(-1, -1)
    outlier_window_def = (
        Window
        .partitionBy("PULocationID", "tpep_pickup_date")
        .orderBy("tpep_pickup_datetime")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    data_with_gap_from_previous_pickup = (
        data
        .withColumn("tpep_pickup_date", F.to_date(F.col("tpep_pickup_datetime")))
        .withColumn("previous_pickup_time", F.first("tpep_pickup_datetime").over(window_def))
        .withColumn(
            "time_difference_minutes",
            F.round((F.unix_timestamp("tpep_pickup_datetime") - F.unix_timestamp("previous_pickup_time")) / 60)
        )
    )

    res_data = (
        data_with_gap_from_previous_pickup
        .withColumn("pickup_gaps_in_window", F.collect_list(F.col("time_difference_minutes")).over(outlier_window_def))
        .withColumn("mean_pickup_gap", F.avg(F.col("time_difference_minutes")).over(outlier_window_def))
        .withColumn("sd_pickup_gap", F.stddev_samp(F.col("time_difference_minutes")).over(outlier_window_def))
        .withColumn("upper_bound", F.col("mean_pickup_gap") + (3 * F.col("sd_pickup_gap")))
        .withColumn(
            "lower_bound",
            F.greatest(F.col("mean_pickup_gap") - (3 * F.col("sd_pickup_gap")), F.lit(0))
        )
        .withColumn("num_rows", F.count(F.lit(1)).over(outlier_window_def))
        .withColumn(
            "is_outlier",
            F.when(
                (F.col("num_rows") > 8) &
                (
                        (F.col("time_difference_minutes") < F.col("lower_bound")) |
                        (F.col("time_difference_minutes") > F.col("upper_bound"))
                ),
                F.lit(True)
            ).otherwise(F.lit(False))
        )
    )

    (
        res_data
        .filter(F.col("is_outlier"))
        .select(
            "PULocationID", "DOLocationID", "tpep_pickup_datetime", "tpep_pickup_date", "tpep_dropoff_datetime",
            "time_difference_minutes", "mean_pickup_gap", "sd_pickup_gap",
            "upper_bound", "lower_bound", "is_outlier", "num_rows"
        )
        .show(truncate=False, n=10)
    )

    (
        res_data
        .filter((F.col("PULocationID") == 4) & (F.col("tpep_pickup_date") == "2021-07-01") & (F.col("num_rows") <= 10))
        .select(
            "PULocationID", "DOLocationID", "tpep_pickup_datetime", "tpep_pickup_date", "tpep_dropoff_datetime",
            "time_difference_minutes", "mean_pickup_gap", "pickup_gaps_in_window", "sd_pickup_gap",
            "upper_bound", "lower_bound", "is_outlier", "num_rows"
        )
        .show(truncate=False, n=50)
    )


if __name__ == '__main__':
    spark = create_spark_session("spark-timeseries-tutorial")
    # joined_with_zones = join_zones_lookup_data(spark=spark)
    # joined_with_zones.repartition(32).write.parquet("data/trips/combined")

    data = spark.read.parquet("data/trips/combined/*.parquet")
    data.show(truncate=False)

    analysis_number_of_rides_per_segment_of_day(data=data)
    # analysis_number_of_rides_per_segment_of_day_with_error(data=data)
    # window_last_15_min_of_dropoff_example(data=data)
    # gap_identification_last_pickup_with_outliers(data=data)
    # gap_identification_last_pickup(data=data)

    # time.sleep(100000000)





