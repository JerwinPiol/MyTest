# Databricks notebook source
# MAGIC %md
# MAGIC ## This notebook performs data cleaning and feature engineering on raw (bronze) NYC taxi data.
# MAGIC
# MAGIC ### Key steps:
# MAGIC - Filters out invalid trips based on distance, duration, amount, and location IDs.
# MAGIC - Creates new features: trip_duration_min, trip_duration_hr, avg_speed_mph.
# MAGIC - Removes outliers and ensures data quality.
# MAGIC - Selects relevant columns for analysis.
# MAGIC - Writes the processed data to a silver table for downstream analytics.

# COMMAND ----------

# Import and load

from pyspark.sql import functions as F

# Load the raw (bronze) taxi data
df_bronze = spark.table("damo_630.default.nyc_taxi_2018_raw")

df_bronze.printSchema()
print("Row count (bronze):", df_bronze.count())
df_bronze.show(10)

# COMMAND ----------

# Basic cleaning

df_clean = df_bronze

# 1) Filter on trip_distance
df_clean = df_clean.filter(
    (F.col("trip_distance") > 0) &
    (F.col("trip_distance") < 100)
)

# 2) Filter on trip_duration (in seconds)
#    Keep trips between 1 minute and 4 hours (240 minutes = 14400 seconds)
df_clean = df_clean.filter(
    (F.col("trip_duration") > 60) &
    (F.col("trip_duration") < 14400)
)

# 3) Filter on total_amount if column exists
if "total_amount" in df_clean.columns:
    df_clean = df_clean.filter(F.col("total_amount") > 0)

# 4) Filter on location IDs: non-null and > 0
df_clean = df_clean.filter(
    (F.col("pickup_location_id").isNotNull()) &
    (F.col("dropoff_location_id").isNotNull()) &
    (F.col("pickup_location_id") > 0) &
    (F.col("dropoff_location_id") > 0)
)

print("Row count after basic cleaning:", df_clean.count())
df_clean.show(10)

# COMMAND ----------

# Feature engineering

# 1) Duration in minutes and hours
df_clean = df_clean.withColumn("trip_duration_min", F.col("trip_duration") / 60.0)
df_clean = df_clean.withColumn("trip_duration_hr", F.col("trip_duration") / 3600.0)

# 2) Average speed in mph: distance (miles) / hours
df_clean = df_clean.withColumn(
    "avg_speed_mph",
    F.col("trip_distance") / F.col("trip_duration_hr")
)

# Replace infinite speeds (division by zero safety, though we filtered duration > 60 sec)
df_clean = df_clean.replace(float("inf"), None, subset=["avg_speed_mph"])

# Drop rows where avg_speed_mph is null
df_clean = df_clean.na.drop(subset=["avg_speed_mph"])

print("Row count after adding duration & speed:", df_clean.count())

# Quick sanity check: show some speeds
df_clean.select("trip_distance", "trip_duration_min", "avg_speed_mph").show(10)

# COMMAND ----------

# Speed over 300
df_clean = df_clean.filter(
    (F.col("avg_speed_mph") > 0) &
    (F.col("avg_speed_mph") < 80)
)

print("Row count after speed filter:", df_clean.count())

# COMMAND ----------

# Build Silver DataFrame

silver_cols = [
    "trip_distance",
    "pickup_location_id",
    "dropoff_location_id",
    "year",
    "month",
    "day",
    "day_of_week",
    "hour_of_day",
    "trip_duration_min",
    "avg_speed_mph",
    "fare_amount",
    "tip_amount",
    "total_amount"
]

# Only keep columns that actually exist (safety)
silver_cols = [c for c in silver_cols if c in df_clean.columns]

df_silver = df_clean.select(*silver_cols)

print("Silver schema:")
df_silver.printSchema()
print("Row count (silver):", df_silver.count())
df_silver.show(10)

# COMMAND ----------

# Write Silver table

df_silver.write.mode("overwrite").saveAsTable("damo_630.default.silver_nyc_taxi")

print("✅ Silver table created: damo_630.default.silver_nyc_taxi")

# COMMAND ----------

df_test = spark.table("damo_630.default.silver_nyc_taxi")
df_test.printSchema()
df_test.show(10)
print("Row count from saved silver table:", df_test.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of Notebook 02: Cleaning and Feature Engineering
# MAGIC
# MAGIC This notebook applies all necessary data cleaning steps and creates engineered features required for downstream analysis. Operations include handling missing values, correcting invalid entries, filtering out unrealistic trips, and generating new columns such as hour of day, day of week, and trip duration.
# MAGIC
# MAGIC ### What This Notebook Includes
# MAGIC - Filtering invalid or inconsistent trip records
# MAGIC - Handling missing, negative, or zero values
# MAGIC - Creating engineered features for time-based analysis
# MAGIC - Preparing a clean Silver dataset
# MAGIC
# MAGIC ### Purpose of This Notebook
# MAGIC The goal is to create a reliable dataset that can be used for meaningful analysis. This notebook produces the Silver table, which becomes the primary input for exploratory analysis and clustering.
# MAGIC
# MAGIC ### Output
# MAGIC A cleaned and feature-rich Silver dataset stored in the project's database.