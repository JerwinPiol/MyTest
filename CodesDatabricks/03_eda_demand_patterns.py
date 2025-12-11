# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks notebook: NYC Taxi Demand Analysis
# MAGIC
# MAGIC This notebook analyzes NYC taxi trip data from a Silver table. 
# MAGIC It explores trip demand patterns by hour, day of week, and location.
# MAGIC
# MAGIC 1. Load the Silver table containing cleaned taxi trip records.
# MAGIC 2. Analyze trip demand by hour of day to identify peak times.
# MAGIC 3. Analyze trip demand by day of week to spot weekly trends.
# MAGIC 4. Create a heatmap of trips by hour and day to visualize temporal patterns.
# MAGIC 5. Identify the top 20 pick-up zones with the highest trip counts.
# MAGIC 6. Find rush-hour hotspots (zones with most trips between 4–7pm).
# MAGIC 7. Save hourly and day-of-week demand results to new tables for reporting.
# MAGIC
# MAGIC Visualizations and aggregations help uncover demand patterns and inform operational decisions.

# COMMAND ----------

# Load the Silver table
from pyspark.sql import functions as F

df_silver = spark.table("damo_630.default.silver_nyc_taxi")

df_silver.printSchema()
print("Rows in silver:", df_silver.count())
df_silver.show(10)

# COMMAND ----------

# Trips per hour of day (0–23)
hourly_demand = (
    df_silver
    .groupBy("hour_of_day")
    .agg(F.count("*").alias("trip_count"))
    .orderBy("hour_of_day")
)

display(hourly_demand)

# COMMAND ----------

# Trips by Day of Week
dow_demand = (
    df_silver
    .groupBy("day_of_week")
    .agg(F.count("*").alias("trip_count"))
    .orderBy("day_of_week")
)

display(dow_demand)

# COMMAND ----------

# Heatmap data for hour_of_day x day_of_week
hour_dow_heatmap = (
    df_silver
    .groupBy("day_of_week", "hour_of_day")
    .agg(F.count("*").alias("trip_count"))
)

display(hour_dow_heatmap)

# COMMAND ----------

# Top Pick-Up Zones
top_pickup_zones = (
    df_silver
    .groupBy("pickup_location_id")
    .agg(F.count("*").alias("trip_count"))
    .orderBy(F.desc("trip_count"))
    .limit(20)   # top 20 zones
)

display(top_pickup_zones)

# COMMAND ----------

# Rush-hour hotspots: zone + hour
rush_hour_df = df_silver.filter(
    (F.col("hour_of_day") >= 16) & (F.col("hour_of_day") <= 19)
)

rush_hour_hotspots = (
    rush_hour_df
    .groupBy("pickup_location_id")
    .agg(F.count("*").alias("trip_count"))
    .orderBy(F.desc("trip_count"))
    .limit(20)
)

display(rush_hour_hotspots)

# COMMAND ----------

# Save hourly demand
hourly_demand.write.mode("overwrite").saveAsTable("damo_630.default.hourly_demand_nyc_taxi")
dow_demand.write.mode("overwrite").saveAsTable("damo_630.default.dow_demand_nyc_taxi")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of Notebook 03: Exploratory Data Analysis (EDA)
# MAGIC
# MAGIC This notebook performs exploratory data analysis to understand patterns, trends, and behaviors in NYC taxi demand. The analysis focuses on time-based patterns such as hourly and weekly demand, geographic hotspots, and relationships between trip characteristics.
# MAGIC
# MAGIC ### What This Notebook Includes
# MAGIC - Descriptive statistics and distribution checks
# MAGIC - Demand patterns across hours, days, and months
# MAGIC - Identification of high-demand pickup zones
# MAGIC - Visual and numerical exploration of trip distances and fares
# MAGIC
# MAGIC ### Purpose of This Notebook
# MAGIC The goal of this notebook is to surface insights about taxi usage that guide the modeling decisions in Notebook 04. EDA also reveals patterns that help interpret the K-Means clusters later in the pipeline.
# MAGIC
# MAGIC ### Output
# MAGIC A set of exploratory findings that inform the clustering strategy used in Notebook 04.