# Databricks notebook source
# MAGIC %md
# MAGIC # 06 – Final Visualizations and Charts
# MAGIC
# MAGIC This notebook collects all charts and summary tables used in the final report and slide deck.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Hourly Demand
# MAGIC ## 2. Day-of-Week Demand
# MAGIC ## 3. Hour x Day Heatmap
# MAGIC ## 4. Top Pickup Zones
# MAGIC ## 5. Cluster Summary Charts
# MAGIC ## 6. Rush-Hour Hotspots
# MAGIC ## 7. Hotspot Tables

# COMMAND ----------

# Hourly Demand Curve (Trips by Hour of Day)
from pyspark.sql import functions as F
hourly_demand = (
    spark.table("damo_630.default.silver_nyc_taxi")
        .groupBy("hour_of_day")
        .agg(F.count("*").alias("trip_count"))
        .orderBy("hour_of_day")
)

display(hourly_demand)

# COMMAND ----------

# Day of Week Demand
from pyspark.sql import functions as F

dow_labels = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun"
}

# Convert dictionary into Spark map literal
mapping_expr = F.create_map([F.lit(x) for kv in dow_labels.items() for x in kv])

dow_demand_named = (
    spark.table("damo_630.default.silver_nyc_taxi")
        .withColumn("day_name", mapping_expr[F.col("day_of_week")])
        .groupBy("day_name")
        .agg(F.count("*").alias("trip_count"))
        .orderBy(
            F.array_position(F.array([F.lit(d) for d in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]]), F.col("day_name"))
        )
)

display(dow_demand_named)

# COMMAND ----------

# Hour × Day-of-Week Heatmap
from pyspark.sql import functions as F

dow_labels = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun"
}

heatmap_df = (
    spark.table("damo_630.default.silver_nyc_taxi")
        .withColumn("day_name", F.create_map([F.lit(x) for kv in dow_labels.items() for x in kv])[F.col("day_of_week")])
        .groupBy("day_name", "hour_of_day")
        .agg(F.count("*").alias("trip_count"))
)

display(heatmap_df)

# COMMAND ----------

# Hourly Average Demand
from pyspark.sql import functions as F

silver = spark.table("damo_630.default.silver_nyc_taxi")

# Trips per hour per day
hourly_per_day = (
    silver
    .groupBy("year", "month", "day", "hour_of_day")
    .agg(F.count("*").alias("trip_count"))
)

# Average trips per hour across all days
hourly_avg = (
    hourly_per_day
    .groupBy("hour_of_day")
    .agg(F.avg("trip_count").alias("avg_trips_per_hour"))
    .orderBy("hour_of_day")
)

display(hourly_avg)

# COMMAND ----------

# Top Pickup Zones (Raw IDs)
from pyspark.sql import functions as F

df_geo = spark.table("damo_630.default.taxi_zone_geo")

top20_named = (
    top20.join(df_geo, top20.pickup_location_id == df_geo.zone_id, "left")
         .select(
             "pickup_location_id",
             "zone_name",
             "borough",
             "trip_count"
         )
         .orderBy(F.desc("trip_count"))
)

display(top20_named)

# COMMAND ----------

# Top Zones WITH Real Names
df_geo = spark.table("damo_630.default.taxi_zone_geo")

top_zones_named = (
    top_zones
        .join(df_geo, top_zones.pickup_location_id == df_geo.zone_id, "left")
        .select(
            "pickup_location_id",
            "zone_name",
            "borough",
            "trip_count"
        )
        .orderBy(F.desc("trip_count"))
)

display(top_zones_named)

# COMMAND ----------

# Cluster summary table
from pyspark.sql import functions as F

gold = spark.table("damo_630.default.gold_nyc_taxi_clusters")

cluster_summary = (
    gold.groupBy("prediction")
        .agg(
            F.sum("total_trips").alias("total_trips_cluster"),
            F.avg("avg_trip_distance").alias("avg_distance"),
            F.avg("avg_fare_amount").alias("avg_fare"),
            F.avg("avg_tip_amount").alias("avg_tip"),
            F.avg("avg_speed").alias("avg_speed"),
            F.avg("avg_duration_min").alias("avg_duration_min")
        )
        .orderBy("prediction")
)

display(cluster_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of Notebook 06: Final Visualizations and Charts
# MAGIC
# MAGIC This notebook compiles all of the visual outputs used in the final report and presentation for the NYC Taxi Demand Clustering Project. The visualizations created here highlight the key patterns identified throughout the analysis and help support the conclusions drawn from the K-Means clustering results.
# MAGIC
# MAGIC ### What This Notebook Includes ###
# MAGIC - Hourly demand chart showing trip volume fluctuations throughout the day
# MAGIC - Day-of-week demand chart illustrating weekly usage trends
# MAGIC - Heatmap combining hour and day-of-week to highlight rush-hour activity
# MAGIC - Top pickup zones based on total trip counts
# MAGIC - Cluster-level comparison charts covering total trips, distance, fare, tips, speed, and duration
# MAGIC
# MAGIC ### Purpose of This Notebook ###
# MAGIC
# MAGIC The goal of this notebook is to present clear, presentation-ready visuals that connect the analytical work from earlier notebooks to the final insights and recommendations. These charts are used directly in both the written report and the slide presentation. They also provide a reproducible way to confirm the patterns found in the Silver and Gold datasets.
# MAGIC
# MAGIC ### Link to Previous Notebooks ###
# MAGIC - Uses the cleaned and processed Silver dataset created in Notebook 02
# MAGIC - Uses the cluster-enriched Gold dataset produced in Notebook 05
# MAGIC - Completes the analytical pipeline by transforming data into interpretable visuals
# MAGIC
# MAGIC ### Output ###
# MAGIC
# MAGIC All screenshots generated from this notebook are stored in the project repository under:
# MAGIC docs/screenshots/

# COMMAND ----------

