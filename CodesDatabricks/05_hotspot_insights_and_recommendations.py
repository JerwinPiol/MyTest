# Databricks notebook source
# MAGIC %md
# MAGIC ## Notebook Overview
# MAGIC
# MAGIC This notebook analyzes New York City taxi trip data to identify distinct demand patterns across different zones using K-Means clustering. It categorizes each taxi zone into one of five clusters based on trip volume, fare characteristics, duration, and speed. The results inform operational recommendations for taxi fleet deployment, aiming to optimize coverage, reduce congestion, and improve service efficiency throughout the city.

# COMMAND ----------

# Load the Gold table
from pyspark.sql import functions as F

df_gold = spark.table("damo_630.default.gold_nyc_taxi_clusters")
df_gold.show(10)
df_gold.printSchema()
print("Zones:", df_gold.count())

# COMMAND ----------

# Cluster summary (high-level insight)
cluster_summary = (
    df_gold.groupBy("prediction")
        .agg(
            F.count("*").alias("num_zones"),
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
# MAGIC
# MAGIC
# MAGIC **Cluster 0** = General NYC Zones - Normal behavior, low-to-medium demand
# MAGIC
# MAGIC **Cluster 1** = LaGuardia Airport - Extremely high volume, short trips, nonstop flow
# MAGIC
# MAGIC **Cluster 2** = Midtown / Manhattan Core - Very high demand, shorter trips, congestion
# MAGIC
# MAGIC **Cluster 3** = JFK Airport - High volume, long-distance trips, high fares
# MAGIC
# MAGIC **Cluster 4** = Secondary Hotspots - Mixed commercial/residential, strong demand
# MAGIC
# MAGIC

# COMMAND ----------

# Top zones per cluster
top_zones_cluster = (
    df_gold
    .select("prediction", "zone_name", "borough", "total_trips")
    .orderBy("prediction", F.desc("total_trips"))
)

display(top_zones_cluster)

# COMMAND ----------

# Compute threshold for hotspot based on 75th percentile (Top Pickup Hotspots)
threshold = df_gold.approxQuantile("total_trips", [0.75], 0.01)[0]

hotspots = (
    df_gold.filter(
        (F.col("prediction") == 0) & 
        (F.col("total_trips") >= threshold)
    )
    .orderBy(F.desc("total_trips"))
)

display(hotspots)

# COMMAND ----------

# Join time-based data for “rush-hour hotspots”
df_silver = spark.table("damo_630.default.silver_nyc_taxi")

# Join cluster into the silver data
df_silver_clustered = df_silver.join(
    df_gold.select("pickup_location_id", "prediction"),
    "pickup_location_id",
    "left"
)

# COMMAND ----------

rush_hour_hotspots = (
    df_silver_clustered
    .filter(F.col("prediction") == 0)
    .groupBy("hour_of_day")
    .agg(F.count("*").alias("trip_count"))
    .orderBy("hour_of_day")
)

display(rush_hour_hotspots)

# COMMAND ----------

# Compute hourly demand per zone for simulation input
hourly_zone_demand = (
    df_silver_clustered
    .groupBy("pickup_location_id", "hour_of_day")
    .agg(F.count("*").alias("trip_count"))
)

display(hourly_zone_demand)

# COMMAND ----------

threshold = hourly_zone_demand.approxQuantile("trip_count", [0.90], 0.0)[0]  # Top 10%

alerts = (
    hourly_zone_demand
    .filter(F.col("trip_count") >= threshold)
    .orderBy(F.desc("trip_count"))
)

display(alerts)

# COMMAND ----------

# MAGIC %md
# MAGIC If this were a live stream, any zone exceeding the 90th percentile demand would trigger an alert to reposition taxis toward that zone.

# COMMAND ----------

# Recommendation Table
recommendations = df_gold.select(
    "zone_name",
    "borough",
    "prediction",
    "total_trips",
    "avg_fare_amount",
    "avg_duration_min"
).orderBy(F.desc("total_trips"))

display(recommendations)

# COMMAND ----------

# MAGIC %md
# MAGIC **Final Recommendations Based on Clustered Demand Patterns**
# MAGIC
# MAGIC This section summarizes operational recommendations derived from the K-Means clustering results and zone-level metrics. Each taxi zone is now categorized into one of five clusters, each representing a distinct pattern of trip volume, fare characteristics, duration, and speed. These clusters provide clear guidance on how taxi fleets should be deployed across New York City to reduce congestion, match demand, and improve wait times.
# MAGIC
# MAGIC **Cluster 1 — LaGuardia Airport (Highest Demand, Single-Zone Cluster)**
# MAGIC
# MAGIC Zones Included: LaGuardia Airport (LGA), Queens
# MAGIC Demand Profile:
# MAGIC - Extremely high trip volume
# MAGIC - Short-to-medium distance trips
# MAGIC - High turnover throughout the day
# MAGIC
# MAGIC Recommendation:
# MAGIC - Assign a dedicated airport taxi fleet to handle continuous demand.
# MAGIC - Maintain peak staffing between 6 AM and 10 PM, reflecting flight schedules.
# MAGIC - Implement real-time queue monitoring to reduce congestion and improve throughput.
# MAGIC
# MAGIC
# MAGIC **Cluster 3 — JFK Airport (High-Demand, Long-Distance Trips)**
# MAGIC
# MAGIC Zones Included: JFK Airport, Queens
# MAGIC Demand Profile:
# MAGIC - High total volume
# MAGIC -Long-distance, high-fare trips
# MAGIC - Consistent demand even during late-night hours
# MAGIC
# MAGIC Recommendation:
# MAGIC - Allocate taxis specializing in longer rides to handle trips into Manhattan and outer boroughs.
# MAGIC - Use flight arrival data to trigger taxi deployment surges.
# MAGIC - Provide overnight fleet coverage due to heavy late-night airport traffic.
# MAGIC
# MAGIC
# MAGIC **Cluster 2 — Manhattan Core Commercial Hotspots**
# MAGIC
# MAGIC Typical Zones: Midtown, Times Square, Financial District
# MAGIC Demand Profile:
# MAGIC - Very high trip density
# MAGIC - Short but frequent trips
# MAGIC - Congestion leads to lower speed
# MAGIC
# MAGIC Recommendation:
# MAGIC - Deploy a concentrated fleet during morning (7–10 AM) and evening (4–8 PM) rush hours.
# MAGIC - Introduce dynamic routing to avoid gridlock around high-traffic intersections.
# MAGIC - Consider surge pricing strategies during peak tourism seasons and events.
# MAGIC
# MAGIC
# MAGIC **Cluster 4 — Secondary High-Activity Corridors**
# MAGIC
# MAGIC Typical Zones: Downtown Brooklyn, Astoria, Long Island City
# MAGIC Demand Profile:
# MAGIC - High overall demand
# MAGIC - Moderate trip distances
# MAGIC - Strong activity during evenings and weekends
# MAGIC
# MAGIC Recommendation:
# MAGIC - Maintain balanced coverage throughout the day.
# MAGIC - Increase fleet allocation during evening commute and weekend nightlife periods.
# MAGIC - Monitor event schedules (sports, concerts) to support timely dispatching.
# MAGIC
# MAGIC
# MAGIC **Cluster 0 — General NYC Zones (Baseline Coverage)**
# MAGIC
# MAGIC Typical Zones: Majority of Bronx, Queens, Brooklyn
# MAGIC Demand Profile:
# MAGIC - Low to moderate trip volume
# MAGIC - Residential and mixed-use areas
# MAGIC - Stable but modest demand patterns
# MAGIC
# MAGIC Recommendation:
# MAGIC - Maintain baseline taxi coverage with flexible overflow from nearby zones.
# MAGIC - Increase availability only during peak commute hours (morning and evening).
# MAGIC - Use demand monitoring alerts to temporarily redeploy taxis to nearby clusters.
# MAGIC
# MAGIC
# MAGIC **Overall Strategic Insights**
# MAGIC - Airports require specialized handling → each forms its own cluster.
# MAGIC - Manhattan zones dominate high-frequency demand and require dense fleet presence.
# MAGIC - Outer boroughs exhibit more predictable patterns, ideal for flexible routing.
# MAGIC - Real-time monitoring supports rapid response to demand spikes.
# MAGIC
# MAGIC This recommendations table provides a clear operational roadmap for taxi fleet planning, ensuring resources are deployed where and when they are most needed.

# COMMAND ----------

# Save insights as a Gold table
recommendations.write.mode("overwrite").saveAsTable(
    "damo_630.default.taxi_recommendations"
)

# COMMAND ----------

