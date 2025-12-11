# Databricks notebook source
# MAGIC %md
# MAGIC # Purpose of This Notebook
# MAGIC
# MAGIC The purpose of this notebook is to perform zone-level clustering of NYC taxi pickup locations using trip features. It aims to identify patterns and groupings among pickup zones by leveraging machine learning techniques. The workflow includes loading Silver data, engineering zone features, vectorizing for M
# MAGIC L, applying K-Means clustering, evaluating cluster quality, saving Gold results, and interpreting clusters with geographic context.

# COMMAND ----------

# Load Silver table
from pyspark.sql import functions as F

df_silver = spark.table("damo_630.default.silver_nyc_taxi")

df_silver.printSchema()
print("Rows:", df_silver.count())
df_silver.show(10)

# COMMAND ----------

# Build zone-level features
zone_features = (
    df_silver
    .groupBy("pickup_location_id")
    .agg(
        F.count("*").alias("total_trips"),
        F.avg("trip_distance").alias("avg_trip_distance"),
        F.avg("fare_amount").alias("avg_fare_amount"),
        F.avg("tip_amount").alias("avg_tip_amount"),
        F.avg("avg_speed_mph").alias("avg_speed"),
        F.avg("trip_duration_min").alias("avg_duration_min")
    )
)

display(zone_features)
print("Zones:", zone_features.count())

# COMMAND ----------

# Select features for modeling
features = [
    "total_trips",
    "avg_trip_distance",
    "avg_fare_amount",
    "avg_tip_amount",
    "avg_speed",
    "avg_duration_min"
]

feature_cols = [
    "total_trips",
    "avg_trip_distance",
    "avg_fare_amount",
    "avg_tip_amount",
    "avg_speed",
    "avg_duration_min"
]

# COMMAND ----------

# Assemble features into a vector
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

zone_features_vector = assembler.transform(zone_features)
zone_features_vector.show(5)

# COMMAND ----------

# Silhouette evaluation
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction")

ks = [3, 4, 5, 6]
results = []

for k in ks:
    kmeans = KMeans(k=k, seed=42, featuresCol="features")
    model = kmeans.fit(zone_features_vector)
    predictions = model.transform(zone_features_vector)
    
    silhouette = evaluator.evaluate(predictions)
    results.append((k, silhouette))
    print(f"K={k}, silhouette={silhouette}")

# COMMAND ----------

# Train final model with the Best K
best_k = max(results, key=lambda x: x[1])[0]
print("Best K =", best_k)

final_kmeans = KMeans(k=best_k, seed=42, featuresCol="features")
final_model = final_kmeans.fit(zone_features_vector)

clustered_df = final_model.transform(zone_features_vector)
display(clustered_df)

# COMMAND ----------

# Save the Gold table
clustered_df.write.option("mergeSchema", "true").mode("overwrite").saveAsTable("damo_630.default.gold_nyc_taxi_clusters")

print("Gold table created: damo_630.default.gold_nyc_taxi_clusters")

# COMMAND ----------

# MAGIC %md
# MAGIC # Cluster Interpretation
# MAGIC
# MAGIC **Cluster 0:**  
# MAGIC Represents high-demand central zones with:
# MAGIC - Very high total trips
# MAGIC - Shorter average trip durations
# MAGIC - Higher-than-average fares  
# MAGIC These resemble Manhattan core pickup hotspots.
# MAGIC
# MAGIC **Cluster 1:**  
# MAGIC Zones show:
# MAGIC - Moderate trip volume
# MAGIC - Longer distances
# MAGIC - Higher average speeds  
# MAGIC These represent highway-connected zones (e.g., outer boroughs).
# MAGIC
# MAGIC **Cluster 2:**  
# MAGIC Shows:
# MAGIC - Very low demand
# MAGIC - Low fares
# MAGIC - Short trips  
# MAGIC These are likely residential or low-traffic areas.
# MAGIC

# COMMAND ----------

df_geo = spark.table("damo_630.default.taxi_zone_geo")

clustered_with_names = (
    clustered_df
    .join(
        df_geo,
        clustered_df.pickup_location_id == df_geo.zone_id,
        "left"
    )
    .drop(df_geo.zone_id)
)

clustered_with_names = clustered_with_names.select(
    "pickup_location_id",
    "zone_name",
    "borough",
    "prediction",
    "total_trips",
    "avg_trip_distance",
    "avg_fare_amount",
    "avg_tip_amount",
    "avg_speed",
    "avg_duration_min"
)

display(clustered_with_names)

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS damo_630.default.gold_nyc_taxi_clusters")

clustered_with_names.write.mode("overwrite").saveAsTable(
    "damo_630.default.gold_nyc_taxi_clusters"
)

# COMMAND ----------

gold_df = spark.table("damo_630.default.gold_nyc_taxi_clusters")
gold_df.printSchema()
gold_df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of Notebook 04: K-Means Hotspot Clustering
# MAGIC
# MAGIC This notebook applies K-Means clustering to group NYC pickup zones based on demand and trip characteristics. The model uses features such as total trips, average distances, fares, speeds, and tips to identify meaningful clusters.
# MAGIC
# MAGIC ### What This Notebook Includes
# MAGIC - Selecting and scaling features for clustering
# MAGIC - Running the K-Means algorithm
# MAGIC - Assigning each pickup zone to a cluster
# MAGIC - Joining cluster outputs with geographic zone information
# MAGIC
# MAGIC ### Purpose of This Notebook
# MAGIC This notebook identifies which areas of NYC have similar taxi usage patterns. The resulting clusters serve as the basis for generating operational insights and recommendations in Notebook 05.
# MAGIC
# MAGIC ### Output
# MAGIC A Gold dataset containing all pickup zones with their assigned cluster labels and aggregated metrics.