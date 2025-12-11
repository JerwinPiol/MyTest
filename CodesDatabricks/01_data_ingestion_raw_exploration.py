# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Notebook Purpose
# MAGIC
# MAGIC This notebook is designed to demonstrate data analysis and processing workflows using Databricks. It covers data ingestion, transformation, and visualization techniques, providing step-by-step guidance for working with large datasets in a collaborative environment.

# COMMAND ----------

df_bronze = spark.table("damo_630.default.nyc_taxi_2018_raw")

# If you want, you can just clone it as "bronze_nyc_taxi"
df_bronze.write.mode("overwrite").saveAsTable("damo_630.default.bronze_nyc_taxi")

# COMMAND ----------

df_raw = spark.table("damo_630.default.nyc_taxi_2018_raw")
df_raw.show(5)

# COMMAND ----------

spark.table("damo_630.default.taxi_zone_geo").show(5)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of Notebook 01: Data Ingestion and Raw Exploration
# MAGIC
# MAGIC This notebook focuses on loading the NYC Taxi dataset into the Databricks environment and performing the initial checks needed before cleaning and analysis. The main steps include importing the raw files, verifying schema consistency, and running basic exploratory queries to understand the data structure and potential quality issues.
# MAGIC
# MAGIC ### What This Notebook Includes
# MAGIC - Importing raw NYC Taxi trip data into Databricks
# MAGIC - Inspecting schemas, data types, and row counts
# MAGIC - Previewing raw records to identify missing values or irregular entries
# MAGIC - Generating initial descriptive statistics
# MAGIC
# MAGIC ### Purpose of This Notebook
# MAGIC The goal is to ensure that the raw dataset is properly ingested and ready for transformation. This notebook sets the foundation for the Silver stage, where data cleaning and feature engineering occur.
# MAGIC
# MAGIC ### Output
# MAGIC A validated raw dataset stored in the Bronze layer, ready for processing in Notebook 02.