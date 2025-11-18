import os
import findspark

findspark.init()

from pyspark.sql.types import *

"""**FULL CLEANING PIPELINE**"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from google.cloud import bigquery

# Configuration via environment variables
PROJECT_ID = os.getenv("GCP_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", "iisc-data-engineering-project"))
DATASET_ID = os.getenv("BQ_DATASET", "ipl_analysis")
MATCH_TABLE = os.getenv("MATCH_TABLE", "match_data")
DELIVERIES_TABLE = os.getenv("DELIVERIES_TABLE", "deliveries_data")
PROCESSED_DATA_TABLE_STAGING = os.getenv("DELIVERIES_TABLE", "processed_ipl_data_staging")
PROCESSED_DATA_TABLE = os.getenv("DELIVERIES_TABLE", "processed_ipl_data")

# Spark Session
spark = SparkSession.builder \
    .appName("IPL_SparkML_Cleaning") \
    .config(
        "spark.jars.packages",
        "com.google.cloud.spark:spark-3.5-bigquery:0.43.1"
    ).getOrCreate()

# Big Query Client
client = bigquery.Client()

# 1. LOAD RAW DATA
matches = (
    spark.read.format("bigquery")
    .option("table", f"{PROJECT_ID}.{DATASET_ID}.{MATCH_TABLE}")
    .load()
)

deliveries = (
    spark.read.format("bigquery")
    .option("table", f"{PROJECT_ID}.{DATASET_ID}.{DELIVERIES_TABLE}")
    .load()
)

print("Loaded matches and deliveries data from BigQuery.")
print(f"Matches count: {matches.count()}")
print(f"Deliveries count: {deliveries.count()}")



## Code to read data from local CSVs (for testing locally)
## Uncomment below lines if reading from local files
# matches_path = "/content/drive/MyDrive/ipl_match_dataset/matches.csv"
# deliveries_path = "/content/drive/MyDrive/ipl_match_dataset/deliveries.csv"

# matches = spark.read.csv(matches_path, header=True, inferSchema=True)
# deliveries = spark.read.csv(deliveries_path, header=True, inferSchema=True)



# 2. MATCHES CLEANING


# Remove missing winner rows
matches = matches.dropna(subset=["winner"])

# Impute player_of_match
matches = matches.fillna({"player_of_match": "unknown"})

# Drop unwanted columns
unwanted_match_cols = ["method", "umpire1", "umpire2"]
for c in unwanted_match_cols:
    if c in matches.columns:
        matches = matches.drop(c)

# Cast numeric columns from string to integer before median imputation
for colname in ["result_margin", "target_runs", "target_overs"]:
    if colname in matches.columns and matches.schema[colname].dataType == StringType():
        matches = matches.withColumn(colname, col(colname).cast(IntegerType()))

# Median imputations
def median_fill(df, colname):
    if colname in df.columns:
        median_val = df.approxQuantile(colname, [0.5], 0.05)[0]
        return df.fillna({colname: median_val})
    return df

for colname in ["result_margin", "target_runs", "target_overs"]:
    matches = median_fill(matches, colname)

# Split season
if "season" in matches.columns:
    matches = matches.withColumn("season_parts", split(col("season"), "/"))
    matches = matches.withColumn("season_start", col("season_parts")[0].cast("int"))
    matches = matches.withColumn(
        "season_end",
        when(length(col("season_parts")[1]) == 2, concat(lit("20"), col("season_parts")[1]))
        .otherwise(col("season_parts")[1])
        .cast("int")
    )
    matches = matches.drop("season_parts")


# Standardize team names
team_name_mapping = {
    'delhi daredevils': 'delhi capitals',
    'kings xi punjab': 'punjab kings',
    'rising pune supergiants': 'pune warriors',
    'rising pune supergiant': 'pune warriors',
    'gujarat lions': 'gujarat titans',
    'deccan chargers': 'sunrisers hyderabad',
    'royal challengers bengaluru': 'royal challengers bangalore'
}

# Corrected mapping_expr creation
mapping_args = []
for k, v in team_name_mapping.items():
    mapping_args.append(lit(k))
    mapping_args.append(lit(v))

mapping_expr = create_map(*mapping_args)

for colname in ["team1", "team2", "toss_winner", "winner"]:
    if colname in matches.columns:
        matches = matches.withColumn(
            colname,
            when(col(colname).isin(list(team_name_mapping.keys())),
                 element_at(mapping_expr, lower(col(colname))))
            .otherwise(col(colname))
        )

# Convert string columns to lowercase
for c, t in matches.dtypes:
    if t == "string":
        matches = matches.withColumn(c, lower(col(c)))

# Ensure match_id exists
if "id" in matches.columns:
    matches = matches.withColumnRenamed("id", "match_id")



# 3. DELIVERIES CLEANING


# Drop unnecessary columns
unwanted_delivery_cols = ["player_dismissed", "dismissal_kind", "fielder"]
for c in unwanted_delivery_cols:
    if c in deliveries.columns:
        deliveries = deliveries.drop(c)

# Lowercase all strings
for c, t in deliveries.dtypes:
    if t == "string":
        deliveries = deliveries.withColumn(c, lower(col(c)))



# 4. MERGE MATCHES + DELIVERIES FOR SPARKML INPUT


# IMPORTANT: select required ball-by-ball features
ball_features = [
    "match_id", "inning", "over", "ball",
    "batting_team", "bowling_team",
    "batsman_runs", "extra_runs", "total_runs", "is_wicket"
]

deliveries_small = deliveries.select([c for c in ball_features if c in deliveries.columns])

# Join matches with deliveries
final_df = deliveries_small.join(matches, on="match_id", how="inner")



# 5. DROP LEAKAGE COLUMNS


leakage_cols = ["win_by_runs", "win_by_wickets", "result", "result_margin"]
for c in leakage_cols:
    if c in final_df.columns:
        final_df = final_df.drop(c)



# 6. SAVE CLEANED DATA

## Code to write data to local CSVs (for testing locally)
## Uncomment below lines if writing to local files
# output_path = "/content/drive/MyDrive/ipl_match_dataset/ipl_cleaned_for_sparkml.csv"
# final_df.write.csv(output_path, header=True, mode="overwrite")
# output_path = "/content/drive/MyDrive/ipl_cleaned_for_sparkml.parquet"
# final_df.write.mode("overwrite").parquet(output_path)
# print("Saved cleaned dataset to:", output_path)

# Show preview
final_df.show(20)

# Write to BigQuery
(
    final_df.write.format("bigquery")
    .option("table", f"{PROJECT_ID}.{DATASET_ID}.{PROCESSED_DATA_TABLE_STAGING}")
    .option("writeMethod", "direct")
    .option("writeAtLeastOnce", "true")
    .mode("append")
    .save()
)

print(f"Saved cleaned dataset to BigQuery table: {PROCESSED_DATA_TABLE_STAGING}")

# Run BigQuery MERGE to upsert from staging to main table
print("Starting BigQuery MERGE to upsert data...")

def q(col):
    return f"`{col}`"

table = client.get_table(f"{PROJECT_ID}.{DATASET_ID}.{PROCESSED_DATA_TABLE}")
cols = [schema.name for schema in table.schema]
update_clause = ", ".join([f"T.{q(c)} = S.{q(c)}" for c in cols])
insert_columns = ", ".join([q(c) for c in cols])
insert_values = ", ".join([f"S.{q(c)}" for c in cols])

merge_query = f"""
MERGE `{PROJECT_ID}.{DATASET_ID}.{PROCESSED_DATA_TABLE}` T
USING `{PROJECT_ID}.{DATASET_ID}.{PROCESSED_DATA_TABLE_STAGING}` S
ON T.match_id = S.match_id
   AND T.inning = S.inning
   AND T.`over` = S.`over`
   AND T.ball = S.ball
WHEN MATCHED THEN
  UPDATE SET {update_clause}
WHEN NOT MATCHED THEN
  INSERT ({insert_columns})
  VALUES ({insert_values})
"""
query_job = client.query(merge_query)
query_job.result()  # Wait for completion
print("BigQuery MERGE completed successfully.")
print(f"Upserted data into BigQuery table: {PROCESSED_DATA_TABLE}")

# Clean up staging table after merge
cleanup_query = f"TRUNCATE TABLE `{PROJECT_ID}.{DATASET_ID}.{PROCESSED_DATA_TABLE_STAGING}`"
cleanup_job = client.query(cleanup_query)
cleanup_job.result()  # Wait for completion
print(f"Cleaned up staging table: {PROCESSED_DATA_TABLE_STAGING}")

