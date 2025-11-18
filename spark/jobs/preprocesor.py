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


output_path = "/content/drive/MyDrive/ipl_match_dataset/ipl_cleaned_for_sparkml.csv"
final_df.write.csv(output_path, header=True, mode="overwrite")
output_path = "/content/drive/MyDrive/ipl_cleaned_for_sparkml.parquet"
final_df.write.mode("overwrite").parquet(output_path)
print("Saved cleaned dataset to:", output_path)

# Show preview
final_df.show(20)