# Pending Tasks
1. Read the data from BigQuery table for preprocessing Spark job
2. Finalize what features we need to generate for data preprocessing
   1. Create the BQ schema for new columns
   2. Create separate BigQuery tables if needed 
3. Store the preprocessed data in BigQuery table
4. Read preprocessed data from BigQuery table for ML model training
   1. Include new features (generated after preprocessing) apart from raw CSVs
5. Create a script to uplaod trained model to GCS bucket
6. Create a script for prediction using the trained model
   1. Call this script method in pubsub consumer
   2. This script will read the model from GCS and make predictions
   3. Create another column in deliveries BQ table and store the predictions there
7. Create Report & Create PPT
8. Push raw data CSVs to git LFS
9. Deduplication of BQ data


# Preprocessing Steps
1. Lowercase for all strings
2. Check historgram for NaN percentage and drop mostly null columns
3. Create table `team_data` team level aggregations which has:
   1. Total runs for team
   2. Total wickets for team
   3. Total matches played by team
   4. Total wins by team
4. Create another table `team_vs_team_data` which team1 vs team2 aggregation accross matches:
   1. team1_score
   2. team2_score
   3. team1_wickets
   4. team2_wickets
   5. team1_wins
   6. team2_wins
