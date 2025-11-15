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


