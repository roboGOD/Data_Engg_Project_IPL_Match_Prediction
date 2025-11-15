gcloud functions deploy match_data_consumer \
  --project iisc-data-engineering-project \
  --runtime python313 \
  --trigger-topic match_data \
  --entry-point pubsub_consumer \
  --region asia-south1 \
  --source . \
  --set-env-vars GCP_PROJECT=iisc-data-engineering-project,BQ_DATASET=ipl_analysis,MATCH_TABLE=match_data,DELIVERIES_TABLE=deliveries_data \
  --service-account iisc-data-engineering-project@iisc-data-engineering-project.iam.gserviceaccount.com

