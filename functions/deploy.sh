# Script to deploy the Pub/Sub consumer Cloud Function
# Assumes gcloud CLI is installed and configured
# Usage: bash deploy.sh

gcloud functions deploy match_data_consumer \
  --project iisc-data-engineering-project \
  --runtime python313 \
  --trigger-topic match_data \
  --entry-point pubsub_consumer \
  --region asia-south1 \
  --source ./pubsub_consumer \
  --memory 256MB \
  --timeout 60s \
  --cpu 0.1 \
  --set-env-vars GCP_PROJECT=iisc-data-engineering-project,BQ_DATASET=ipl_analysis,MATCH_TABLE=match_data,DELIVERIES_TABLE=deliveries_data \
  --service-account iisc-data-engineering-project@iisc-data-engineering-project.iam.gserviceaccount.com \
  --ingress-settings internal-only \
  --concurrency 1 \
  --max-instances 1

# For first time deployment bind the invoker role
# gcloud functions add-invoker-policy-binding match_data_consumer \
#   --member=serviceAccount:iisc-data-engineering-project@iisc-data-engineering-project.iam.gserviceaccount.com \
#   --region=asia-south1 \
#   --project=iisc-data-engineering-project
