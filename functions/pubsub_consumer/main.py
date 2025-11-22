"""Cloud Function Pub/Sub consumer.

This module exposes `pubsub_consumer(event, context)` which can be deployed
to Google Cloud Functions and triggered by Pub/Sub messages. The function
decodes the Pub/Sub payload (base64), expects a JSON object with the
structure {"eventType": "match"|"deliveries", "payload": {...}}, and
inserts the payload as a row into the appropriate BigQuery table.

Deployment note:
gcloud functions deploy pubsub_consumer --runtime python39 --trigger-topic YOUR_TOPIC --entry-point pubsub_consumer
or deploy with --trigger-resource and --trigger-event for subscription-based triggers.
"""

import os
import base64
import json
from google.cloud import bigquery
from google.cloud import storage
from feature_generator import generate_features, clean_match_record, clean_delivery_record, merge_match_delivery
import pandas as pd
import joblib

# Configuration via environment variables (set these in Cloud Functions env)
PROJECT_ID = os.getenv("GCP_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", "iisc-data-engineering-project"))
DATASET_ID = os.getenv("BQ_DATASET", "ipl_analysis")
MATCH_TABLE = os.getenv("MATCH_TABLE", "match_data")
DELIVERIES_TABLE = os.getenv("DELIVERIES_TABLE", "deliveries_data")

# Initialize BigQuery client once per execution environment (reused across invocations)
bq_client = bigquery.Client()


def _get_table_ref(event_type: str):
    """Return BigQuery TableReference for a given event type."""
    if event_type == "deliveries":
        return bq_client.dataset(DATASET_ID).table(DELIVERIES_TABLE)
    if event_type == "match":
        return bq_client.dataset(DATASET_ID).table(MATCH_TABLE)
    raise ValueError(f"Unknown eventType: {event_type}")

def download_model_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        f"Blob {source_blob_name} downloaded to {destination_file_name}."
    )

    return joblib.load(destination_file_name)

def predict_winner(pipeline, match_data, delivery_data):
    match_df = clean_match_record(match_data)
    delivery_df = clean_delivery_record(delivery_data)
    merged_df = merge_match_delivery(match_df, delivery_df)
    features_df = generate_features(merged_df)
    prediction = pipeline.predict(features_df)
    print(f"Prediction: {prediction[0]}")
    return int(prediction[0])

# Fetch match data from BigQuery for given delivery
def fetch_match_for_delivery(delivery: dict) -> dict:
    match_id = delivery.get("match_id")
    if match_id is None:
        raise ValueError("Delivery record missing 'match_id' field")

    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.{MATCH_TABLE}`
        WHERE id = @match_id
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("match_id", "STRING", match_id)
        ]
    )

    query_job = bq_client.query(query, job_config=job_config)
    results = query_job.result()
    match_records = [dict(row) for row in results]

    if not match_records:
        raise ValueError(f"No match record found for match_id: {match_id}")
    
    print (f"\n\nFetched match record for match_id {match_id}: {match_records[0]}")

    return match_records[0]


def pubsub_consumer(event, context):
    """Cloud Function entry point for Pub/Sub messages.

    Args:
        event (dict): The dictionary with data specific to this type of event.
            The `data` field contains the Pub/Sub message body as a base64-encoded string.
        context (google.cloud.functions.Context): Metadata of triggering event.

    Raises:
        Exception: Raising an exception signals a failure and may cause Pub/Sub
                   to retry the message depending on configuration.
    """
    print(f"Context: {context}")
    if 'data' not in event:
        raise ValueError("No data field in Pub/Sub message")

    try:
        raw = base64.b64decode(event['data']).decode('utf-8')
        print(f"Raw message data: {raw}")
        msg = json.loads(raw)

        event_type = msg.get("eventType")
        if not event_type:
            raise ValueError("Missing 'eventType' in message")

        table_ref = _get_table_ref(event_type)

        payload = msg.get("payload")
        if payload is None:
            raise ValueError("Missing 'payload' in message")
        
        if event_type == "deliveries":
            match_data = fetch_match_for_delivery(payload)
            pipeline = download_model_from_gcs(
                bucket_name="ipl-data-models",
                source_blob_name="models/ipl_winner_xgb_powerplay_model.pkl",
                destination_file_name="/tmp/ipl_winner_xgb_powerplay_model.pkl"
            )
            predicted_winner = predict_winner(pipeline, match_data, payload)
            payload["predicted_winner"] = predicted_winner
            print (f"\n\nUpdated delivery payload with prediction: {payload}")

        # Insert into BigQuery
        errors = bq_client.insert_rows_json(table_ref, [payload])
        if errors:
            # Log and raise so Pub/Sub can retry or dead-letter depending on config
            print(f"BigQuery insert errors: {errors}")
            raise RuntimeError(f"BigQuery insert errors: {errors}")

        # Choose the first available identifier. The payload is expected to be a dict
        # (decoded from JSON), but we defensively support attribute access too.
        if isinstance(payload, dict):
            row_id = payload.get('id') if payload.get('id') is not None else payload.get('match_id')
        else:
            row_id = getattr(payload, 'id', None) or getattr(payload, 'match_id', None)

        if row_id is None:
            print("Inserted row into BigQuery successfully (no id or match_id present).")
        else:
            print(f"Inserted row with ID {row_id} into BigQuery successfully.")

    except Exception:
        # Re-raise to surface the error to Cloud Functions / Pub/Sub
        print("Failed to process Pub/Sub message", flush=True)
        raise

