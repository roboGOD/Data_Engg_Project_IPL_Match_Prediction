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

