from google.cloud import pubsub_v1, bigquery
import json

project_id = "iisc-data-engineering-project"
subscription_id = "match_data-sub"
dataset_id = "ipl_analysis"
match_data_table_id = "match_data"
deliveries_data_table_id = "deliveries_data"

# Initialize clients
subscriber = pubsub_v1.SubscriberClient()
bq_client = bigquery.Client()

subscription_path = subscriber.subscription_path(project_id, subscription_id)
match_data_table_ref = bq_client.dataset(dataset_id).table(match_data_table_id)
deliveries_data_table_ref = bq_client.dataset(dataset_id).table(deliveries_data_table_id)

def callback(message: pubsub_v1.subscriber.message.Message):
    print(f"Received message: {message.data}")
    try:
        # Convert Pub/Sub data (bytes) to dict
        data = json.loads(message.data.decode("utf-8"))
        
        # Determine target table based on eventType
        if ("eventType" in data) and (data["eventType"] == "deliveries"):
            table_ref = deliveries_data_table_ref
        elif ("eventType" in data) and (data["eventType"] == "match"):
            table_ref = match_data_table_ref
        else:
            raise ValueError("Unknown eventType in message data")
        
        # Extract payload
        if ("payload" in data):
            data = data["payload"]
        else:
            raise ValueError("No payload found in message data")
        
        # Insert into BigQuery
        errors = bq_client.insert_rows_json(table_ref, [data])
        if errors:
            print(f"BigQuery insert errors: {errors}")
        else:
            print("Inserted row into BigQuery successfully.")
        
        message.ack()  # Acknowledge successful processing

    except Exception as e:
        print(f"Error: {e}")
        message.nack()

subscriber.subscribe(subscription_path, callback=callback)

print("Listening for messages...")
while True:
    pass
