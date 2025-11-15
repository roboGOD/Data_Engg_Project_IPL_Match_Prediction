# Data_Engg_Project_IPL_Match_Prediction
This project implements an **end-to-end real-time data engineering pipeline** for IPL (Indian Premier League) match prediction.  
It uses **Google Cloud Pub/Sub** for event streaming, **Cloud Bigtable** for scalable data storage, and **Apache Spark Streaming** for real-time data processing and machine-learning model training.



## Running the Cloud Function locally

A lightweight local runner is provided to simulate Pub/Sub events and invoke the
Cloud Function handler in `consumer.py`.

1. Create or edit `sample_event.json` with the payload you want to test.
2. Run the local runner:

```bash
python3 run_local_consumer.py --file sample_event.json
```

Alternatively you can pass inline JSON:

```bash
python3 run_local_consumer.py --json '{"eventType":"deliveries","payload":{...}}'
```

Notes:
- The runner builds a fake Pub/Sub event (base64-encoded `data` field) and calls
	the `pubsub_consumer(event, context)` function directly.
- The BigQuery client in `consumer.py` will attempt to connect to GCP unless you
	mock it or provide credentials. For simple local-only testing you can modify
	`consumer.py` to skip the insert when an environment variable (e.g. `BQ_DRY_RUN`)
	is set.


