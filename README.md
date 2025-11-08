# Data_Engg_Project_IPL_Match_Prediction
This project implements an **end-to-end real-time data engineering pipeline** for IPL (Indian Premier League) match prediction.  
It uses **Google Cloud Pub/Sub** for event streaming, **Cloud Bigtable** for scalable data storage, and **Apache Spark Streaming** for real-time data processing and machine-learning model training.

## 🚀 System Architecture

```mermaid
flowchart LR
    A[Raw IPL Data (matches.csv, deliveries.csv)] -->|CSV Ingestion| B[Pub/Sub Publisher (Python)]
    B -->|Stream of JSON events| C[(Pub/Sub Topic)]
    C -->|Subscriber → Sink| D[(Cloud Bigtable)]
    D -->|Read API| E[Spark Streaming]
    E -->|Feature Engineering & Aggregation| F[ML Pipeline (Spark MLlib)]
    F -->|Model Output| G[Predictions Dashboard or BigQuery]
