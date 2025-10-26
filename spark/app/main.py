# spark/app/main.py
import json
import os
import uuid
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType
from sqlalchemy import create_engine, text

DB_URL = os.environ["DB_URL"]
KAFKA_BROKER = os.environ["KAFKA_BROKER"]
KAFKA_TOPIC  = os.environ.get("KAFKA_TOPIC", "aoi_jobs")

S3_BUCKET    = os.environ.get("S3_BUCKET", "aoi")
S3_ENDPOINT  = os.environ.get("S3_ENDPOINT", "http://minio:9000")

def update_job_status(aoi_id: str, status: str, error_text: str = None):
    eng = create_engine(DB_URL)
    with eng.begin() as conn:
        if status == "running":
            conn.execute(text("UPDATE aoi_jobs SET started_at=now(), status=:s WHERE aoi_id=:id"),
                         {"s": status, "id": aoi_id})
        elif status in ("completed", "failed"):
            conn.execute(text("UPDATE aoi_jobs SET finished_at=now(), status=:s, error_text=:e WHERE aoi_id=:id"),
                         {"s": status, "e": error_text, "id": aoi_id})
        else:
            conn.execute(text("UPDATE aoi_jobs SET status=:s WHERE aoi_id=:id"),
                         {"s": status, "id": aoi_id})

def record_artifact(aoi_id: str, kind: str, uri: str, size_bytes: int = 0):
    eng = create_engine(DB_URL)
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO artifacts(aoi_id, kind, storage_uri, size_bytes)
            VALUES(:a,:k,:u,:b)
        """), {"a": aoi_id, "k": kind, "u": uri, "b": size_bytes})

def process_payload(payload: str):
    """
    Minimal placeholder: parse job, mark running, pretend we produced an overlay.
    Replace this with: tile planning -> fetch -> mosaic -> (optional) upscaling -> detection -> overlay.
    """
    job = json.loads(payload)
    aoi_id = job.get("aoi_id") or str(uuid.uuid4())
    try:
        update_job_status(aoi_id, "running")

        # TODO: your pipeline here -> produce URIs like s3a://aoi/mosaics/{aoi_id}/raw.tif etc.
        demo_overlay_uri = f"s3a://{S3_BUCKET}/detections/{aoi_id}/overlay.png"
        record_artifact(aoi_id, "overlay", demo_overlay_uri, 0)

        update_job_status(aoi_id, "completed")
    except Exception as e:
        update_job_status(aoi_id, "failed", error_text=str(e))

if __name__ == "__main__":
    spark = (SparkSession.builder
             .appName("AOI-Streaming")
             .getOrCreate())

    # Read from Kafka topic
    df = (spark.readStream
          .format("kafka")
          .option("kafka.bootstrap.servers", KAFKA_BROKER)
          .option("subscribe", KAFKA_TOPIC)
          .option("startingOffsets", "latest")
          .load())

    # value is binary -> string
    values = df.select(F.col("value").cast(StringType()).alias("json"))

    # Process each message on the driver (for simplicity first; you can mapPartitions later)
    # WARNING: foreachBatch runs on the driver for each microbatch—scale it with care.
    def handle_batch(batch_df, batch_id):
        rows = [r["json"] for r in batch_df.select("json").collect()]
        for payload in rows:
            if payload:
                process_payload(payload)

    query = (values.writeStream
             .outputMode("update")
             .foreachBatch(handle_batch)
             .start())

    query.awaitTermination()
