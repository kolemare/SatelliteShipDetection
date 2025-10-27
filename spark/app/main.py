# spark/app/main.py
import json
import os
import uuid
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType
from sqlalchemy import create_engine, text
from typing import Optional 

DB_URL       = os.environ["DB_URL"]
KAFKA_BROKER = os.environ["KAFKA_BROKER"]
KAFKA_TOPIC  = os.environ.get("KAFKA_TOPIC", "aoi_jobs")

S3_BUCKET   = os.environ.get("S3_BUCKET", "aoi")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio:9000")  # not used here, but handy if you need it


def update_job_status(aoi_id: str, status: str) -> None:
    """Update job status/timestamps per the simplified schema (no error_text)."""
    eng = create_engine(DB_URL)
    with eng.begin() as conn:
        if status == "running":
            conn.execute(
                text("UPDATE aoi_jobs SET started_at=now(), status=:s WHERE aoi_id=:id"),
                {"s": status, "id": aoi_id},
            )
        elif status in ("completed", "failed"):
            conn.execute(
                text("UPDATE aoi_jobs SET finished_at=now(), status=:s WHERE aoi_id=:id"),
                {"s": status, "id": aoi_id},
            )
        else:
            conn.execute(
                text("UPDATE aoi_jobs SET status=:s WHERE aoi_id=:id"),
                {"s": status, "id": aoi_id},
            )


def upsert_results(
    aoi_id: str,
    submit_name: str,
    raw_uri: Optional[str] = None,
    upscaled_uri: Optional[str] = None,
    detection_uri: Optional[str] = None,
) -> None:
    """
    Insert-or-update into results (one row per job). Keeps it simple:
    - Insert new row if not exists.
    - If exists, update any provided URIs (leave others as-is).
    """
    eng = create_engine(DB_URL)
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO results (aoi_id, submit_name, raw_uri, upscaled_uri, detection_uri)
                VALUES (:a, :n, :r, :u, :d)
                ON CONFLICT (aoi_id)
                DO UPDATE SET
                  submit_name   = EXCLUDED.submit_name,
                  raw_uri       = COALESCE(EXCLUDED.raw_uri, results.raw_uri),
                  upscaled_uri  = COALESCE(EXCLUDED.upscaled_uri, results.upscaled_uri),
                  detection_uri = COALESCE(EXCLUDED.detection_uri, results.detection_uri)
            """),
            {"a": aoi_id, "n": submit_name, "r": raw_uri, "u": upscaled_uri, "d": detection_uri},
        )


def process_payload(payload: str) -> None:
    """
    Minimal placeholder pipeline:
      - parse job
      - mark running
      - (do your real work here)
      - write URIs to `results`
      - mark completed
    """
    job = json.loads(payload or "{}")
    aoi_id = job.get("aoi_id") or str(uuid.uuid4())
    submit_name = job.get("submit_name") or aoi_id  # fall back to aoi_id if not provided

    try:
        update_job_status(aoi_id, "running")

        # TODO: replace these demo URIs with real outputs from your pipeline.
        raw_uri       = f"s3a://{S3_BUCKET}/mosaics/{aoi_id}/raw.tif"
        upscaled_uri  = f"s3a://{S3_BUCKET}/mosaics/{aoi_id}/upscaled.tif" if job.get("upscaled", True) else None
        detection_uri = f"s3a://{S3_BUCKET}/detections/{aoi_id}/overlay.png"

        upsert_results(
            aoi_id=aoi_id,
            submit_name=submit_name,
            raw_uri=raw_uri,
            upscaled_uri=upscaled_uri,
            detection_uri=detection_uri,
        )

        update_job_status(aoi_id, "completed")
    except Exception as e:
        print(f"[ERROR] processing {aoi_id}: {e}", flush=True)
        update_job_status(aoi_id, "failed")


if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("AOI-Streaming")
        .getOrCreate()
    )

    # Read from Kafka topic
    df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", KAFKA_TOPIC)
        # 'latest' = only messages after the stream starts; switch to 'earliest' if you want to replay
        .option("startingOffsets", "latest")
        .load()
    )

    # value (binary) -> string JSON
    values = df.select(F.col("value").cast(StringType()).alias("json"))

    # Process each micro-batch on the driver (simple to start; scale later)
    def handle_batch(batch_df, batch_id):
        rows = [r["json"] for r in batch_df.select("json").collect()]
        for payload in rows:
            if payload:
                process_payload(payload)

    query = (
        values.writeStream
        .foreachBatch(handle_batch)
        .start()
    )

    query.awaitTermination()
