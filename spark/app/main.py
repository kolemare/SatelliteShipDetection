#!/usr/bin/env python3
import os, json, uuid, cv2, traceback
import numpy as np
from sqlalchemy import create_engine, text
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType

from ConvNextInference import detect_and_draw
from realESRGAN import SuperSampler
from tiling import AOIRequest, TileStitcher
from providers import ProviderCatalog, ProviderKey

DB_URL        = os.environ["DB_URL"]
KAFKA_BROKER  = os.environ["KAFKA_BROKER"]
KAFKA_TOPIC   = os.environ.get("KAFKA_TOPIC", "aoi_jobs")

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/opt/spark-events/checkpoints/aoi-stream")

# ---------------------------------------------------------------------
def _png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()

_engine = None
_ss = None
def _init_partition():
    """Lazy-init SQLAlchemy engine and RealESRGAN once per Python worker."""
    global _engine, _ss
    if _engine is None:
        print("[init] creating SQLAlchemy engine", flush=True)
        _engine = create_engine(DB_URL, pool_pre_ping=True)
    if _ss is None:
        print("[init] creating SuperSampler (RealESRGAN)", flush=True)
        _ss = SuperSampler()
    return _engine, _ss

def update_status(conn, aoi_id, status):
    conn.execute(
        text("UPDATE aoi_jobs SET status=:s, updated_at=now() WHERE aoi_id=:id"),
        {"s": status, "id": aoi_id},
    )

def fail_status(conn, aoi_id, msg: str):
    conn.execute(
        text("UPDATE aoi_jobs SET status='failed', updated_at=now() WHERE aoi_id=:id"),
        {"id": aoi_id},
    )
    print(f"[fail] aoi_id={aoi_id}: {msg}", flush=True)

def upsert_results(conn, aoi_id, submit_name, n_found, raw_png, det_png, up_png=None):
    conn.execute(
        text(
            """
        INSERT INTO results
          (aoi_id, submit_name, num_detections,
           raw_image, raw_mime,
           upscaled_image, upscaled_mime,
           detection_image, detection_mime)
        VALUES
          (:a, :n, :num,
           :raw, 'image/png',
           :up,  CASE WHEN :up IS NULL THEN NULL ELSE 'image/png' END,
           :det, 'image/png')
        ON CONFLICT (aoi_id) DO UPDATE SET
          num_detections=EXCLUDED.num_detections,
          raw_image=EXCLUDED.raw_image,
          detection_image=EXCLUDED.detection_image,
          upscaled_image=EXCLUDED.upscaled_image
        """
        ),
        {"a": aoi_id, "n": submit_name, "num": n_found, "raw": raw_png, "det": det_png, "up": up_png},
    )

def _map_provider(provider_str: str) -> ProviderKey:
    # Accept strings like "ESRI", "esri", etc.
    try:
        return ProviderKey[provider_str.upper()]
    except Exception:
        return ProviderKey.ESRI

def process_payload(payload: str):
    """Process a single Kafka message (JSON string)."""
    job = json.loads(payload or "{}")
    aoi_id      = job.get("aoi_id", str(uuid.uuid4()))
    submit_name = job.get("submit_name", aoi_id)
    provider    = job.get("provider", "ESRI")
    zoom        = int(job.get("zoom", 15))
    stride      = int(job.get("stride", 112))
    thresh      = float(job.get("thresh", 0.5))
    upscale     = bool(job.get("upscaled", True))
    bbox_wkt    = job.get("bbox_wkt")

    # Back-compat: allow "upscale_mode": "x4" to imply upscale True
    if "upscale_mode" in job and isinstance(job["upscale_mode"], str):
        upscale = job["upscale_mode"].lower() in ("x4", "true", "yes", "y", "1")

    print(f"[job] aoi_id={aoi_id} name={submit_name} zoom={zoom} stride={stride} thresh={thresh} upscale={upscale}", flush=True)

    eng, ss = _init_partition()
    with eng.begin() as conn:
        try:
            update_status(conn, aoi_id, "queued")

            if not bbox_wkt:
                fail_status(conn, aoi_id, "missing bbox_wkt")
                return

            # Resolve provider from payload (fallback to ESRI)
            pc = ProviderCatalog()
            prov_key = _map_provider(provider)
            provider_obj = pc.get(prov_key)

            # Download
            update_status(conn, aoi_id, "downloading")
            print(f"[download] provider={provider_obj.name} bbox={bbox_wkt}", flush=True)

            # Build AOI from WKT + provider
            aoi = AOIRequest.from_wkt(bbox_wkt, zoom=zoom, provider=provider_obj)

            # Stitch tiles -> PIL.Image(RGB) -> np.ndarray (uint8)
            stitcher = TileStitcher(tile_size=provider_obj.tile_size)
            raw_img = stitcher.stitch(aoi)                         # PIL.Image
            raw_rgb = np.array(raw_img, dtype=np.uint8)            # (H, W, 3), RGB
            raw_png = _png_bytes(cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))

            # Optional upscale
            up_png = None
            if upscale:
                update_status(conn, aoi_id, "upscaling")
                print("[upscale] running RealESRGAN x4", flush=True)
                up_bgr = ss.enhance_np(cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))
                up_png = _png_bytes(up_bgr)
                detect_rgb = cv2.cvtColor(up_bgr, cv2.COLOR_BGR2RGB)
            else:
                detect_rgb = raw_rgb

            # Detect
            update_status(conn, aoi_id, "detecting")
            print("[detect] running ConvNeXt inference & drawing", flush=True)
            overlay_rgb, n_found = detect_and_draw(detect_rgb, None, 224, stride, thresh)
            det_png = _png_bytes(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            # Store
            print(f"[store] detections={n_found}; writing blobs to DB", flush=True)
            upsert_results(conn, aoi_id, submit_name, n_found, raw_png=raw_png, det_png=det_png, up_png=up_png)
            update_status(conn, aoi_id, "completed")
            print(f"[done] aoi_id={aoi_id} ✅ completed", flush=True)

        except Exception as e:
            traceback.print_exc()
            fail_status(conn, aoi_id, f"exception: {e}")

def foreach_partition(rows):
    """Partition handler for foreachPartition: iterate rows, call process_payload."""
    count = 0
    for row in rows:
        try:
            payload = row["json"]
            print(f"[partition] processing record: {payload}", flush=True)
            process_payload(payload)
            count += 1
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            traceback.print_exc()
    print(f"[partition] processed {count} records", flush=True)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("🔧 Starting AOI-Streaming driver…", flush=True)
    print(f"    Kafka broker={KAFKA_BROKER} topic={KAFKA_TOPIC}", flush=True)
    print(f"    DB_URL={DB_URL}", flush=True)
    print(f"    checkpoint={CHECKPOINT_DIR}", flush=True)

    spark = (
        SparkSession.builder
        .appName("AOI-Streaming")
        .config("spark.executor.cores", "1")
        .config("spark.task.cpus", "1")
        .config("spark.dynamicAllocation.enabled", "false")
        .getOrCreate()
    )

    # Kafka source — use earliest for test so existing messages are consumed once
    df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")              # first clean run; ignored once checkpoint exists
        # .option("kafka.group.id", "…")                    # DO NOT set: Spark manages its own group id
        .option("failOnDataLoss", "false")
        .load()
    )

    # Parse JSON payloads and spread work across executors
    values = df.select(F.col("value").cast(StringType()).alias("json")).repartition(3)

    def handle_batch(batch_df, batch_id):
        # Avoid double actions on the same micro-batch; just process it.
        print(f"📦 foreachBatch: batch_id={batch_id}", flush=True)
        batch_df.foreachPartition(foreach_partition)

    query = (
        values.writeStream
        .foreachBatch(handle_batch)
        .outputMode("append")                              # outputMode is ignored by foreachBatch; append is fine
        .trigger(processingTime="5 seconds")
        .option("checkpointLocation", CHECKPOINT_DIR)      # Use a FRESH path after topic/partition changes
        .start()
    )

    print("✅ Stream started. Waiting for data…", flush=True)
    query.awaitTermination()
