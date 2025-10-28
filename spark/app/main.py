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

# --------------------------
# DB helpers (schema-correct)
# --------------------------
def ensure_job_row(conn, *, aoi_id, submit_name, zoom, bbox_wkt, upscaled, stride, thresh):
    """
    Insert the aoi_jobs row if it doesn't exist yet.
    Uses your exact columns; no updated_at.
    submit_name is UNIQUE, so ON CONFLICT on aoi_id is safest.
    """
    sql = text("""
        INSERT INTO aoi_jobs (aoi_id, submit_name, zoom, bbox_wkt, upscaled, stride, thresh)
        VALUES (:aoi_id, :submit_name, :zoom, :bbox_wkt, :upscaled, :stride, :thresh)
        ON CONFLICT (aoi_id) DO NOTHING
    """)
    conn.execute(sql, {
        "aoi_id": aoi_id,
        "submit_name": submit_name,
        "zoom": zoom,
        "bbox_wkt": bbox_wkt,
        "upscaled": upscaled,
        "stride": stride,
        "thresh": thresh,
    })

def mark_status(conn, aoi_id: str, status: str):
    """
    Only uses columns in your schema: status, started_at, finished_at.
    """
    sql = text("""
        UPDATE aoi_jobs
        SET
          status = :status,
          started_at  = CASE
                           WHEN :status = 'running' AND started_at IS NULL
                           THEN now()
                           ELSE started_at
                         END,
          finished_at = CASE
                           WHEN :status IN ('completed','failed')
                           THEN now()
                           ELSE finished_at
                         END
        WHERE aoi_id = :aoi_id
    """)
    conn.execute(sql, {"status": status, "aoi_id": aoi_id})

def mark_started(conn, aoi_id: str):
    mark_status(conn, aoi_id, "running")

def mark_completed(conn, aoi_id: str):
    mark_status(conn, aoi_id, "completed")

def mark_failed(conn, aoi_id: str):
    mark_status(conn, aoi_id, "failed")

def upsert_results(conn, *, aoi_id, submit_name, num_detections,
                   raw_png: bytes, det_png: bytes, up_png: bytes | None):
    """
    Uses only your 'results' columns. Sets *_mime to 'image/png' (or NULL for upscaled).
    """
    sql = text("""
        INSERT INTO results (
          aoi_id, submit_name, num_detections,
          raw_image, raw_mime,
          upscaled_image, upscaled_mime,
          detection_image, detection_mime
        )
        VALUES (
          :aoi_id, :submit_name, :num_det,
          :raw_img, 'image/png',
          :up_img,  CASE WHEN :up_img IS NULL THEN NULL ELSE 'image/png' END,
          :det_img, 'image/png'
        )
        ON CONFLICT (aoi_id) DO UPDATE SET
          submit_name     = EXCLUDED.submit_name,
          num_detections  = EXCLUDED.num_detections,
          raw_image       = EXCLUDED.raw_image,
          raw_mime        = EXCLUDED.raw_mime,
          upscaled_image  = EXCLUDED.upscaled_image,
          upscaled_mime   = EXCLUDED.upscaled_mime,
          detection_image = EXCLUDED.detection_image,
          detection_mime  = EXCLUDED.detection_mime
    """)
    conn.execute(sql, {
        "aoi_id": aoi_id,
        "submit_name": submit_name,
        "num_det": num_detections,
        "raw_img": raw_png,
        "up_img": up_png,
        "det_img": det_png,
    })

# --------------------------
# misc
# --------------------------
def _map_provider(provider_str: str) -> ProviderKey:
    try:
        return ProviderKey[provider_str.upper()]
    except Exception:
        return ProviderKey.ESRI

# --------------------------
# core job
# --------------------------
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

    try:
        # 1) Ensure row exists (commit immediately so others can see it)
        with eng.begin() as conn:
            ensure_job_row(
                conn,
                aoi_id=aoi_id,
                submit_name=submit_name,
                zoom=zoom,
                bbox_wkt=bbox_wkt,
                upscaled=upscale,
                stride=stride,
                thresh=thresh,
            )
        print("[status] ensured aoi_jobs row exists", flush=True)

        # Validate inputs early and mark failed if needed
        if not bbox_wkt:
            print("[ERROR] missing bbox_wkt", flush=True)
            with eng.begin() as conn:
                mark_failed(conn, aoi_id)
            return

        # 2) Mark queued → running (separate commit so UI/psql sees progress)
        with eng.begin() as conn:
            mark_status(conn, aoi_id, "queued")
            mark_started(conn, aoi_id)
        print(f"[status] aoi_id={aoi_id} → running", flush=True)

        # 3) Heavy work (no DB writes here)
        pc = ProviderCatalog()
        prov_key = _map_provider(provider)
        provider_obj = pc.get(prov_key)

        print(f"[download] provider={provider_obj.name} bbox={bbox_wkt}", flush=True)
        aoi = AOIRequest.from_wkt(bbox_wkt, zoom=zoom, provider=provider_obj)

        stitcher = TileStitcher(tile_size=provider_obj.tile_size)
        raw_img = stitcher.stitch(aoi)                         # PIL.Image
        raw_rgb = np.array(raw_img, dtype=np.uint8)            # (H, W, 3), RGB
        raw_png = _png_bytes(cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))

        # Optional upscale
        up_png = None
        if upscale:
            print("[upscale] RealESRGAN x4", flush=True)
            up_bgr = ss.enhance_np(cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))
            up_png = _png_bytes(up_bgr)
            detect_rgb = cv2.cvtColor(up_bgr, cv2.COLOR_BGR2RGB)
        else:
            detect_rgb = raw_rgb

        # Detect
        print("[detect] ConvNeXt inference & drawing", flush=True)
        overlay_rgb, n_found = detect_and_draw(detect_rgb, None, 224, stride, thresh)
        det_png = _png_bytes(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

        # 4) Store results and mark completed (single commit)
        with eng.begin() as conn:
            upsert_results(
                conn,
                aoi_id=aoi_id,
                submit_name=submit_name,
                num_detections=n_found,
                raw_png=raw_png,
                det_png=det_png,
                up_png=up_png,
            )
            mark_completed(conn, aoi_id)

        print(f"[done] aoi_id={aoi_id} ✅ completed (detections={n_found})", flush=True)

    except Exception as e:
        traceback.print_exc()
        # Ensure failure status is committed even if above failed mid-way
        try:
            with eng.begin() as conn:
                mark_failed(conn, aoi_id)
        except Exception:
            traceback.print_exc()
        print(f"[fail] aoi_id={aoi_id}: {e}", flush=True)

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
        .option("startingOffsets", "earliest")  # ignored once checkpoint exists
        .option("failOnDataLoss", "false")
        .load()
    )

    # Parse JSON payloads and spread work across executors
    values = df.select(F.col("value").cast(StringType()).alias("json")).repartition(3)

    def handle_batch(batch_df, batch_id):
        print(f"📦 foreachBatch: batch_id={batch_id}", flush=True)
        batch_df.foreachPartition(foreach_partition)

    query = (
        values.writeStream
        .foreachBatch(handle_batch)
        .outputMode("append")  # ignored by foreachBatch; OK
        .trigger(processingTime="5 seconds")
        .option("checkpointLocation", CHECKPOINT_DIR)
        .start()
    )

    print("✅ Stream started. Waiting for data…", flush=True)
    query.awaitTermination()
