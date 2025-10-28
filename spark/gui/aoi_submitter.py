#!/usr/bin/env python3
# Streamlit AOI job submitter + status/preview
# - Enqueue jobs (DB row + Kafka message with full payload expected by Spark)
# - Refresh to update status list
# - Preview stored images (raw, optional upscaled, detection) for completed jobs
#
# deps: streamlit, streamlit-folium, folium, sqlalchemy, psycopg2-binary, kafka-python

from __future__ import annotations
import os, json, uuid, re
from typing import Optional, Tuple, List, Dict

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from kafka import KafkaProducer

# -------------------------
# Config (env with defaults)
# -------------------------
DB_URL        = os.getenv("DB_URL",        "postgresql+psycopg2://aoi:aoi@postgres:5432/aoi")
KAFKA_BROKER  = os.getenv("KAFKA_BROKER",  "redpanda:9092")
KAFKA_TOPIC   = os.getenv("KAFKA_TOPIC",   "aoi_jobs")

# -------------------------
# Helpers: general
# -------------------------
def _to_bytes(x):
    """Convert psycopg2 BYTEA (memoryview) or None to bytes/None."""
    return None if x is None else (x if isinstance(x, (bytes, bytearray)) else bytes(x))

# -------------------------
# Helpers: geometry
# -------------------------
def extract_bbox(draw: dict) -> Optional[Tuple[float, float, float, float]]:
    """Extract bounding box (west,south,east,north) from folium draw tool output."""
    if not draw:
        return None
    last = draw.get("last_active_drawing")
    if not last:
        all_drawings = draw.get("all_drawings") or []
        if not all_drawings:
            return None
        last = all_drawings[-1]
    geom = (last or {}).get("geometry", {})
    if geom.get("type") != "Polygon":
        return None
    coords = geom["coordinates"][0]
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    return (min(lons), min(lats), max(lons), max(lats))

def bbox_to_rect_wkt(w, s, e, n) -> str:
    """Convert bbox to WKT polygon."""
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"

NAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,64}$")

# -------------------------
# DB Helpers
# -------------------------
def engine() -> Engine:
    return create_engine(DB_URL, future=True, pool_pre_ping=True)

def job_name_exists(conn, submit_name: str) -> bool:
    row = conn.execute(
        text("SELECT 1 FROM aoi_jobs WHERE submit_name=:n LIMIT 1"),
        {"n": submit_name}
    ).fetchone()
    return row is not None

def insert_job(conn, aoi_id: uuid.UUID, submit_name: str,
               zoom: int, bbox_wkt: str, upscaled: bool, stride: int, thresh: float):
    conn.execute(
        text("""
        INSERT INTO aoi_jobs (
          aoi_id, submit_name, submitted_at, status, zoom, bbox_wkt, upscaled, stride, thresh
        )
        VALUES (:id, :name, now(), 'queued', :zoom, :bbox, :upscaled, :stride, :thresh)
        """),
        {
            "id": str(aoi_id),
            "name": submit_name,
            "zoom": zoom,
            "bbox": bbox_wkt,
            "upscaled": upscaled,
            "stride": stride,
            "thresh": thresh
        }
    )

def recent_jobs(conn, limit: int = 10) -> List[Dict]:
    rows = conn.execute(text("""
        SELECT submit_name, status, submitted_at
        FROM aoi_jobs
        ORDER BY submitted_at DESC
        LIMIT :lim
    """), {"lim": limit}).fetchall()
    return [dict(r._mapping) for r in rows]

def completed_jobs(conn, limit: int = 50) -> List[Dict]:
    rows = conn.execute(text("""
        SELECT j.submit_name, j.aoi_id, j.finished_at, j.submitted_at
        FROM aoi_jobs j
        WHERE j.status = 'completed'
        ORDER BY COALESCE(j.finished_at, j.submitted_at) DESC
        LIMIT :lim
    """), {"lim": limit}).fetchall()
    return [dict(r._mapping) for r in rows]

def load_result_images_by_submit_name(conn, submit_name: str) -> Optional[Dict[str, bytes]]:
    """
    Return dict with keys: raw, upscaled, detection -> tuples (bytes, mime|None).
    Converts BYTEA (memoryview) to bytes for Streamlit compatibility.
    """
    row = conn.execute(text("""
        SELECT r.raw_image, r.raw_mime,
               r.upscaled_image, r.upscaled_mime,
               r.detection_image, r.detection_mime
        FROM results r
        JOIN aoi_jobs j ON j.aoi_id = r.aoi_id
        WHERE j.submit_name = :name
        LIMIT 1
    """), {"name": submit_name}).fetchone()
    if not row:
        return None
    m = row._mapping
    raw_bytes      = _to_bytes(m["raw_image"])
    upscaled_bytes = _to_bytes(m["upscaled_image"])
    det_bytes      = _to_bytes(m["detection_image"])
    return {
        "raw":       (raw_bytes,      (m.get("raw_mime") or "image/png") if raw_bytes        else None),
        "upscaled":  (upscaled_bytes, (m.get("upscaled_mime") or "image/png") if upscaled_bytes else None),
        "detection": (det_bytes,      (m.get("detection_mime") or "image/png") if det_bytes    else None),
    }

# -------------------------
# Kafka
# -------------------------
def produce_kafka(payload: dict):
    """Produce the full job payload to Kafka."""
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda v: (v or "").encode("utf-8"),
        linger_ms=10,
        retries=3,
    )
    key = payload.get("aoi_id", "")
    fut = producer.send(KAFKA_TOPIC, key=key, value=payload)
    rec = fut.get(timeout=10)
    producer.flush(5)
    producer.close(5)
    return rec

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="AOI Job Submitter", layout="wide")
st.title("🛰 AOI → Enqueue job for Spark processor")

# ---- Sidebar: job params + submit
with st.sidebar:
    st.markdown("### Job Parameters")
    submit_name = st.text_input(
        "Job name (unique)",
        placeholder="e.g. port_of_shanghai_2025_10_27",
        help="Allowed: letters, numbers, dots, underscores, hyphens (3–64 chars)."
    )
    zoom = st.slider("Zoom", 1, 22, 15)
    upscaled = st.checkbox("Upscale (RealESRGAN ×4)", value=True)
    stride = st.select_slider("Stride", options=[56, 64, 96, 112, 128, 160, 192], value=112)
    thresh = st.slider("Ship threshold", 0.1, 0.95, 0.5, 0.05)
    # Optional provider choice (processor defaults to ESRI if omitted)
    provider = st.selectbox("Provider", options=["ESRI", "OSM", "SENTINEL_EOX"], index=0)

    submit = st.button("🚀 Submit Job", use_container_width=True)

left, right = st.columns([3, 2], gap="large")

# ---- Map draw on the left
with left:
    m = folium.Map(location=(20, 0), zoom_start=3,
                   tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                   attr="© OpenStreetMap contributors",
                   control_scale=True)

    Draw(
        draw_options={"polyline": False, "polygon": False, "circle": False,
                      "marker": False, "circlemarker": False, "rectangle": True},
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    draw = st_folium(m, height=560, width=None,
                     returned_objects=["last_active_drawing", "all_drawings"])

# ---- Status + Preview on the right
with right:
    st.subheader("Status")
    top_row = st.columns([1, 1])
    with top_row[0]:
        refresh = st.button("🔄 Refresh statuses", use_container_width=True)
    with top_row[1]:
        st.caption(f"DB_URL={DB_URL} · KAFKA_BROKER={KAFKA_BROKER} · TOPIC={KAFKA_TOPIC}")

    status = st.empty()
    info = st.empty()

    st.markdown("### Recent Jobs")
    jobs_box = st.empty()

    st.markdown("---")
    st.markdown("### Preview Completed Job")
    sel_col, btn_col = st.columns([2, 1])

    # Will fill these after we talk to DB
    completed = []
    recent = []

    try:
        eng = engine()
        with eng.begin() as conn:
            recent = recent_jobs(conn, limit=12)
            completed = completed_jobs(conn, limit=50)
    except Exception:
        pass

    # Recent table
    if recent:
        jobs_box.table(recent)
    else:
        jobs_box.caption("No jobs yet or DB not reachable.")

    # Completed selector
    completed_names = [row["submit_name"] for row in completed] if completed else []
    with sel_col:
        chosen = st.selectbox(
            "Select a completed job to preview",
            options=completed_names,
            index=0 if completed_names else None,
            placeholder="Completed jobs will appear here…",
        )
    with btn_col:
        show = st.button("👁️ Show images", use_container_width=True, disabled=not bool(chosen))

    # Preview panel
    preview_area = st.container()

    if refresh:
        st.rerun()

    if show and chosen:
        try:
            with engine().begin() as conn:
                blobs = load_result_images_by_submit_name(conn, chosen)
            if not blobs:
                preview_area.warning("No results found for that job yet.")
            else:
                raw, raw_mime = blobs["raw"]
                up_tuple = blobs["upscaled"]
                det, det_mime = blobs["detection"]

                st.success(f"Showing stored images for **{chosen}**")
                img_cols = st.columns(3)

                # Raw
                with img_cols[0]:
                    st.markdown("**Raw**")
                    if raw:
                        st.image(raw, caption="raw_image", use_column_width=True)
                    else:
                        st.caption("— missing —")

                # Upscaled (optional)
                with img_cols[1]:
                    st.markdown("**Upscaled (optional)**")
                    if up_tuple and up_tuple[0]:
                        st.image(up_tuple[0], caption="upscaled_image", use_column_width=True)
                    else:
                        st.caption("— not generated —")

                # Detection
                with img_cols[2]:
                    st.markdown("**Detection**")
                    if det:
                        st.image(det, caption="detection_image", use_column_width=True)
                    else:
                        st.caption("— missing —")

        except Exception as e:
            preview_area.error(f"Failed to load results: {e}")

# -------------------------
# Submit Action
# -------------------------
if submit:
    name = (submit_name or "").strip()
    if not NAME_RE.match(name):
        st.error("Invalid job name. Use 3–64 chars: letters, numbers, '.', '_', '-'.")
        st.stop()

    bbox = extract_bbox(draw)
    if not bbox:
        st.error("Please draw a rectangle on the map first.")
        st.stop()

    w, s, e, n = bbox
    bbox_wkt = bbox_to_rect_wkt(w, s, e, n)
    aoi_id = uuid.uuid4()

    # Insert a row in DB first (status='queued')
    try:
        eng = engine()
        with eng.begin() as conn:
            if job_name_exists(conn, name):
                st.error("That job name already exists. Choose another.")
                st.stop()
            insert_job(conn, aoi_id, name, int(zoom), bbox_wkt, bool(upscaled), int(stride), float(thresh))
        st.success("DB: job row inserted (status='queued').")
    except Exception as db_err:
        st.error(f"DB insert failed: {db_err}")
        st.stop()

    # Build FULL payload (this is what the Spark processor expects)
    payload = {
        "aoi_id": str(aoi_id),
        "submit_name": name,
        "bbox_wkt": bbox_wkt,
        "zoom": int(zoom),
        "upscaled": bool(upscaled),
        "stride": int(stride),
        "thresh": float(thresh),
        "provider": provider or "ESRI",   # processor defaults to ESRI if missing
    }

    # Produce to Kafka
    try:
        meta = produce_kafka(payload)
        st.success(f"Kafka: produced to {KAFKA_TOPIC} @ partition {meta.partition}, offset {meta.offset}.")
        st.info(f"✅ Enqueued job {name} (aoi_id={aoi_id})")
        st.code(json.dumps(payload, indent=2), language="json")
        st.rerun()
    except Exception as k_err:
        st.error(f"Kafka produce failed: {k_err}")
        st.stop()
