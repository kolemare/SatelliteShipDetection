#!/usr/bin/env python3
# Streamlit AOI job submitter (DB seed + Kafka produce + job list)
# Uses Esri base map with OSM toggle, no labels overlay.
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
# Helpers
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

def job_name_exists(engine: Engine, submit_name: str) -> bool:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT 1 FROM aoi_jobs WHERE submit_name=:n LIMIT 1"),
            {"n": submit_name}
        ).fetchone()
    return row is not None

def insert_job(engine: Engine, aoi_id: uuid.UUID, submit_name: str,
               zoom: int, bbox_wkt: str, upscaled: bool, stride: int, thresh: float):
    with engine.begin() as conn:
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

def produce_kafka(aoi_id: uuid.UUID, submit_name: str):
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda v: v.encode("utf-8") if v else None,
        linger_ms=10,
        retries=3,
    )
    payload = {"aoi_id": str(aoi_id), "submit_name": submit_name}
    future = producer.send(KAFKA_TOPIC, key=str(aoi_id), value=payload)
    rec = future.get(timeout=10)
    producer.flush(5)
    producer.close(5)
    return rec

def recent_jobs(engine: Engine, limit: int = 10) -> List[Dict]:
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT submit_name, status, submitted_at
            FROM aoi_jobs
            ORDER BY submitted_at DESC
            LIMIT :lim
        """), {"lim": limit}).fetchall()
    return [dict(r._mapping) for r in rows]

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="AOI Job Submitter", layout="wide")
st.title("🛰 AOI → Enqueue job for Spark processor")

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
    submit = st.button("🚀 Submit Job", use_container_width=True)

left, right = st.columns([3, 2], gap="large")

with left:
    # Use OpenStreetMap for clarity (city names), but processing uses Esri imagery.
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

with right:
    st.subheader("Status")
    status = st.empty()
    info = st.empty()

    st.markdown("### Recent Jobs")
    jobs_box = st.empty()

    def render_jobs():
        try:
            _engine = create_engine(DB_URL, future=True)
            rows = recent_jobs(_engine, limit=10)
            if rows:
                jobs_box.table(rows)
            else:
                jobs_box.caption("No jobs yet.")
        except Exception:
            jobs_box.caption("Can't load recent jobs (DB not ready).")

    render_jobs()

# -------------------------
# Action
# -------------------------
if submit:
    name = (submit_name or "").strip()
    if not NAME_RE.match(name):
        status.error("Invalid job name. Use 3–64 chars: letters, numbers, '.', '_', '-'.")
        st.stop()

    bbox = extract_bbox(draw)
    if not bbox:
        status.error("Please draw a rectangle on the map first.")
        st.stop()

    w, s, e, n = bbox
    bbox_wkt = bbox_to_rect_wkt(w, s, e, n)
    aoi_id = uuid.uuid4()

    status.info("Connecting to Postgres…")
    try:
        engine = create_engine(DB_URL, future=True)
        if job_name_exists(engine, name):
            status.error("That job name already exists. Choose another.")
            st.stop()
        insert_job(engine, aoi_id, name, zoom, bbox_wkt, upscaled, stride, thresh)
        status.success("DB: job row inserted (status='queued').")
    except Exception as db_err:
        status.error(f"DB insert failed: {db_err}")
        st.stop()

    status.info("Producing Kafka message…")
    try:
        meta = produce_kafka(aoi_id, name)
        status.success(f"Kafka: produced to {KAFKA_TOPIC} @ partition {meta.partition}, offset {meta.offset}.")
        info.success(f"✅ Enqueued job {name} (aoi_id={aoi_id})")
        st.code(json.dumps({
            "aoi_id": str(aoi_id),
            "submit_name": name,
            "zoom": zoom, "upscaled": upscaled, "stride": stride, "thresh": thresh,
            "bbox_wkt": bbox_wkt
        }, indent=2), language="json")

        render_jobs()
        st.rerun()
    except Exception as k_err:
        status.error(f"Kafka produce failed: {k_err}")
        st.stop()

st.caption(f"DB_URL={DB_URL} · KAFKA_BROKER={KAFKA_BROKER} · TOPIC={KAFKA_TOPIC}")
