CREATE TABLE IF NOT EXISTS aoi_jobs (
  aoi_id UUID PRIMARY KEY,
  submitted_at TIMESTAMPTZ DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  status TEXT NOT NULL DEFAULT 'queued',  -- queued | running | completed | failed
  user_id TEXT,
  provider TEXT,
  zoom INT,
  bbox_wkt TEXT,
  upscale_mode TEXT,  -- none | x4 |
  stride INT,
  thresh FLOAT,
  error_text TEXT
);

CREATE TABLE IF NOT EXISTS mosaics (
  aoi_id UUID REFERENCES aoi_jobs(aoi_id),
  source TEXT,           -- raw | x4 |
  width INT,
  height INT,
  crs TEXT,
  storage_uri TEXT,
  etag TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS detections (
  aoi_id UUID REFERENCES aoi_jobs(aoi_id),
  chip_id TEXT,
  geom_wkt TEXT,         -- polygon/point as WKT
  score FLOAT,
  class TEXT DEFAULT 'ship',
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS artifacts (
  aoi_id UUID REFERENCES aoi_jobs(aoi_id),
  kind TEXT,             -- overlay | thumbnail | raw | upscaled
  storage_uri TEXT,
  size_bytes BIGINT,
  created_at TIMESTAMPTZ DEFAULT now()
);
