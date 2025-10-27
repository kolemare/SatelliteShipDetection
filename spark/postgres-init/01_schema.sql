-- 1) Submissions
CREATE TABLE IF NOT EXISTS aoi_jobs (
  aoi_id       UUID PRIMARY KEY,
  submit_name  TEXT NOT NULL UNIQUE,        -- user-chosen unique name
  submitted_at TIMESTAMPTZ DEFAULT now(),
  started_at   TIMESTAMPTZ,
  finished_at  TIMESTAMPTZ,
  status       TEXT NOT NULL DEFAULT 'queued',  -- queued|running|completed|failed
  zoom         INT,
  bbox_wkt     TEXT,
  upscaled     BOOLEAN DEFAULT false,
  stride       INT,
  thresh       FLOAT
);

-- 2) Results (one row per job)
CREATE TABLE IF NOT EXISTS results (
  aoi_id        UUID PRIMARY KEY REFERENCES aoi_jobs(aoi_id) ON DELETE CASCADE,
  submit_name   TEXT NOT NULL,
  raw_uri       TEXT,
  upscaled_uri  TEXT,
  detection_uri TEXT,
  created_at    TIMESTAMPTZ DEFAULT now()
);
