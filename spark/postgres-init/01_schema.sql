-- 1) Submissions
CREATE TABLE IF NOT EXISTS aoi_jobs (
  aoi_id       UUID PRIMARY KEY,
  submit_name  TEXT NOT NULL UNIQUE,              -- user-chosen unique name
  submitted_at TIMESTAMPTZ DEFAULT now(),
  started_at   TIMESTAMPTZ,
  finished_at  TIMESTAMPTZ,
  status       TEXT NOT NULL DEFAULT 'queued',    -- queued|running|completed|failed
  zoom         INT,
  bbox_wkt     TEXT,
  upscaled     BOOLEAN DEFAULT false,
  stride       INT,
  thresh       FLOAT
);

-- 2) Results (one row per job) — images stored inline
CREATE TABLE IF NOT EXISTS results (
  aoi_id            UUID PRIMARY KEY
                    REFERENCES aoi_jobs(aoi_id) ON DELETE CASCADE,
  submit_name       TEXT NOT NULL,
  num_detections    INT  NOT NULL DEFAULT 0,

  -- raw is mandatory
  raw_image         BYTEA NOT NULL,
  raw_mime          TEXT  NOT NULL DEFAULT 'image/png',

  -- upscaled is optional
  upscaled_image    BYTEA,
  upscaled_mime     TEXT,

  -- detection is mandatory
  detection_image   BYTEA NOT NULL,
  detection_mime    TEXT  NOT NULL DEFAULT 'image/png',

  created_at        TIMESTAMPTZ DEFAULT now()
);

-- Optional helpful index if you query by submit name often
CREATE INDEX IF NOT EXISTS idx_results_submit_name ON results(submit_name);
