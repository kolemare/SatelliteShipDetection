# Streamlit AOI job submitter
FROM python:3.11-slim

# Avoid interactive tzdata etc.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install runtime libs that psycopg2-binary sometimes needs (ca-certificates, curl are handy)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy and install deps first (better caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app
COPY aoi_submitter.py .

# Streamlit listens on 8501
EXPOSE 8501

# Environment defaults (override in compose)
ENV DB_URL=postgresql+psycopg2://aoi:aoi@postgres:5432/aoi \
    KAFKA_BROKER=redpanda:9092 \
    KAFKA_TOPIC=aoi_jobs

# Run Streamlit
CMD ["streamlit", "run", "aoi_submitter.py", "--server.port=8501", "--server.address=0.0.0.0"]
