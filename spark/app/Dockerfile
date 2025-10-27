# ./app/Dockerfile
FROM apache/spark:4.0.1-scala2.13-java17-python3-r-ubuntu

USER root
# OpenCV runtime etc.
RUN apt-get update && apt-get install -y python3-pip python3-opencv libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Your app + weights
COPY . /opt/app
