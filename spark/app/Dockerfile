FROM apache/spark:3.5.3
USER root
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . /opt/app