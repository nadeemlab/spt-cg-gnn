# Use cuda.Dockerfile if you have a CUDA-enabled GPU
FROM python:3.11-slim-buster
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
ENV PIP_NO_CACHE_DIR=1
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install dgl -f https://data.dgl.ai/wheels/repo.html
RUN pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
ENV DGLBACKEND=pytorch
RUN pip install \
    spatialprofilingtoolbox[cggnn]>=0.17.3 \
    cg-gnn>=0.3.1
ADD . /app
RUN chmod +x train.py
RUN mv train.py /usr/local/bin/spt-plugin-train-on-graphs
EXPOSE 80
