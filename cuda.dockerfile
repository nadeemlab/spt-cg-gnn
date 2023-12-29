FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
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
RUN pip install \
    spatialprofilingtoolbox[cggnn]>=0.17.4 \
    cg-gnn>=0.3.1
RUN pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
RUN pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
ENV DGLBACKEND=pytorch
RUN pip install \
    spatialprofilingtoolbox[cggnn]>=0.17.3 \
    cg-gnn>=0.3.1
ADD . /app
RUN chmod +x train.py
RUN mv train.py /usr/local/bin/spt-plugin-train-on-graphs
RUN chmod +x /app/print_graph_config.sh
RUN mv /app/print_graph_config.sh /usr/local/bin/spt-plugin-print-graph-request-configuration
EXPOSE 80
