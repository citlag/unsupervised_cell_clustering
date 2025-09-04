FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    build-essential \
	python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh \
    && bash miniconda.sh -b -p $CONDA_DIR \
    && rm miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create a working directory
WORKDIR /workspace

# Copy environment and requirements
COPY requirements.txt ./

# Create conda environment
RUN conda create -n myenv python=3.9
#RUN conda run -n myenv pip install -r requirements.txt

# Activate environment by default
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
RUN mkdir -p /lib
COPY ./lib/ /lib/
RUN mkdir -p /data
COPY ./data/ /data/
RUN mkdir -p /hibou-L
COPY ./hibou-L/ /hibou-L/

# Copy scripts
COPY read_cell_detections_perform_clustering.py config.yaml ./

# Expose Jupyter port (optional, if you plan to use notebooks inside container)
EXPOSE 8888

# Run script
CMD ["python", "read_cell_detections_perform_clustering.py", "-config", "/config.yaml", "-output", "/data", "-wsi", "/data/TCGA-V5-A7RE-11A-01-TS1.57401526-EF9E-49AC-8FF6-B4F9652311CE.svs", "-anno", "/data/debug_cells.geojson"]