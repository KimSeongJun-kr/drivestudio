FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Initialize conda
RUN /opt/conda/bin/conda init bash

# Setup conda environment
RUN conda create -n drivestudio python=3.9 -y

# Activate conda environment
SHELL ["conda", "run", "-n", "drivestudio", "/bin/bash", "-c"]

# Add conda env bin directory to PATH
ENV PATH="/opt/conda/envs/drivestudio/bin:$PATH"

# Default working directory
WORKDIR /workspace/drivestudio

# Copy requirements.txt
COPY requirements.txt .

# ENV for pytorch3d install
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.6"
# For general settings, use below
# TORCH_CUDA_ARCH_LIST="Turing Ampere Ada Hopper"

# Install packages in requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0 && \
    pip install git+https://github.com/facebookresearch/pytorch3d.git && \
    pip install git+https://github.com/NVlabs/nvdiffrast

# Copy $ install third_party/smplx
COPY third_party/smplx ./third_party/smplx
WORKDIR /workspace/drivestudio/third_party/smplx
RUN pip install .

# init work directory
WORKDIR /workspace/drivestudio

CMD ["/bin/bash"]
