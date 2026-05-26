FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget unzip build-essential cmake ninja-build \
    libgl1-mesa-glx libglib2.0-0 libglew-dev libassimp-dev \
    libboost-all-dev libgtk-3-dev libopencv-dev \
    python3 python3-pip python-is-python3 tzdata \
    && ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# R2-Gaussian
WORKDIR /workspace/ct-metal-reduction-prerelease

RUN pip install -r requirements.txt

RUN pip install -e submodules/simple-knn && \
    pip install -e submodules/xray-gaussian-rasterization-voxelization

# TIGRE
WORKDIR /workspace

RUN wget https://github.com/CERN/TIGRE/archive/refs/tags/v2.3.zip && \
    unzip v2.3.zip && \
    pip install TIGRE-2.3/Python --no-build-isolation && \
    rm v2.3.zip

# XrayPhysics
RUN git clone https://github.com/kylechampley/XrayPhysics.git /workspace/XrayPhysics

WORKDIR /workspace/XrayPhysics

RUN pip install -v .

WORKDIR /workspace/ct-metal-reduction-prerelease

CMD ["/bin/bash"]