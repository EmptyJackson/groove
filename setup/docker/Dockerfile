FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Set timezone
ENV TZ=Europe/London DEBIAN_FRONTEND=noninteractive

# Add old libraries (Python 3.8) to Ubuntu 22.04
RUN apt update
RUN apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y \
    git \
    python3.8 \
    python3-pip \
    python3.8-venv \
    python3-setuptools \
    python3-wheel

# Create local user
# https://jtreminio.com/blog/running-docker-containers-as-current-host-user/
ARG UID
ARG GID
RUN if [ ${UID:-0} -ne 0 ] && [ ${GID:-0} -ne 0 ]; then \
    groupadd -g ${GID} duser &&\
    useradd -l -u ${UID} -g duser duser &&\
    install -d -m 0755 -o duser -g duser /home/duser &&\
    chown --changes --silent --no-dereference --recursive ${UID}:${GID} /home/duser \
    ;fi

USER duser
WORKDIR /home/duser

# Install Python packages
ENV PATH="/home/duser/.local/bin:$PATH"
RUN python3.8 -m pip install --upgrade pip
ARG REQS
RUN python3.8 -m pip install $REQS -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /home/duser/groove
