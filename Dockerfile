# syntax=docker/dockerfile:experimental

# from
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# apt source
RUN apt-get update && \
    apt-get -y upgrade

# apt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git libgl1-mesa-dev libgtk2.0-dev tzdata locales && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# locale
ENV PYTHONUNBUFFERED=1

# cache
RUN mkdir -p /root/.cache/torch/checkpoints/ && \
    mkdir -p /root/.cache/pip/

# workspace
RUN mkdir -p /workspace
WORKDIR /workspace

# install packages
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
