FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    sudo \
    python3-pip \
    python3-venv \
    apt-utils \
    git \
    wget \
    curl \
    ffmpeg \
    zip \
    unzip \
    parallel 


ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

RUN pip install -U pip setuptools wheel poetry
RUN python3 -m poetry config virtualenvs.in-project true

