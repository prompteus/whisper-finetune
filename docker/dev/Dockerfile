FROM condaforge/mambaforge:22.9.0-1

ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    sudo \
    apt-utils \
    git \
    wget \
    curl \
    jq \
    micro \
    zip \
    unzip \
    openssh-client \
    parallel \
    bat \
    fd-find \
    rsync \
    libmagic-dev \
    ffmpeg

RUN ln -s $(which fdfind) /usr/bin/fd \
 && ln -s $(which batcat) /usr/bin/bat


ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN conda update -y -n base -c conda-forge conda 

USER $USERNAME

COPY conda-env.yaml .
RUN conda create -n venv \
 && mamba env update -n venv --file conda-env.yaml \
 && echo "conda activate venv" >> ~/.bashrc

SHELL ["conda","run", "-n", "venv", "/bin/bash", "-c"]

RUN python -m pip install -U pip setuptools wheel poetry poetry-types \
 && python -m poetry config virtualenvs.options.system-site-packages true \
 && python -m poetry self add poetry-types
