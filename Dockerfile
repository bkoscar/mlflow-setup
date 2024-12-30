FROM ubuntu:22.04 AS mlflow_base 

LABEL maintainer="Omar" \
      description="MLflow base image" \
      version="1.0"

ENV MLFLOW_USER=mlflow
ENV HOME_DIR=/home/${MLFLOW_USER}
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos "" ${MLFLOW_USER} \
    && usermod -aG sudo ${MLFLOW_USER} \
    && echo "${MLFLOW_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p ${HOME_DIR} \
    && chown -R ${MLFLOW_USER}:${MLFLOW_USER} ${HOME_DIR}

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    zlib1g-dev \
    wget \
    curl \
    sudo \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER ${MLFLOW_USER}

WORKDIR ${HOME_DIR}

COPY src ${HOME_DIR}/src

# Crear el directorio mlruns
# RUN sudo mkdir -p ${HOME_DIR}/src/mlruns

# Establecer MLFLOW_TRACKING_URI como variable de entorno
ENV MLFLOW_TRACKING_URI=file://${HOME_DIR}/src/mlruns

RUN python3.11 -m venv ${HOME_DIR}/mlflow_venv \
    && source ${HOME_DIR}/mlflow_venv/bin/activate \
    && pip install --upgrade pip \
    && pip install mlflow \
    && pip install mlflow[extras]
    # && pip install --no-cache-dir -r ${HOME_DIR}/src/requirements.txt

EXPOSE 5001

# CMD ["/bin/bash", "-c", "source ${HOME_DIR}/mlflow_venv/bin/activate && mlflow ui --host 0.0.0.0 --port 5001"]
CMD ["/bin/bash", "-c", "source ${HOME_DIR}/mlflow_venv/bin/activate && mlflow ui --host 0.0.0.0 --port 5001 --backend-store-uri file:///home/mlflow/src/mlruns"]
