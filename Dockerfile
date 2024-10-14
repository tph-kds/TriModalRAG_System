# Arguments Initialization
ARG PYTHON_VERSION=3.9
ARG RAG_VERSION=0.0.1
ARG ROOT_DIR=/trim_rag
ARG PYTHONUNBUFFERED=1



FROM python:${PYTHON_VERSION}-slim-buster as base

RUN set -ex \
  && apt-get update \
  && apt-get upgrade -y \
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*


FROM base as builder

ENV PYTHONUNBUFFERED=${PYTHONUNBUFFERED} \
    ROOT_DIR=${ROOT_DIR}

RUN mdkir -p /ROOT_DIR
WORKDIR /ROOT_DIR

COPY . /ROOT_DIR

RUN pip install --no-cache-dir -r requirements.txt








