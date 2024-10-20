ARG PYTHON_VERSION=3.9
ARG PYTHONUNBUFFERED=1
ARG ROOT_NAME=gradio
ARG GRADIO_SERVER_NAME="0.0.0.0"
ARG EXPOSE_CONTAINER_PORT=7860

FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=${PYTHONUNBUFFERED} \
ROOT_NAME=${ROOT_NAME}

RUN mkdir -p /usr/serving/${ROOT_NAME}

COPY requirements.txt ./

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./serving/gradio /usr/serving/${ROOT_NAME}


WORKDIR /usr/serving/${ROOT_NAME}

RUN pip install -r requirements.txt

EXPOSE ${EXPOSE_CONTAINER_PORT}

ENV GRADIO_SERVER_NAME=${GRADIO_SERVER_NAME}

CMD ["python", "gradio.app.py"]