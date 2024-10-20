ARG PYTHON_VERSION=3.9
ARG ROOT_NAME=streamlit_app
ARG EXPOSE_CONTAINER_PORT=8501
ARG SEVER_STREAMLIT_ADDRESS=0.0.0.0



FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    ROOT_NAME=${ROOT_NAME} \
    EXPOSE_CONTAINER_PORT=${EXPOSE_CONTAINER_PORT} \
    SEVER_STREAMLIT_ADDRESS=${SEVER_STREAMLIT_ADDRESS}

RUN mkdir -p /usr/serving/${ROOT_NAME}

COPY requirements.txt ./usr/serving/${ROOT_NAME}

RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
git \
&& rm -rf /var/lib/apt/lists/*

COPY ./serving/streamlit_app /usr/serving/${ROOT_NAME}

WORKDIR /usr/serving/${ROOT_NAME}

# Copy requirements.txt and install dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

EXPOSE ${EXPOSE_CONTAINER_PORT}

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=${EXPOSE_CONTAINER_PORT}", "--server.address=${SEVER_STREAMLIT_ADDRESS}"]