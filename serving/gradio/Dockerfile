FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

RUN mdkdir -p /usr/serving/gradio

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./app /usr/serving/gradio

WORKDIR /usr/serving/gradio

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "gradio.app.py"]