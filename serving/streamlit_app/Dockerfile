FROM python:3.9-slim

RUN mdkdir -p /usr/serving/streamlit_app

RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
git \
&& rm -rf /var/lib/apt/lists/*

COPY ./app /usr/serving/streamlit_app

WORKDIR /usr/serving/streamlit_app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]