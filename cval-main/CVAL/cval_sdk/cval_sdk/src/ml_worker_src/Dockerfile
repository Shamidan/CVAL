FROM python:3.10-slim

WORKDIR /app

COPY cval_sdk/src/annotation/requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y iputils-ping
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Копируем код сервиса
COPY cval_sdk/src/ /app

ENV PYTHONPATH /app

# Запускаем Python-скрипт
CMD ["python", "ml_worker_src/ml_worker.py"]
# python ml_worker_src/ml_worker.py