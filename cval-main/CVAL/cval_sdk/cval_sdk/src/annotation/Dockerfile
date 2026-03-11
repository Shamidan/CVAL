FROM python:3.10-slim

WORKDIR /app

# Копируем файл с зависимостями
COPY cval_sdk/src/annotation/requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y iputils-ping
RUN pip install -r requirements.txt


# Копируем код сервиса
COPY cval_sdk/src/ /app

ENV PYTHONPATH /app

CMD ["uvicorn", "annotation.cvat_hook.cvat_hook:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "debug"]