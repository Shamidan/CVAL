FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями
COPY req.txt ./req.txt
RUN apt-get update && apt-get install -y iputils-ping
# Устанавливаем Python-зависимости
RUN pip install -r req.txt

ENV PYTHONPATH /app

COPY . /app

CMD ["python", "uncertanly_cval_web_service/app.py"]