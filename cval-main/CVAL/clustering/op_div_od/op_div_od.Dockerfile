# Используем базовый образ с Python 3.10 для CPU
FROM python:3.10-slim

# Устанавливаем переменную окружения для работы apt-get без ввода
ENV DEBIAN_FRONTEND=noninteractive

# Устанавливаем рабочую директорию
WORKDIR /scripts

# Копируем файл с зависимостями
COPY op_div_od/req.txt req.txt

# Устанавливаем системные зависимости
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python-библиотеки из req.txt
RUN pip install --no-cache-dir -r req.txt

# Копируем исходный код проекта
COPY op_div_od/scripts /scripts

# Открываем порт 5004 для сервера FastAPI
EXPOSE 5004

# Запускаем сервер FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5004"]

