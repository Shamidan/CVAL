FROM python:3.10-slim

WORKDIR /app

COPY src/requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENV PYTHONPATH /app

COPY . /app

# Указываем команду запуска
CMD ["python", "src/app.py"]