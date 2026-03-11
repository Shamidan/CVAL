FROM python:3.10-slim

WORKDIR /src

COPY src/req.txt ./req.txt
RUN pip install -r req.txt

# Копируем код сервиса
COPY src /src

ENV PYTHONPATH /embedings

EXPOSE 8005

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8005"]