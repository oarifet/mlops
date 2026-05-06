FROM python:3.10-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir \
    mlflow==2.11.1 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.0

EXPOSE 5000

