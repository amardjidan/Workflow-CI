FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir mlflow pandas numpy scikit-learn joblib


CMD ["python", "modelling.py"]