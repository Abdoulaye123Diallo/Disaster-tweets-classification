FROM python:3.10-slim

WORKDIR /app

# Installer les bibliothèques nécessaires
RUN pip install --no-cache-dir \
    flask \
    joblib==1.5.1 \
    pandas \
    numpy \
    scipy \
    scikit-learn==1.7.0

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 5000

CMD ["python", "src/endpoints/app.py"]
