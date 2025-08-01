FROM python:3.10-slim

WORKDIR /app

# Installer les bibliothèques nécessaires
RUN pip install --no-cache-dir \
    flask \
    flasgger \
    joblib==1.5.1 \
    pandas \
    numpy \
    scipy \
    scikit-learn==1.7.0
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "src.endpoints.app:app"]
