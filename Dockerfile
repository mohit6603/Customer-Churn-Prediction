# Serves the churn model with FastAPI.
#   docker build -t churn-api .
#   docker run -p 8000:8000 -v "$(pwd)/logs:/app/logs" churn-api
# Requires a trained model in models/ (run `python -m src.train_pipeline` first).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY configs/ configs/
COPY src/ src/
COPY api/ api/
COPY models/churn_model.joblib models/
COPY reports/ reports/

# Run as a non-root user; logs/ holds the prediction log (mount it to persist).
RUN useradd --create-home appuser && mkdir -p logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
