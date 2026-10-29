FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu121.py310

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY predictor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predictor/app.py .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-8080}"]
