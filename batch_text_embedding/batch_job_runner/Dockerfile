FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY batch_job_runner/requirements.txt .
RUN pip install -r requirements.txt

COPY batch_job_runner/run_batch_prediction.py ./run_batch_prediction.py
COPY shared/embedding_workflow ./embedding_workflow

CMD ["python", "run_batch_prediction.py"]
