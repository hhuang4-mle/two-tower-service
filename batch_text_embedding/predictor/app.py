import os
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
from sentence_transformers import SentenceTransformer


model: SentenceTransformer | None = None
MODEL_DIR = Path("/tmp/model")


def download_model_artifacts(uri: str, destination: Path) -> None:
    parsed_uri = urlparse(uri)
    if parsed_uri.scheme != "gs" or not parsed_uri.netloc:
        raise ValueError(f"Expected a gs:// model URI, got: {uri}")

    prefix = parsed_uri.path.lstrip("/").rstrip("/")
    if not prefix:
        raise ValueError(f"Model URI must point to a non-empty prefix: {uri}")

    destination.mkdir(parents=True, exist_ok=True)
    bucket = storage.Client().bucket(parsed_uri.netloc)
    downloaded_files = 0

    for blob in bucket.list_blobs(prefix=f"{prefix}/"):
        relative_path = blob.name[len(prefix) :].lstrip("/")
        if not relative_path:
            continue

        local_path = destination / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        downloaded_files += 1

    if downloaded_files == 0:
        raise FileNotFoundError(f"No model artifacts found at {uri}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model

    download_model_artifacts(os.environ["AIP_STORAGE_URI"], MODEL_DIR)
    model = SentenceTransformer(
        str(MODEL_DIR),
        device="cuda",
        trust_remote_code=True,
    )
    yield
    model = None


class HealthCheckLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) == 5:
            _, method, path, _, status = args
            if method == "GET" and path == "/health" and status == 200:
                return False
        return True


logging.getLogger("uvicorn.access").addFilter(HealthCheckLogFilter())

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok"}


def parse_instance(instance: Any, index: int) -> dict[str, Any]:
    if isinstance(instance, list) and len(instance) == 1:
        instance = instance[0]

    if isinstance(instance, str):
        try:
            instance = json.loads(instance)
        except json.JSONDecodeError as error:
            raise HTTPException(
                status_code=400,
                detail=f"instances[{index}] must be a JSON object with a string 'text' field. - json.loads() error. {instance}",
            ) from error

    if not isinstance(instance, dict) or not isinstance(instance.get("text"), str):
        raise HTTPException(
            status_code=400,
            detail=f"instances[{index}] must be a JSON object with a string 'text' field. - instance error. - {type(instance)} - {instance}",
        )
    return instance


@app.post("/predict")
async def predict(request: Request) -> dict[str, list[list[float]]]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        payload = await request.json()
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="Expected a JSON prediction request.",
        ) from error

    # Vertex AI normally sends {"instances": [...]}; accepting a raw list also
    # supports the batch-prediction payload form without FastAPI rejecting it.
    instances = payload.get("instances") if isinstance(payload, dict) else payload
    parameters = payload.get("parameters", {}) if isinstance(payload, dict) else {}
    if not isinstance(parameters, dict):
        raise HTTPException(
            status_code=400,
            detail="'parameters' must be a JSON object.",
        )
    default_truncate_dim = int(os.environ.get("TRUNCATE_DIM", "768"))
    truncate_dim = parameters.get("truncate_dim", default_truncate_dim)
    if isinstance(truncate_dim, bool) or not isinstance(truncate_dim, int) or truncate_dim <= 0:
        raise HTTPException(
            status_code=400,
            detail="'parameters.truncate_dim' must be a positive integer.",
        )
    if not isinstance(instances, list):
        raise HTTPException(
            status_code=400,
            detail="Expected a JSON object containing an 'instances' array.",
        )

    parsed_instances = [
        parse_instance(instance, index) for index, instance in enumerate(instances)
    ]

    embeddings = model.encode_document(
        [instance["text"] for instance in parsed_instances],
        batch_size=32,
        truncate_dim=truncate_dim,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return {
        "predictions": [
            embedding.tolist() for embedding in embeddings
        ]
    }
