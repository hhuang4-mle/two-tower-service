import os
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
from sentence_transformers import SentenceTransformer


model: SentenceTransformer | None = None
MODEL_DIR = Path("/tmp/model")
MAX_SEQUENCE_LENGTH = 2048
encoder_batch_size = 32
default_truncate_dim = 768


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
    global model, encoder_batch_size, default_truncate_dim

    model = None
    encoder_batch_size = int(os.environ.get("ENCODER_BATCH_SIZE", "32"))
    default_truncate_dim = int(os.environ.get("TRUNCATE_DIM", "768"))
    if encoder_batch_size <= 0 or default_truncate_dim <= 0:
        raise ValueError("ENCODER_BATCH_SIZE and TRUNCATE_DIM must be positive integers")
    download_model_artifacts(os.environ["AIP_STORAGE_URI"], MODEL_DIR)
    if not (MODEL_DIR / "modules.json").is_file():
        raise FileNotFoundError("Expected a complete saved Sentence Transformers artifact")
    loaded_model = SentenceTransformer(
        str(MODEL_DIR),
        device="cuda",
        trust_remote_code=True,
        local_files_only=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    loaded_model.max_seq_length = min(loaded_model.max_seq_length, MAX_SEQUENCE_LENGTH)
    loaded_model.tokenizer.truncation_side = "right"
    loaded_model.eval()
    logging.getLogger("uvicorn.error").info(
        "EmbeddingGemma ready: dtype=%s, max_seq_length=%d tokens, microbatch_size=%d",
        next(loaded_model.parameters()).dtype,
        loaded_model.max_seq_length,
        encoder_batch_size,
    )
    model = loaded_model
    try:
        yield
    finally:
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
                detail=f"instances[{index}] must be a JSON object with a string 'text' field.",
            ) from error

    if not isinstance(instance, dict) or not isinstance(instance.get("text"), str):
        raise HTTPException(
            status_code=400,
            detail=f"instances[{index}] must be a JSON object with a string 'text' field.",
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
    if not parsed_instances:
        return {"predictions": []}

    try:
        with torch.inference_mode():
            embeddings = model.encode(
                [instance["text"] for instance in parsed_instances],
                prompt="",
                batch_size=encoder_batch_size,
                truncate_dim=truncate_dim,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=False,
            ).float().cpu()
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(parsed_instances)
            or embeddings.shape[1] == 0
            or not torch.isfinite(embeddings).all()
            or (torch.linalg.vector_norm(embeddings, dim=1) == 0).any()
        ):
            raise HTTPException(
                status_code=500, detail="Model returned invalid embeddings."
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500, detail="Text embedding inference failed."
        ) from None

    return {"predictions": embeddings.tolist()}
