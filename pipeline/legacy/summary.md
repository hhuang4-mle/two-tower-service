# EmbeddingGemma Vertex AI Batch Prediction

## Introduction

This document provides an overview of an end-to-end implementation for generating product text
embeddings at scale with Google's `embeddinggemma-300m` model on Vertex AI.
It packages the model behind a custom FastAPI prediction container, registers
that container and its model artifacts in Vertex AI Model Registry, and
submits a GPU-backed batch prediction job using BigQuery as both the input and
output integration point. The document describes the architecture, processing
workflow, serving container, and operational considerations for this solution.

Each input record contains an `article_nbr` and product text. The batch job
returns a normalized, 512-dimensional embedding while preserving the
corresponding article number, making the results suitable for downstream
feature-store, similarity-search, and machine learning workflows.

## Solution overview

The solution consists of four components:

| Component | Purpose |
| --- | --- |
| Hugging Face model artifacts | Supplies the `google/embeddinggemma-300m` model and tokenizer files. |
| Cloud Storage | Stores the model artifacts used when Vertex AI loads the custom model. |
| Custom serving container (`app.py`) | Loads model artifacts from the configured Cloud Storage location, loads the model onto a GPU, exposes health and prediction endpoints, and returns embeddings. |
| Vertex AI batch prediction | Runs the serving container at scale against BigQuery input and writes predictions to BigQuery. |

The target configuration uses the following Google Cloud resources:

| Setting | Value |
| --- | --- |
| Google Cloud project | `gcp-wow-rwds-ai-super-plo-dev` |
| Vertex AI location | `us-central1` |
| Model artifact location | `gs://wx-e1710759-2e30-40b2-a876-11588a57cb50/text-embedding-models/embeddinggemma-300m` |
| Serving image | `gcr.io/gcp-wow-rwds-ai-super-plo-dev/text-embedding-batch-prediction` |
| Input table | `gcp-wow-rwds-ai-super-plo-dev.two_tower.test_input_embeddings` |
| Output dataset | `gcp-wow-rwds-ai-super-plo-dev.two_tower` |

## End-to-end workflow

### 1. Download and store model artifacts

The model is downloaded from Hugging Face, limited to the files required for
inference, and uploaded to Cloud Storage. The directory structure must be
preserved so that `SentenceTransformer` can load the model correctly.

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="google/embeddinggemma-300m",
    local_dir="/tmp/embeddinggemma-300m",
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.model",
        "*.txt",
        "*.md",
    ],
)
```

The downloaded files are uploaded to:

```text
gs://wx-e1710759-2e30-40b2-a876-11588a57cb50/text-embedding-models/embeddinggemma-300m
```

### 2. Prepare the BigQuery batch input

Vertex AI BigQuery batch prediction accepts a JSON instance in each input row.
The input table is created with a single `instances` column that encodes the
article identifier and product text:

```sql
CREATE OR REPLACE TABLE `gcp-wow-rwds-ai-super-plo-dev.two_tower.test_input_embeddings` AS
SELECT
  TO_JSON_STRING(STRUCT(article_nbr AS article_nbr, product_embedding_text AS text)) AS instances
FROM
  `gcp-wow-rwds-ai-super-plo-dev.two_tower.super_product_text_final_17082026`;
```

An example value in the `instances` column is:

```json
{"article_nbr":"123456","text":"Product name and descriptive product text"}
```

### 3. Build and publish the serving container

The predictor container is built from `../../batch_text_embedding/Dockerfile` and pushed to Artifact
Registry. It uses a CUDA 12.1 Python 3.10 base image and runs the FastAPI
application with Uvicorn on the Vertex AI HTTP port.

```bash
make push
```

The published image is:

```text
gcr.io/gcp-wow-rwds-ai-super-plo-dev/text-embedding-batch-prediction
```

### 4. Register the Vertex AI model

The shared model-resolution logic in `shared/embedding_workflow/util.py` reuses the most recently
created matching model, uploads the custom model if no match exists, or uses
the explicitly configured model resource name. Registration associates the
model artifact location with the serving container and configures:

| Configuration | Value |
| --- | --- |
| Display name | `embeddinggemma-300m-tei` |
| Prediction route | `/predict` |
| Health route | `/health` |
| Container port | `8080` |
| Container environment | `MODEL_ID=/tmp/model` |

Set `MODEL_RESOURCE_NAME` to select a specific registered model.

### 5. Submit the batch prediction job

The submission script calls `run_workflow()` from
`shared/embedding_workflow/util.py`, which prepares
the BigQuery input table, creates a Vertex AI v1 `BatchPredictionJob`, waits
for completion, and postprocesses the generated prediction table. The job
requests two `g2-standard-4` replicas, each equipped with one NVIDIA L4 GPU.

The Vertex AI v1 API is used rather than the high-level
`Model.batch_predict()` helper because it exposes
`disable_container_logging=True`. This setting prevents prediction-container
stdout and stderr from being written to Cloud Logging, reducing logging costs.
It does not disable Vertex AI control-plane or Cloud Audit Logs.

```bash
PYTHONPATH=shared python3 batch_job_runner/run_batch_prediction.py
```

The script always waits for the prediction job before running postprocessing.

Vertex AI creates a `prediction_*` table in the configured output dataset for
raw predictions. Postprocessing writes the configured result table.

## Serving container implementation

The serving container is implemented in `../../batch_text_embedding/app.py`. It is a FastAPI application
that implements the Vertex AI custom prediction container contract.

### Model initialization

At application startup, the container reads the `AIP_STORAGE_URI` environment
variable provided by Vertex AI for the registered model artifact location. It
then accesses the configured GCS bucket, lists all files below that prefix,
and downloads them to `/tmp/model`, preserving subdirectories. The model is
loaded from these local artifacts using `SentenceTransformer` with CUDA as the
target device and `trust_remote_code=True`.

The model is held in process memory for the lifetime of the serving
application. The `/health` endpoint reports HTTP 503 until loading is
complete, allowing Vertex AI to wait until the replica is ready.

### Prediction endpoint

The container exposes `POST /predict`. It accepts either the standard Vertex
AI request envelope:

```json
{
  "instances": [
    {
      "article_nbr": "123456",
      "text": "Product name and descriptive product text"
    }
  ]
}
```

or the raw list form that can be used by batch workers:

```json
[
  {
    "article_nbr": "123456",
    "text": "Product name and descriptive product text"
  }
]
```

Each instance can also be a JSON-encoded string, which supports the BigQuery
`instances` column generated during input preparation. The application
validates that every parsed instance is an object with a string `text` field.
Malformed requests receive an HTTP 400 response.

The model generates embeddings with the following settings:

| Setting | Value |
| --- | --- |
| Sentence Transformers method | `encode_document` |
| Inference batch size | `32` |
| Output dimensions | `512` (`truncate_dim=512`) |
| Vector normalization | Enabled |
| Output representation | JSON arrays |

For every input record, the prediction response retains `article_nbr` and
returns its embedding:

```json
{
  "predictions": [
    {
      "article_nbr": "123456",
      "embedding": [0.0123, -0.0456]
    }
  ]
}
```

### Health endpoint

The container exposes `GET /health`. It returns:

```json
{"status":"ok"}
```

after the model has loaded. If the model is not yet available, it returns HTTP
503. Vertex AI uses this endpoint to determine whether a prediction replica is
ready to receive traffic.

## Model resource name handling

The Vertex AI SDK may return a model resource name whose project segment is a
numeric project number. The batch prediction request instead requires the
string project ID:

```text
projects/gcp-wow-rwds-ai-super-plo-dev/locations/us-central1/models/MODEL_ID
```

The submission script extracts the terminal model ID from the SDK model
resource name and reconstructs the batch-job model URI with the configured
string project ID. This prevents a mismatch between a numeric project number
and the project identifier expected by the batch prediction request.
