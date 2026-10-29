# Vertex AI Text Embedding Pipeline (Native Components)

This directory contains the Kubeflow Pipelines (KFP) implementation of the text embedding workflow, using **Native GCP Components** and **External SQL Files**. 

This approach eliminates the need to maintain redundant Python logic inside the pipeline graph. 

## Project Structure
* **`sql/`**: Contains the standard BigQuery SQL queries used for preprocessing and postprocessing. You get full syntax highlighting and IDE support for your queries.
* **`pipeline.py`**: The pipeline definition graph. It uses native `BigqueryQueryJobOp` and `ModelBatchPredictOp` components. The only Python component left is a tiny string formatter for the post-processing query.
* **`submit_pipeline.py`**: Handles checking/uploading the Vertex AI model *before* the pipeline runs, formats the SQL strings, and submits the job.
* **`requirements.txt`**: The dependencies needed to submit the pipeline.

## Text-based embedding contract

The pipeline uses `TEXT_COLUMN`, not an ID column. It stages one
`{"text": "..."}` instance per distinct input string, converting null text to
`""` for inference only. Empty and whitespace-only strings are included; other
text is unchanged and compared exactly, including for collated source columns.
Null and empty-string rows share the empty-string embedding.

Postprocessing preserves every original source row and column and appends
`embedding` as `ARRAY<FLOAT64>`. It validates prediction uniqueness,
staged-input membership, finite numeric values, consistent dimensions, and
final row count before replacing the result table. Successful Vertex jobs with
missing predictions still publish every source row, using `embedding = []`
for missing predictions. The postprocessing script returns `missing_texts` and
`missing_rows` counts; the Cloud Run runner logs a warning for partial results.
Fully failed Vertex jobs still stop execution.
Empty source tables skip prediction and
still produce an empty result with the embedding column. Cleanup runs only
after successful postprocessing.

`ID_COLUMN` has been removed. Keep the source unchanged for the duration of a
run, use separate input/result tables for concurrent runs, and do not use a
source that already has an `embedding` column. Data-dependent task caching is
disabled so unchanged table names cannot reuse stale predictions.

## Getting Started

### 1. Install Dependencies

Use Python 3.12 in a fresh, project-specific virtual environment. KFP 2.7
requires `protobuf<5`, which conflicts with newer versions required by some
other Google Cloud client libraries; installing into a shared environment
containing those clients can produce dependency-conflict warnings.

```bash
python3.12 -m venv .venv-py312
source .venv-py312/bin/activate
pip install -r requirements.txt
```

### 2. Compile the Pipeline (Optional)

Compile the Python code into a `embedding_pipeline.yaml` representation without submitting it.

```bash
python submit_pipeline.py --compile-only
```

### 3. Submit the Pipeline

You can submit the pipeline to Vertex AI. The script reads the same environment variables as your original setup.
The default `PIPELINE_ROOT` is
`gs://wx-1b759ae5-bd00-4de8-bf4d-427baa30e931/fs/vertex-ai-pipeline`.
Override it with `PIPELINE_ROOT` if needed. The bucket must exist; the
submitting user needs permission to inspect it, and the pipeline execution
service account needs read/write access to its objects. Unless a service
account is configured for the job, Vertex AI uses the project's Compute Engine
default service account.

```bash
# Submit with the default pipeline root
python submit_pipeline.py

# Submit with specific project overrides
PROJECT_ID="gcp-wow-rwds-ai-super-plo-dev" \
LOCATION="us-central1" \
python submit_pipeline.py

# Optionally override the pipeline root
PIPELINE_ROOT="gs://another-existing-bucket/vertex-pipeline-artifacts" \
python submit_pipeline.py
```

agy --conversation=efb5c6ac-82b3-4019-b6be-010ef6603c5f
