# Run an EmbeddingGemma batch example

1. Build and push the Docker image.
2. Register the model in Vertex AI.
3. Prepare a small BigQuery input table.
4. Submit a batch prediction job.
5. Check the job and predictions.

**No online endpoint deployment or Cloud Run Job is needed.** Vertex starts
the GPU containers for the batch job. The full Vertex AI Pipeline comes later.

Before starting, authenticate `gcloud` and `bq`, enable Cloud Build and Vertex
AI, and confirm L4 quota and image/GCS/BigQuery permissions. The example uses
`us-central1` and a BigQuery `US` dataset; verify location compatibility.
Running these commands creates billable resources. This procedure has not yet
been executed.

Run all commands in the same shell from `batch_text_embedding/`.

## 1. Build and push the image

Use a unique tag so the test cannot accidentally reuse an older image:

```bash
export PROJECT_ID=gcp-wow-rwds-ai-super-plo-dev
export LOCATION=us-central1
export RUN_ID=embedding_smoke_$(date -u +%Y%m%d_%H%M%S)
export IMAGE_URI=gcr.io/$PROJECT_ID/embeddinggemma-300m-bf16:$RUN_ID
export ARTIFACT_URI=gs://wx-e1710759-2e30-40b2-a876-11588a57cb50/text-embedding-models/embeddinggemma-300m
export DATASET=$RUN_ID

make push PROJECT_ID="$PROJECT_ID" IMAGE_TAG="$RUN_ID"
```

`make push` builds and publishes the image through Cloud Build; no local
Docker build or model download is required.

## 2. Register the model

Link the new image to the model artifacts already in GCS:

```bash
gcloud ai models upload \
  --project="$PROJECT_ID" --region="$LOCATION" \
  --model-id="$RUN_ID" --display-name=embeddinggemma-300m-bf16 \
  --artifact-uri="$ARTIFACT_URI" \
  --container-image-uri="$IMAGE_URI" \
  --container-predict-route=/predict \
  --container-health-route=/health \
  --container-ports=8080 \
  --container-env-vars=TRUNCATE_DIM=768,ENCODER_BATCH_SIZE=32

export MODEL_RESOURCE_NAME=projects/$PROJECT_ID/locations/$LOCATION/models/$RUN_ID
```

The display name is `embeddinggemma-300m-bf16`; the unique resource ID isolates
this test from existing models. The artifact must include the complete saved
Sentence Transformers model, including `modules.json` and tokenizer files.

## 3. Prepare BigQuery input

Create a separate test dataset and six input rows: ordinary text, a duplicate,
empty text, NULL normalized to empty text, Unicode, and oversized text.
Use the serialized `instances` format supported by the copied workflow:

```bash
bq --project_id="$PROJECT_ID" mk --dataset --location=US \
  "$PROJECT_ID:$DATASET"

bq --project_id="$PROJECT_ID" --location=US query --use_legacy_sql=false <<SQL
CREATE TABLE \`$PROJECT_ID.$DATASET.input\` AS
SELECT TO_JSON_STRING(STRUCT(COALESCE(text, '') AS text)) AS instances
FROM UNNEST([
  'Milk bread eggs',
  'Milk bread eggs',
  '',
  CAST(NULL AS STRING),
  CODE_POINTS_TO_STRING([19990, 30028]),
  REPEAT('milk ', 4096)
]) AS text;
SQL
```

Duplicates are intentional for this example. The future pipeline will own
deduplication and preprocessing.

## 4. Submit the batch prediction job

This example uses one L4 GPU and writes predictions back to the test dataset:

```bash
curl --fail-with-body --silent --show-error \
  -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://$LOCATION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$LOCATION/batchPredictionJobs" \
  --data-binary @- <<JSON
{
  "displayName": "$RUN_ID",
  "model": "$MODEL_RESOURCE_NAME",
  "inputConfig": {
    "instancesFormat": "bigquery",
    "bigquerySource": {"inputUri": "bq://$PROJECT_ID.$DATASET.input"}
  },
  "outputConfig": {
    "predictionsFormat": "bigquery",
    "bigqueryDestination": {"outputUri": "bq://$PROJECT_ID.$DATASET"}
  },
  "dedicatedResources": {
    "machineSpec": {
      "machineType": "g2-standard-4",
      "acceleratorType": "NVIDIA_L4",
      "acceleratorCount": 1
    },
    "startingReplicaCount": 1,
    "maxReplicaCount": 1
  },
  "manualBatchTuningParameters": {"batchSize": 32},
  "disableContainerLogging": false
}
JSON
```

Save the returned job `name`. This submits the job; it does not wait for it.

## 5. Check the job and predictions

Open **Vertex AI > Batch predictions**, select the job, and wait for success.
Check for partial failures as well: a successful job can still have missing
predictions.

In the container logs, look for `dtype=torch.bfloat16` and a sequence limit
of at most **2,048 tokens**, not bytes. In the job details, find the actual
BigQuery output table; do not guess its generated name.

Set that table below without the `bq://` prefix:

```bash
export PREDICTION_TABLE=REPLACE_PROJECT.REPLACE_DATASET.REPLACE_TABLE
bq show --schema --format=prettyjson "$PREDICTION_TABLE"

bq --project_id="$PROJECT_ID" --location=US query --use_legacy_sql=false <<SQL
WITH results AS (
  SELECT
    JSON_VALUE(instances, '$.text') AS text,
    JSON_EXTRACT_ARRAY(prediction) AS elements
  FROM \`$PREDICTION_TABLE\`
)
SELECT
  COUNT(*) AS predictions,
  COUNTIF(
    text IS NOT NULL AND ARRAY_LENGTH(elements) = 768
    AND NOT EXISTS (
      SELECT 1 FROM UNNEST(elements) AS element
      WHERE SAFE_CAST(element AS FLOAT64) IS NULL
         OR IS_NAN(SAFE_CAST(element AS FLOAT64))
         OR IS_INF(SAFE_CAST(element AS FLOAT64))
    )
  ) AS valid_predictions
FROM results;
SQL
```

Expect **6 predictions and 6 valid predictions**. Confirm that the original
text keys and duplicate counts are preserved; never join by output row order.
The query uses the copied workflow's `instances`/`prediction` schema. If the
actual schema differs, resolve that mismatch before proceeding.

## After the first successful run

Before adopting named-field input, repeat with a table containing a `text`
column and add `"instanceConfig": {"instanceType": "object"}` to the job
request. Verify preserved text keys and complete predictions again.

This smoke test checks basic inference and BigQuery integration, not the full
pipeline or training compatibility. Compare against training-generated vectors
before production use. Keep the test resources until results are reconciled,
then remove only the test resources created by this run.
