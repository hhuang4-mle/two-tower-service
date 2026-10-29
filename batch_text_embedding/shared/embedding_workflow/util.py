"""Run the BigQuery -> Vertex AI -> BigQuery embedding workflow."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass

from google.api_core.client_options import ClientOptions
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from google.cloud import bigquery


DEFAULT_PROJECT_ID = "gcp-wow-rwds-ai-super-plo-dev"
DEFAULT_LOCATION = "us-central1"
DEFAULT_DATASET = "two_tower"
DEFAULT_MODEL_DISPLAY_NAME = "embeddinggemma-300m-tei"
DEFAULT_MODEL_ARTIFACT_URI = (
    "gs://wx-e1710759-2e30-40b2-a876-11588a57cb50/"
    "text-embedding-models/embeddinggemma-300m"
)
DEFAULT_SERVING_CONTAINER_IMAGE_URI = (
    "gcr.io/gcp-wow-rwds-ai-super-plo-dev/"
    "text-embedding-batch-prediction"
)
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
LOGGER = logging.getLogger(__name__)
PREDICTION_JOB_POLL_INTERVAL_SECONDS = 30
FAILED_BATCH_JOB_STATES = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


@dataclass(frozen=True)
class WorkflowConfig:
    project_id: str
    location: str
    source_table: str
    text_column: str
    input_table: str
    output_dataset: str
    result_table: str
    model_artifact_uri: str
    serving_container_image_uri: str
    model_display_name: str
    model_resource_name: str | None
    truncate_dim: str | None
    display_name: str
    cleanup_input_table: bool

    @classmethod
    def from_environment(cls) -> WorkflowConfig:
        project_id = os.getenv("PROJECT_ID", DEFAULT_PROJECT_ID)
        dataset = f"{project_id}.{os.getenv('DATASET', DEFAULT_DATASET)}"
        return cls(
            project_id=project_id,
            location=os.getenv("LOCATION", os.getenv("REGION", DEFAULT_LOCATION)),
            source_table=os.getenv(
                "SOURCE_TABLE", f"{dataset}.super_product_text_final_17082026"
            ),
            text_column=os.getenv("TEXT_COLUMN", "product_embedding_text"),
            input_table=os.getenv("INPUT_TABLE", f"{dataset}.test_input_embeddings"),
            output_dataset=os.getenv("OUTPUT_DATASET", dataset),
            result_table=os.getenv("RESULT_TABLE", f"{dataset}.product_embeddings"),
            model_artifact_uri=os.getenv(
                "MODEL_ARTIFACT_URI", DEFAULT_MODEL_ARTIFACT_URI
            ),
            serving_container_image_uri=os.getenv(
                "SERVING_CONTAINER_IMAGE_URI",
                DEFAULT_SERVING_CONTAINER_IMAGE_URI,
            ),
            model_display_name=os.getenv(
                "MODEL_DISPLAY_NAME", DEFAULT_MODEL_DISPLAY_NAME
            ),
            model_resource_name=os.getenv("MODEL_RESOURCE_NAME") or None,
            truncate_dim=os.getenv("TRUNCATE_DIM") or None,
            display_name=os.getenv(
                "BATCH_PREDICTION_DISPLAY_NAME",
                "embeddinggemma-300m-batch-prediction",
            ),
            cleanup_input_table=os.getenv("CLEANUP_INPUT_TABLE", "")
            .strip()
            .lower()
            in {"1", "true", "yes"},
        )


def quote_table(table_id: str) -> str:
    """Validate a project.dataset.table ID and return it backtick-quoted."""
    cleaned = table_id.strip().replace("`", "")
    parts = cleaned.split(".")
    if len(parts) != 3 or not all(
        IDENTIFIER_PATTERN.fullmatch(part) for part in parts
    ):
        raise ValueError(
            "Expected a BigQuery table ID in project.dataset.table format, "
            f"got {table_id!r}"
        )
    return f"`{cleaned}`"


def quote_column(column_name: str) -> str:
    """Validate a BigQuery column name and return it backtick-quoted."""
    cleaned = column_name.strip().replace("`", "")
    if not IDENTIFIER_PATTERN.fullmatch(cleaned):
        raise ValueError(f"Invalid BigQuery column name: {column_name!r}")
    return f"`{cleaned}`"


def preprocess_input(client: bigquery.Client, config: WorkflowConfig) -> int:
    """Stage one text-only instance per distinct model input."""
    query = f"""
CREATE OR REPLACE TABLE {quote_table(config.input_table)} AS
SELECT TO_JSON_STRING(STRUCT(text AS text)) AS instances
FROM (
  SELECT DISTINCT
    COLLATE(COALESCE(CAST(source.{quote_column(config.text_column)} AS STRING), ''), '') AS text
  FROM {quote_table(config.source_table)} AS source
)
"""
    client.query(query).result()
    return client.get_table(config.input_table).num_rows


def _get_model(config: WorkflowConfig) -> aiplatform.Model:
    aiplatform.init(project=config.project_id, location=config.location)

    if config.model_resource_name:
        model_id = config.model_resource_name.rsplit("/", maxsplit=1)[-1]
        model_name = _model_resource_name_for_id(model_id, config)
        LOGGER.info("Using configured Vertex AI model %s", model_name)
        return aiplatform.Model(model_name)

    escaped_display_name = (
        config.model_display_name.replace("\\", "\\\\").replace('"', '\\"')
    )
    existing_models = aiplatform.Model.list(
        filter=f'display_name="{escaped_display_name}"',
        order_by="create_time desc",
    )
    existing_models = [
        model
        for model in existing_models
        if model.display_name == config.model_display_name
    ]
    if existing_models:
        model = existing_models[0]
        model_name = _model_resource_name(model, config)
        LOGGER.info("Reusing Vertex AI model %s", model_name)
        if len(existing_models) > 1:
            LOGGER.warning(
                "Found %d Vertex AI models named %r; reusing the most "
                "recently created model %s. Set MODEL_RESOURCE_NAME to "
                "select a different model.",
                len(existing_models),
                config.model_display_name,
                model_name,
            )
        return aiplatform.Model(model_name)

    LOGGER.info(
        "No Vertex AI model named %r found; uploading model",
        config.model_display_name,
    )
    environment_variables = {"MODEL_ID": "/tmp/model"}
    if config.truncate_dim:
        environment_variables["TRUNCATE_DIM"] = config.truncate_dim

    uploaded_model = aiplatform.Model.upload(
        display_name=config.model_display_name,
        serving_container_image_uri=config.serving_container_image_uri,
        artifact_uri=config.model_artifact_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        serving_container_environment_variables=environment_variables,
        sync=True,
    )
    model_name = _model_resource_name(uploaded_model, config)
    LOGGER.info("Uploaded Vertex AI model %s", model_name)
    return aiplatform.Model(model_name)


def _model_resource_name_for_id(
    model_id: str,
    config: WorkflowConfig,
) -> str:
    return (
        f"projects/{config.project_id}/locations/{config.location}/"
        f"models/{model_id}"
    )


def _model_resource_name(
    model: aiplatform.Model,
    config: WorkflowConfig,
) -> str:
    # The SDK may return a numeric project number, but the batch job requires
    # the string project ID.
    model_id = model.resource_name.rsplit("/", maxsplit=1)[-1]
    return _model_resource_name_for_id(model_id, config)


def submit_batch_prediction(
    config: WorkflowConfig,
) -> aiplatform.BatchPredictionJob:
    """Submit a BigQuery-backed Vertex AI batch prediction job."""
    aiplatform.init(project=config.project_id, location=config.location)
    model_name = _model_resource_name(_get_model(config), config)

    # The high-level batch_predict() helper does not expose
    # disable_container_logging, so create the job through the v1 client.
    job = aiplatform_v1.types.BatchPredictionJob(
        display_name=config.display_name,
        model=model_name,
        input_config=aiplatform_v1.types.BatchPredictionJob.InputConfig(
            instances_format="bigquery",
            bigquery_source=aiplatform_v1.types.BigQuerySource(
                input_uri=f"bq://{config.input_table}",
            ),
        ),
        output_config=aiplatform_v1.types.BatchPredictionJob.OutputConfig(
            predictions_format="bigquery",
            bigquery_destination=aiplatform_v1.types.BigQueryDestination(
                output_uri=f"bq://{config.output_dataset}",
            ),
        ),
        dedicated_resources=aiplatform_v1.types.BatchDedicatedResources(
            machine_spec=aiplatform_v1.types.MachineSpec(
                machine_type="g2-standard-4",
                accelerator_type="NVIDIA_L4",
                accelerator_count=1,
            ),
            starting_replica_count=4,
            max_replica_count=8,
        ),
        manual_batch_tuning_parameters=aiplatform_v1.types.ManualBatchTuningParameters(
            batch_size=16
        ),
        disable_container_logging=True,
    )
    job_client = aiplatform_v1.JobServiceClient(
        client_options=ClientOptions(
            api_endpoint=f"{config.location}-aiplatform.googleapis.com"
        )
    )
    response = job_client.create_batch_prediction_job(
        parent=f"projects/{config.project_id}/locations/{config.location}",
        batch_prediction_job=job,
    )
    return aiplatform.BatchPredictionJob(response.name)


def wait_for_prediction_table(batch_job: aiplatform.BatchPredictionJob) -> str:
    """Wait for completion and return the generated BigQuery table ID."""
    while True:
        state = batch_job.state.name
        if state == "JOB_STATE_SUCCEEDED":
            break
        if state in FAILED_BATCH_JOB_STATES:
            raise RuntimeError(
                "Vertex batch prediction finished with state "
                f"{state}: {batch_job.error}"
            )
        time.sleep(PREDICTION_JOB_POLL_INTERVAL_SECONDS)

    # Vertex reports the dataset as bq://project.dataset and the table as a
    # bare table name, so the full table ID has to be reassembled.
    output_info = batch_job.output_info
    dataset = (output_info.bigquery_output_dataset or "").removeprefix("bq://")
    table = output_info.bigquery_output_table or ""
    if not dataset or not table:
        raise RuntimeError(
            "Vertex batch prediction succeeded without reporting a BigQuery "
            f"output table: {output_info}"
        )
    return f"{dataset}.{table}"


def postprocess_output(
    client: bigquery.Client,
    prediction_table: str | None,
    config: WorkflowConfig,
) -> int:
    """Preserve every source row, warning when predictions are missing."""
    prediction_source = (
        f"SELECT instances, prediction FROM {quote_table(prediction_table)}"
        if prediction_table
        else "SELECT CAST(NULL AS STRING) AS instances, "
        "CAST(NULL AS STRING) AS prediction WHERE FALSE"
    )
    query = f"""
CREATE TEMP TABLE embedding_predictions AS
SELECT
  COLLATE(JSON_VALUE(instances, '$.text'), '') AS text,
  JSON_EXTRACT_ARRAY(prediction) AS elements
FROM ({prediction_source});

ASSERT NOT EXISTS (
  SELECT 1 FROM embedding_predictions
  WHERE text IS NULL
    OR elements IS NULL
    OR ARRAY_LENGTH(elements) = 0
    OR EXISTS (
      SELECT 1 FROM UNNEST(elements) AS element
      WHERE SAFE_CAST(element AS FLOAT64) IS NULL
        OR IS_NAN(SAFE_CAST(element AS FLOAT64))
        OR IS_INF(SAFE_CAST(element AS FLOAT64))
    )
) AS 'Predictions must contain text and nonempty finite numeric embeddings';

ASSERT NOT EXISTS (
  SELECT 1 FROM embedding_predictions
  GROUP BY text HAVING COUNT(*) != 1
) AS 'Expected exactly one prediction per text';

ASSERT (
  SELECT COUNT(DISTINCT ARRAY_LENGTH(elements)) <= 1
  FROM embedding_predictions
) AS 'Embedding dimensions must be consistent';

ASSERT NOT EXISTS (
  SELECT 1
  FROM (
    SELECT COLLATE(JSON_VALUE(instances, '$.text'), '') AS text
    FROM {quote_table(config.input_table)}
  ) AS inputs
  RIGHT JOIN embedding_predictions AS predictions
    ON inputs.text = predictions.text
  WHERE inputs.text IS NULL
) AS 'Prediction texts must belong to staged input texts';

CREATE TEMP TABLE embedding_result AS
SELECT
  source.*,
  ARRAY(
    SELECT CAST(element AS FLOAT64)
    FROM UNNEST(predictions.elements) AS element WITH OFFSET AS position
    ORDER BY position
  ) AS embedding
FROM {quote_table(config.source_table)} AS source
LEFT JOIN embedding_predictions AS predictions
  ON COLLATE(COALESCE(CAST(source.{quote_column(config.text_column)} AS STRING), ''), '')
    = predictions.text;

ASSERT (
  (SELECT COUNT(*) FROM embedding_result)
    = (SELECT COUNT(*) FROM {quote_table(config.source_table)})
) AS 'Result must preserve the source row count';

CREATE OR REPLACE TABLE {quote_table(config.result_table)} AS
SELECT * FROM embedding_result;

SELECT
  (
    SELECT COUNT(*)
    FROM {quote_table(config.input_table)} AS inputs
    LEFT JOIN embedding_predictions AS predictions
      ON COLLATE(JSON_VALUE(inputs.instances, '$.text'), '') = predictions.text
    WHERE predictions.text IS NULL
  ) AS missing_texts,
  (
    SELECT COUNT(*) FROM embedding_result WHERE ARRAY_LENGTH(embedding) = 0
  ) AS missing_rows;
"""
    coverage = next(iter(client.query(query).result()))
    if coverage.missing_texts or coverage.missing_rows:
        LOGGER.warning(
            "Published partial results to %s: %d staged texts have no "
            "prediction; %d source rows have embedding=[]. "
            "Raw prediction table: %s. Retry failed texts to fill these gaps.",
            config.result_table,
            coverage.missing_texts,
            coverage.missing_rows,
            prediction_table,
        )
    return client.get_table(config.result_table).num_rows


def run_workflow(config: WorkflowConfig | None = None) -> dict[str, str | int]:
    """Run preprocessing, batch prediction, and postprocessing."""
    config = config or WorkflowConfig.from_environment()
    bq_client = bigquery.Client(project=config.project_id)

    LOGGER.info(
        "Preprocessing source table %s into %s",
        config.source_table,
        config.input_table,
    )
    input_rows = preprocess_input(bq_client, config)
    LOGGER.info("Preprocessing complete: %d distinct text inputs", input_rows)

    batch_job = None
    prediction_table = None
    if input_rows:
        LOGGER.info("Submitting batch prediction job")
        batch_job = submit_batch_prediction(config)
        LOGGER.info("Submitted batch prediction job %s", batch_job.resource_name)

        LOGGER.info("Waiting for batch prediction job to finish")
        prediction_table = wait_for_prediction_table(batch_job)
        LOGGER.info(
            "Batch prediction complete; raw predictions are in %s",
            prediction_table,
        )
    else:
        LOGGER.info("Source is empty; skipping batch prediction")

    LOGGER.info("Postprocessing predictions into %s", config.result_table)
    result_rows = postprocess_output(bq_client, prediction_table, config)
    LOGGER.info(
        "Postprocessing complete: %d source/result rows, %d distinct inputs, "
        "%d repeated inputs avoided",
        result_rows,
        input_rows,
        result_rows - input_rows,
    )

    if config.cleanup_input_table:
        LOGGER.info("Deleting temporary input table %s", config.input_table)
        bq_client.delete_table(config.input_table, not_found_ok=True)

    return {
        "input_table": config.input_table,
        "input_rows": input_rows,
        "batch_prediction_job": batch_job.resource_name if batch_job else "",
        "prediction_table": prediction_table or "",
        "result_table": config.result_table,
        "result_rows": result_rows,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = WorkflowConfig.from_environment()
    run_workflow(config)

if __name__ == "__main__":
    main()
