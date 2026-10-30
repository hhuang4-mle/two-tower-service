from typing import List

from kfp import dsl


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["google-cloud-bigquery==3.31.0"],
)
def initialize_run(
    project_id: str,
    vertex_location: str,
    bigquery_location: str,
    entity_type: str,
    source_table: str,
    text_columns: List[str],
    embedding_model_resource_name: str,
    service_account: str,
    request_batch_size: int,
    replica_count: int,
    retry_rounds: int,
    cleanup_intermediates: bool,
) -> str:
    import json
    import re
    import uuid

    from google.cloud import bigquery

    if entity_type not in {"customer", "product"}:
        raise ValueError("entity_type must be customer or product")
    expected = 2 if entity_type == "customer" else 1
    if len(text_columns) != expected or len({name.lower() for name in text_columns}) != expected:
        raise ValueError("Supply two distinct ordered customer columns or one product column")
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", source_table):
        raise ValueError("source_table must be a plain project.dataset.table identifier")
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in text_columns):
        raise ValueError("Text column names must be plain top-level identifiers")
    if not re.fullmatch(r"projects/[A-Za-z0-9_-]+/locations/[a-z0-9-]+/models/[A-Za-z0-9_@.-]+", embedding_model_resource_name):
        raise ValueError("Supply the full registered Vertex model resource name")
    if embedding_model_resource_name.split("/")[3] != vertex_location:
        raise ValueError("Model and batch job must use the same Vertex location")
    if request_batch_size <= 0 or replica_count <= 0 or not 0 <= retry_rounds <= 2:
        raise ValueError("Batch/replica counts must be positive; retry_rounds must be 0..2")

    client = bigquery.Client(project=project_id, location=bigquery_location)
    dataset = source_table.rsplit(".", 1)[0]
    if client.get_dataset(dataset).location.lower() != bigquery_location.lower():
        raise ValueError("bigquery_location does not match the source dataset")
    source = client.get_table(source_table)
    schema = {field.name.lower(): field for field in source.schema}
    for name in text_columns:
        field = schema.get(name.lower())
        if field is None or field.field_type != "STRING" or field.mode == "REPEATED":
            raise ValueError("Configured text columns must be scalar STRING fields")
    embedding_columns = (
        ["long_term_embedding", "mid_term_embedding"]
        if entity_type == "customer" else ["product_embedding"]
    )
    if any(name in schema for name in embedding_columns):
        raise ValueError("Source already contains a reserved output embedding column")

    run_id = uuid.uuid4().hex
    prefix = f"{source_table}_d768"
    config = {
        "project_id": project_id,
        "vertex_location": vertex_location,
        "bigquery_location": bigquery_location,
        "entity_type": entity_type,
        "source_table": source_table,
        "source_columns": [field.name for field in source.schema],
        "text_columns": text_columns,
        "embedding_columns": embedding_columns,
        "model": embedding_model_resource_name,
        "service_account": service_account,
        "request_batch_size": request_batch_size,
        "replica_count": replica_count,
        "retry_rounds": retry_rounds,
        "cleanup_intermediates": cleanup_intermediates,
        "run_id": run_id,
        "dataset": dataset,
        "output_table": prefix,
        "failure_table": f"{prefix}_failures",
        "input_table": f"{prefix}_input_{run_id}",
        "embedding_table": f"{prefix}_embeddings_{run_id}",
        "manifest_table": f"{prefix}_jobs_{run_id}",
        "lock_table": f"{prefix}_lock",
    }
    if any(len(config[key].split(".")[-1]) > 1016 for key in (
        "input_table", "embedding_table", "manifest_table", "failure_table", "lock_table"
    )):
        raise ValueError("Source table name is too long for derived table suffixes")
    lock = bigquery.Table(
        config["lock_table"], schema=[bigquery.SchemaField("run_id", "STRING")]
    )
    lock.labels = {"embedding_run": run_id}
    lock.description = (
        "Exclusive encoder publication lock. Remove manually only after confirming "
        "the owning pipeline and its batch jobs have stopped."
    )
    # Atomic creation rejects overlapping runs for the same source.
    client.create_table(lock)
    return json.dumps(config)


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["google-cloud-bigquery==3.31.0"],
)
def prepare_texts(config: str, sql_template: str) -> str:
    import json

    from google.cloud import bigquery

    cfg = json.loads(config)
    client = bigquery.Client(project=cfg["project_id"], location=cfg["bigquery_location"])
    selects = [
        f"SELECT COLLATE(COALESCE(`{column}`, ''), '') AS text "
        f"FROM `{cfg['source_table']}`"
        for column in cfg["text_columns"]
    ]
    union = "\nUNION ALL\n".join(selects)
    client.query(sql_template.format(text_union=union, **cfg)).result()
    return config


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=[
        "google-cloud-aiplatform==1.93.0",
        "google-cloud-bigquery==3.31.0",
    ],
)
def predict_texts(config: str, attempt: int, previous: str = "") -> str:
    import json
    import time

    from google.api_core.client_options import ClientOptions
    from google.cloud import aiplatform_v1, bigquery

    cfg = json.loads(config)
    client = bigquery.Client(project=cfg["project_id"], location=cfg["bigquery_location"])
    input_table = cfg["input_table"] if attempt == 0 else f"{cfg['input_table']}_retry_{attempt}"
    if attempt > cfg["retry_rounds"]:
        return json.dumps({"skipped": True})
    if attempt:
        if not previous:
            raise ValueError("Retry requires a reconciled previous attempt")
        client.query(f"""
CREATE TABLE `{input_table}` AS
SELECT inputs.instances FROM `{cfg['input_table']}` AS inputs
LEFT JOIN `{cfg['embedding_table']}` AS valid
ON COLLATE(JSON_VALUE(inputs.instances, '$.text'), '') = valid.text
WHERE valid.text IS NULL;
""").result()
    if client.get_table(input_table).num_rows == 0:
        return json.dumps({"skipped": True})

    api = aiplatform_v1.JobServiceClient(client_options=ClientOptions(
        api_endpoint=f"{cfg['vertex_location']}-aiplatform.googleapis.com"
    ))
    job = aiplatform_v1.BatchPredictionJob(
        display_name=f"embedding-{cfg['run_id']}-attempt-{attempt}",
        model=cfg["model"],
        service_account=cfg["service_account"],
        input_config={
            "instances_format": "bigquery",
            "bigquery_source": {"input_uri": f"bq://{input_table}"},
        },
        output_config={
            "predictions_format": "bigquery",
            "bigquery_destination": {"output_uri": f"bq://{cfg['dataset']}"},
        },
        dedicated_resources={
            "machine_spec": {
                "machine_type": "g2-standard-4",
                "accelerator_type": "NVIDIA_L4",
                "accelerator_count": 1,
            },
            "starting_replica_count": cfg["replica_count"],
            "max_replica_count": cfg["replica_count"],
        },
        manual_batch_tuning_parameters={
            "batch_size": max(1, cfg["request_batch_size"] // (2 ** attempt))
        },
        disable_container_logging=False,
        labels={"embedding_run": cfg["run_id"]},
    )
    created = api.create_batch_prediction_job(
        parent=f"projects/{cfg['project_id']}/locations/{cfg['vertex_location']}",
        batch_prediction_job=job,
    )
    states = aiplatform_v1.JobState
    print(f"Submitted Vertex batch job: {created.name}")
    errors = client.insert_rows_json(cfg["manifest_table"], [{
        "attempt": attempt, "job_name": created.name,
        "state": states(created.state).name, "prediction_table": "",
        "failed_count": 0, "incomplete_count": 0,
    }])
    if errors:
        raise RuntimeError(f"Could not record submitted batch job: {created.name}")
    terminal = {
        states.JOB_STATE_SUCCEEDED, states.JOB_STATE_FAILED,
        states.JOB_STATE_CANCELLED, states.JOB_STATE_EXPIRED,
        states.JOB_STATE_PARTIALLY_SUCCEEDED,
    }
    while True:
        job = api.get_batch_prediction_job(name=created.name)
        if job.state in terminal:
            break
        time.sleep(30)
    output = job.output_info
    dataset = output.bigquery_output_dataset.removeprefix("bq://")
    table = output.bigquery_output_table
    prediction_table = f"{dataset}.{table}" if dataset and table else ""
    row = {
        "attempt": attempt, "job_name": job.name,
        "state": states(job.state).name, "prediction_table": prediction_table,
        "failed_count": int(job.completion_stats.failed_count),
        "incomplete_count": int(job.completion_stats.incomplete_count),
    }
    errors = client.insert_rows_json(cfg["manifest_table"], [row])
    if errors:
        raise RuntimeError(f"Could not record Vertex batch output metadata: {job.name}")
    if job.state not in {states.JOB_STATE_SUCCEEDED, states.JOB_STATE_PARTIALLY_SUCCEEDED}:
        raise RuntimeError(
            f"Vertex stage failed: {job.name}, {states(job.state).name}; "
            "inspect Vertex logs before resubmitting"
        )
    if not prediction_table:
        raise RuntimeError("Vertex completed without a BigQuery prediction table")
    return json.dumps({"skipped": False, "prediction_table": prediction_table})


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["google-cloud-bigquery==3.31.0"],
)
def reconcile_texts(config: str, prediction: str, sql_template: str) -> str:
    import json
    import re

    from google.cloud import bigquery

    cfg = json.loads(config)
    result = json.loads(prediction)
    if result["skipped"]:
        return "reconciled"
    table = result["prediction_table"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+", table):
        raise ValueError("Unexpected Vertex output table identifier")
    client = bigquery.Client(project=cfg["project_id"], location=cfg["bigquery_location"])
    client.query(sql_template.format(prediction_table=table, **cfg)).result()
    return "reconciled"


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["google-cloud-bigquery==3.31.0"],
)
def publish_embeddings(config: str, reconciled: str, sql_template: str) -> str:
    import json

    from google.cloud import bigquery

    cfg = json.loads(config)
    if reconciled != "reconciled":
        raise ValueError("Prediction reconciliation must complete before publication")
    client = bigquery.Client(project=cfg["project_id"], location=cfg["bigquery_location"])
    joins = []
    fields = []
    failures = []
    for index, (text_column, embedding_column) in enumerate(zip(
        cfg["text_columns"], cfg["embedding_columns"]
    )):
        alias = f"e{index}"
        joins.append(
            f"LEFT JOIN `{cfg['embedding_table']}` AS {alias} "
            f"ON COLLATE(COALESCE(source.`{text_column}`, ''), '') = {alias}.text"
        )
        fields.append(f"{alias}.embedding AS `{embedding_column}`")
        failures.append(f"IF({alias}.text IS NULL, '{text_column}', NULL)")
    join_sql = "\n".join(joins)
    fields_sql = ", ".join(fields)
    failures_sql = ", ".join(failures)
    select_embeddings = ", ".join(f"`{name}`" for name in cfg["embedding_columns"])
    source_fields = ", ".join(
        "source_record.`" + name.replace("`", "\\`") + "`"
        for name in cfg["source_columns"]
    )
    output_columns = ", ".join(
        "`" + name.replace("`", "\\`") + "`"
        for name in cfg["source_columns"] + cfg["embedding_columns"]
    )
    unstaged = " OR ".join(
        f"COLLATE(COALESCE(source.`{column}`, ''), '') NOT IN "
        f"(SELECT COLLATE(JSON_VALUE(instances, '$.text'), '') FROM `{cfg['input_table']}`)"
        for column in cfg["text_columns"]
    )
    client.query(sql_template.format(
        embedding_fields=fields_sql,
        failed_fields=failures_sql,
        embedding_joins=join_sql,
        unstaged_texts=unstaged,
        embedding_columns=select_embeddings,
        source_fields=source_fields,
        output_columns=output_columns,
        attempts=cfg["retry_rounds"] + 1,
        **{key: value for key, value in cfg.items() if key != "embedding_columns"},
    )).result()
    if cfg["cleanup_intermediates"]:
        for table in (
            cfg["input_table"], cfg["embedding_table"],
            *(f"{cfg['input_table']}_retry_{attempt}" for attempt in range(1, 3)),
        ):
            client.delete_table(table, not_found_ok=True)
    return cfg["output_table"]


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["google-cloud-bigquery==3.31.0"],
)
def release_run(config: str = ""):
    import json

    from google.cloud import bigquery

    if not config:
        return
    cfg = json.loads(config)
    client = bigquery.Client(project=cfg["project_id"], location=cfg["bigquery_location"])
    lock = client.get_table(cfg["lock_table"])
    if lock.labels.get("embedding_run") != cfg["run_id"]:
        raise RuntimeError("Refusing to release another run's source lock")
    client.delete_table(lock)
