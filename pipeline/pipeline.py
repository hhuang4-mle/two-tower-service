from pathlib import Path
from typing import List

from kfp import dsl

from components import (
    initialize_run,
    predict_texts,
    prepare_texts,
    publish_embeddings,
    reconcile_texts,
    release_run,
)

SQL_DIRECTORY = Path(__file__).with_name("sql")


@dsl.pipeline(
    name="embeddinggemma-text-embedding",
    description="BigQuery text preparation, Vertex encoding, reconciliation and source mapping",
)
def embedding_pipeline(
    entity_type: str,
    source_table: str,
    text_columns: List[str],
    embedding_model_resource_name: str,
    bigquery_location: str,
    project_id: str = "gcp-wow-rwds-ai-super-plo-dev",
    vertex_location: str = "us-central1",
    service_account: str = "super-plo-dev@gcp-wow-rwds-ai-super-plo-dev.iam.gserviceaccount.com",
    request_batch_size: int = 32,
    replica_count: int = 4,
    retry_rounds: int = 2,
    cleanup_intermediates: bool = False,
):
    initialize = initialize_run(
        project_id=project_id,
        vertex_location=vertex_location,
        bigquery_location=bigquery_location,
        entity_type=entity_type,
        source_table=source_table,
        text_columns=text_columns,
        embedding_model_resource_name=embedding_model_resource_name,
        service_account=service_account,
        request_batch_size=request_batch_size,
        replica_count=replica_count,
        retry_rounds=retry_rounds,
        cleanup_intermediates=cleanup_intermediates,
    )
    initialize.set_caching_options(False)
    prepare = prepare_texts(
        config=initialize.output,
        sql_template=(SQL_DIRECTORY / "prepare.sql").read_text(),
    )
    prepare.set_caching_options(False)
    previous = ""
    for attempt in range(3):
        predict = predict_texts(
            config=prepare.output, attempt=attempt, previous=previous
        )
        predict.set_display_name(f"Encode texts: attempt {attempt + 1}")
        predict.set_caching_options(False)
        reconcile = reconcile_texts(
            config=prepare.output, prediction=predict.output,
            sql_template=(SQL_DIRECTORY / "reconcile.sql").read_text(),
        )
        reconcile.set_display_name(f"Reconcile texts: attempt {attempt + 1}")
        reconcile.set_caching_options(False)
        previous = reconcile.output
    publish = publish_embeddings(
        config=prepare.output, reconciled=previous,
        sql_template=(SQL_DIRECTORY / "publish.sql").read_text(),
    )
    publish.set_caching_options(False)
    release = release_run(config=initialize.output)
    release.after(publish).ignore_upstream_failure()
    release.set_caching_options(False)
